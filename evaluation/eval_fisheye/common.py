"""Shared Stage 2 fisheye eval utilities: model loading / data loading / fisheye unprojection.

Shared helpers imported by eval_fisheye.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

# Allow running `python -m evaluation.eval_fisheye.xxx` directly
try:
    from xlens.models import XLensNet
    from evaluation._support.datasets import (
        OmniSceneDataset, LoadOmniFrame, OmniResize,
        ColorAugmentation, Normalize, collate_omni_frames,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from xlens.models import XLensNet
    from evaluation._support.datasets import (
        OmniSceneDataset, LoadOmniFrame, OmniResize,
        ColorAugmentation, Normalize, collate_omni_frames,
    )

logger = logging.getLogger(__name__)

# Fallback defaults (from stage2_fisheye.yaml)
DEFAULT_DATA_ROOT = "/path/to/data/foundationstage2/test"
DEFAULT_LUT_DIR = "/path/to/data/texture/texture"
DEFAULT_CONFIG = str(
    Path(__file__).resolve().parents[2] / "configs" / "stage2_fisheye.yaml"
)
DEFAULT_IMAGE_HW = (504, 798)   # stage2_fisheye.yaml: image_size [[504, 798]]
DEFAULT_CHECKPOINT_DIR = "/path/to/DRVFM/checkpoints"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

def load_yaml_config(path: Optional[str]) -> Dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning(f"config yaml not found: {p}, ignoring")
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def pick(cli_val, yaml_cfg: Dict, key: str, default):
    """Priority: CLI (not None) > eval yaml (not None) > hardcoded default."""
    if cli_val is not None:
        return cli_val
    if key in yaml_cfg and yaml_cfg[key] is not None:
        return yaml_cfg[key]
    return default


def resolve_image_hw(yaml_cfg: Dict,
                     target_h: Optional[int],
                     target_w: Optional[int]) -> Tuple[int, int]:
    """Prefer CLI target_h/w, else first yaml image_size, else default 504x798."""
    if target_h and target_w:
        hw = (int(target_h), int(target_w))
    else:
        img_sizes = yaml_cfg.get("image_size")
        if img_sizes:
            first = img_sizes[0]
            hw = (int(first[0]), int(first[1]))
        else:
            hw = DEFAULT_IMAGE_HW
    if hw[0] % 14 != 0 or hw[1] % 14 != 0:
        raise ValueError(f"target_hw={hw} is not a multiple of 14 (DINOv2 patch requirement)")
    return hw


# ----------------------------------------------------------------------------
# Model loading (config in ckpt takes priority, yaml fallback)
# ----------------------------------------------------------------------------

def build_model_from_ckpt(
    ckpt_path: Path,
    config_yaml: Optional[Path],
    device: torch.device,
) -> Tuple[XLensNet, Dict]:
    """Build model + load weights. Returns (model, effective config dict).

    The released checkpoint is a bare .safetensors (no embedded arch config), so
    pass its arch via config_yaml (configs/xlens_vits.yaml). A .pth ckpt embeds
    its own config, which takes priority.
    """
    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt = {"model": load_file(str(ckpt_path))}   # weights-only; arch from config_yaml
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # config in ckpt matches weight shapes, so prefer it; yaml fallback
    cfg = ckpt.get("config", None)
    if cfg is None and config_yaml is not None:
        cfg = load_yaml_config(str(config_yaml))
    if not cfg:
        cfg = {
            "backbone": "vits",
            "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
            "head_features": 128,
            "head_out_channels": [128, 256, 384, 384],
            "predict_mask": True,
            "scale_head_mode": "attn_pool",
            "scale_head_num_queries": 4,
            "scale_head_num_heads": 6,
        }

    logger.info(
        f"model config backbone={cfg.get('backbone')} "
        f"scale_head_mode={cfg.get('scale_head_mode')} "
        f"predict_mask={cfg.get('predict_mask')} "
        f"use_calib_tokens={cfg.get('use_calib_tokens', False)} "
        f"n_cam_types={cfg.get('n_cam_types', 0)} "
        f"use_dwc={cfg.get('use_dwc', False)}"
    )

    # Pass through only the params XLensNet accepts (avoid dropping stage2/3-only keys like
    # calib_tokens / distortion_bias, which would otherwise cause missing/unexpected weight shapes)
    import inspect
    accepted = {k for k in inspect.signature(XLensNet.__init__).parameters
                if k != "self"}
    kwargs = {k: v for k, v in cfg.items() if k in accepted}
    if "backbone" in cfg and "backbone_name" not in kwargs:
        kwargs["backbone_name"] = cfg["backbone"]
    if "head_out_channels" in kwargs:
        kwargs["head_out_channels"] = tuple(kwargs["head_out_channels"])
    kwargs.setdefault("checkpoint_dir", cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
    kwargs["freeze_backbone"] = False     # eval calls .eval() immediately, freezing is moot

    model = XLensNet(**kwargs).to(device)

    info = model.load_state_dict(state_dict, strict=False)
    # unexpected keys almost always mean architecture mismatch (e.g. use_dwc / calib_tokens)
    if info.missing_keys:
        logger.warning(f"missing {len(info.missing_keys)} keys (first 5: {info.missing_keys[:5]})")
    if info.unexpected_keys:
        logger.error(f"unexpected {len(info.unexpected_keys)} keys (first 5: "
                     f"{info.unexpected_keys[:5]}) - check config matches the ckpt's training config")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model loaded, {n_params / 1e6:.1f}M params, "
                f"global_step={ckpt.get('global_step', '?')}")
    model.eval()
    return model, (cfg or {})


def read_global_step(ckpt_path: Path) -> object:
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        gs = raw.get("global_step", "unknown")
        del raw
        return gs
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------

def build_val_pipeline(target_hw: Tuple[int, int]) -> List:
    """Same as training val (no color aug)."""
    return [
        LoadOmniFrame(),
        OmniResize(target_size=target_hw, patch_size=14),
        ColorAugmentation(enabled=False),
        Normalize(),
    ]


def list_test_scenes(data_root: Path) -> List[str]:
    """All scene dirs under data_root containing common/ (or one level nested)."""
    scenes = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "common").exists() or (d / d.name / "common").exists():
            scenes.append(d.name)
    return scenes


# Fisheye-only eval uses exactly these 4 fisheye cameras. Some scenes mix pinhole cameras
# (CAM_Front / CAM_Back) into rgb/; without an explicit filter the loader would treat them as
# cameras, look for a nonexistent pinhole LUT, and skip the whole scene. Hardcode 4 fisheye,
# ignore extra cameras.
FISHEYE_CAMS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")


def build_scene_dataset(
    data_root: Path,
    lut_dir: Path,
    scene_name: str,
    target_hw: Tuple[int, int],
    image_format: str = "jpg",
    cameras_filter: Optional[Sequence[str]] = FISHEYE_CAMS,
) -> OmniSceneDataset:
    pipeline = build_val_pipeline(target_hw)
    return OmniSceneDataset(
        scene_dir=str(data_root / scene_name),
        lut_dir=str(lut_dir),
        pipeline=pipeline,
        image_format=image_format,
        view_sampling="single_frame",
        num_views=4,
        cameras_filter=cameras_filter,   # keep only 4 fisheye, ignore mixed-in pinhole
        use_cleaned_valid_mask=True,   # auto-enabled if the test set has valid_mask/
    )


def load_one_sample(
    data_root: Path,
    lut_dir: Path,
    scene_name: str,
    target_hw: Tuple[int, int],
    frame_name: Optional[str] = None,
    frame_idx: int = 0,
    image_format: str = "jpg",
) -> Tuple[Dict, List[str], str]:
    """Get a single-sample (B=1, S=4) collate batch. Returns (batch, cam_short_names, frame_used)."""
    ds = build_scene_dataset(data_root, lut_dir, scene_name, target_hw, image_format)
    if frame_name is None:
        if frame_idx >= len(ds.frame_names):
            raise IndexError(f"frame_idx={frame_idx} out of range, scene has only {len(ds.frame_names)} frames")
        frame_name = ds.frame_names[frame_idx]
    if frame_name not in ds.frame_names:
        raise KeyError(f"scene {scene_name} has no frame {frame_name}. First 5: {ds.frame_names[:5]}")
    actual_idx = ds.frame_names.index(frame_name)
    # Single-resolution pipeline: pass an int (a tuple triggers the retry-fallback idx+retry TypeError)
    sample = ds[actual_idx]
    if sample is None:
        raise RuntimeError(f"sampling failed: {scene_name}/{frame_name}")
    batch = collate_omni_frames([sample])
    return batch, ds.cameras_name, frame_name


# ----------------------------------------------------------------------------
# Ray-map input (matches the training ray-map convention)
# ----------------------------------------------------------------------------

def build_ray_map_input(batch: Dict, mode: str, device: torch.device):
    """mode: full (intrinsics+extrinsics) / icam (intrinsics only) / none. Eval default full."""
    if mode == "none":
        return None, None
    d_cam = batch["d_cam"].to(device)                       # (B, S, 3, H, W)
    if mode == "full":
        return batch["ray_map"].to(device), d_cam
    if mode == "icam":
        zeros_t = torch.zeros_like(d_cam)
        return torch.cat([d_cam, zeros_t], dim=2), d_cam     # (B, S, 6, H, W)
    raise ValueError(f"unknown ray_mode={mode}")


# ----------------------------------------------------------------------------
# Image de-normalization / fisheye unprojection
# ----------------------------------------------------------------------------

def img_norm_to_rgb_uint8(img_chw: np.ndarray) -> np.ndarray:
    """ImageNet-normalized (3, H, W) -> uint8 RGB (H, W, 3)."""
    img = img_chw.transpose(1, 2, 0).astype(np.float32)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


def unproject_to_view0(
    depth: np.ndarray,                  # (S, H, W) meters
    d_cam: np.ndarray,                  # (S, 3, H, W) camera-frame unit ray (OpenCV)
    extrinsics_c2w: np.ndarray,         # (S, 4, 4) c2w (view0=I)
    depth_mode: str = "z_depth",
) -> np.ndarray:
    """Fisheye unprojection to view0 frame. Returns (S, H, W, 3).

    z_depth (default/correct, matches training loss): pt = (depth/|d_cam.z|) * d_cam_unit
    radial: pt = depth * d_cam_unit (sanity only)
    """
    S, H, W = depth.shape
    out = np.empty((S, H, W, 3), dtype=np.float32)
    for s in range(S):
        d = d_cam[s].transpose(1, 2, 0).astype(np.float64)      # (H, W, 3) OpenCV
        z = depth[s].astype(np.float64)
        if depth_mode == "z_depth":
            dz_abs = np.abs(d[..., 2])
            scale = np.where(dz_abs > 1e-3, z / np.maximum(dz_abs, 1e-3), 0.0)
            pts_cam = scale[..., None] * d
        elif depth_mode == "radial":
            pts_cam = z[..., None] * d
        else:
            raise ValueError(f"unknown depth_mode={depth_mode}")
        c2w = extrinsics_c2w[s].astype(np.float64)
        R, t = c2w[:3, :3], c2w[:3, 3]
        out[s] = (pts_cam @ R.T + t).astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: XLensNet,
    batch: Dict,
    device: torch.device,
    ray_mode: str = "full",
    amp_dtype: str = "bf16",
) -> Dict[str, object]:
    """Run one batch, return numpy (with batch dim). Works for single or multiple samples."""
    images = batch["images"].to(device)
    ray_map, d_cam = build_ray_map_input(batch, ray_mode, device)

    use_amp = (device.type == "cuda")
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    cam_types = batch.get("cam_types")
    if cam_types is not None:
        cam_types = cam_types.to(device)
    with torch.autocast("cuda", enabled=use_amp, dtype=dtype):
        out = model(images, ray_map=ray_map, d_cam=d_cam, cam_types=cam_types)

    res: Dict[str, object] = {
        "pred_depth_metric": out["depth_metric"].float().cpu().numpy(),   # (B, S, H, W)
        "metric_scaling_factor": out["metric_scaling_factor"].float().cpu().numpy(),  # (B,)
    }
    if "depth_conf" in out:
        res["pred_conf"] = out["depth_conf"].float().cpu().numpy()
    if "mask" in out:
        res["pred_mask"] = out["mask"].float().cpu().numpy()
    elif "mask_logits" in out:
        res["pred_mask"] = torch.sigmoid(out["mask_logits"]).float().cpu().numpy()
    return res


# ----------------------------------------------------------------------------
# Valid-pixel mask (shared by depth eval / unprojection)
# ----------------------------------------------------------------------------

def build_valid_mask(
    gt_depth: np.ndarray,          # (S, H, W)
    view_mask: np.ndarray,         # (S,) bool
    min_depth: float,
    max_depth: float,
    nam: Optional[np.ndarray] = None,        # (S, H, W) lens / non-ambiguous mask
    has_nam: Optional[np.ndarray] = None,    # (S,) bool
    use_lens_mask: bool = True,
) -> np.ndarray:
    """Return (S, H, W) bool: GT depth finite and in (min,max], view valid, inside lens."""
    valid = (gt_depth > min_depth) & (gt_depth < max_depth) & np.isfinite(gt_depth)
    valid &= view_mask[:, None, None].astype(bool)
    if use_lens_mask and nam is not None and has_nam is not None:
        ones = np.ones_like(nam, dtype=bool)
        lens = np.where(has_nam[:, None, None].astype(bool), nam.astype(bool), ones)
        valid &= lens
    return valid
