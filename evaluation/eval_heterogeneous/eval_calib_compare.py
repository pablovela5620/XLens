#!/usr/bin/env python3
"""X-Lens heterogeneous 6-camera (4 fisheye + 2 pinhole) calibrated depth evaluation.

  * Data: foundationstage3/test, 6 cameras per scene (CAM_A/B/C/D fisheye cam_type=0
    via LUT; CAM_Front/CAM_Back pinhole cam_type=1, d_cam from K, no LUT).
  * The model is given GT intrinsics and extrinsics:
      - X-Lens: ray_map (= d_world + camera center t, view0-canonical) + d_cam +
                cam_types. ray_map encodes intrinsics (per-pixel ray direction)
                and extrinsics (rotation into d_world, translation into t), so
                ray_mode=full supplies both.
  * Mask: lens_mask + sky_mask both apply. OmniSceneDataset(use_sky_mask=True) folds
    sky into non_ambiguous_mask; valid pixel = GT depth in (min,max] & finite & view
    valid & non_ambiguous (lens & ~sky).
  * Metrics (relative depth: each frame aligns pred to GT by a median ratio, scale-free):
      - scale_absrel : |mean(pred_metric) - mean(gt)| / mean(gt)  (metric scale error, unaligned)
      - depth_absrel : mean(|pred_rel - gt| / gt)
      - rmse         : sqrt(mean((pred_rel - gt)^2))
      - tau (d<1.03) : mean( max(pred_rel/gt, gt/pred_rel) < 1.03 )  inlier ratio @1.03
      - delta1(1.25) : mean( max(pred_rel/gt, gt/pred_rel) < 1.25 )
    Alignment scale is estimated per view (median ratio), accumulated separately for
    fisheye / pinhole / overall so one failing camera type does not contaminate the
    other's alignment.

Usage:
    # released checkpoint (.safetensors): pass the shipped arch config
    python -m evaluation.eval_heterogeneous.eval_calib_compare \
        --stage3_ckpt /path/to/model.safetensors \
        --config configs/xlens_vits.yaml \
        --frames_per_scene 8 --seqs_per_scene 2 \
        --out_dir ./eval_heterogeneous_calib_out

    # a .pth ckpt embeds its own arch, so --config is optional; a bare
    # .safetensors without --config falls back to weight-inferred arch.
    python -m ...eval_calib_compare --stage3_ckpt .../best_model.pth --max_scenes 2 --frames_per_scene 2
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from xlens.models import XLensNet
    from evaluation._support.datasets import (
        OmniSceneDataset, collate_omni_frames,
        LoadOmniFrame, OmniResize, ColorAugmentation, Normalize,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from xlens.models import XLensNet
    from evaluation._support.datasets import (
        OmniSceneDataset, collate_omni_frames,
        LoadOmniFrame, OmniResize, ColorAugmentation, Normalize,
    )

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_heterogeneous_calib")

# ----------------------------------------------------------------------------
# Default paths / constants
# ----------------------------------------------------------------------------
DEFAULT_DATA = "/path/to/data/foundationstage3/test"
DEFAULT_LUT = "/path/to/data/texture/texture"
HW = (504, 798)
PINHOLE_CAMS = ("CAM_Front", "CAM_Back")
FISHEYE_CAMS = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")   # 4 fisheye (--fisheye_only)
TAU_THR = 1.03      # inlier ratio threshold
DELTA1_THR = 1.25   # delta1 threshold
MIN_DEPTH = 0.05
MAX_DEPTH = 80.0


# ============================================================================
# Model loading (mirrors the config->kwargs mapping in engine/trainer.py)
# ============================================================================
def _infer_arch_from_state(state: Dict) -> Dict:
    """Infer the architecture from the weights, so a bare .safetensors (no embedded
    config) loads correctly. cfg (embedded in a .pth, or a user --config) overrides."""
    a: Dict = {}
    qkv = next((v for k, v in state.items() if k.endswith("blocks.0.attn.qkv.weight")), None)
    if qkv is not None:
        a["backbone"] = {384: "vits", 768: "vitb", 1024: "vitl", 1536: "vitg"}.get(int(qkv.shape[0]) // 3, "vits")
    hp = state.get("head.projects.0.weight")
    if hp is not None:
        a["head_features"] = int(hp.shape[0])
    a["scale_head_mode"] = "attn_pool" if any("scale_head.attn_layers" in k for k in state) else "cls"
    # depth-head output channels: 2 = depth+conf (predict_mask off), 3 = depth+conf+mask (on)
    oc = state.get("head.scratch.output_conv2.2.weight")
    if oc is not None:
        a["predict_mask"] = int(oc.shape[0]) >= 3
    tok = state.get("backbone.pretrained.calib_tokens.tokens")
    if tok is not None:                                    # (L, n_cam_types, K, dim)
        a["use_calib_tokens"] = True
        a["n_cam_types"] = int(tok.shape[1])
        a["calib_tokens_per_type"] = int(tok.shape[2])
    else:
        a["use_calib_tokens"] = False
    a["use_distortion_bias"] = any("distortion_bias" in k for k in state)
    db = state.get("backbone.pretrained.distortion_bias.mlp.0.weight")
    if db is not None:
        a["distortion_bias_hidden_dim"] = int(db.shape[0])
    return a


def build_stage3_model(ckpt_path: Path, device: torch.device,
                       config: Optional[Dict] = None) -> Tuple[XLensNet, Dict]:
    """Build + load the model.

    Arch resolution priority: user ``config`` (e.g. configs/xlens_vits.yaml) >
    config embedded in a .pth ckpt > weight-inference (``_infer_arch_from_state``,
    the fallback for a bare .safetensors with no config).
    For the released .safetensors, pass ``--config configs/xlens_vits.yaml``.
    """
    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt = {"model": load_file(str(ckpt_path)), "config": {}}   # weights-only; arch inferred below
    else:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    cfg = ckpt.get("config", {}) or {}
    if config:                                            # user --config overrides the embedded cfg
        cfg = {**cfg, **config}
    arch = _infer_arch_from_state(state_dict)              # weight-inferred defaults (bare safetensors)

    def g(key, default):                                  # cfg wins, then weight-inferred, then hardcoded
        return cfg.get(key, arch.get(key, default))

    calib_inject_types = tuple(cfg.get("calib_token_inject_types", [0]))
    kwargs = dict(
        backbone_name=g("backbone", "vits"),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints"),
        head_features=g("head_features", 128),
        head_out_channels=tuple(cfg.get("head_out_channels", [128, 256, 384, 384])),
        freeze_backbone=False,
        predict_mask=g("predict_mask", False),
        scale_head_mode=g("scale_head_mode", "attn_pool"),
        scale_head_num_queries=cfg.get("scale_head_num_queries", 4),
        scale_head_num_heads=cfg.get("scale_head_num_heads", 6),
        n_cam_types=g("n_cam_types", 2),
        use_calib_tokens=g("use_calib_tokens", False),
        calib_tokens_per_type=g("calib_tokens_per_type", 4),
        calib_inject_types=calib_inject_types,
        use_distortion_bias=g("use_distortion_bias", False),
        distortion_bias_layers=cfg.get("distortion_bias_layers", "global_only"),
        distortion_bias_hidden_dim=g("distortion_bias_hidden_dim", 64),
        distortion_bias_chunk_size=cfg.get("distortion_bias_chunk_size", 1024),
        distortion_bias_only=cfg.get("distortion_bias_only", False),
    )
    if "use_dwc" in cfg:
        kwargs["use_dwc"] = cfg["use_dwc"]
        if cfg["use_dwc"]:
            kwargs["dwc_kernel_size"] = cfg.get("dwc_kernel_size", 3)

    try:
        model = XLensNet(**kwargs).to(device)
    except TypeError:
        kwargs.pop("use_dwc", None); kwargs.pop("dwc_kernel_size", None)
        model = XLensNet(**kwargs).to(device)

    info = model.load_state_dict(state_dict, strict=False)
    miss = [k for k in info.missing_keys if "backbone" not in k]
    if miss:
        logger.warning(f"[stage3] missing {len(info.missing_keys)} keys (non-backbone e.g. {miss[:5]})")
    if info.unexpected_keys:
        logger.warning(f"[stage3] unexpected {len(info.unexpected_keys)} keys (e.g. {info.unexpected_keys[:5]})")
    # calib_tokens injected for fisheye only (matches training; re-set defensively)
    ci = getattr(getattr(model.backbone, "pretrained", None), "calib_tokens", None)
    if ci is not None and hasattr(ci, "inject_types"):
        ci.inject_types = calib_inject_types
    model.eval()
    gs = ckpt.get("global_step", "?")
    logger.info(f"[stage3] loaded step={gs}  n_cam_types={kwargs['n_cam_types']} "
                f"calib_tokens={kwargs['use_calib_tokens']} dist_bias={kwargs['use_distortion_bias']}")
    return model, {"global_step": gs, "config": cfg}


def build_val_pipeline():
    return [LoadOmniFrame(), OmniResize(target_size=HW, patch_size=14),
            ColorAugmentation(enabled=False), Normalize()]


# ============================================================================
# Dump: depth maps + synthesized point clouds (gt + each X-Lens ckpt)
# ============================================================================
def _lens_sky_mask(batch):
    """Extract the lens & ~sky mask (S,H,W bool) from the batch, matching build_valid.
    non_ambiguous_mask = lens & ~sky (sky folded in when OmniSceneDataset
    use_sky_mask=True); views with has_non_ambiguous_mask=False default to all True."""
    nam = batch.get("non_ambiguous_mask")
    has = batch.get("has_non_ambiguous_mask")
    if nam is None or has is None:
        return None
    nam = nam[0].bool().numpy()                       # (S,H,W)
    has = has[0].bool().numpy()                       # (S,)
    return np.where(has[:, None, None], nam, np.ones_like(nam))


def load_lens_masks(scene_dir, cam_names, H, W):
    """Read per-view GT lens masks: <scene_dir>/mask/<cam>_mask.png (>0 = inside lens).
    Pinhole / missing-file views default to all True. Return (S,H,W) bool or None
    (if none found)."""
    import os
    import cv2
    masks, got = [], False
    for cam in cam_names:
        mp = os.path.join(str(scene_dir), "mask", f"{cam}_mask.png")
        if os.path.isfile(mp):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                masks.append(m > 0); got = True; continue
        masks.append(np.ones((H, W), dtype=bool))
    return np.stack(masks, 0) if got else None


def dump_frame_set(dump_dir, frame_key, gt, frame_preds, batch, valid, subsample,
                   frame_masks=None, frame_confs=None, conf_drop_pct=None,
                   lens_masks=None, fov_deg=None, save_pcd=True):
    """Dump gt + each X-Lens prediction as depth maps + synthesized point clouds.
    Depth maps / point clouds use the GT lens mask (lens_masks) to remove outside-lens
    regions; each X-Lens pred point cloud additionally applies the model's own predicted
    sky mask and confidence cutoff; GT uses the eval valid mask."""
    from evaluation.dump_utils import dump_frame
    dcam = batch["d_cam"][0].numpy()
    c2w = batch["extrinsics_world"][0].numpy()
    imgs = batch["images"][0].numpy()
    vm = batch["view_mask"][0].numpy().astype(bool)
    lens_sky = _lens_sky_mask(batch)                  # fallback when model has no mask head
    frame_masks = frame_masks or {}
    frame_confs = frame_confs or {}
    fd = dump_dir / frame_key
    dump_frame(fd, "gt", gt, dcam, c2w, imgs, view_mask=vm,
               pcd_mask_shw=valid, lens_shw=lens_masks, max_depth=MAX_DEPTH, subsample=subsample,
               fov_deg=fov_deg, save_pcd=save_pcd)
    for lbl, pred in frame_preds.items():
        pm = frame_masks.get(lbl)
        pm = pm if pm is not None else lens_sky       # model's predicted mask, else dataset mask
        dump_frame(fd, lbl, pred, dcam, c2w, imgs, view_mask=vm,
                   pcd_mask_shw=pm, lens_shw=lens_masks, conf_shw=frame_confs.get(lbl),
                   conf_drop_pct=conf_drop_pct, max_depth=MAX_DEPTH, subsample=subsample,
                   fov_deg=fov_deg, save_pcd=save_pcd)


# ============================================================================
# Multi-ckpt parsing (evaluate several stage3 weights; data read once)
# ============================================================================
def default_ckpt_label(p) -> str:
    """Derive a short label from a ckpt path: '<parent dir>_<stem>'."""
    p = Path(p)
    return f"{p.parent.name}_{p.stem}"


def parse_ckpt_specs(items: List[str]) -> List[Tuple[str, Path]]:
    """Parse ['label=path', 'path', ...] into [(label, Path), ...] with unique labels.

    Each item is 'label=/abs/path.pth' (explicit label) or a bare path (label derived).
    Split on the first '=' (paths do not contain '=').
    """
    parsed: List[Tuple[str, Path]] = []
    for it in items:
        if "=" in it:
            lbl, path = it.split("=", 1)
            lbl = lbl.strip()
        else:
            path = it
            lbl = default_ckpt_label(path)
        parsed.append((lbl, Path(path.strip())))
    # de-duplicate labels (append #n on collision)
    seen: Dict[str, int] = {}
    out: List[Tuple[str, Path]] = []
    for lbl, p in parsed:
        if lbl in seen:
            seen[lbl] += 1
            lbl = f"{lbl}#{seen[lbl]}"
        else:
            seen[lbl] = 0
        out.append((lbl, p))
    return out


# ============================================================================
# Relative-depth metric accumulator
# ============================================================================
class RelDepthAccumulator:
    """Streaming accumulation of aligned relative depth. Stores sums / counts only."""

    def __init__(self):
        self.n = 0
        self.sum_absrel = 0.0
        self.sum_sq = 0.0          # (pred_rel - gt)^2
        self.n_tau = 0             # max(r,1/r) < 1.03
        self.n_d1 = 0              # < 1.25
        # scale: per-frame |mean(pred_metric)-mean(gt)|/mean(gt), averaged over frames
        self.scale_absrel_sum = 0.0
        self.scale_frames = 0

    def update_rel(self, pred_rel: np.ndarray, gt: np.ndarray):
        """pred_rel / gt: (M,) aligned relative-depth valid pixels."""
        if pred_rel.size == 0:
            return
        pred_rel = np.maximum(pred_rel.astype(np.float64), 1e-6)
        gt = np.maximum(gt.astype(np.float64), 1e-6)
        diff = pred_rel - gt
        ratio = np.maximum(pred_rel / gt, gt / pred_rel)
        self.n += pred_rel.size
        self.sum_absrel += float((np.abs(diff) / gt).sum())
        self.sum_sq += float((diff * diff).sum())
        self.n_tau += int((ratio < TAU_THR).sum())
        self.n_d1 += int((ratio < DELTA1_THR).sum())

    def update_scale(self, pred_metric: np.ndarray, gt: np.ndarray):
        """Per-frame metric scale error: |mean(pred)-mean(gt)|/mean(gt), over the
        frame-pooled pixels of that view type. metric_scaling_factor is a per-frame
        global scalar, so scale must be measured at frame level; per-view would be
        inflated by viewing-direction differences."""
        if pred_metric.size == 0:
            return
        mg = float(np.mean(gt))
        mp = float(np.mean(pred_metric))
        if mg > 1e-6:
            self.scale_absrel_sum += abs(mp - mg) / mg
            self.scale_frames += 1

    def result(self) -> Dict[str, float]:
        if self.n == 0:
            return {k: float("nan") for k in
                    ["scale_absrel", "depth_absrel", "rmse", "tau_1.03", "delta1_1.25", "n_pix", "n_frames"]}
        return {
            "scale_absrel": (self.scale_absrel_sum / self.scale_frames) if self.scale_frames else float("nan"),
            "depth_absrel": self.sum_absrel / self.n,
            "rmse": math.sqrt(self.sum_sq / self.n),
            "tau_1.03": self.n_tau / self.n,
            "delta1_1.25": self.n_d1 / self.n,
            "n_pix": self.n,
            "n_frames": self.scale_frames,
        }


class ModelAccumulators:
    """One per model: fisheye / pinhole / overall groups."""
    def __init__(self):
        self.fisheye = RelDepthAccumulator()
        self.pinhole = RelDepthAccumulator()
        self.overall = RelDepthAccumulator()

    def result(self):
        return {"fisheye": self.fisheye.result(),
                "pinhole": self.pinhole.result(),
                "overall": self.overall.result()}


def accumulate_frame(acc: ModelAccumulators, pred_metric, gt, valid, cam_types):
    """One frame (S,H,W) of pred_metric/gt/valid + cam_types(S,). Relative depth is
    aligned per view by median ratio (not frame-pooled), so a failing camera type does
    not contaminate the other's alignment; this is the standard scale-invariant
    monocular convention. The scale metric uses unaligned metric depth.
    """
    S = pred_metric.shape[0]
    # relative depth: accumulate after per-view median alignment; scale: once per frame after pooling
    pools = {"fisheye": ([], []), "pinhole": ([], [])}
    for s in range(S):
        m = valid[s]
        if not m.any():
            continue
        pm = pred_metric[s][m]; g = gt[s][m]
        med_p = np.median(pm); med_g = np.median(g)
        scale = (med_g / med_p) if med_p > 1e-8 else 1.0
        pred_rel = pm * scale
        key = "pinhole" if int(cam_types[s]) == 1 else "fisheye"
        bucket = acc.pinhole if key == "pinhole" else acc.fisheye
        bucket.update_rel(pred_rel, g)
        acc.overall.update_rel(pred_rel, g)
        pools[key][0].append(pm); pools[key][1].append(g)
    # one scale error per frame per type / overall (fit of the global metric scale on that domain)
    pm_all, gt_all = [], []
    for key, accb in (("fisheye", acc.fisheye), ("pinhole", acc.pinhole)):
        if pools[key][0]:
            p = np.concatenate(pools[key][0]); g = np.concatenate(pools[key][1])
            accb.update_scale(p, g); pm_all.append(p); gt_all.append(g)
    if pm_all:
        acc.overall.update_scale(np.concatenate(pm_all), np.concatenate(gt_all))


# ============================================================================
# Inference: X-Lens (given GT intrinsics + extrinsics)
# ============================================================================
@torch.no_grad()
def infer_stage3(model, batch, device, amp_dtype=torch.bfloat16, return_mask=False):
    images = batch["images"].to(device)
    ray_map = batch["ray_map"].to(device)
    d_cam = batch["d_cam"].to(device)
    cam_types = batch["cam_types"].to(device)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = model(images, ray_map=ray_map, d_cam=d_cam, cam_types=cam_types)
    depth = out["depth_metric"].float().cpu().numpy()   # (B,S,H,W) m
    if not return_mask:
        return depth
    # model's predicted non-ambiguous mask: sigmoid(mask_logits)>0.5 = non-sky/valid
    ml = out.get("mask_logits")
    mask = (torch.sigmoid(ml.float()) > 0.5).cpu().numpy() if ml is not None else None
    # depth confidence (B,S,H,W) for point-cloud cutoff on dump; None if ckpt has no conf head
    dc = out.get("depth_conf")
    conf = dc.float().cpu().numpy() if dc is not None else None
    return depth, mask, conf


# ============================================================================
# Valid-pixel mask (lens + sky both apply)
# ============================================================================
def build_valid(batch, device) -> np.ndarray:
    gt = batch["depths"].to(device)                               # (B,S,H,W), 0/inf=invalid
    view_mask = batch["view_mask"].to(device)                    # (B,S)
    nam = batch.get("non_ambiguous_mask")                       # (B,S,H,W) lens & ~sky
    has_nam = batch.get("has_non_ambiguous_mask")              # (B,S)
    valid = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & torch.isfinite(gt)
    valid &= view_mask[..., None, None]
    if nam is not None and has_nam is not None:
        nam = nam.to(device); has_nam = has_nam.to(device)
        ones = torch.ones_like(nam, dtype=torch.bool)
        lens = torch.where(has_nam[..., None, None], nam.bool(), ones)
        valid &= lens
    return valid.cpu().numpy()


# ============================================================================
# Scene list (keep N sequences per unique scene)
# ============================================================================
def list_scenes(data_root: Path, seqs_per_scene: int, max_scenes: int) -> List[str]:
    all_dirs = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if seqs_per_scene:
        groups: Dict[str, List[str]] = {}
        for d in all_dirs:
            base = re.sub(r"_\d+_\d+_\d+$", "", d)
            groups.setdefault(base, []).append(d)
        scenes = []
        for base in sorted(groups):
            scenes.extend(sorted(groups[base])[:seqs_per_scene])
    else:
        scenes = all_dirs
    if max_scenes:
        scenes = scenes[:max_scenes]
    return scenes


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3_ckpt", type=Path, default=None,
                    help="single ckpt (legacy). For multiple, use --stage3_ckpts.")
    ap.add_argument("--config", type=Path, default=None,
                    help="model arch config yaml; overrides weight-inference. "
                         "For the released .safetensors pass configs/xlens_vits.yaml.")
    ap.add_argument("--stage3_ckpts", nargs="+", default=None,
                    help="evaluate several stage3 ckpts; each 'label=path' or bare 'path'. "
                         "Data loading runs once, shared across ckpts.")
    ap.add_argument("--data_root", default=DEFAULT_DATA, type=Path)
    ap.add_argument("--lut_dir", default=DEFAULT_LUT, type=Path)
    ap.add_argument("--num_views", type=int, default=6)
    ap.add_argument("--fisheye_only", action="store_true",
                    help="evaluate the 4 fisheye cameras (CAM_A/B/C/D) only, excluding "
                         "pinhole CAM_Front/CAM_Back. Forces num_views=4, cameras_filter=4 "
                         "fisheye, pinhole_cams empty.")
    ap.add_argument("--num_workers", type=int, default=8,
                    help="DataLoader prefetch workers per scene (0=synchronous main process). "
                         "Increase when data loading is the bottleneck to overlap 6-camera "
                         "jpg/depth/mask reads with GPU forward.")
    ap.add_argument("--frames_per_scene", type=int, default=8)
    ap.add_argument("--seqs_per_scene", type=int, default=2, help="sequences kept per unique scene, 0=all")
    ap.add_argument("--max_scenes", type=int, default=0, help="0=all")
    ap.add_argument("--out_dir", default="./eval_heterogeneous_calib_out", type=Path)
    ap.add_argument("--dump_dir", type=Path, default=None,
                    help="root dir for dumping depth maps (.npy+.png) + point clouds (.ply). "
                         "Omit to skip dumping. Per frame: gt + each X-Lens ckpt, "
                         "multi-view combined into one n-camera point cloud.")
    ap.add_argument("--dump_max_frames", type=int, default=0,
                    help="max frames dumped per scene (0=all evaluated frames)")
    ap.add_argument("--dump_subsample", type=int, default=1,
                    help="point-cloud pixel subsample stride (1=keep all; larger=smaller ply)")
    ap.add_argument("--dump_conf_drop_pct", type=float, default=20.0,
                    help="X-Lens point-cloud confidence cutoff: drop the lowest-confidence percent "
                         "per view (default 20=keep top 80%; 0=no cutoff).")
    ap.add_argument("--dump_fov_deg", type=float, default=0.0,
                    help="point-cloud FoV cone crop (half-angle degrees): keep only rays within "
                         "this angle of the optical axis (dz=|d_cam.z|>=cos(fov_deg)); trims the "
                         "unreliable fisheye periphery. 0=no crop. Applies to every dumped cloud.")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="restrict to these exact scene dir names (overrides seqs_per_scene/"
                         "max_scenes discovery); missing dirs are skipped with a warning.")
    ap.add_argument("--dump_depth_only", action="store_true",
                    help="dump depth maps (.npy+.png) only, skip the fused point cloud (.ply)")
    a = ap.parse_args()

    # ---- Optional arch config (overrides weight-inference) ----
    arch_cfg = None
    if a.config is not None:
        import yaml
        with open(a.config, "r") as f:
            arch_cfg = yaml.safe_load(f) or {}
        logger.info(f"[stage3] using arch config {a.config}")

    # ---- Resolve ckpt list (one or more) ----
    if a.stage3_ckpts:
        ckpt_specs = parse_ckpt_specs(a.stage3_ckpts)
    elif a.stage3_ckpt is not None:
        ckpt_specs = [(default_ckpt_label(a.stage3_ckpt), a.stage3_ckpt)]
    else:
        ap.error("--stage3_ckpt or --stage3_ckpts required")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA unavailable, slow")

    if a.scenes:
        scenes = [s for s in a.scenes if (a.data_root / s).is_dir()]
        missing = [s for s in a.scenes if s not in scenes]
        if missing:
            logger.warning(f"--scenes: {len(missing)} not found under {a.data_root}, skipping: {missing}")
    else:
        scenes = list_scenes(a.data_root, a.seqs_per_scene, a.max_scenes)
    logger.info(f"{len(scenes)} scene dirs, {a.frames_per_scene} frames each, "
                f"{len(ckpt_specs)} X-Lens ckpt(s)")

    # ---- Build models (all resident; vits is small, data read once) ----
    stage3_models: List[Tuple[str, XLensNet, Dict]] = []
    for lbl, ckpt in ckpt_specs:
        logger.info(f"[stage3] loading {lbl} <- {ckpt}")
        m, meta = build_stage3_model(ckpt, device, config=arch_cfg)
        meta["ckpt"] = str(ckpt)
        stage3_models.append((lbl, m, meta))

    pipeline = build_val_pipeline()
    labels = [lbl for lbl, _, _ in stage3_models]
    acc_s3 = {lbl: ModelAccumulators() for lbl in labels}   # one per ckpt
    per_scene = {}
    n_frames_done = 0
    # FPS timing (per frame = one 6-camera forward, skipping warmup); one per ckpt
    timer = {lbl: [0.0, 0] for lbl in labels}   # [total sec, frames]
    warmup_left = 1
    t0 = time.time()

    def timed(fn):
        """Run one inference and return (result, seconds). CUDA sync for accurate timing."""
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = time.time()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out, time.time() - ts

    # fisheye-only: keep CAM_A/B/C/D, drop pinhole; single-frame 4 cameras -> num_views=4
    cameras_filter = list(FISHEYE_CAMS) if a.fisheye_only else None
    pinhole_cams = () if a.fisheye_only else PINHOLE_CAMS
    num_views = 4 if a.fisheye_only else a.num_views

    for si, scene in enumerate(scenes):
        try:
            ds = OmniSceneDataset(
                scene_dir=str(a.data_root / scene), lut_dir=str(a.lut_dir),
                pipeline=pipeline, image_format="jpg", view_sampling="single_frame",
                num_views=num_views, pinhole_cams=pinhole_cams,
                cameras_filter=cameras_filter,
                use_cleaned_valid_mask=True, use_sky_mask=True)
        except Exception as e:
            logger.warning(f"  [skip] {scene}: dataset build failed {e}")
            continue
        n = len(ds)
        if n == 0:
            continue
        stride = max(1, n // a.frames_per_scene)
        idxs = list(range(0, n, stride))[:a.frames_per_scene]

        s3_scene = {lbl: ModelAccumulators() for lbl in labels}
        far_pix = tot_pix = 0
        p95_list = []
        scene_dumped = 0
        # Multi-worker prefetch: wrap the sampled idxs in a Subset DataLoader so 6-camera
        # reads overlap the main-process GPU forward (batch_size=1 keeps single-frame inference).
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idxs),
            batch_size=1, shuffle=False, num_workers=a.num_workers,
            pin_memory=False, collate_fn=collate_omni_frames,
        )
        for fi, batch in enumerate(loader):
            idx = idxs[fi]
            valid = build_valid(batch, device)[0]                 # (S,H,W)
            gt = batch["depths"].numpy()[0]                       # (S,H,W)
            cam_types = batch["cam_types"].numpy()[0]             # (S,)
            # Scene-scale proxy: GT depth spread (p95) + far-field fraction. maxD (80m)
            # saturates, so p95 of valid depth represents openness.
            gv = gt[valid]
            if gv.size:
                tot_pix += gv.size
                far_pix += int((gv > 30.0).sum())
                p95_list.append(float(np.percentile(gv, 95)))

            warm = warmup_left > 0
            # Fetch model masks only when dumping (metrics are unaffected, still use dataset GT mask)
            do_dump = a.dump_dir is not None and (a.dump_max_frames == 0 or scene_dumped < a.dump_max_frames)
            frame_preds, frame_masks, frame_confs = {}, {}, {}

            # ============ Multi-view path (X-Lens, all cameras jointly) ============
            for lbl, model, _ in stage3_models:
                if do_dump:
                    out, dt = timed(lambda m=model: infer_stage3(m, batch, device, return_mask=True))
                    pred_s3 = out[0][0]
                    frame_masks[lbl] = out[1][0] if out[1] is not None else None
                    frame_confs[lbl] = out[2][0] if out[2] is not None else None
                else:
                    pred_s3, dt = timed(lambda m=model: infer_stage3(m, batch, device)[0])
                if not warm:
                    timer[lbl][0] += dt; timer[lbl][1] += 1
                accumulate_frame(acc_s3[lbl], pred_s3, gt, valid, cam_types)
                accumulate_frame(s3_scene[lbl], pred_s3, gt, valid, cam_types)
                frame_preds[lbl] = pred_s3

            if do_dump:
                try:
                    cam_names = batch["cameras_name"][0]              # per-view camera names for this frame
                    Himg, Wimg = gt.shape[-2:]
                    lens_masks = load_lens_masks(a.data_root / scene, cam_names, Himg, Wimg)
                    dump_frame_set(a.dump_dir, f"{scene}/frame{idx}", gt, frame_preds,
                                   batch, valid, a.dump_subsample,
                                   frame_masks=frame_masks,
                                   frame_confs=frame_confs, conf_drop_pct=a.dump_conf_drop_pct,
                                   lens_masks=lens_masks,
                                   fov_deg=a.dump_fov_deg, save_pcd=not a.dump_depth_only)
                    scene_dumped += 1
                except Exception as e:
                    logger.warning(f"  [dump] {scene} frame {idx} dump failed: {e}")

            if warmup_left > 0:
                warmup_left -= 1
            n_frames_done += 1

        far_frac = (far_pix / tot_pix) if tot_pix else 0.0
        p95 = float(np.median(p95_list)) if p95_list else 0.0
        # No explicit indoor/outdoor label; approximate by GT depth spread:
        # p95>12m or far fraction>3% -> outdoor.
        domain = "outdoor" if (p95 > 12.0 or far_frac > 0.03) else "indoor"
        per_scene[scene] = {"domain": domain, "far_frac": round(far_frac, 4),
                            "p95_depth": round(p95, 1),
                            "stage3": {lbl: s3_scene[lbl].result() for lbl in labels},
                            "n_frames": len(idxs)}
        # one progress line per ckpt (overall bucket)
        for lbl in labels:
            r = s3_scene[lbl].result()["overall"]
            logger.info(f"  ({si+1}/{len(scenes)}) [{domain:7}] {scene} [{lbl}]: {len(idxs)} frames "
                        f"absrel={r['depth_absrel']:.4f} scale={r['scale_absrel']:.3f} "
                        f"d1={r['delta1_1.25']:.4f} | {n_frames_done/(time.time()-t0):.2f} f/s")

    fps = {lbl: (timer[lbl][1] / timer[lbl][0]) if timer[lbl][0] > 0 else float("nan")
           for lbl in labels}

    # ---- Summary ----
    stage3_models_out = {
        lbl: {"ckpt": meta["ckpt"], "global_step": meta["global_step"],
              "result": acc_s3[lbl].result()}
        for lbl, _, meta in stage3_models
    }
    summary = {
        "stage3_ckpts": {lbl: meta["ckpt"] for lbl, _, meta in stage3_models},
        "stage3_models": stage3_models_out,
        "n_scenes": len(per_scene), "n_frames": n_frames_done,
        "fps_per_6cam_frame": fps,
        "n_indoor": sum(1 for v in per_scene.values() if v["domain"] == "indoor"),
        "n_outdoor": sum(1 for v in per_scene.values() if v["domain"] == "outdoor"),
        "per_scene": per_scene,
    }

    out_json = a.out_dir / "summary_calib_compare.json"
    # Resume-friendly: merge with any existing summary on disk; ckpts not run this time
    # (other X-Lens ckpts) keep their prior results.
    if out_json.exists():
        try:
            old = json.loads(out_json.read_text())
        except Exception:
            old = {}
        if old:
            for k in ("stage3_models", "stage3_ckpts", "fps_per_6cam_frame"):
                mm = dict(old.get(k, {})); mm.update(summary.get(k, {})); summary[k] = mm
            old_ps, new_ps = old.get("per_scene", {}), summary.get("per_scene", {})
            pm = {}
            for sc in set(old_ps) | set(new_ps):
                ms = dict(old_ps.get(sc, {})); ms.update(new_ps.get(sc, {}))   # keep old, overwrite/add new
                s3 = dict(old_ps.get(sc, {}).get("stage3", {}))
                s3.update(new_ps.get(sc, {}).get("stage3", {}))
                if s3:
                    ms["stage3"] = s3
                pm[sc] = ms
            summary["per_scene"] = pm
            kept = [k for k in old if k not in summary]    # old top-level keys not produced this run
            for k in kept:
                summary[k] = old[k]
            if kept:
                logger.info(f"  merged existing summary, kept {len(kept)} un-rerun keys: {kept}")
    out_json.write_text(json.dumps(summary, indent=2))

    # Also write one file per ckpt (legacy single-ckpt schema)
    for lbl, _, meta in stage3_models:
        sub = {
            "stage3_ckpt": meta["ckpt"],
            "stage3_global_step": meta["global_step"],
            "n_scenes": len(per_scene), "n_frames": n_frames_done,
            "fps_per_6cam_frame": {"stage3": fps.get(lbl)},
            "n_indoor": summary["n_indoor"], "n_outdoor": summary["n_outdoor"],
            "stage3": acc_s3[lbl].result(),
            "per_scene": {sc: {"domain": v["domain"], "far_frac": v["far_frac"],
                               "p95_depth": v["p95_depth"], "n_frames": v["n_frames"],
                               "stage3": v["stage3"][lbl]}
                          for sc, v in per_scene.items()},
        }
        (a.out_dir / f"summary_calib_compare__{lbl}.json").write_text(json.dumps(sub, indent=2))

    for lbl in labels:
        print_table(f"stage3[{lbl}]", acc_s3[lbl].result())
    print_fps(fps)
    print_per_scene(per_scene, labels)
    logger.info(f"results written to {out_json}")


def print_table(name: str, res: Dict[str, Dict[str, float]]):
    print(f"\n===== {name} (relative depth, lens+sky mask) =====")
    hdr = f"  {'group':8} {'scale_absrel':>13} {'depth_absrel':>13} {'rmse':>9} {'tau@1.03':>9} {'d1@1.25':>9} {'n_pix':>12}"
    print(hdr)
    for grp in ["fisheye", "pinhole", "overall"]:
        r = res[grp]
        if not r["n_pix"]:
            print(f"  {grp:8} {'--':>13}")
            continue
        print(f"  {grp:8} {r['scale_absrel']:>13.4f} {r['depth_absrel']:>13.4f} "
              f"{r['rmse']:>9.4f} {r['tau_1.03']:>9.4f} {r['delta1_1.25']:>9.4f} {r['n_pix']:>12,}")


def print_fps(fps: Dict[str, float]):
    print(f"\n===== FPS (one 6-camera forward per frame, warmup skipped, {('cuda' if torch.cuda.is_available() else 'cpu')}) =====")
    for k, v in fps.items():
        print(f"  {k:12} {v:.2f} frame/s  ({1000.0/v:.0f} ms/frame)" if v == v else f"  {k:12} --")


def print_per_scene(per_scene: Dict, labels: List[str]):
    """Per-scene overall metrics (sorted by domain + scale). One table per ckpt."""
    for lbl in labels:
        print(f"\n===== per-scene (overall, stage3[{lbl}]; sorted by domain + scale_absrel) =====")
        hdr = (f"  {'domain':7} {'p95m':>6} {'scene':40} {'absrel':>7} {'rmse':>7} "
               f"{'d1':>6} {'scale':>6}")
        print(hdr)
        rows = sorted(per_scene.items(),
                      key=lambda kv: (kv[1]["domain"], -kv[1]["stage3"][lbl]["overall"]["scale_absrel"]))
        for scene, v in rows:
            r = v["stage3"][lbl]["overall"]
            line = (f"  {v['domain']:7} {v['p95_depth']:6.1f} {scene[:40]:40} "
                    f"{r['depth_absrel']:7.4f} {r['rmse']:7.4f} {r['delta1_1.25']:6.3f} {r['scale_absrel']:6.3f}")
            print(line)


if __name__ == "__main__":
    main()
