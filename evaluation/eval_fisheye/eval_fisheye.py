"""Fisheye-only (4-cam omni) evaluation.

Two datasets (the second skips metrics and only saves depth maps, toggled by the config
valid section):
  - test   : compute depth metrics + save [GT | pred | error] 3-panel figures.
  - valid  : same structure as test (has GT depth), saves only [RGB | GT depth | Pred depth]
             3-panel figures, no metrics. (save_depth_only_omni)

The test set does two things:
  1. Compute depth metrics (main: MAE, plus AbsRel / RMSE / delta), aggregate per scene + overall, write JSON.
  2. For each (scene, frame), sample one frame and save a 3-panel depth comparison per camera:
         [ GT depth | pred depth | error map ]
     - GT / pred share one depth colormap (turbo) and vmin/vmax for direct visual comparison.
     - error map = |GT - pred|, clipped to 1m: 0m=dark, >=1m=brightest (--err_clip).
       Errors >1m are clamped to brightest so outliers do not blow out the color scale.
     - Error and drawing use valid pixels only (GT valid + lens mask + view valid); invalid = black.

All metrics and visualizations use a mask: valid GT depth range + fisheye lens mask + (optional) valid_mask.

Usage:
    python -m evaluation.eval_fisheye.eval_fisheye \
        --ckpt /path/to/model.safetensors \
        --model_config configs/xlens_vits.yaml \
        --config evaluation/eval_fisheye/eval_fisheye_config.yaml \
        --output_dir ./eval_fisheye_out

    # Only a few scenes + limit visualizations
    python -m ... --scenes taobao_VictorianLivingRoom_50_40_10 --max_vis_frames 3

    # Metrics only, no figures (faster)
    python -m ... --no_vis
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


def _get_cmap(name):
    # matplotlib 3.7+ API, avoids the cm.get_cmap deprecation warning
    try:
        return matplotlib.colormaps[name]
    except Exception:
        return cm.get_cmap(name)

from evaluation.eval_fisheye.common import (
    DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_LUT_DIR,
    build_model_from_ckpt, read_global_step, load_yaml_config, resolve_image_hw,
    list_test_scenes, build_scene_dataset, build_ray_map_input,
    img_norm_to_rgb_uint8, build_valid_mask, pick,
)
from evaluation._support.datasets import collate_omni_frames

# Default eval config yaml shipped in this directory
DEFAULT_EVAL_CONFIG = str(Path(__file__).resolve().parent / "eval_fisheye_config.yaml")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEFAULT_MIN_DEPTH = 0.05
DEFAULT_MAX_DEPTH = 80.0
DISTANCE_BUCKETS = [5.0, 10.0, 15.0, 20.0, 30.0, 80.0]


# ============================================================================
# Metric accumulator (streaming, stores only sum / count). Main metric: MAE.
# ============================================================================

class DepthMetricAccumulator:
    DELTA_THRESHOLDS = (1.25, 1.25 ** 2, 1.25 ** 3)

    def __init__(self, bucket_edges: List[float] = DISTANCE_BUCKETS):
        self.bucket_edges = list(bucket_edges)
        labels, prev = [], 0.0
        for hi in self.bucket_edges:
            labels.append(f"{prev:g}-{hi:g}m")
            prev = hi
        self.bucket_labels = labels
        self.all_labels = self.bucket_labels + ["overall"]
        self.stats = {
            lab: {"n_pix": 0, "sum_abs_diff": 0.0, "sum_sq_diff": 0.0,
                  "sum_abs_rel": 0.0, "sum_log_diff_sq": 0.0,
                  "n_delta1": 0, "n_delta2": 0, "n_delta3": 0}
            for lab in self.all_labels
        }

    def update(self, pred: np.ndarray, gt: np.ndarray) -> int:
        if pred.size == 0:
            return 0
        gt = gt.astype(np.float64)
        pred = pred.astype(np.float64)
        gt_safe = np.maximum(gt, 1e-6)
        pred_safe = np.maximum(pred, 1e-6)
        abs_diff = np.abs(pred - gt)
        sq_diff = (pred - gt) ** 2
        abs_rel = abs_diff / gt_safe
        log_diff_sq = (np.log(pred_safe) - np.log(gt_safe)) ** 2
        ratio = np.maximum(pred_safe / gt_safe, gt_safe / pred_safe)
        self._acc("overall", abs_diff, sq_diff, abs_rel, log_diff_sq, ratio)
        prev = 0.0
        for hi, label in zip(self.bucket_edges, self.bucket_labels):
            m = (gt > prev) & (gt <= hi)
            if m.any():
                self._acc(label, abs_diff[m], sq_diff[m], abs_rel[m], log_diff_sq[m], ratio[m])
            prev = hi
        return int(pred.size)

    def _acc(self, label, abs_diff, sq_diff, abs_rel, log_diff_sq, ratio):
        s = self.stats[label]
        s["n_pix"] += int(abs_diff.size)
        s["sum_abs_diff"] += float(abs_diff.sum())
        s["sum_sq_diff"] += float(sq_diff.sum())
        s["sum_abs_rel"] += float(abs_rel.sum())
        s["sum_log_diff_sq"] += float(log_diff_sq.sum())
        s["n_delta1"] += int((ratio < self.DELTA_THRESHOLDS[0]).sum())
        s["n_delta2"] += int((ratio < self.DELTA_THRESHOLDS[1]).sum())
        s["n_delta3"] += int((ratio < self.DELTA_THRESHOLDS[2]).sum())

    def result(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for label in self.all_labels:
            s = self.stats[label]
            n = s["n_pix"]
            if n == 0:
                out[label] = {"n_pix": 0, "mae": float("nan"), "abs_rel": float("nan"),
                              "rmse": float("nan"), "rmse_log": float("nan"),
                              "delta_1.25": float("nan"), "delta_1.25_2": float("nan"),
                              "delta_1.25_3": float("nan")}
                continue
            out[label] = {
                "n_pix": n,
                "mae": s["sum_abs_diff"] / n,
                "abs_rel": s["sum_abs_rel"] / n,
                "rmse": math.sqrt(s["sum_sq_diff"] / n),
                "rmse_log": math.sqrt(s["sum_log_diff_sq"] / n),
                "delta_1.25": s["n_delta1"] / n,
                "delta_1.25_2": s["n_delta2"] / n,
                "delta_1.25_3": s["n_delta3"] / n,
            }
        return out


# ============================================================================
# 3-panel depth comparison figure
# ============================================================================

def _colorize_depth(depth: np.ndarray, valid: np.ndarray,
                    vmin: float, vmax: float, cmap_name: str = "turbo") -> np.ndarray:
    """depth (H,W) -> RGB uint8 (H,W,3). Invalid pixels painted black."""
    d = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = (_get_cmap(cmap_name)(d)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _colorize_error(err_abs: np.ndarray, valid: np.ndarray,
                    err_clip: float, cmap_name: str = "inferno") -> np.ndarray:
    """|GT-pred| (H,W) -> RGB uint8. Error clipped to err_clip(=1m): 0m=dark, >=1m=brightest.
    Invalid pixels painted black."""
    e = np.clip(err_abs / max(err_clip, 1e-6), 0.0, 1.0)   # >1m clamped to 1.0 (brightest)
    rgb = (_get_cmap(cmap_name)(e)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def save_depth_compare_figure(
    gt_depth: np.ndarray,       # (H, W) meters
    pred_depth: np.ndarray,     # (H, W) meters
    valid: np.ndarray,          # (H, W) bool
    out_path: Path,
    title: str,
    err_clip: float = 1.0,
) -> Optional[float]:
    """Draw [GT | pred | error] 3-panel figure. Return MAE over valid pixels (None if none)."""
    if not valid.any():
        return None
    gt_v = gt_depth[valid]
    pred_v = pred_depth[valid]
    # Depth color scale from GT valid pixels' 2-98 percentile to avoid outlier blowout
    vmin = float(np.percentile(gt_v, 2))
    vmax = float(np.percentile(gt_v, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0

    err_abs = np.abs(gt_depth - pred_depth)
    mae = float(err_abs[valid].mean())

    gt_rgb = _colorize_depth(gt_depth, valid, vmin, vmax)
    pred_rgb = _colorize_depth(pred_depth, valid, vmin, vmax)
    err_rgb = _colorize_error(err_abs, valid, err_clip)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))
    axes[0].imshow(gt_rgb)
    axes[0].set_title(f"GT depth  [{vmin:.2f}, {vmax:.2f}]m")
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Pred depth (same scale)")
    axes[2].imshow(err_rgb)
    axes[2].set_title(f"|GT - Pred|  clip@{err_clip:.1f}m (bright=large err)  MAE={mae:.3f}m")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)

    # Error colorbar (0 -> err_clip)
    sm = cm.ScalarMappable(cmap="inferno",
                           norm=plt.Normalize(vmin=0.0, vmax=err_clip))
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("abs error (m)")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return mae


def save_depth_only_figure(
    rgb_uint8: np.ndarray,      # (H, W, 3) uint8, de-normalized input image
    pred_depth: np.ndarray,     # (H, W) meters
    valid: np.ndarray,          # (H, W) bool
    out_path: Path,
    title: str,
    gt_depth: Optional[np.ndarray] = None,   # (H, W) meters; None -> no GT panel
) -> bool:
    """Save depth figures only (no metrics).

    - gt_depth given (valid set): 3-panel [RGB | GT depth | Pred depth].
    - gt_depth=None: 2-panel [RGB | Pred depth].
    GT / pred share one depth colormap (turbo) and vmin/vmax for direct comparison.
    Returns False (no figure) if there are no valid pixels."""
    if not valid.any():
        return False
    # Color-scale reference: GT valid pixels if available, else pred valid pixels' 2-98 percentile
    ref = (gt_depth if gt_depth is not None else pred_depth)[valid]
    vmin = float(np.percentile(ref, 2))
    vmax = float(np.percentile(ref, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0

    panels = [("RGB", rgb_uint8)]
    if gt_depth is not None:
        panels.append((f"GT depth  [{vmin:.2f}, {vmax:.2f}]m",
                       _colorize_depth(gt_depth, valid, vmin, vmax)))
    panels.append(("Pred depth (same scale)" if gt_depth is not None
                   else f"Pred depth  [{vmin:.2f}, {vmax:.2f}]m",
                   _colorize_depth(pred_depth, valid, vmin, vmax)))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (t, im) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(t)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


# ============================================================================
# Full-frame 4-fisheye comparison: one figure showing all 4 as [RGB | GT depth | (pred-gt) error]
#   rows = RGB / GT / error, columns = 4 fisheye cameras.
#   - error drawn only on valid pixels with GT in (0, near_clip] (default 5m), rest grayed.
#   - error is signed (pred - gt) with a diverging colormap (blue=underestimate, red=overestimate).
# ============================================================================

def save_fisheye_grid_figure(
    rgbs: List[np.ndarray],     # S x (H, W, 3) uint8
    gts: List[np.ndarray],      # S x (H, W) meters
    preds: List[np.ndarray],    # S x (H, W) meters
    valids: List[np.ndarray],   # S x (H, W) bool
    cam_names: List[str],
    out_path: Path,
    title: str,
    near_clip: float = 5.0,     # only GT 0~near_clip meters
    err_clip: float = 1.0,      # error scale +/-err_clip meters
) -> bool:
    """3xS grid comparison figure for the 4 fisheye. Emits a figure if at least one view is valid."""
    S = len(cam_names)
    if not any(v.any() for v in valids):
        return False

    depth_cmap = _get_cmap("turbo").copy()
    depth_cmap.set_bad("0.12")                 # invalid pixels: dark gray
    err_cmap = _get_cmap("RdBu_r").copy()
    err_cmap.set_bad("0.12")

    # Column width from fisheye aspect ratio, leaving room for row labels / colorbar
    H, W = gts[0].shape
    cell_w = 2.6
    cell_h = cell_w * H / W
    fig, axes = plt.subplots(
        3, S, figsize=(cell_w * S + 1.4, cell_h * 3 + 0.8),
        constrained_layout=True,
    )
    if S == 1:
        axes = axes.reshape(3, 1)

    depth_im = err_im = None
    for s in range(S):
        valid = valids[s]
        gt = gts[s]
        near = valid & (gt > 0.0) & (gt <= near_clip)     # error counts only 0~near points
        signed_err = np.where(near, preds[s] - gt, np.nan)
        mae5 = float(np.abs(preds[s] - gt)[near].mean()) if near.any() else float("nan")

        # row 0: RGB
        axes[0, s].imshow(rgbs[s])
        axes[0, s].set_title(cam_names[s], fontsize=10)

        # row 1: GT depth (0~near_clip turbo, saturated beyond, invalid gray)
        gt_m = np.ma.masked_where(~valid, np.clip(gt, 0.0, near_clip))
        depth_im = axes[1, s].imshow(gt_m, cmap=depth_cmap, vmin=0.0, vmax=near_clip)

        # row 2: signed error (pred-gt), GT 0~near only, diverging colormap
        err_m = np.ma.masked_invalid(signed_err)
        err_im = axes[2, s].imshow(err_m, cmap=err_cmap, vmin=-err_clip, vmax=err_clip)
        axes[2, s].set_title(f"MAE<={near_clip:g}m: {mae5:.3f}", fontsize=9)

    for r in range(3):
        for s in range(S):
            axes[r, s].set_xticks([]); axes[r, s].set_yticks([])
    axes[0, 0].set_ylabel("RGB", fontsize=11)
    axes[1, 0].set_ylabel("GT depth", fontsize=11)
    axes[2, 0].set_ylabel("pred − GT", fontsize=11)

    if depth_im is not None:
        cb = fig.colorbar(depth_im, ax=axes[1, :].tolist(), fraction=0.025, pad=0.01)
        cb.set_label("depth (m)")
    if err_im is not None:
        cb = fig.colorbar(err_im, ax=axes[2, :].tolist(), fraction=0.025, pad=0.01)
        cb.set_label("pred − GT (m)")

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


# ============================================================================
# Main evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    model,
    data_root: Path,
    lut_dir: Path,
    scenes: List[str],
    target_hw,
    device: torch.device,
    ray_mode: str,
    min_depth: float,
    max_depth: float,
    amp_dtype: str,
    batch_size: int,
    num_workers: int,
    max_samples_per_scene: Optional[int],
    use_lens_mask: bool,
    vis: bool,
    vis_dir: Path,
    max_vis_frames: int,
    err_clip: float,
    near_clip: float = 5.0,
    bucket_edges: List[float] = DISTANCE_BUCKETS,
) -> Dict:
    overall = DepthMetricAccumulator(bucket_edges)
    per_scene: Dict[str, DepthMetricAccumulator] = {}
    amp_dtype_t = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_amp = (device.type == "cuda")

    n_samples_total = 0
    sum_scale, n_scale = 0.0, 0
    t0 = time.time()

    for scene in scenes:
        try:
            ds = build_scene_dataset(data_root, lut_dir, scene, target_hw)
        except Exception as e:
            logger.warning(f"skipping scene {scene}: {e}")
            continue
        acc = DepthMetricAccumulator(bucket_edges)
        per_scene[scene] = acc
        cam_names = ds.cameras_name
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, drop_last=False, collate_fn=collate_omni_frames,
        )
        logger.info(f"=== scene {scene}: {len(ds)} frames, cams={cam_names} ===")

        seen, vis_done = 0, 0
        for batch in loader:
            B = batch["images"].shape[0]
            images = batch["images"].to(device, non_blocking=True)
            gt_depth_t = batch["depths"].to(device, non_blocking=True)
            view_mask_t = batch["view_mask"]
            nam_t = batch.get("non_ambiguous_mask")
            has_nam_t = batch.get("has_non_ambiguous_mask")

            ray_map, d_cam_in = build_ray_map_input(batch, ray_mode, device)
            cam_types = batch.get("cam_types")
            if cam_types is not None:
                cam_types = cam_types.to(device)
            with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype_t):
                out = model(images, ray_map=ray_map, d_cam=d_cam_in, cam_types=cam_types)

            pred_np = out["depth_metric"].float().cpu().numpy()       # (B, S, H, W)
            sf_np = out["metric_scaling_factor"].float().cpu().numpy()
            gt_np = gt_depth_t.cpu().numpy()
            view_np = view_mask_t.numpy()
            nam_np = nam_t.numpy() if nam_t is not None else None
            has_nam_np = has_nam_t.numpy() if has_nam_t is not None else None
            imgs_np = batch["images"].numpy()                         # (B, S, 3, H, W)

            for b in range(B):
                if max_samples_per_scene is not None and seen >= max_samples_per_scene:
                    break
                valid_b = build_valid_mask(
                    gt_np[b], view_np[b], min_depth, max_depth,
                    nam=nam_np[b] if nam_np is not None else None,
                    has_nam=has_nam_np[b] if has_nam_np is not None else None,
                    use_lens_mask=use_lens_mask,
                )                                                     # (S, H, W)
                if valid_b.any():
                    acc.update(pred_np[b][valid_b], gt_np[b][valid_b])
                    overall.update(pred_np[b][valid_b], gt_np[b][valid_b])
                sum_scale += float(sf_np[b]); n_scale += 1

                # Visualization: one figure showing all 4 fisheye as [RGB | GT | pred-GT]
                if vis and vis_done < max_vis_frames:
                    S = pred_np.shape[1]
                    cams = [cam_names[s] if s < len(cam_names) else f"VIEW_{s}"
                            for s in range(S)]
                    save_fisheye_grid_figure(
                        rgbs=[img_norm_to_rgb_uint8(imgs_np[b, s]) for s in range(S)],
                        gts=[gt_np[b, s] for s in range(S)],
                        preds=[pred_np[b, s] for s in range(S)],
                        valids=[valid_b[s] for s in range(S)],
                        cam_names=cams,
                        out_path=vis_dir / scene / f"sample{seen:04d}.png",
                        title=f"{scene}  sample{seen}",
                        near_clip=near_clip,
                        err_clip=err_clip,
                    )
                    vis_done += 1

                seen += 1
                n_samples_total += 1
            if max_samples_per_scene is not None and seen >= max_samples_per_scene:
                break

        res = acc.result()["overall"]
        logger.info(f"  [{scene}] MAE={res['mae']:.4f}m  AbsRel={res['abs_rel']:.4f}  "
                    f"RMSE={res['rmse']:.4f}  d<1.25={res['delta_1.25']:.4f}  "
                    f"({res['n_pix']:,} pix)")

    elapsed = time.time() - t0
    logger.info(f"evaluation done: {n_samples_total} samples, {elapsed:.1f}s")

    result = {
        "n_samples": n_samples_total,
        "ray_mode": ray_mode,
        "min_depth": min_depth, "max_depth": max_depth,
        "err_clip_m": err_clip,
        "mean_metric_scaling_factor": (sum_scale / n_scale) if n_scale else None,
        "overall": overall.result(),
        "per_scene": {s: acc.result() for s, acc in per_scene.items()},
    }
    return result


# ============================================================================
# Save depth figures only (no metrics): valid set (3-panel rgb|gt|pred)
# ============================================================================

@torch.no_grad()
def save_depth_only_omni(
    model,
    data_root: Path,
    lut_dir: Path,
    scenes: List[str],
    target_hw,
    device: torch.device,
    ray_mode: str,
    min_depth: float,
    max_depth: float,
    amp_dtype: str,
    batch_size: int,
    num_workers: int,
    max_samples_per_scene: Optional[int],
    use_lens_mask: bool,
    out_dir: Path,
    max_vis_frames: int,
) -> int:
    """valid set: uses OmniSceneDataset (has GT depth), saves a 3-panel [RGB | GT depth | Pred depth]
    figure per (scene, frame, cam). No metrics."""
    amp_dtype_t = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_amp = (device.type == "cuda")
    n_saved = 0
    t0 = time.time()

    for scene in scenes:
        try:
            ds = build_scene_dataset(data_root, lut_dir, scene, target_hw)
        except Exception as e:
            logger.warning(f"[valid] skipping scene {scene}: {e}")
            continue
        cam_names = ds.cameras_name
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, drop_last=False, collate_fn=collate_omni_frames,
        )
        logger.info(f"=== [valid] scene {scene}: {len(ds)} frames, cams={cam_names} ===")

        seen, vis_done = 0, 0
        for batch in loader:
            if vis_done >= max_vis_frames:
                break
            B = batch["images"].shape[0]
            images = batch["images"].to(device, non_blocking=True)
            gt_depth_t = batch["depths"].to(device, non_blocking=True)
            view_np = batch["view_mask"].numpy()
            nam_t = batch.get("non_ambiguous_mask")
            has_nam_t = batch.get("has_non_ambiguous_mask")
            nam_np = nam_t.numpy() if nam_t is not None else None
            has_nam_np = has_nam_t.numpy() if has_nam_t is not None else None

            ray_map, d_cam_in = build_ray_map_input(batch, ray_mode, device)
            cam_types = batch.get("cam_types")
            if cam_types is not None:
                cam_types = cam_types.to(device)
            with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype_t):
                out = model(images, ray_map=ray_map, d_cam=d_cam_in, cam_types=cam_types)

            pred_np = out["depth_metric"].float().cpu().numpy()       # (B, S, H, W)
            gt_np = gt_depth_t.cpu().numpy()
            imgs_np = batch["images"].numpy()                         # (B, S, 3, H, W)

            for b in range(B):
                if max_samples_per_scene is not None and seen >= max_samples_per_scene:
                    break
                if vis_done >= max_vis_frames:
                    break
                valid_b = build_valid_mask(
                    gt_np[b], view_np[b], min_depth, max_depth,
                    nam=nam_np[b] if nam_np is not None else None,
                    has_nam=has_nam_np[b] if has_nam_np is not None else None,
                    use_lens_mask=use_lens_mask,
                )                                                     # (S, H, W)
                S = pred_np.shape[1]
                for s in range(S):
                    if not valid_b[s].any():
                        continue
                    cam = cam_names[s] if s < len(cam_names) else f"VIEW_{s}"
                    rgb = img_norm_to_rgb_uint8(imgs_np[b, s])
                    if save_depth_only_figure(
                        rgb, pred_np[b, s], valid_b[s],
                        out_dir / scene / f"sample{seen:04d}_{cam}.png",
                        title=f"[valid] {scene}  sample{seen}  {cam}",
                        gt_depth=gt_np[b, s],
                    ):
                        n_saved += 1
                vis_done += 1
                seen += 1

    logger.info(f"[valid] depth figures saved: {n_saved}, {time.time() - t0:.1f}s -> {out_dir}")
    return n_saved


def print_depth_table(name: str, depth_res: Dict[str, Dict[str, float]]):
    logger.info(f"--- depth metrics: {name} ---")
    headers = ["bucket", "n_pix", "mae(m)", "abs_rel", "rmse(m)", "d<1.25", "d<1.25^2", "d<1.25^3"]
    logger.info("  " + " | ".join(f"{h:>10}" for h in headers))
    for bucket, m in depth_res.items():
        if m["n_pix"] == 0:
            continue
        logger.info("  " +
                    f"{bucket:>10} | {m['n_pix']:>10,d} | {m['mae']:>10.4f} | "
                    f"{m['abs_rel']:>10.4f} | {m['rmse']:>10.4f} | "
                    f"{m['delta_1.25']:>10.4f} | {m['delta_1.25_2']:>10.4f} | "
                    f"{m['delta_1.25_3']:>10.4f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage2 fisheye-only test-set eval (MAE + depth figures)")
    ap.add_argument("--config", type=str, default=DEFAULT_EVAL_CONFIG,
                    help="eval config yaml (default eval_fisheye_config.yaml). "
                         "Unset parameters below are read from here")
    # All default=None: fall back to yaml, then to hardcoded defaults
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint .pth (overrides yaml)")
    ap.add_argument("--model_config", type=str, default=None,
                    help="model arch config yaml (config embedded in a .pth ckpt takes "
                         "priority). For the released .safetensors pass configs/xlens_vits.yaml.")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--lut_dir", type=str, default=None)
    ap.add_argument("--scenes", type=str, nargs="*", default=None,
                    help="scene dir names to evaluate; unset uses yaml, then all")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--target_h", type=int, default=None)
    ap.add_argument("--target_w", type=int, default=None)
    ap.add_argument("--ray_mode", choices=["full", "icam", "none"], default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--min_depth", type=float, default=None)
    ap.add_argument("--max_depth", type=float, default=None)
    ap.add_argument("--amp_dtype", choices=["bf16", "fp16"], default=None)
    ap.add_argument("--max_samples_per_scene", type=int, default=None,
                    help="max N frames per scene (sanity)")
    ap.add_argument("--no_lens_mask", action="store_true", help="disable lens mask filtering (overrides yaml)")
    ap.add_argument("--no_vis", action="store_true", help="skip depth figures (overrides yaml)")
    ap.add_argument("--max_vis_frames", type=int, default=None)
    ap.add_argument("--err_clip", type=float, default=None,
                    help="error figure scale +/-this value (m). signed error clamped at both ends")
    ap.add_argument("--near_clip", type=float, default=None,
                    help="error figure counts only GT within 0~this value (m) (default 5)")
    ap.add_argument("--device", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    ycfg = load_yaml_config(args.config)   # eval config yaml

    # Resolve all parameters (CLI > yaml > default)
    ckpt = pick(args.ckpt, ycfg, "ckpt", None)
    if not ckpt:
        raise ValueError("no ckpt: set ckpt in --config yaml, or pass --ckpt")
    model_config = pick(args.model_config, ycfg, "model_config", DEFAULT_CONFIG)
    data_root = Path(pick(args.data_root, ycfg, "data_root", DEFAULT_DATA_ROOT))
    lut_dir = Path(pick(args.lut_dir, ycfg, "lut_dir", DEFAULT_LUT_DIR))
    scenes = pick(args.scenes, ycfg, "scenes", None)
    output_dir = pick(args.output_dir, ycfg, "output_dir", "./eval_fisheye_out")
    ray_mode = pick(args.ray_mode, ycfg, "ray_mode", "full")
    batch_size = pick(args.batch_size, ycfg, "batch_size", 2)
    num_workers = pick(args.num_workers, ycfg, "num_workers", 4)
    min_depth = pick(args.min_depth, ycfg, "min_depth", DEFAULT_MIN_DEPTH)
    max_depth = pick(args.max_depth, ycfg, "max_depth", DEFAULT_MAX_DEPTH)
    amp_dtype = pick(args.amp_dtype, ycfg, "amp_dtype", "bf16")
    max_samples_per_scene = pick(args.max_samples_per_scene, ycfg, "max_samples_per_scene", None)
    max_vis_frames = pick(args.max_vis_frames, ycfg, "max_vis_frames", 3)
    err_clip = pick(args.err_clip, ycfg, "err_clip", 1.0)
    near_clip = pick(args.near_clip, ycfg, "near_clip", 5.0)
    bucket_edges = [float(x) for x in ycfg.get("distance_buckets") or DISTANCE_BUCKETS]
    device_str = pick(args.device, ycfg, "device", "cuda")
    # Booleans: CLI --no_* forces off, otherwise use yaml (default True)
    use_lens_mask = (not args.no_lens_mask) and ycfg.get("use_lens_mask", True)
    vis = (not args.no_vis) and ycfg.get("vis", True)

    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA unavailable, falling back to CPU, eval will be slow.")

    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    model_cfg_yaml = load_yaml_config(model_config)
    target_hw = resolve_image_hw(model_cfg_yaml, args.target_h or ycfg.get("target_h"),
                                 args.target_w or ycfg.get("target_w"))
    logger.info(f"target_hw={target_hw}  data_root={data_root}  ray_mode={ray_mode}")

    model, _ = build_model_from_ckpt(ckpt_path, Path(model_config) if model_config else None, device)
    global_step = read_global_step(ckpt_path)

    scenes = scenes or list_test_scenes(data_root)
    if not scenes:
        raise RuntimeError(f"no scenes found under data_root: {data_root}")
    logger.info(f"evaluating {len(scenes)} scenes: {scenes}")

    out_dir = Path(output_dir)
    vis_dir = out_dir / "depth_compare"

    result = evaluate(
        model=model, data_root=data_root, lut_dir=lut_dir, scenes=scenes,
        target_hw=target_hw, device=device, ray_mode=ray_mode,
        min_depth=min_depth, max_depth=max_depth, amp_dtype=amp_dtype,
        batch_size=batch_size, num_workers=num_workers,
        max_samples_per_scene=max_samples_per_scene,
        use_lens_mask=use_lens_mask,
        vis=vis, vis_dir=vis_dir, max_vis_frames=max_vis_frames,
        err_clip=err_clip, near_clip=near_clip, bucket_edges=bucket_edges,
    )
    result["ckpt"] = str(ckpt_path)
    result["global_step"] = global_step
    result["scenes"] = scenes
    result["target_hw"] = list(target_hw)

    logger.info("\n" + "=" * 60)
    print_depth_table("overall", result["overall"])
    for scene, m in result["per_scene"].items():
        logger.info("")
        print_depth_table(scene, m)

    out_dir.mkdir(parents=True, exist_ok=True)
    step_tag = global_step if global_step != "unknown" else "unk"
    json_path = out_dir / f"eval_fisheye_step{step_tag}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nmetrics JSON: {json_path}")
    if vis:
        logger.info(f"depth figures: {vis_dir}/<scene>/sample####_CAM_X.png")
    om = result["overall"]["overall"]
    logger.info(f"\n>>> overall MAE = {om['mae']:.4f} m   "
                f"AbsRel = {om['abs_rel']:.4f}   d<1.25 = {om['delta_1.25']:.4f}")

    # ------------------------------------------------------------------
    # valid set (depth figures only, 3-panel rgb|gt|pred, no metrics)
    # ------------------------------------------------------------------
    valid_cfg = ycfg.get("valid") or {}
    if valid_cfg.get("enabled", False):
        v_root = Path(valid_cfg.get("data_root", DEFAULT_DATA_ROOT))
        if not v_root.exists():
            logger.warning(f"[valid] data_root not found, skipping: {v_root}")
        else:
            v_scenes = valid_cfg.get("scenes") or list_test_scenes(v_root)
            v_out = Path(valid_cfg.get("output_dir") or (out_dir / "valid_depth"))
            logger.info("\n" + "=" * 60 + f"\n[valid] {len(v_scenes)} scenes -> {v_out}")
            save_depth_only_omni(
                model=model, data_root=v_root, lut_dir=lut_dir, scenes=v_scenes,
                target_hw=target_hw, device=device, ray_mode=ray_mode,
                min_depth=min_depth, max_depth=max_depth, amp_dtype=amp_dtype,
                batch_size=batch_size, num_workers=num_workers,
                max_samples_per_scene=valid_cfg.get("max_samples_per_scene",
                                                    max_samples_per_scene),
                use_lens_mask=use_lens_mask,
                out_dir=v_out,
                max_vis_frames=valid_cfg.get("max_vis_frames", max_vis_frames),
            )


if __name__ == "__main__":
    main()
