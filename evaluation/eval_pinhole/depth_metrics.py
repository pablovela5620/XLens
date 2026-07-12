"""
Standard mono/multi-view depth metrics (eth3d / scannetpp / tav2 benchmark convention).

Computed per frame over valid pixels, then averaged over frames (equal weight). Metrics:
    abs_rel:   mean(|pred-gt| / gt)
    sq_rel:    mean((pred-gt)^2 / gt)
    rmse:      sqrt(mean((pred-gt)^2))            (m)
    rmse_log:  sqrt(mean((ln pred - ln gt)^2))
    mae:       mean(|pred-gt|)                    (m)
    delta1/2/3: mean( max(pred/gt, gt/pred) < 1.25^k )
    n_pix:     number of valid pixels

Scale alignment (pred_depth is in model normalized space, align to meters first):
    self_scale: pred * model scale_head metric_scaling_factor. Tests absolute metric ability.
    median:     pred * median(gt/pred) per frame. Scale-invariant, structure only.
    none:       pred as-is (assumed meters)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


METRIC_KEYS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "mae",
               "delta1", "delta2", "delta3", "tau_103", "scale_abs_rel"]

# MAE bucketed by GT depth (m). Bucket = (lo, hi], first bucket starts at 0.
DISTANCE_BUCKETS = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0),
                    (15.0, 20.0), (20.0, 25.0), (25.0, 80.0)]
BUCKET_LABELS = [f"{int(lo)}-{int(hi)}m" for lo, hi in DISTANCE_BUCKETS]


def align_pred(
    pred: np.ndarray,            # (H, W) normalized depth
    gt: np.ndarray,              # (H, W) meters
    valid: np.ndarray,          # (H, W) bool
    mode: str,
    model_sf: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Align pred to meters. Returns aligned pred (H, W), or None if alignment fails."""
    if mode == "self_scale":
        if model_sf is None or not np.isfinite(model_sf):
            raise ValueError("self_scale requires model metric_scaling_factor, but model has no scale head")
        return pred * float(model_sf)
    if mode == "median":
        p = pred[valid]
        g = gt[valid]
        keep = (p > 1e-6) & (g > 1e-6) & np.isfinite(p) & np.isfinite(g)
        if keep.sum() < 10:
            return None
        s = float(np.median(g[keep] / p[keep]))
        return pred * s
    if mode == "none":
        return pred
    raise ValueError(f"unknown alignment={mode}")


def compute_frame_depth_metrics(
    pred_metric: np.ndarray,     # (H, W) aligned to meters (pass median-aligned for relative-scale eval)
    gt: np.ndarray,              # (H, W) meters
    valid: np.ndarray,          # (H, W) bool
    min_valid_pix: int = 100,
    scale_abs_rel: Optional[float] = None,   # separate absolute-scale error (|sf - s*|/s*), precomputed by caller
) -> Optional[Dict[str, float]]:
    """Per-frame depth metrics. Returns None if too few valid pixels.
    abs_rel/rmse/tau/delta computed on the given pred_metric scale (pass median-aligned for
    relative-scale eval); scale_abs_rel is a separate scale-independent absolute-scale error."""
    p = pred_metric[valid]
    g = gt[valid]
    keep = (p > 1e-3) & (g > 1e-3) & np.isfinite(p) & np.isfinite(g)
    p = p[keep].astype(np.float64)
    g = g[keep].astype(np.float64)
    if p.size < min_valid_pix:
        return None

    diff = p - g
    ratio = np.maximum(p / g, g / p)
    m = {
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "sq_rel": float(np.mean(diff ** 2 / g)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
        "tau_103": float(np.mean(ratio < 1.03)),       # tau = inlier@1.03 (tight-threshold accuracy)
        "scale_abs_rel": float(scale_abs_rel) if scale_abs_rel is not None else float("nan"),
        "n_pix": int(p.size),
    }
    # MAE bucketed by GT depth (m) + per-bucket pixel count (pixel-weighted at aggregation)
    abs_diff = np.abs(p - g)
    for (lo, hi), label in zip(DISTANCE_BUCKETS, BUCKET_LABELS):
        bm = (g > lo) & (g <= hi)
        n_b = int(bm.sum())
        m[f"mae@{label}"] = float(np.mean(abs_diff[bm])) if n_b > 0 else float("nan")
        m[f"npix@{label}"] = n_b
    return m


def aggregate_frame_metrics(frame_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate multi-frame metrics with equal per-frame weight. n_pix summed, n_frames counted."""
    frame_metrics = [m for m in frame_metrics if m is not None]
    empty = ({k: float("nan") for k in METRIC_KEYS}
             | {f"mae@{lb}": float("nan") for lb in BUCKET_LABELS}
             | {"n_pix": 0, "n_frames": 0})
    if not frame_metrics:
        return empty
    out: Dict[str, float] = {}
    # Global metrics: equal per-frame average
    for k in METRIC_KEYS:
        vals = [m[k] for m in frame_metrics if np.isfinite(m.get(k, np.nan))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    # Bucketed MAE: weighted by per-bucket pixel count
    for lb in BUCKET_LABELS:
        num = den = 0.0
        for m in frame_metrics:
            n_b = m.get(f"npix@{lb}", 0)
            v = m.get(f"mae@{lb}", float("nan"))
            if n_b > 0 and np.isfinite(v):
                num += v * n_b
                den += n_b
        out[f"mae@{lb}"] = float(num / den) if den > 0 else float("nan")
        out[f"npix@{lb}"] = int(den)
    out["n_pix"] = int(sum(m["n_pix"] for m in frame_metrics))
    out["n_frames"] = len(frame_metrics)
    return out
