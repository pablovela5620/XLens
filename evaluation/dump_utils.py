"""Shared utilities for dumping depth maps + point clouds during eval
(used by eval_pinhole/run_eval and eval_heterogeneous/*).

Design:
  * Pure-numpy core (save_ply / unprojection / d_cam_from_K); no torch or model deps.
  * Depth: per-view raw .npy (float32 meters) + a color .png preview (best-effort,
    skipped if no plotting library).
  * Point cloud: unproject each view to the view0 world frame via GT intrinsics/extrinsics,
    merge all views into a single .ply (xyz+rgb). One file for pred, one for gt.

Conventions (consistent across eval):
  * depth = z-depth (meters), d_cam = OpenCV camera-frame unit ray (X right/Y down/Z forward),
    c2w = view0-canonical.
  * Unprojection: pt_cam = (depth / |d_cam.z|) * d_cam_unit; pt_world = c2w @ pt_cam
    (works for both fisheye and pinhole).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_PNG_BACKEND_WARNED = False


# ============================================================================
# PLY writer (binary little-endian, optional RGB)
# ============================================================================
def save_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    has_color = colors is not None and len(colors) == N
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {N}",
              "property float x", "property float y", "property float z"]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header\n")
    if has_color:
        colors = np.asarray(colors, dtype=np.uint8)
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        buf = np.empty(N, dtype=dt)
        buf["x"], buf["y"], buf["z"] = points[:, 0], points[:, 1], points[:, 2]
        buf["r"], buf["g"], buf["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    else:
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        buf = np.empty(N, dtype=dt)
        buf["x"], buf["y"], buf["z"] = points[:, 0], points[:, 1], points[:, 2]
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        buf.tofile(f)


def img_norm_to_rgb(img_chw: np.ndarray) -> np.ndarray:
    """ImageNet-normalized (3,H,W) → uint8 RGB (H,W,3)."""
    img = img_chw.transpose(1, 2, 0).astype(np.float32) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


# ============================================================================
# Depth color preview (best-effort: turbo colormap via matplotlib, else skip)
# ============================================================================
def save_depth_png(path: Path, depth_hw: np.ndarray, valid_hw: Optional[np.ndarray],
                   max_depth: float = 80.0) -> bool:
    global _PNG_BACKEND_WARNED
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        if not _PNG_BACKEND_WARNED:
            logger.warning("[dump] matplotlib unavailable, skipping depth .png preview (raw .npy still saved)")
            _PNG_BACKEND_WARNED = True
        return False
    d = depth_hw.astype(np.float32)
    m = valid_hw.astype(bool) if valid_hw is not None else (np.isfinite(d) & (d > 0))
    if m.sum() < 10:
        vmin, vmax = 0.0, max_depth
    else:
        vmin = float(np.percentile(d[m], 2))
        vmax = float(min(max_depth, np.percentile(d[m], 98)))
        if vmax <= vmin:
            vmax = vmin + 1e-3
    vis = np.where(m, d, np.nan)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), vis, cmap="turbo", vmin=vmin, vmax=vmax)
    return True


# ============================================================================
# Unprojection
# ============================================================================
def d_cam_from_intrinsics(K: np.ndarray, H: int, W: int) -> np.ndarray:
    """Pinhole intrinsics -> camera-frame unit rays (3,H,W) OpenCV. For block1 (no LUT d_cam)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vv, uu = np.meshgrid(np.arange(H, dtype=np.float64),
                         np.arange(W, dtype=np.float64), indexing="ij")
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)
    d = np.stack([x, y, z], 0)                       # (3,H,W)
    return (d / np.linalg.norm(d, axis=0, keepdims=True)).astype(np.float32)


def unproject_world(depth_shw: np.ndarray, d_cam_s3hw: np.ndarray,
                    c2w_s44: np.ndarray) -> np.ndarray:
    """z-depth + unit ray + c2w -> view0 world points (S,H,W,3). pt = (z/|d.z|)*d, then c2w."""
    S, H, W = depth_shw.shape
    out = np.empty((S, H, W, 3), dtype=np.float32)
    for s in range(S):
        d = d_cam_s3hw[s].transpose(1, 2, 0).astype(np.float64)   # (H,W,3)
        z = depth_shw[s].astype(np.float64)
        dz = np.abs(d[..., 2])
        scale = np.where(dz > 1e-3, z / np.maximum(dz, 1e-3), 0.0)
        pts_cam = scale[..., None] * d                            # (H,W,3)
        c2w = c2w_s44[s].astype(np.float64)
        out[s] = (pts_cam @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)
    return out


# ============================================================================
# Single-frame dump: depth maps (per view) + point cloud (merged across views)
# ============================================================================
def dump_frame(
    frame_dir: Path,
    model_name: str,
    depth_shw: np.ndarray,            # (S,H,W) z-depth meters (pred or gt)
    d_cam_s3hw: np.ndarray,           # (S,3,H,W) camera-frame unit rays
    c2w_s44: np.ndarray,              # (S,4,4) view0-canonical
    images_s3hw: np.ndarray,          # (S,3,H,W) ImageNet-normalized
    *,
    view_mask: Optional[np.ndarray] = None,   # (S,) bool, None=all valid
    pcd_mask_shw: Optional[np.ndarray] = None,  # (S,H,W) bool point mask; None=infer from finite+range
    lens_shw: Optional[np.ndarray] = None,      # (S,H,W) bool GT lens FOV mask; used to drop outside-lens pixels
    conf_shw: Optional[np.ndarray] = None,      # (S,H,W) per-pixel confidence; thresholded by conf_drop_pct
    conf_drop_pct: Optional[float] = None,      # drop the lowest-confidence percent (e.g. 20 = drop lowest 20%)
    fov_deg: Optional[float] = None,            # point-cloud FoV cone crop: keep rays within this half-angle
                                                # from the optical axis (dz=|d_cam.z|>=cos(fov_deg)); None/0=no crop
    min_depth: float = 0.05,
    max_depth: float = 80.0,
    save_depth: bool = True,
    save_pcd: bool = True,
    subsample: int = 1,               # keep one point every `subsample` pixels (1=full density)
) -> None:
    """Dump this model's depth maps (per-view npy+png) and merged point cloud (one ply)."""
    S, H, W = depth_shw.shape
    vm = view_mask.astype(bool) if view_mask is not None else np.ones(S, bool)
    frame_dir.mkdir(parents=True, exist_ok=True)

    all_pts: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    pts_world = unproject_world(depth_shw, d_cam_s3hw, c2w_s44) if save_pcd else None

    for s in range(S):
        if not vm[s]:
            continue
        d = depth_shw[s]
        # Lens FOV mask: prefer GT lens_mask; otherwise infer from d_cam (outside-lens norm ~0).
        if lens_shw is not None:
            lens = lens_shw[s].astype(bool)
        else:
            lens = np.linalg.norm(d_cam_s3hw[s], axis=0) > 0.5
        if save_depth:
            # Apply lens mask to depth too (npy 0, png blank) to drop outside-lens noise
            d_lens = np.where(lens, d, 0.0).astype(np.float32)
            np.save(frame_dir / f"{model_name}__depth_v{s}.npy", d_lens)
            disp = np.isfinite(d) & (d > 0) & lens
            save_depth_png(frame_dir / f"{model_name}__depth_v{s}.png", d, disp, max_depth)
        if save_pcd:
            # Base geometry mask: finite depth & in range & inside lens FOV & dz>1e-3 (numerical guard).
            # AND with caller-provided pcd_mask (pred: model sky mask; gt: eval valid).
            dz = np.abs(d_cam_s3hw[s, 2])
            m = np.isfinite(d) & (d > min_depth) & (d <= max_depth) & (dz > 1e-3) & lens
            # FoV cone crop: keep rays within `fov_deg` half-angle of the optical axis
            # (dz = |d_cam.z| = |cos theta|). Trims unreliable fisheye periphery.
            if fov_deg is not None and fov_deg > 0:
                m = m & (dz >= float(np.cos(np.deg2rad(fov_deg))))
            if pcd_mask_shw is not None:
                m = m & pcd_mask_shw[s].astype(bool)
            # Confidence threshold: among geometry+sky candidates, drop lowest conf_drop_pct%
            if conf_shw is not None and conf_drop_pct and m.any():
                cv = conf_shw[s][m].astype(np.float64)
                cv = cv[np.isfinite(cv)]
                if cv.size:
                    thr = float(np.percentile(cv, conf_drop_pct))
                    m = m & (conf_shw[s] >= thr)
            if subsample > 1:
                sub = np.zeros_like(m)
                sub[::subsample, ::subsample] = True
                m = m & sub
            if m.any():
                all_pts.append(pts_world[s][m])                       # (n,3)
                all_rgb.append(img_norm_to_rgb(images_s3hw[s])[m])    # (n,3)

    if save_pcd and all_pts:
        pts = np.concatenate(all_pts, 0)
        rgb = np.concatenate(all_rgb, 0)
        # Merged cloud: all views already in view0 world frame, concatenate directly
        save_ply(frame_dir / f"{model_name}__pcd.ply", pts, rgb)
