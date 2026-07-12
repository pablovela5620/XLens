"""OmniOCC 6-cam pinhole depth eval loader (for eval_pinhole.run_eval).

OmniOCC GT is a merged cam0-frame lidar point cloud (no per-view depth maps). The point cloud is
projected into each view as z-depth (same convention as WAI GT z-depth), producing the batch
eval_scene expects:
    images / intrinsics / extrinsics_world / cam_types
    + gt_depth (S,H,W, z meters) / gt_valid (S,H,W, bool)
OmniOCC then uses the exact same metrics as WAI test (abs_rel/rmse/delta...), output separately.

Reuses eval_pinhole.omniocc_loader for images/calibration/point cloud (6 views = cam0/1/2 x left/right).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from evaluation.eval_pinhole import omniocc_loader as ol
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from evaluation.eval_pinhole import omniocc_loader as ol


def set_omniocc_root(root: Optional[str]) -> None:
    """Override the omniocc_loader data root."""
    if not root:
        return
    r = Path(root)
    ol.OMNIOCC_ROOT = r
    ol.SCENES_ROOT = r / "occ_real" / "occ_real_imagepcd"
    ol.CALIB_PATH = r / "extrinsic" / "real_calib.npy"
    ol.LIDAR_EXTRINSIC_DIR = r / "extrinsic"


def list_omniocc_scenes() -> List[str]:
    return ol.list_scenes(ol.SCENES_ROOT)


def load_calib():
    # Pass CALIB_PATH explicitly: ol.load_calib default is bound at import time; set_omniocc_root
    # only updates the module global, so the current value must be passed explicitly.
    return ol.load_calib(calib_path=ol.CALIB_PATH)


def _project_lidar_to_depths(
    gt_lidar_cam0: np.ndarray,   # (N,3) in cam0 frame
    intr: np.ndarray,            # (S,3,3)
    c2w: np.ndarray,             # (S,4,4) c2w in cam0 frame
    H: int, W: int,
    min_depth: float, max_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Point cloud -> per-view sparse z-depth (S,H,W) + valid (S,H,W).
    P_s = inv(c2w_s) @ P_cam0; pinhole projection, keep z in (min,max] and in-frame; nearest point per pixel."""
    S = intr.shape[0]
    gt_depth = np.zeros((S, H, W), dtype=np.float32)
    gt_valid = np.zeros((S, H, W), dtype=bool)
    pts0 = gt_lidar_cam0.astype(np.float64)                     # (N,3)
    pts0_h = np.concatenate([pts0, np.ones((pts0.shape[0], 1))], axis=1)  # (N,4)
    lo = max(min_depth, 1e-6)
    for s in range(S):
        w2c = np.linalg.inv(c2w[s].astype(np.float64))          # cam0 -> cam_s
        pc = (w2c @ pts0_h.T).T[:, :3]                          # (N,3) in cam_s
        z = pc[:, 2]
        front = z > lo
        if not front.any():
            continue
        K = intr[s].astype(np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x, y, zz = pc[front, 0], pc[front, 1], z[front]
        u = np.round(fx * x / zz + cx).astype(np.int64)
        v = np.round(fy * y / zz + cy).astype(np.int64)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zz <= max_depth)
        u, v, zz = u[inb], v[inb], zz[inb]
        if u.size == 0:
            continue
        # z-buffer: write far->near sorted, so numpy keeps the last (nearest) per pixel
        order = np.argsort(-zz)
        flat = (v[order] * W + u[order])
        dd = gt_depth[s].reshape(-1)
        vv = gt_valid[s].reshape(-1)
        dd[flat] = zz[order].astype(np.float32)
        vv[flat] = True
    return gt_depth, gt_valid


def load_omniocc_scene_batch(
    scene_name: str,
    target_hw: Tuple[int, int],
    min_depth: float = 0.05,
    max_depth: float = 80.0,
    calib: Optional[Dict] = None,
) -> Dict:
    """Return a scene_batch compatible with eval_pinhole.eval_scene (6 pinhole views + projected z-depth GT)."""
    sb = ol.load_scene_batch(scene_name, target_hw=target_hw,
                             scenes_root=ol.SCENES_ROOT, calib=calib)
    H, W = int(target_hw[0]), int(target_hw[1])
    intr = sb["intrinsics"][0].numpy()            # (S,3,3)
    c2w = sb["extrinsics_world"][0].numpy()        # (S,4,4)
    gt_lidar = sb["gt_lidar_cam0"].numpy()         # (N,3)
    gt_depth, gt_valid = _project_lidar_to_depths(
        gt_lidar, intr, c2w, H, W, min_depth, max_depth)
    S = intr.shape[0]
    return {
        "scene_name": scene_name,
        "dataset_name": "omniocc",
        "images": sb["images"],                    # (1,S,3,H,W)
        "intrinsics": sb["intrinsics"],            # (1,S,3,3)
        "extrinsics_world": sb["extrinsics_world"],  # (1,S,4,4)
        "cam_types": torch.ones(1, S, dtype=torch.int64),  # all pinhole = 1
        "gt_depth": gt_depth,                       # (S,H,W) z meters
        "gt_valid": gt_valid,                       # (S,H,W) bool
        "target_hw": (H, W),
        "num_views": S,
        "view_tags": sb.get("view_tags"),
    }
