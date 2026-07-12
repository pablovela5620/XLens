"""
OmniOCC 6-camera data loading (cam0/cam1/cam2 x left/right, skipping cam3).

A scene yields a batch dict matching the xlens training dataloader:
    images:           (1, 6, 3, H, W)   ImageNet-normalized float tensor
    intrinsics:       (1, 6, 3, 3)      GT K, rescaled
    extrinsics_world: (1, 6, 4, 4)      c2w in view0 frame (cam0 = I)
    valid_views:      (1,) = [6]

Plus GT geometry:
    gt_lidar_cam0:    (N, 3)            merged lidar cloud in cam0 frame (meters)

View order (view0 = cam0_left):
    [cam0_left, cam0_right, cam1_left, cam1_right, cam2_left, cam2_right]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# Constants

# OmniOCC dataset root; override with OMNIOCC_ROOT:
#   export OMNIOCC_ROOT=/your/local/path/to/OmniOCC/real
import os as _os
OMNIOCC_ROOT = Path(_os.environ.get(
    "OMNIOCC_ROOT",
    "/path/to/data/OmniOCC/real",
))
SCENES_ROOT = OMNIOCC_ROOT / "occ_real" / "occ_real_imagepcd"
CALIB_PATH = OMNIOCC_ROOT / "extrinsic" / "real_calib.npy"
LIDAR_EXTRINSIC_DIR = OMNIOCC_ROOT / "extrinsic"

# Skip cam3; use cam0/cam1/cam2 x left+right = 6 views
USE_CAM_IDS: List[int] = [0, 1, 2]
USE_SIDES: List[str] = ["left", "right"]

# ImageNet normalization (DINOv2 / X-Lens)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# Calibration
# ============================================================================

def load_calib(calib_path: Path = CALIB_PATH) -> Dict[str, np.ndarray]:
    """
    Returns:
        intrinsics: dict[int, (3,3)]       cam_id -> K
        baselink2cam: dict[int, (4,4)]     cam_id -> w2c-style (baselink as world)
        cam2lidar: dict[int, (4,4)]        cam_id -> 4x4 (from cam{i}_to_lidar.npy)
    """
    calib = np.load(calib_path, allow_pickle=True).item()
    out: Dict[str, np.ndarray] = {
        "intrinsics": {},
        "baselink2cam": {},
        "cam2lidar": {},
    }
    for cid in USE_CAM_IDS:
        out["intrinsics"][cid] = np.asarray(calib[f"cam{cid}_intrinsic"], dtype=np.float64)
        out["baselink2cam"][cid] = np.asarray(calib[f"baselink2cam{cid}"], dtype=np.float64)
        # cam{i}_to_lidar.npy provided separately
        cam2lidar_path = LIDAR_EXTRINSIC_DIR / f"cam{cid}_to_lidar.npy"
        out["cam2lidar"][cid] = np.load(cam2lidar_path).astype(np.float64)
    return out


def compute_c2w_in_cam0_frame(calib: Dict, cam_id: int) -> np.ndarray:
    """
    Compose cam_id -> cam0 c2w via the cam{i}_to_lidar chain:
        P_cam0 = inv(T_c0->L) @ T_ci->L @ P_cam_i
        => c2w_cam_i_in_cam0 = inv(T_cam0_to_lidar) @ T_cam_i_to_lidar

    Use cam_to_lidar (not baselink2cam): both are independent calibrations and
    disagree by several degrees per camera. GT lidar and pred share the same
    cam_to_lidar chain, keeping the whole pipeline self-consistent.

    cam_id == 0: c2w = I.
    """
    T_c0_to_L = calib["cam2lidar"][0]
    T_ci_to_L = calib["cam2lidar"][cam_id]
    return np.linalg.inv(T_c0_to_L) @ T_ci_to_L


def compute_lidar_to_cam0(calib: Dict) -> np.ndarray:
    """lidar -> cam0 frame transform (4x4).

    inv(cam0_to_lidar); shares the cam_to_lidar extrinsics with
    compute_c2w_in_cam0_frame so GT lidar and pred live in the same cam0 frame.
    """
    return np.linalg.inv(calib["cam2lidar"][0])


# Scene list

def list_scenes(scenes_root: Path = SCENES_ROOT) -> List[str]:
    return sorted([d.name for d in scenes_root.iterdir() if d.is_dir()])


# Single-scene loading

def _load_image(path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    """Read JPG -> resize -> ImageNet normalize. Returns (3, H, W) float32."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target_h, target_w = target_hw
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(img, (2, 0, 1))  # (3, H, W)


def _scale_intrinsics(K: np.ndarray, orig_hw: Tuple[int, int], target_hw: Tuple[int, int]) -> np.ndarray:
    """Rescale K by the resize ratio: fx, cx * (W_new/W_old); fy, cy * (H_new/H_old)."""
    oh, ow = orig_hw
    th, tw = target_hw
    sx = tw / ow
    sy = th / oh
    K_new = K.copy()
    K_new[0, 0] *= sx  # fx
    K_new[0, 2] *= sx  # cx
    K_new[1, 1] *= sy  # fy
    K_new[1, 2] *= sy  # cy
    return K_new


def _load_lidar_pts(scene_dir: Path, scene_name: str) -> np.ndarray:
    """
    Read scene_dir/<scene_name>_merged.npy, return (N, 3) xyz in lidar frame.
    File is (N, 4) or (N, 5); columns 0-2 are xyz, the rest intensity / label.
    """
    npy_path = scene_dir / f"{scene_name}_merged.npy"
    if not npy_path.exists():
        # Some scenes only have Example_<name>_merged.npy
        npy_path = scene_dir / f"Example_{scene_name}_merged.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"lidar cloud not found: {scene_dir}/<{scene_name}>_merged.npy")
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"invalid lidar file shape: {arr.shape}")
    xyz = arr[:, :3].astype(np.float32)
    # Drop NaN / Inf
    keep = np.isfinite(xyz).all(axis=1)
    return xyz[keep]


def load_scene_batch(
    scene_name: str,
    target_hw: Tuple[int, int] = (504, 588),   # (H, W), ~1.18 aspect vs 1088x1280
    scenes_root: Path = SCENES_ROOT,
    calib: Dict = None,
) -> Dict:
    """
    Load one scene, returning a training-style batch dict + GT geometry.

    Args:
        scene_name: e.g. '4f_ba1'
        target_hw:  inference resolution (H, W). Default 504x588 (~4:3)
        scenes_root: occ_real_imagepcd directory
        calib:      preloaded calibration dict

    Returns:
        dict:
            images: torch.FloatTensor (1, 6, 3, H, W)
            intrinsics: torch.FloatTensor (1, 6, 3, 3)  rescaled to target_hw
            extrinsics_world: torch.FloatTensor (1, 6, 4, 4)  c2w in cam0 frame
            depths: torch.FloatTensor (1, 6, H, W)  (zeros, interface placeholder)
            valid_views: torch.IntTensor (1,) = [6]
            gt_lidar_cam0: torch.FloatTensor (N, 3)  merged lidar in cam0 frame (meters)
            scene_name: str
            orig_hw: (1088, 1280)
            target_hw: (H, W)
            view_tags: list[str], e.g. ['cam0_left', 'cam0_right', ...]
    """
    if calib is None:
        calib = load_calib()
    scene_dir = Path(scenes_root) / scene_name

    # ---- Images + K + c2w per view ----
    view_tags: List[str] = []
    imgs: List[np.ndarray] = []
    Ks: List[np.ndarray] = []
    c2ws: List[np.ndarray] = []

    # Probe the first image for original resolution
    probe = cv2.imread(str(scene_dir / "000000_cam0_left.jpg"), cv2.IMREAD_COLOR)
    if probe is None:
        raise FileNotFoundError(f"scene missing cam0_left: {scene_dir}")
    orig_h, orig_w = probe.shape[:2]

    for cid in USE_CAM_IDS:
        K_scaled = _scale_intrinsics(calib["intrinsics"][cid], (orig_h, orig_w), target_hw)
        c2w = compute_c2w_in_cam0_frame(calib, cid)

        for side in USE_SIDES:
            tag = f"cam{cid}_{side}"
            img_path = scene_dir / f"000000_{tag}.jpg"
            imgs.append(_load_image(img_path, target_hw))
            Ks.append(K_scaled.copy())
            # left/right share c2w (calib only gives camera-level pose; ~cm baseline ignored)
            c2ws.append(c2w.copy())
            view_tags.append(tag)

    images = torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(0).float()       # (1, 6, 3, H, W)
    intrinsics = torch.from_numpy(np.stack(Ks, axis=0)).unsqueeze(0).float()      # (1, 6, 3, 3)
    extrinsics = torch.from_numpy(np.stack(c2ws, axis=0)).unsqueeze(0).float()   # (1, 6, 4, 4)
    depths = torch.zeros(1, 6, target_hw[0], target_hw[1], dtype=torch.float32)  # placeholder

    # ---- Lidar cloud -> cam0 frame ----
    lidar_pts = _load_lidar_pts(scene_dir, scene_name)                            # (N, 3) in lidar frame
    T_lidar2cam0 = compute_lidar_to_cam0(calib)                                   # 4×4
    R = T_lidar2cam0[:3, :3]
    t = T_lidar2cam0[:3, 3]
    gt_lidar_cam0 = (R @ lidar_pts.T).T + t                                       # (N, 3)

    # cam_types: OmniOCC is all pinhole; training collate defaults to 1 (pinhole).
    # Pass it in eval too, otherwise cam_type_embed=None diverges from training.
    cam_types = torch.ones(1, 6, dtype=torch.long)

    return {
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics_world": extrinsics,
        "depths": depths,
        "valid_views": torch.tensor([6], dtype=torch.int64),
        "cam_types": cam_types,
        "gt_lidar_cam0": torch.from_numpy(gt_lidar_cam0).float(),
        "scene_name": scene_name,
        "orig_hw": (orig_h, orig_w),
        "target_hw": tuple(target_hw),
        "view_tags": view_tags,
    }
