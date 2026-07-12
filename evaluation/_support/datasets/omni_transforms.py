"""Omni fisheye dataset transforms.

LoadOmniFrame: load RGB / depth (inf->0) / c2w / K / static mask, and read per-pixel d_cam
               (camera-frame unit ray direction) from the LUT EXR.
OmniResize:    subclass of Resize that also transforms the d_cam tensor during resize+crop.

LUT loading uses a global cache (keyed by lut_dir + cam_name), read once per worker process.
"""

import os.path as osp
import logging
from threading import Lock

import cv2
import numpy as np

from .transforms import BaseTransform, Resize

logger = logging.getLogger(__name__)


# Global LUT cache: {lut_dir: {cam_name: (H, W, 3) np.float32}}
_LUT_CACHE: dict = {}
_LUT_LOCK = Lock()


def _load_exr_rgb(path: str) -> np.ndarray:
    """Read OpenEXR (R, G, B) channels -> (H, W, 3) float32."""
    try:
        import OpenEXR
        import Imath
    except ImportError as e:
        raise ImportError(
            "OpenEXR is required to read LUT files (pip install OpenEXR Imath)"
        ) from e

    f = OpenEXR.InputFile(path)
    header = f.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    chans = []
    for c in ("R", "G", "B"):
        raw = f.channel(c, pt)
        chans.append(np.frombuffer(raw, dtype=np.float32).reshape(h, w))
    return np.stack(chans, axis=-1)


def _get_lut(lut_dir: str, cam_name: str) -> np.ndarray:
    """Return the d_cam LUT (H, W, 3) float32 in OpenCV camera convention, globally cached.

    The LUT EXR files are baked in RTX convention (X right, Y up, Z backward), while the stored
    extrinsics_world (c2w) are OpenCV (X right, Y down, Z forward). All downstream code
    (collate d_world = R @ d_cam, depth-loss normals, eval) assumes OpenCV, so the LUT is
    converted to OpenCV once here by negating Y and Z. Edge pixels (0,0,0) stay (0,0,0).
    """
    with _LUT_LOCK:
        per_dir = _LUT_CACHE.setdefault(lut_dir, {})
        if cam_name not in per_dir:
            path = osp.join(lut_dir, f"{cam_name}_rayEnterDirection.exr")
            arr = _load_exr_rgb(path).astype(np.float32)
            # RTX -> OpenCV: negate Y and Z.
            arr[..., 1] = -arr[..., 1]
            arr[..., 2] = -arr[..., 2]
            per_dir[cam_name] = arr
        return per_dir[cam_name]


def _pinhole_d_cam(K: np.ndarray, h: int, w: int) -> np.ndarray:
    """Per-pixel viewing direction d_cam (H, W, 3) float32 unit vectors for a pinhole camera.

    Same convention as the fisheye LUT (OpenCV: X right, Y down, Z forward). No LUT is needed;
    computed from K as d = normalize(K^{-1} @ [u+0.5, v+0.5, 1]^T) (+0.5 = pixel center).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us = (np.arange(w, dtype=np.float32) + 0.5 - cx) / fx
    vs = (np.arange(h, dtype=np.float32) + 0.5 - cy) / fy
    uu, vv = np.meshgrid(us, vs)                          # (H, W)
    dirs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-6)
    return dirs.astype(np.float32)


class LoadOmniFrame(BaseTransform):
    """Load an Omni fisheye frame.

    Differences from LoadMultiViewFrame:
      - rgb directory is 'rgb/'
      - depth npy contains inf, replaced with 0
      - loads LUT EXR -> d_cam (H, W, 3) camera-frame unit vectors
      - loads static per-camera mask (CAM_X_mask.png, lens valid region)

    Output data dict:
      cameras_name:                  List[str] view_id
      cameras_rgb:                   {view_id: (H, W, 3) uint8}
      cameras_depth:                 {view_id: (H, W) float32, inf cleared}
      cameras_intrinsics_extrinsics: {view_id: {intrinsics, extrinsics_world}}
      cameras_d_cam:                 {view_id: (H, W, 3) float32 unit vectors}
      cameras_non_ambiguous_mask:    {view_id: (H, W) bool lens valid region}  (optional)
    """

    def transform(self, sample_info: dict) -> dict:
        views = sample_info["views"]

        scene_dir = sample_info["scene_dir"]
        image_dir = osp.join(scene_dir, sample_info["image_dir"])
        depth_dir = osp.join(scene_dir, sample_info["depth_dir"])
        common_dir = osp.join(scene_dir, sample_info["common_dir"])
        mask_dir = osp.join(scene_dir, sample_info.get("mask_dir", "mask"))
        # per-frame cleaned mask directory; None means disabled.
        vm_subdir = sample_info.get("valid_mask_dir", None)
        valid_mask_dir = osp.join(scene_dir, vm_subdir) if vm_subdir else None
        lut_dir = sample_info["lut_dir"]

        image_format = sample_info["image_format"]
        depth_format = sample_info["depth_format"]
        common_format = sample_info["common_format"]

        # Pinhole camera set: no LUT, d_cam from K, cam_type=1. All others are fisheye
        # (LUT, cam_type=0). Empty for pure-fisheye datasets.
        pinhole_cams = set(sample_info.get("pinhole_cams") or [])

        # Cameras of one frame share the common npy; each camera's static mask is read once.
        common_cache = {}
        mask_cache = {}

        cameras_name = []
        cameras_rgb = {}
        cameras_depth = {}
        cameras_intrinsics_extrinsics = {}
        cameras_d_cam = {}
        cameras_mask = {}
        cameras_type = {}

        for frame_name, cam_name in views:
            view_id = f"{frame_name}_{cam_name}"
            cameras_name.append(view_id)

            # ---- RGB ----
            img_path = osp.join(image_dir, cam_name, f"{frame_name}.{image_format}")
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            assert img is not None, f"Failed to read: {img_path}"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cameras_rgb[view_id] = img

            # ---- Depth (with inf) ----
            dpth_path = osp.join(depth_dir, cam_name, f"{frame_name}.{depth_format}")
            depth = np.load(dpth_path, allow_pickle=True).astype(np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth[depth < 0] = 0.0
            cameras_depth[view_id] = depth

            # ---- Intrinsics & Extrinsics ----
            if frame_name not in common_cache:
                cp = osp.join(common_dir, f"{frame_name}.{common_format}")
                common_cache[frame_name] = np.load(cp, allow_pickle=True).item()
            cam_data = common_cache[frame_name][cam_name]
            K = cam_data["intrinsics"].astype(np.float32)
            c2w = cam_data["extrinsics_world"].astype(np.float32)
            assert K.shape == (3, 3), f"K shape: {K.shape}"
            assert c2w.shape == (4, 4), f"c2w shape: {c2w.shape}"
            assert np.isfinite(c2w).all(), f"c2w has NaN/inf, view={view_id}"
            cameras_intrinsics_extrinsics[view_id] = {
                "intrinsics": K,
                "extrinsics_world": c2w,
            }

            # ---- d_cam: fisheye from LUT, pinhole from K ----
            if cam_name in pinhole_cams:
                # Pinhole: freshly computed array (not cached), no copy needed.
                cameras_d_cam[view_id] = _pinhole_d_cam(K, img.shape[0], img.shape[1])
                cameras_type[view_id] = 1   # pinhole
            else:
                d_cam = _get_lut(lut_dir, cam_name)  # (H, W, 3)
                assert d_cam.shape[:2] == img.shape[:2], (
                    f"LUT {d_cam.shape} does not match image {img.shape}, cam={cam_name}"
                )
                # copy: resize mutates in place, must not pollute the global cache.
                cameras_d_cam[view_id] = d_cam.copy()
                cameras_type[view_id] = 0   # fisheye

            # ---- static lens mask (cached per cam, shared across frames) ----
            if cam_name not in mask_cache:
                mp = osp.join(mask_dir, f"{cam_name}_mask.png")
                if osp.exists(mp):
                    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    mask_cache[cam_name] = (m > 0)
                else:
                    mask_cache[cam_name] = None
            m = mask_cache[cam_name]
            if m is not None:
                # Copy per view since resize mutates in place.
                final_mask = m.copy()
            else:
                final_mask = None

            # ---- per-frame cleaned valid_mask (sky / inf / far-range filter, ANDed with lens) ----
            #   <scene>/valid_mask/CAM_X/{frame}.npy   (bool)
            #   <scene>/valid_mask/CAM_X/{frame}.npz   (packed bits + shape)
            if valid_mask_dir is not None:
                vm_npy = osp.join(valid_mask_dir, cam_name, f"{frame_name}.npy")
                vm_npz = osp.join(valid_mask_dir, cam_name, f"{frame_name}.npz")
                per_frame_mask = None
                if osp.exists(vm_npy):
                    per_frame_mask = np.load(vm_npy).astype(bool)
                elif osp.exists(vm_npz):
                    z = np.load(vm_npz)
                    pf = np.unpackbits(z["packed"])[: int(np.prod(z["shape"]))]
                    per_frame_mask = pf.reshape(tuple(z["shape"])).astype(bool)
                if per_frame_mask is not None:
                    if final_mask is None:
                        final_mask = per_frame_mask.copy()
                    else:
                        # Nearest-neighbor resize to the lens mask shape on mismatch.
                        if per_frame_mask.shape != final_mask.shape:
                            per_frame_mask = cv2.resize(
                                per_frame_mask.astype(np.uint8),
                                (final_mask.shape[1], final_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        final_mask &= per_frame_mask

            # ---- per-frame sky mask (sky from depth>thr, for outdoor scenes) ----
            # <scene>/sky_mask/CAM_X/{frame}.png (255=sky). Folded into non_ambiguous_mask
            # (sky set to 0): removes sky from depth supervision (avoids treating the far skybox
            # as GT) and provides "sky=invalid" as mask-loss BCE GT. Pinhole cameras without a
            # lens mask get their first mask here.
            sky_subdir = sample_info.get("sky_mask_dir", "sky_mask")
            if sky_subdir:
                sky_path = osp.join(scene_dir, sky_subdir, cam_name, f"{frame_name}.png")
                if osp.exists(sky_path):
                    sky_img = cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)
                    if sky_img is not None:
                        sky_bool = sky_img > 0
                        if final_mask is None:
                            final_mask = ~sky_bool
                        else:
                            if sky_bool.shape != final_mask.shape:
                                sky_bool = cv2.resize(
                                    sky_bool.astype(np.uint8),
                                    (final_mask.shape[1], final_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(bool)
                            final_mask &= ~sky_bool

            if final_mask is not None:
                cameras_mask[view_id] = final_mask

        out = {
            "frame_info": sample_info,
            "cameras_name": cameras_name,
            "cameras_rgb": cameras_rgb,
            "cameras_depth": cameras_depth,
            "cameras_intrinsics_extrinsics": cameras_intrinsics_extrinsics,
            "cameras_d_cam": cameras_d_cam,
            "cameras_type": cameras_type,   # {view_id: 0=fisheye / 1=pinhole}, read by collate for cam_types
        }
        if cameras_mask:
            out["cameras_non_ambiguous_mask"] = cameras_mask
        return out


class OmniResize(Resize):
    """Resize subclass that also transforms cameras_d_cam on each resize+crop:
      - INTER_LINEAR interpolation (vector field)
      - re-normalize to unit vectors afterward (interpolation breaks unit length)
      - pixels originally (0,0,0) stay 0 when the norm is below threshold

    K is still updated with the pinhole formula and kept as metadata; the actual geometry flows
    through ray_map and does not depend on K.
    """

    def transform(self, data: dict) -> dict:
        cameras_name = data["cameras_name"]
        cameras_d_cam = data.get("cameras_d_cam")

        # Select resolution (as in the parent class).
        import random as _random
        resolution_idx = data.get("resolution_idx", None)
        if resolution_idx is not None:
            resolution_idx = min(resolution_idx, len(self.resolutions) - 1)
            target_h, target_w = self.resolutions[resolution_idx]
        else:
            target_h, target_w = _random.choice(self.resolutions)

        cameras_mask = data.get("cameras_non_ambiguous_mask", None)

        for camera_name in cameras_name:
            rgb = data["cameras_rgb"][camera_name]
            depth = data["cameras_depth"][camera_name]
            orig_h, orig_w = rgb.shape[:2]
            K = data["cameras_intrinsics_extrinsics"][camera_name]["intrinsics"].copy()

            # Aspect-preserving resize with new_h >= target_h and new_w >= target_w.
            scale = max(target_h / orig_h, target_w / orig_w)
            new_h = max(int(round(orig_h * scale)), target_h)
            new_w = max(int(round(orig_w * scale)), target_w)
            sx = new_w / orig_w
            sy = new_h / orig_h

            resized_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            mask = cameras_mask.get(camera_name) if cameras_mask is not None else None
            resized_mask = None
            if mask is not None:
                resized_mask = cv2.resize(
                    mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST,
                ) > 0

            # d_cam (3-channel vector field)
            resized_d_cam = None
            if cameras_d_cam is not None and camera_name in cameras_d_cam:
                d = cameras_d_cam[camera_name]
                resized_d_cam = cv2.resize(d, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Re-normalize to unit vectors; zero vectors stay zero (avoid divide-by-zero).
                norm = np.linalg.norm(resized_d_cam, axis=-1, keepdims=True)
                safe_norm = np.maximum(norm, 1e-6)
                resized_d_cam = (resized_d_cam / safe_norm).astype(np.float32)

            # Scale K.
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy

            # Center-crop to the exact target size.
            crop_top = (new_h - target_h) // 2
            crop_left = (new_w - target_w) // 2
            cropped_rgb = resized_rgb[crop_top:crop_top + target_h, crop_left:crop_left + target_w]
            cropped_depth = resized_depth[crop_top:crop_top + target_h, crop_left:crop_left + target_w]
            cropped_mask = (
                resized_mask[crop_top:crop_top + target_h, crop_left:crop_left + target_w]
                if resized_mask is not None else None
            )
            cropped_d_cam = (
                resized_d_cam[crop_top:crop_top + target_h, crop_left:crop_left + target_w]
                if resized_d_cam is not None else None
            )

            K[0, 2] -= crop_left
            K[1, 2] -= crop_top

            data["cameras_rgb"][camera_name] = cropped_rgb
            data["cameras_depth"][camera_name] = cropped_depth
            if cameras_mask is not None:
                cameras_mask[camera_name] = cropped_mask
            if cropped_d_cam is not None:
                cameras_d_cam[camera_name] = cropped_d_cam
            data["cameras_intrinsics_extrinsics"][camera_name]["intrinsics"] = K.astype(np.float32)

        return data
