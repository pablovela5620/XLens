"""Data transforms for loading and preprocessing multi-view data."""

from typing import Optional, Sequence, Callable, List, Dict, Union
from abc import ABCMeta, abstractmethod
import os.path as osp

import cv2
import numpy as np


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms: Optional[Sequence[Callable]]):
        self.transforms: List[Callable] = []
        if transforms is None:
            transforms = []
        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"transform must be callable, got {type(transform)}")

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


class BaseTransform(metaclass=ABCMeta):
    """Transform base class."""
    def __call__(self, results):
        return self.transform(results)

    @abstractmethod
    def transform(self, results):
        pass


class LoadMultiViewFrame(BaseTransform):
    """Load multi-view frame data.

    For each (frame_name, camera_name) in views, load RGB, depth, intrinsics, and extrinsics.

    Cleaning:
    1. Depth: NaN/inf -> 0, negatives -> 0
    2. Intrinsics: validate 3x3 pinhole, no skew
    3. Extrinsics: validate 4x4 c2w, all finite
    """

    def transform(self, sample_info: dict) -> dict:
        views = sample_info["views"]  # [(frame_name, cam_name), ...]

        image_dir = osp.join(sample_info["scene_dir"], sample_info["image_dir"])
        depth_dir = osp.join(sample_info["scene_dir"], sample_info["depth_dir"])
        common_dir = osp.join(sample_info["scene_dir"], sample_info["common_dir"])

        image_format = sample_info["image_format"]
        depth_format = sample_info["depth_format"]
        common_format = sample_info["common_format"]

        # Cache loaded common data (shared across cameras of the same frame).
        common_cache = {}

        cameras_name = []
        cameras_rgb = {}
        cameras_depth = {}
        cameras_intrinsics_extrinsics = {}

        for frame_name, cam_name in views:
            view_id = f"{frame_name}_{cam_name}"
            cameras_name.append(view_id)

            # RGB
            image_path = osp.join(image_dir, cam_name, f"{frame_name}.{image_format}")
            image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            cameras_rgb[view_id] = image_data

            # Depth + cleaning: NaN/inf/negatives -> 0
            depth_path = osp.join(depth_dir, cam_name, f"{frame_name}.{depth_format}")
            depth_data = np.load(depth_path, allow_pickle=True).astype(np.float32)
            depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
            depth_data[depth_data < 0] = 0.0
            cameras_depth[view_id] = depth_data

            # Intrinsics/extrinsics + validation
            if frame_name not in common_cache:
                common_path = osp.join(common_dir, f"{frame_name}.{common_format}")
                common_cache[frame_name] = np.load(common_path, allow_pickle=True).item()

            cam_data = common_cache[frame_name][cam_name]
            K = cam_data["intrinsics"].astype(np.float32)
            c2w = cam_data["extrinsics_world"].astype(np.float32)

            assert K.shape == (3, 3), f"Intrinsics must be 3x3, got {K.shape}, view={view_id}"
            assert K[0, 1] == 0.0 and K[1, 0] == 0.0, f"Intrinsics have skew, view={view_id}"

            assert c2w.shape == (4, 4), f"Extrinsics must be 4x4, got {c2w.shape}, view={view_id}"
            assert np.isfinite(c2w).all(), f"Extrinsics have NaN/inf, view={view_id}"

            cameras_intrinsics_extrinsics[view_id] = {
                "intrinsics": K,
                "extrinsics_world": c2w,
            }

        return {
            "frame_info": sample_info,
            "cameras_name": cameras_name,
            "cameras_rgb": cameras_rgb,
            "cameras_depth": cameras_depth,
            "cameras_intrinsics_extrinsics": cameras_intrinsics_extrinsics,
        }


class Resize(BaseTransform):
    """Aspect-preserving resize + center crop to a target resolution.

    Resize so both dimensions are >= the target, then center-crop to the exact target. No padding;
    the backbone sees only valid pixels. K is updated throughout (scale + crop offset) for
    geometric consistency.

    Supports:
    1. Fixed resolution: target_size is a single (H, W) tuple.
    2. Random multi-resolution: target_size is a list [(H1, W1), ...].

    K updates:
        Step 1 (resize): K[0,0]*=sx, K[1,1]*=sy, K[0,2]*=sx, K[1,2]*=sy; sx=new_w/orig_w, sy=new_h/orig_h
        Step 2 (crop):   K[0,2]-=crop_left, K[1,2]-=crop_top
    """

    # Training resolutions.
    XLENS_RESOLUTIONS = [
        (504, 504), (504, 378), (504, 336), (504, 280),
        (336, 504), (896, 504), (756, 504), (672, 504),
    ]

    def __init__(
        self,
        target_size: Union[tuple, List[tuple]] = (504, 504),
        patch_size: int = 14,
        interpolation: str = "bilinear",
    ):
        self.patch_size = patch_size
        self.interpolation = interpolation

        # Normalize to a list of resolutions.
        if isinstance(target_size, list):
            self.resolutions = [tuple(s) for s in target_size]
        else:
            self.resolutions = [tuple(target_size)]

        # All resolutions must be divisible by patch_size (backbone token alignment).
        for h, w in self.resolutions:
            assert h % patch_size == 0 and w % patch_size == 0, \
                f"Resolution ({h}, {w}) not divisible by patch_size={patch_size}"

    def transform(self, data: dict) -> dict:
        cameras_name = data["cameras_name"]

        # Resolution: use the sampler's resolution_idx if given, else random.
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

            # Step 1: aspect-preserving resize so both dims >= target (scale = max), then crop.
            scale = max(target_h / orig_h, target_w / orig_w)
            new_h = max(int(round(orig_h * scale)), target_h)
            new_w = max(int(round(orig_w * scale)), target_w)

            # Compute sx/sy separately since rounding may perturb the ideal scale.
            sx = new_w / orig_w
            sy = new_h / orig_h

            resized_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Resize the non-ambiguous mask too (NEAREST preserves binary values).
            mask = cameras_mask.get(camera_name) if cameras_mask is not None else None
            resized_mask = None
            if mask is not None:
                resized_mask = cv2.resize(
                    mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST,
                ) > 0

            # K update - resize
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy

            # Step 2: center-crop to the exact target (target_h, target_w).
            crop_top = (new_h - target_h) // 2
            crop_left = (new_w - target_w) // 2
            cropped_rgb = resized_rgb[
                crop_top : crop_top + target_h,
                crop_left : crop_left + target_w,
            ]
            cropped_depth = resized_depth[
                crop_top : crop_top + target_h,
                crop_left : crop_left + target_w,
            ]
            cropped_mask = None
            if resized_mask is not None:
                cropped_mask = resized_mask[
                    crop_top : crop_top + target_h,
                    crop_left : crop_left + target_w,
                ]

            # K update - crop offset
            K[0, 2] -= crop_left
            K[1, 2] -= crop_top

            data["cameras_rgb"][camera_name] = cropped_rgb
            data["cameras_depth"][camera_name] = cropped_depth
            if cameras_mask is not None:
                cameras_mask[camera_name] = cropped_mask
            data["cameras_intrinsics_extrinsics"][camera_name]["intrinsics"] = K.astype(np.float32)

        return data


class ColorAugmentation(BaseTransform):
    """Color augmentation.

    Applies only to RGB; depth and camera parameters are untouched. Pure PIL/numpy.

    - ColorJitter(brightness=0.3, contrast=0.4, saturation=0.2, hue=0.1) - p=0.75
    - RandomGrayscale - p=0.05
    - GaussianBlur - p=0.05
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def _color_jitter(self, img, brightness=0.3, contrast=0.4, saturation=0.2, hue=0.1):
        """PIL-based color jitter."""
        from PIL import ImageEnhance
        import random as _rng

        # brightness
        if brightness > 0:
            factor = _rng.uniform(max(0, 1 - brightness), 1 + brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        # contrast
        if contrast > 0:
            factor = _rng.uniform(max(0, 1 - contrast), 1 + contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        # saturation
        if saturation > 0:
            factor = _rng.uniform(max(0, 1 - saturation), 1 + saturation)
            img = ImageEnhance.Color(img).enhance(factor)
        # hue: shift in HSV space
        if hue > 0:
            arr = np.array(img)
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + _rng.uniform(-hue, hue) * 180) % 180
            arr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            from PIL import Image
            img = Image.fromarray(arr)
        return img

    def transform(self, data: dict) -> dict:
        if not self.enabled:
            return data

        import random as _rng
        from PIL import Image, ImageFilter

        for camera_name in data["cameras_name"]:
            rgb = data["cameras_rgb"][camera_name]
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(rgb)

            # ColorJitter, p=0.75
            if _rng.random() < 0.75:
                pil_img = self._color_jitter(pil_img, 0.3, 0.4, 0.2, 0.1)

            # Grayscale, p=0.05
            if _rng.random() < 0.05:
                pil_img = pil_img.convert("L").convert("RGB")

            # GaussianBlur, p=0.05
            if _rng.random() < 0.05:
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=2))

            data["cameras_rgb"][camera_name] = np.array(pil_img)

        return data


class Normalize(BaseTransform):
    """ImageNet normalization."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def transform(self, data: dict) -> dict:
        for camera_name in data["cameras_name"]:
            rgb = data["cameras_rgb"][camera_name].astype(np.float32) / 255.0
            rgb = (rgb - self.IMAGENET_MEAN) / self.IMAGENET_STD
            data["cameras_rgb"][camera_name] = rgb
        return data
