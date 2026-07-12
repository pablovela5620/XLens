"""Collate multi-view samples from the DataLoader into batched tensors."""

from typing import Sequence
import numpy as np
import torch


def _affine_inverse(T: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of a 4x4 affine (c2w) matrix.

    [R | t] -> [R^T | -R^T t]. Supports batch dims (..., 4, 4).
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    Rt = R.transpose(-2, -1)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def _canonicalize_to_view0(extrinsics_c2w: torch.Tensor) -> torch.Tensor:
    """Transform all c2w extrinsics into the view-0 frame: T_i_new = inv(T_0) @ T_i.

    View 0 becomes the identity; view i is expressed relative to view 0. Standard
    multi-view preprocessing; without it camera_loss is ill-defined across samples.

    Args:
        extrinsics_c2w: (B, S, 4, 4) c2w matrices.

    Returns:
        (B, S, 4, 4) with [:, 0] equal to identity.
    """
    T0 = extrinsics_c2w[:, 0:1]              # (B, 1, 4, 4)
    T0_inv = _affine_inverse(T0)             # (B, 1, 4, 4)
    return T0_inv @ extrinsics_c2w           # (B, S, 4, 4)


def collate_multiview_frames(data_batch: Sequence[dict]) -> dict:
    """Collate multi-view samples into a batch.

    Supports variable view counts and resolutions: pads to the batch max with 0
    (images/intrinsics) or inf (depths).

    Output:
        images: (B, S_max, 3, H_max, W_max) float32
        depths: (B, S_max, H_max, W_max) float32
        intrinsics: (B, S_max, 3, 3) float32
        extrinsics_world: (B, S_max, 4, 4) float32
        valid_views: (B,) number of valid views per sample
        frames_info: List[dict]
        cameras_name: List[List[str]]
    """
    batch_size = len(data_batch)

    # Batch-max view count and resolution.
    max_views = 0
    max_h, max_w = 0, 0
    for item in data_batch:
        n_views = len(item["cameras_name"])
        max_views = max(max_views, n_views)
        first_cam = item["cameras_name"][0]
        h, w = item["cameras_rgb"][first_cam].shape[:2]
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    images = torch.zeros(batch_size, max_views, 3, max_h, max_w, dtype=torch.float32)
    depths = torch.full((batch_size, max_views, max_h, max_w), float("inf"), dtype=torch.float32)
    intrinsics = torch.zeros(batch_size, max_views, 3, 3, dtype=torch.float32)
    extrinsics_world = torch.zeros(batch_size, max_views, 4, 4, dtype=torch.float32)
    valid_views = torch.zeros(batch_size, dtype=torch.long)
    # Non-ambiguous mask GT. Padding views are 0 and filtered by view_mask in the loss.
    non_ambiguous_mask = torch.zeros(batch_size, max_views, max_h, max_w, dtype=torch.bool)
    # Whether each view actually has a GT mask; lets depth-loss AND it into valid_mask
    # only for views that provide one.
    has_non_ambiguous_mask = torch.zeros(batch_size, max_views, dtype=torch.bool)

    frames_info = []
    all_cameras_name = []

    for batch_idx, data_item in enumerate(data_batch):
        frames_info.append(data_item.get("frame_info", {}))
        all_cameras_name.append(data_item["cameras_name"])
        n_views = len(data_item["cameras_name"])
        valid_views[batch_idx] = n_views
        cameras_mask = data_item.get("cameras_non_ambiguous_mask", None)

        for cam_idx, view_id in enumerate(data_item["cameras_name"]):
            rgb = data_item["cameras_rgb"][view_id]
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            h, w = rgb.shape[:2]
            images[batch_idx, cam_idx, :, :h, :w] = torch.from_numpy(rgb).permute(2, 0, 1)

            depth = data_item["cameras_depth"][view_id]
            depths[batch_idx, cam_idx, :h, :w] = torch.from_numpy(np.ascontiguousarray(depth))

            if cameras_mask is not None:
                m = cameras_mask.get(view_id)
                if m is not None:
                    non_ambiguous_mask[batch_idx, cam_idx, :h, :w] = torch.from_numpy(
                        np.ascontiguousarray(m.astype(np.bool_))
                    )
                    has_non_ambiguous_mask[batch_idx, cam_idx] = True

            ie = data_item["cameras_intrinsics_extrinsics"][view_id]
            intrinsics[batch_idx, cam_idx] = torch.from_numpy(np.ascontiguousarray(ie["intrinsics"]))
            extrinsics_world[batch_idx, cam_idx] = torch.from_numpy(np.ascontiguousarray(ie["extrinsics_world"]))

    # Canonicalize extrinsics to view 0 of each sample. Done here so upstream
    # (transforms/sampler) stays agnostic and downstream (trainer/model/loss) always
    # receives view-0 canonical extrinsics. Padding views have all-zero T; T0_inv @ 0 = 0
    # stays invalid and is masked downstream via valid_views.
    extrinsics_world = _canonicalize_to_view0(extrinsics_world)

    # Per-sample dataset metadata for tier routing in trainer/loss.
    dataset_names = [
        fi.get("dataset_name", "") if isinstance(fi, dict) else "" for fi in frames_info
    ]
    quality_tiers = [
        fi.get("quality_tier", "A") if isinstance(fi, dict) else "A" for fi in frames_info
    ]
    is_metric_flags = [
        bool(fi.get("is_metric", False)) if isinstance(fi, dict) else False for fi in frames_info
    ]

    # view_mask: (B, S) True for valid (non-padding) views; MaskLoss uses it to filter padding.
    view_mask = (
        torch.arange(max_views).unsqueeze(0) < valid_views.unsqueeze(1)
    )  # (B, S) bool

    # cam_types: (B, S) camera type id. 0=fisheye, 1=pinhole. Defaults to pinhole;
    # a sample may override per view via the 'cameras_type' dict.
    cam_types_tensor = torch.ones(batch_size, max_views, dtype=torch.long)
    for bi, data_item in enumerate(data_batch):
        cam_type_map = data_item.get("cameras_type", None)
        if cam_type_map is None:
            continue
        for vi, view_id in enumerate(data_item["cameras_name"]):
            if view_id in cam_type_map:
                cam_types_tensor[bi, vi] = int(cam_type_map[view_id])

    return {
        "images": images,
        "depths": depths,
        "intrinsics": intrinsics,
        "extrinsics_world": extrinsics_world,
        "valid_views": valid_views,
        "view_mask": view_mask,
        "non_ambiguous_mask": non_ambiguous_mask,
        "has_non_ambiguous_mask": has_non_ambiguous_mask,
        "cam_types": cam_types_tensor,
        "frames_info": frames_info,
        "cameras_name": all_cameras_name,
        "dataset_names": dataset_names,
        "quality_tiers": quality_tiers,
        "is_metric": is_metric_flags,
    }
