"""Omni fisheye dataset collate.

Extends collate_multiview_frames (which canonicalizes extrinsics to view 0 = world) with:

  ray_map (B, S, 6, H, W) float32:
      channels 0..2 = d_world (world-frame per-pixel unit ray direction)
      channels 3..5 = t (camera center, broadcast over the image, sample-normalized)

  d_cam (B, S, 3, H, W) float32:
      camera-frame unit rays (LUT values, resized); used by dropout mode 2 (intrinsics only).

  pose_scale_factor (B,) float32:
      normalization factor for t (mean ||t_i|| over non-view-0 views, clamped >= 1e-6).

World-frame convention matches _canonicalize_to_view0: view 0 -> I, so view 0 has t = 0 and
d_world = d_cam.
"""

import torch

from .collate import collate_multiview_frames


def collate_omni_frames(data_batch):
    """Return all collate_multiview_frames fields plus ray_map / d_cam / pose_scale_factor."""
    B = len(data_batch)

    # Extract cameras_d_cam first; the parent collate does not recognize it. Order matches batch.
    d_cam_per_sample = [item.pop("cameras_d_cam", None) for item in data_batch]

    out = collate_multiview_frames(data_batch)

    extrinsics = out["extrinsics_world"]   # (B, S, 4, 4) c2w canonical (view0=I)
    _, S_max, H_max, W_max = out["depths"].shape

    # 1. d_cam tensor (B, S, 3, H, W)
    d_cam_tensor = torch.zeros(B, S_max, 3, H_max, W_max, dtype=torch.float32)
    for bi in range(B):
        per_view = d_cam_per_sample[bi]
        if per_view is None:
            continue
        for vi, view_id in enumerate(out["cameras_name"][bi]):
            d = per_view.get(view_id)
            if d is None:
                continue
            h, w = d.shape[:2]
            # d shape: (h, w, 3) -> (3, h, w)
            d_cam_tensor[bi, vi, :, :h, :w] = torch.from_numpy(d).permute(2, 0, 1)

    # 2. d_world = R @ d_cam (per-pixel)
    R = extrinsics[..., :3, :3].contiguous()                # (B, S, 3, 3)
    t = extrinsics[..., :3, 3].contiguous()                 # (B, S, 3)
    d_world = torch.einsum("bsij,bsjhw->bsihw", R, d_cam_tensor)  # (B, S, 3, H, W)

    # 3. Normalize t, excluding padding views and view 0 (whose t is always 0).
    view_mask = out["view_mask"]                            # (B, S) bool
    t_norm = torch.linalg.norm(t, dim=-1)                   # (B, S)
    nonzero_view_mask = view_mask & (
        torch.arange(S_max, device=view_mask.device).unsqueeze(0) > 0
    )                                                       # (B, S)
    safe_count = nonzero_view_mask.sum(dim=-1).clamp(min=1).float()
    pose_scale = (t_norm * nonzero_view_mask).sum(dim=-1) / safe_count   # (B,)
    pose_scale = pose_scale.clamp(min=1e-6).float()

    t_normalized = t / pose_scale[:, None, None]            # (B, S, 3)

    # 4. ray_map (B, S, 6, H, W): cat([d_world, broadcast(t)])
    t_broadcast = t_normalized[..., None, None].expand(B, S_max, 3, H_max, W_max)
    ray_map = torch.cat([d_world, t_broadcast], dim=2).contiguous()

    # 5. Zero pixels outside the lens FOV. d_world is already 0 there (LUT = (0,0,0)), but t is
    # broadcast over the image, leaving ray_map = [0,0,0, t] outside the FOV. Gate the whole vector
    # with the GT lens mask, matching the depth-loss valid_mask semantics.
    nam = out.get("non_ambiguous_mask")           # (B, S, H, W) bool
    has = out.get("has_non_ambiguous_mask")       # (B, S) bool
    if nam is not None and has is not None:
        gate = torch.where(
            has[:, :, None, None], nam, torch.ones_like(nam)
        ).unsqueeze(2).float()                    # (B, S, 1, H, W)
        ray_map = ray_map * gate
        d_cam_tensor = d_cam_tensor * gate

    out["ray_map"] = ray_map
    out["d_cam"] = d_cam_tensor
    out["pose_scale_factor"] = pose_scale

    # cam_types: camera type id (B, S) int64. 0 = fisheye (default), 1 = pinhole.
    # Use the sample's cameras_type dict if present, else all 0.
    cam_types_tensor = torch.zeros(B, S_max, dtype=torch.long)
    for bi in range(B):
        cam_type_map = data_batch[bi].get("cameras_type", None)
        if cam_type_map is None:
            continue
        for vi, view_id in enumerate(out["cameras_name"][bi]):
            t_id = cam_type_map.get(view_id, 0)
            cam_types_tensor[bi, vi] = int(t_id)
    out["cam_types"] = cam_types_tensor
    return out
