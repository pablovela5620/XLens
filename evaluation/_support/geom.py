"""Geometry helper extracted verbatim from the training engine.

Only ``_compute_pinhole_ray_map`` is needed by the evaluation code (it derives the
per-pixel camera-frame unit rays ``d_cam`` and the 6-channel ``ray_map`` fed to the
RayMapEncoder from pinhole intrinsics + c2w extrinsics). It is wrapped in a thin
``RayMapHelper`` shim class so the eval call sites
(``RayMapHelper._compute_pinhole_ray_map(...)``) remain unchanged.

The function body is a faithful copy of
``xlens/engine/trainer.py::RayMapHelper._compute_pinhole_ray_map``;
numerics are identical.
"""

import torch
import torch.nn.functional as F


class RayMapHelper:
    @staticmethod
    def _compute_pinhole_ray_map(intrinsics, extrinsics_world, H, W, device,
                                  dtype=torch.float32, view_mask=None):
        """Compute d_cam (camera-frame unit rays) and ray_map from pinhole intrinsics + c2w extrinsics.

        For each pixel (u, v):
            d_cam = normalize([(u - cx) / fx, (v - cy) / fy, 1])
            d_world = R_c2w @ d_cam
            ray_map = cat([d_world, broadcast(t_normalized)], dim=channel)

        t normalization matches omni_collate.collate_omni_frames:
          pose_scale = mean(||t_i||) over valid views i > 0
          t_normalized = t / pose_scale
        so pinhole and fisheye RayMapEncoder inputs have matching t magnitudes in a mixed batch.

        Args:
            intrinsics:        (B, S, 3, 3)
            extrinsics_world:  (B, S, 4, 4) c2w (collate canonicalizes to view 0)
            H, W:              image resolution
            view_mask:         (B, S) bool valid-view mask. None treated as all True.
        Returns:
            d_cam:       (B, S, 3, H, W)
            ray_map:     (B, S, 6, H, W) - channels 0..2=d_world, 3..5=t_normalized broadcast
            pose_scale:  (B,) scale used to normalize t; also usable for depth-scale consistency
        """
        B, S = intrinsics.shape[:2]
        K = intrinsics.to(device=device, dtype=dtype)             # (B, S, 3, 3)
        c2w = extrinsics_world.to(device=device, dtype=dtype)     # (B, S, 4, 4)

        # 1) pixel grid (W is horizontal = u, H is vertical = v)
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )                                                          # (H, W) each
        u_grid = u_grid + 0.5                                      # pixel center
        v_grid = v_grid + 0.5

        fx = K[..., 0, 0].unsqueeze(-1).unsqueeze(-1)              # (B, S, 1, 1)
        fy = K[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
        cx = K[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
        cy = K[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

        x = (u_grid - cx) / fx.clamp(min=1e-6)                     # (B, S, H, W)
        y = (v_grid - cy) / fy.clamp(min=1e-6)
        z = torch.ones_like(x)
        d_cam = torch.stack([x, y, z], dim=2)                      # (B, S, 3, H, W)
        d_cam = F.normalize(d_cam, dim=2, eps=1e-6)

        # 2) d_world = R_c2w @ d_cam
        R = c2w[..., :3, :3]                                       # (B, S, 3, 3)
        d_cam_flat = d_cam.reshape(B, S, 3, H * W)                 # (B, S, 3, HW)
        d_world_flat = torch.matmul(R, d_cam_flat)                 # (B, S, 3, HW)
        d_world = d_world_flat.reshape(B, S, 3, H, W)

        # 3) normalize t (matches omni_collate): mean ||t|| over valid non-view-0 views
        t = c2w[..., :3, 3]                                        # (B, S, 3)
        if view_mask is None:
            view_mask = torch.ones(B, S, dtype=torch.bool, device=device)
        else:
            view_mask = view_mask.to(device=device, dtype=torch.bool)
        # view 0 is canonical (t=0), excluded from the baseline mean
        nonzero_view_mask = view_mask & (
            torch.arange(S, device=device).unsqueeze(0) > 0
        )                                                          # (B, S)
        t_norm = torch.linalg.norm(t, dim=-1)                      # (B, S)
        safe_count = nonzero_view_mask.sum(dim=-1).clamp(min=1).to(dtype)
        pose_scale = (t_norm * nonzero_view_mask).sum(dim=-1) / safe_count
        pose_scale = pose_scale.clamp(min=1e-6).to(dtype)          # (B,)

        t_normalized = t / pose_scale[:, None, None]               # (B, S, 3)
        t_broadcast = t_normalized.reshape(B, S, 3, 1, 1).expand(B, S, 3, H, W).contiguous()

        ray_map = torch.cat([d_world, t_broadcast], dim=2)         # (B, S, 6, H, W)
        return d_cam, ray_map, pose_scale
