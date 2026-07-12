#!/usr/bin/env python3
"""X-Lens inference CLI.

Run one multi-view scene through a released checkpoint and dump per-view metric
depth (.npy) plus a fused world-frame point cloud (.ply).

The scene is described by a small JSON manifest listing, per view, an image and
its camera calibration. Three modes are supported and can be mixed in one scene:

    pinhole        view has an intrinsics matrix "K" (3x3)          -> cam_type 1
    fisheye        view has a calibration LUT "lut" (.npy/.exr)     -> cam_type 0
    heterogeneous  a manifest containing both kinds of views

Example manifest (examples/scene_pinhole.json):
    {
      "image_size": [504, 798],
      "views": [
        {"image": "imgs/cam0.jpg", "K": [[f,0,cx],[0,f,cy],[0,0,1]], "c2w": [[...4x4...]]},
        {"image": "imgs/cam1.jpg", "K": [[...]],                       "c2w": [[...]]}
      ]
    }
For fisheye views replace "K" with "lut": "calib/cam0_lut.npy". "c2w" is optional
(if omitted for all views, the geometry prior / ray_map is skipped).

Usage:
    python inference.py --ckpt checkpoints/xlens_vits_stage3.pth \\
        --manifest examples/scene_hetero.json --out out/scene0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from xlens.inference import XLensInference
from xlens.inference.preprocess import (
    CAM_TYPE_FISHEYE, CAM_TYPE_PINHOLE,
    pinhole_d_cam, load_fisheye_lut, resize_d_cam, assemble_batch,
)
from xlens.inference.geometry import fuse_point_cloud, save_ply, save_depth_preview

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_image(path: Path, hw):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if hw is not None:
        img = img.resize((hw[1], hw[0]), Image.BILINEAR)   # PIL is (W, H)
    return np.asarray(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(description="X-Lens inference")
    ap.add_argument("--ckpt", required=True, help="path to a released .pth or .safetensors checkpoint")
    ap.add_argument("--config", default=None, help="arch config YAML (required for a bare .safetensors)")
    ap.add_argument("--manifest", required=True, help="scene JSON manifest")
    ap.add_argument("--out", default="out/scene", help="output directory prefix")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf-thresh", type=float, default=0.0, help="drop points with confidence below this absolute value")
    ap.add_argument("--conf-drop-pct", type=float, default=-1.0,
                    help="drop the lowest N%% of points by confidence (0-100). "
                         "Default: 8 when the scene has fisheye views, else 0.")
    ap.add_argument("--fov-max", type=float, default=-1.0,
                    help="drop pixels more than this many degrees off the optical axis. "
                         "Default: 85 when the scene has fisheye views, else off. Trims high-distortion "
                         "fisheye periphery where depth is unreliable.")
    ap.add_argument("--no-fisheye-clean", action="store_true",
                    help="disable the recommended automatic fisheye cleanup (FoV<=85 + drop lowest 8%% confidence)")
    ap.add_argument("--max-depth", type=float, default=None, help="drop points beyond this euclidean range (m), e.g. sky")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.manifest).parent
    hw = manifest.get("image_size")            # [H, W] or None (keep native)

    images, d_cams, cam_types, c2ws, masks = [], [], [], [], []
    for v in manifest["views"]:
        img = _load_image(root / v["image"], hw)
        H, W = img.shape[:2]
        if "dcam" in v:                         # precomputed per-pixel unit rays (any camera model)
            d = np.load(root / v["dcam"]).astype(np.float32)
            d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
            cam_types.append(int(v.get("cam_type", CAM_TYPE_FISHEYE)))
        elif "lut" in v:                        # fisheye view (LUT of unit rays)
            d = load_fisheye_lut(str(root / v["lut"]), H, W)
            cam_types.append(int(v.get("cam_type", CAM_TYPE_FISHEYE)))
        elif "K" in v:                          # pinhole view
            d = pinhole_d_cam(np.asarray(v["K"], np.float32), H, W)
            cam_types.append(int(v.get("cam_type", CAM_TYPE_PINHOLE)))
        else:
            raise ValueError("each view needs 'K' (pinhole), 'lut' (fisheye), or 'dcam' (precomputed rays)")
        images.append(img)
        d_cams.append(d)
        if "c2w" in v:
            c2ws.append(np.asarray(v["c2w"], np.float32))
        if "mask" in v:
            from PIL import Image
            mk = Image.open(root / v["mask"]).convert("L")
            if hw is not None:
                mk = mk.resize((hw[1], hw[0]), Image.NEAREST)
            masks.append(np.asarray(mk) > 127)

    c2w = np.stack(c2ws) if len(c2ws) == len(images) else None
    mode = ("heterogeneous" if len(set(cam_types)) > 1
            else "pinhole" if cam_types[0] == CAM_TYPE_PINHOLE else "fisheye")
    logging.info("scene: %d views, mode=%s, ray_map=%s", len(images), mode, c2w is not None)

    batch = assemble_batch(images, d_cams, cam_types, c2w=c2w, device=args.device)
    model = XLensInference(args.ckpt, device=args.device, config=args.config)
    out = model(batch)

    depth = out["depth_metric"][0].numpy()                      # (S, H, W)
    conf = out["depth_conf"][0].numpy() if "depth_conf" in out else None
    d_cam_np = batch["d_cam"][0].cpu().numpy().transpose(0, 2, 3, 1)   # (S, H, W, 3)
    rgb = np.stack(images)                                       # (S, H, W, 3)

    masks_np = np.stack(masks) if len(masks) == len(images) else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "depth_metric.npy", depth)
    logging.info("saved depth -> %s", out_dir / "depth_metric.npy")
    if save_depth_preview(out_dir / "depth_preview.png", depth, valid=masks_np):
        logging.info("saved depth preview -> %s", out_dir / "depth_preview.png")

    if c2w is not None:
        # Recommended fisheye cleanup: high lens distortion makes peripheral depth unreliable, so
        # by default trim rays >85 deg off-axis and drop the lowest 8% confidence when any fisheye
        # view is present. Override with --fov-max / --conf-drop-pct, or disable via --no-fisheye-clean.
        has_fisheye = any(t == CAM_TYPE_FISHEYE for t in cam_types)
        fisheye_clean = has_fisheye and not args.no_fisheye_clean
        fov_max = args.fov_max if args.fov_max >= 0 else (85.0 if fisheye_clean else None)
        conf_drop = args.conf_drop_pct if args.conf_drop_pct >= 0 else (8.0 if fisheye_clean else 0.0)
        if fisheye_clean and args.fov_max < 0 and args.conf_drop_pct < 0:
            logging.info("fisheye views present -> applying recommended cleanup (FoV<=85 deg + drop lowest 8% confidence)")

        points, colors = fuse_point_cloud(depth, d_cam_np, c2w, rgb=rgb,
                                          conf=conf, conf_thresh=args.conf_thresh,
                                          conf_drop_pct=conf_drop, fov_max_deg=fov_max,
                                          masks=masks_np, max_depth=args.max_depth)
        save_ply(out_dir / "points.ply", points, colors)
        logging.info("saved %d points -> %s", len(points), out_dir / "points.ply")
    else:
        logging.info("no c2w in manifest -> skipping point-cloud fusion (depth only)")


if __name__ == "__main__":
    main()
