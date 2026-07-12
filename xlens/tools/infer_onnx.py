#!/usr/bin/env python3
"""Run a geometry-baked X-Lens ONNX model (from ``export_onnx_geo.py``).

The ONNX graph has all calibration baked in, so the network's only runtime input
is ``images``. This CLI mirrors ``inference.py`` but swaps the PyTorch model for
an ONNX Runtime session: it reads the same ``scene.json`` manifest, feeds only the
images, and writes per-view metric depth (.npy + preview) plus a fused world-frame
point cloud (.ply).

The manifest's per-view ``dcam`` / ``K`` / ``c2w`` are used ONLY to (a) resize the
images to the baked resolution / order and (b) unproject depth into a point cloud
outside the graph — they are never fed to the network. The rig must match the one
the ONNX was exported for.

Usage:
    python -m xlens.tools.infer_onnx \
        --onnx onnx/xlens_hetero_loft.onnx \
        --manifest examples/hetero_loft/scene.json \
        --out out/hetero_onnx --provider cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from xlens.inference.preprocess import (
    CAM_TYPE_FISHEYE, CAM_TYPE_PINHOLE,
    pinhole_d_cam, load_fisheye_lut, assemble_batch,
)
from xlens.inference.geometry import fuse_point_cloud, save_ply, save_depth_preview

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_image(path, hw):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if hw is not None:
        img = img.resize((hw[1], hw[0]), Image.BILINEAR)   # PIL is (W, H)
    return np.asarray(img, dtype=np.uint8)


def load_scene(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest_path).parent
    hw = manifest.get("image_size")
    images, d_cams, cam_types, c2ws, masks = [], [], [], [], []
    for v in manifest["views"]:
        img = _load_image(root / v["image"], hw)
        H, W = img.shape[:2]
        if "dcam" in v:
            d = np.load(root / v["dcam"]).astype(np.float32)
            d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
            cam_types.append(int(v.get("cam_type", CAM_TYPE_FISHEYE)))
        elif "lut" in v:
            d = load_fisheye_lut(str(root / v["lut"]), H, W)
            cam_types.append(int(v.get("cam_type", CAM_TYPE_FISHEYE)))
        elif "K" in v:
            d = pinhole_d_cam(np.asarray(v["K"], np.float32), H, W)
            cam_types.append(int(v.get("cam_type", CAM_TYPE_PINHOLE)))
        else:
            raise ValueError("each view needs 'K', 'lut', or 'dcam'")
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
    masks_np = np.stack(masks) if len(masks) == len(images) else None
    return images, d_cams, cam_types, c2w, masks_np


def make_session(onnx_path, provider):
    import onnxruntime as ort
    pref = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }[provider]
    avail = ort.get_available_providers()
    chosen = [p for p in pref if p in avail] or ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    # The baked calib masks are huge constants; skip graph optimizers that would
    # try to fold them and blow up memory.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=chosen)
    logging.info("onnxruntime providers: %s", sess.get_providers())
    return sess


def main():
    ap = argparse.ArgumentParser(description="X-Lens ONNX inference (geometry baked in)")
    ap.add_argument("--onnx", required=True, help="geometry-baked .onnx from export_onnx_geo.py")
    ap.add_argument("--manifest", required=True, help="scene JSON manifest (same rig as the export)")
    ap.add_argument("--out", default="out/onnx_scene", help="output directory")
    ap.add_argument("--provider", default="cuda", choices=["cuda", "trt", "cpu"],
                    help="ONNX Runtime execution provider (default: cuda)")
    ap.add_argument("--conf-drop-pct", type=float, default=-1.0,
                    help="drop the lowest N%% of points by confidence. Default: 8 with fisheye, else 0")
    ap.add_argument("--fov-max", type=float, default=-1.0,
                    help="drop rays > this many deg off-axis. Default: 85 with fisheye, else off")
    ap.add_argument("--no-fisheye-clean", action="store_true",
                    help="disable the default fisheye cleanup (FoV<=85 + drop lowest 8%% confidence)")
    ap.add_argument("--max-depth", type=float, default=None, help="drop points beyond this euclidean range (m)")
    ap.add_argument("--no-fusion", action="store_true", help="depth only, skip point-cloud fusion")
    args = ap.parse_args()

    images, d_cams, cam_types, c2w, masks = load_scene(args.manifest)
    S, (H, W) = len(images), images[0].shape[:2]
    logging.info("scene: %d views %s, HxW=%dx%d", S, cam_types, H, W)

    # Build the model input. Only `images` is fed to the graph; d_cam / c2w here
    # are for fusion + preview only.
    batch = assemble_batch(images, d_cams, cam_types, c2w=c2w, device="cpu")
    img_in = batch["images"].numpy().astype(np.float32)   # (1,S,3,H,W)

    sess = make_session(args.onnx, args.provider)
    exp = sess.get_inputs()[0].shape
    if list(exp) != list(img_in.shape):
        raise ValueError(f"ONNX expects images {exp} but manifest gives {list(img_in.shape)} "
                         "— the rig / resolution / view count must match the export.")
    onames = [o.name for o in sess.get_outputs()]
    out = dict(zip(onames, sess.run(None, {"images": img_in})))
    depth = out["depth_metric"][0]                                # (S,H,W)
    conf = out["depth_conf"][0] if "depth_conf" in out else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "depth_metric.npy", depth)
    logging.info("saved depth -> %s", out_dir / "depth_metric.npy")
    if save_depth_preview(out_dir / "depth_preview.png", depth, valid=masks):
        logging.info("saved depth preview -> %s", out_dir / "depth_preview.png")

    if args.no_fusion or c2w is None:
        logging.info("no point-cloud fusion (%s)", "requested" if args.no_fusion else "manifest has no c2w")
        return

    has_fisheye = any(t == CAM_TYPE_FISHEYE for t in cam_types)
    clean = has_fisheye and not args.no_fisheye_clean
    fov_max = args.fov_max if args.fov_max >= 0 else (85.0 if clean else None)
    conf_drop = args.conf_drop_pct if args.conf_drop_pct >= 0 else (8.0 if clean else 0.0)
    d_cam_np = batch["d_cam"][0].numpy().transpose(0, 2, 3, 1)    # (S,H,W,3)
    rgb = np.stack(images)
    points, colors = fuse_point_cloud(depth, d_cam_np, c2w, rgb=rgb, conf=conf,
                                      conf_drop_pct=conf_drop, fov_max_deg=fov_max,
                                      masks=masks, max_depth=args.max_depth)
    save_ply(out_dir / "points.ply", points, colors)
    logging.info("saved %d points -> %s", len(points), out_dir / "points.ply")


if __name__ == "__main__":
    main()
