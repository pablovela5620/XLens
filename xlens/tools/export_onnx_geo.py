#!/usr/bin/env python3
"""Export XLensNet to ONNX with ALL geometry baked in as constants.

Unlike ``export_onnx.py`` (images-only, geometry priors OFF), this exporter
bakes the full geometry of a fixed rig into the graph, so the ONNX model
reproduces the real ``inference.py`` path bit-for-bit:

    ray_map / d_cam / cam_types  -> constants (register_buffer)
    ray_feat = ray_encoder(ray_map)  -> precomputed constant (ray_encoder removed
                                        from the graph; keeps the first residual
                                        stable for downstream int quantization)

The only runtime input is ``images`` of shape (1, S, 3, H, W), ImageNet-normalized
outside the graph. S / H / W / camera order are all fixed by the manifest.

Geometry is taken from the SAME code the PyTorch inference uses
(``xlens.inference.preprocess.assemble_batch`` -> ray_map/d_cam/cam_types), so the
baked constants are identical to a real forward pass. The manifest is the usual
scene.json (see examples/hetero_loft).

Outputs: depth (normalized), depth_metric, depth_conf, metric_scaling_factor
(+ mask if the checkpoint has predict_mask).

Usage:
    python -m xlens.tools.export_onnx_geo \
        --ckpt DRVFM/output/final/model.safetensors \
        --config configs/xlens_vits.yaml \
        --manifest examples/hetero_loft/scene.json \
        --out out/hetero_loft/model_geo.onnx
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make `import xlens` resolvable when run as a script from anywhere.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xlens.inference import XLensInference                       # noqa: E402
from xlens.inference.preprocess import (                          # noqa: E402
    CAM_TYPE_FISHEYE, CAM_TYPE_PINHOLE,
    pinhole_d_cam, load_fisheye_lut, assemble_batch,
)
from xlens.inference.geometry import fuse_point_cloud, save_ply, save_depth_preview  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_onnx_geo")

PATCH = 14   # DINOv2 vit*14 patch size


# ---------------------------------------------------------------------------
# Manifest -> per-view arrays (mirrors inference.py)
# ---------------------------------------------------------------------------
def load_scene(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest_path).parent
    hw = manifest.get("image_size")            # [H, W] or None (keep native)

    def _load_image(path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if hw is not None:
            img = img.resize((hw[1], hw[0]), Image.BILINEAR)   # PIL is (W, H)
        return np.asarray(img, dtype=np.uint8)

    images, d_cams, cam_types, c2ws, masks, names = [], [], [], [], [], []
    for v in manifest["views"]:
        img = _load_image(root / v["image"])
        H, W = img.shape[:2]
        if "dcam" in v:                         # precomputed per-pixel unit rays
            d = np.load(root / v["dcam"]).astype(np.float32)
            d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
            cam_types.append(int(v.get("cam_type", CAM_TYPE_FISHEYE)))
        elif "lut" in v:                        # fisheye LUT
            d = load_fisheye_lut(str(root / v["lut"]), H, W)
            cam_types.append(int(v.get("cam_type", CAM_TYPE_FISHEYE)))
        elif "K" in v:                          # pinhole
            d = pinhole_d_cam(np.asarray(v["K"], np.float32), H, W)
            cam_types.append(int(v.get("cam_type", CAM_TYPE_PINHOLE)))
        else:
            raise ValueError("each view needs 'K', 'lut', or 'dcam'")
        images.append(img)
        d_cams.append(d)
        names.append(v.get("camera", v.get("image")))
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
    if c2w is None:
        raise ValueError("manifest has no c2w for all views; geometry cannot be baked "
                         "(ray_map requires poses). Add 'c2w' to every view.")
    return images, d_cams, cam_types, c2w, masks_np, names


# ---------------------------------------------------------------------------
# FishRoPE NaN fix: fully-out fisheye patches -> (0,0,1)
# ---------------------------------------------------------------------------
def fix_fully_out_patches(d_cam_t):
    """d_cam_t: (1,S,3,H,W) float. Fisheye LUT edges are (0,0,0); a 14x14 patch
    that is entirely out-of-lens avg-pools to (0,0,0), and FishRoPE's atan2(0,0)
    is 0 in PyTorch but NaN in ONNX. Fill only fully-out patches with (0,0,1):
    avg-pool -> az=el=0, identical to the PyTorch atan2(0,0)=0 result, so the
    graph output is unchanged. Boundary patches keep per-pixel zeros (bit-aligned
    with the PyTorch path). Pinhole views never hit this (z=1 everywhere).
    """
    d = d_cam_t.detach().cpu().numpy().copy()          # (1,S,3,H,W)
    _, S, _, H, W = d.shape
    Hp, Wp = H // PATCH, W // PATCH
    n_fixed = 0
    for s in range(S):
        rays = d[0, s]                                  # (3,H,W)
        norm = np.linalg.norm(rays, axis=0)             # (H,W)
        valid = norm > 1e-3                             # in-lens pixel
        cnt = valid.reshape(Hp, PATCH, Wp, PATCH).sum(axis=(1, 3))   # (Hp,Wp)
        fully_out = np.repeat(np.repeat(cnt == 0, PATCH, 0), PATCH, 1)  # (H,W)
        if fully_out.any():
            d[0, s, 0][fully_out] = 0.0
            d[0, s, 1][fully_out] = 0.0
            d[0, s, 2][fully_out] = 1.0
            n_fixed += int((cnt == 0).sum())
    logger.info("FishRoPE fix: filled %d fully-out patches with (0,0,1)", n_fixed)
    return torch.from_numpy(d).to(d_cam_t.device, d_cam_t.dtype)


# ---------------------------------------------------------------------------
# Static calib attention mask (removes NonZero / boolean-index from the graph)
# ---------------------------------------------------------------------------
def patch_static_calib_mask(device):
    """Replace vision_transformer.build_calib_attention_mask with a version that
    builds the additive mask in numpy from a *detached* inject_mask and returns a
    fresh ``torch.from_numpy`` constant.

    inject_mask is derived from the baked (constant) cam_types, so the mask is
    static. The original builds it with boolean-indexed assignment
    (``mask[not_inject] = -inf``), which the ONNX tracer lowers to NonZero +
    ScatterND with a data-dependent shape -> ORT shape-inference rejects the graph.
    Returning a from_numpy constant severs it from traced inputs. Values are
    identical (finfo.min), so the PyTorch reference is unchanged; slim_onnx later
    clamps finfo.min -> -1e4 for softmax stability. Results are cached and the same
    tensor object is reused across identical layers to bound trace-time memory.
    """
    import xlens.models.dinov2.vision_transformer as VT

    cache = {}
    NEG = float(np.finfo(np.float32).min)

    def static_build(K_per_view, inject_mask, N_non_calib, attn_type, device=device,
                     dtype=torch.float32):
        if K_per_view <= 0:
            return None
        im = inject_mask.detach().cpu().numpy().astype(bool)          # (B, S)
        if not im.any():
            return None
        B, S = im.shape
        key = (attn_type, int(K_per_view), int(N_non_calib), B, S, im.tobytes())
        if key in cache:
            return cache[key]
        Lpv = N_non_calib + K_per_view
        if attn_type == "local":
            Bp = B * S
            mask = np.zeros((Bp, Lpv, Lpv), np.float32)
            not_inject = (~im).reshape(Bp)
            if not_inject.any():
                mask[not_inject, N_non_calib:, :] = NEG
                mask[not_inject, :, N_non_calib:] = NEG
        elif attn_type == "global":
            L = S * Lpv
            mask = np.zeros((B, L, L), np.float32)
            for b in range(B):
                for v in range(S):
                    lo = v * Lpv + N_non_calib
                    hi = (v + 1) * Lpv
                    if not im[b, v]:
                        mask[b, lo:hi, :] = NEG
                        mask[b, :, lo:hi] = NEG
                    else:
                        for vo in range(S):
                            if vo == v:
                                continue
                            o0, o1 = vo * Lpv, (vo + 1) * Lpv
                            mask[b, lo:hi, o0:o1] = NEG
                            mask[b, o0:o1, lo:hi] = NEG
        else:
            raise ValueError(f"unknown attn_type: {attn_type}")
        t = torch.from_numpy(mask).to(device=device, dtype=dtype)
        cache[key] = t
        return t

    VT.build_calib_attention_mask = static_build
    logger.info("patched build_calib_attention_mask -> static numpy constant")


# ---------------------------------------------------------------------------
# ONNX-friendliness patches (fused attention -> manual softmax)
# ---------------------------------------------------------------------------
def make_onnx_friendly(model):
    # 1) DINOv2 Attention: turn off F.scaled_dot_product_attention (opset<20).
    try:
        from xlens.models.dinov2.layers.attention import Attention as DinoAttn
        n = 0
        for m in model.modules():
            if isinstance(m, DinoAttn) and getattr(m, "fused_attn", False):
                m.fused_attn = False
                n += 1
        logger.info("disabled fused_attn on %d DINOv2 Attention modules", n)
    except ImportError:
        pass

    # 2) nn.MultiheadAttention (scale head): its eval fastpath emits
    #    aten::_native_multi_head_attention, unsupported in ONNX. Replace with a
    #    plain matmul+softmax that is numerically equivalent (batch_first, self-attn).
    def _manual_mha_forward(self, query, key, value, key_padding_mask=None,
                            need_weights=True, attn_mask=None,
                            average_attn_weights=True, is_causal=False):
        B, L, E = query.shape
        Hh = self.num_heads
        Dh = E // Hh
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, Hh, Dh).transpose(1, 2)
        k = k.view(B, L, Hh, Dh).transpose(1, 2)
        v = v.view(B, L, Hh, Dh).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (Dh ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(out), None

    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.MultiheadAttention):
            m.forward = types.MethodType(_manual_mha_forward, m)
            n += 1
    if n:
        logger.info("patched %d nn.MultiheadAttention.forward (manual softmax)", n)


# ---------------------------------------------------------------------------
# ONNX wrapper: bake geometry, precompute ray_feat
# ---------------------------------------------------------------------------
class OnnxWrapper(nn.Module):
    def __init__(self, model, ray_map, d_cam, cam_types, out_keys):
        super().__init__()
        self.model = model
        with torch.no_grad():
            ray_feat = model.ray_encoder(ray_map.float())   # constant
        self.register_buffer("ray_feat", ray_feat)
        self.register_buffer("d_cam", d_cam)
        self.register_buffer("cam_types", cam_types)
        self.out_keys = out_keys

    def forward(self, images):
        out = self.model(images, ray_feat=self.ray_feat,
                         d_cam=self.d_cam, cam_types=self.cam_types)
        return tuple(out[k] for k in self.out_keys)


# ---------------------------------------------------------------------------
# ONNX slimming: dedup big constants + clamp -inf/finfo.min masks
# ---------------------------------------------------------------------------
def _compress_masks(g, numpy_helper, onnx, fp16=True):
    """Shrink the baked calib attention masks (0 / -1e4 additive, ~99% zeros).

    Two lossless edits, both verified against the broadcast `Add` that consumes
    each mask (out = attn_scores + mask):
      1) head-squeeze: masks are stored (B', H, L, L) but identical across the H
         heads (a broadcast of one 3D mask that the exporter materialized). Slice
         to (B', 1, L, L); the Add broadcasts over heads -> bit-identical, H x
         smaller.
      2) fp16 (optional): 0 and -1e4 are both exact in fp16, and softmax(-1e4)=0
         regardless, so casting the mask to fp16 is bit-identical after softmax.
         A Cast->float32 is inserted before each consumer to keep the Add valid.
    Only touches tensors that look like additive masks (FLOAT, 4D, square LxL,
    large, values in [<=0, min<-1e3]); model weights and distortion bias untouched.
    """
    # consumers: input_name -> list of (node, input_index)
    consumers = {}
    for node in g.node:
        for i, inp in enumerate(node.input):
            consumers.setdefault(inp, []).append((node, i))

    n_sq, n_fp16, saved = 0, 0, 0
    new_casts = []
    for node in g.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        t = next((a.t for a in node.attribute if a.name == "value"), None)
        if t is None or t.data_type != onnx.TensorProto.FLOAT or len(t.dims) != 4:
            continue
        d = list(t.dims)
        if d[2] != d[3] or d[2] * d[3] < 1_000_000:
            continue
        arr = numpy_helper.to_array(t)
        if not (arr.min() < -1e3 and arr.max() <= 0.0):
            continue                      # not an additive mask
        before = arr.nbytes

        # (1) head-squeeze — only if all heads identical
        if d[1] > 1 and np.array_equal(arr[:, :1], arr[:, 1:2]) and \
           np.array_equal(arr[:, :1].repeat(d[1], 1), arr):
            arr = arr[:, :1]
            n_sq += 1

        # (2) fp16 + Cast
        if fp16:
            arr16 = arr.astype(np.float16)
            t.CopyFrom(numpy_helper.from_array(arr16, t.name))
            cast_out = node.output[0] + "_f32"
            cast = onnx.helper.make_node("Cast", [node.output[0]], [cast_out],
                                         name=node.name + "_castf32",
                                         to=onnx.TensorProto.FLOAT)
            for cn, ci in consumers.get(node.output[0], []):
                if cn is not cast:
                    cn.input[ci] = cast_out
            new_casts.append(cast)
            n_fp16 += 1
        else:
            t.CopyFrom(numpy_helper.from_array(arr, t.name))
        saved += before - arr.astype(np.float16 if fp16 else np.float32).nbytes

    g.node.extend(new_casts)
    logger.info("compress masks: head-squeezed %d, fp16 %d, saved ~%.0f MB",
                n_sq, n_fp16, saved / 1e6)


def slim_onnx(src_model_path, out_path, fp16_mask=True):
    import hashlib
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(src_model_path))
    g = m.graph

    # (a) dedup identical large Constants (per-layer calib masks are identical)
    seen, remap, rm = {}, {}, []
    for node in g.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            t = next((a.t for a in node.attribute if a.name == "value"), None)
            if t is None or not t.raw_data or len(t.raw_data) < 1_000_000:
                continue
            key = (tuple(t.dims), t.data_type, hashlib.md5(t.raw_data).hexdigest())
            if key in seen:
                remap[node.output[0]] = seen[key]; rm.append(node)
            else:
                seen[key] = node.output[0]
    for n in rm:
        g.node.remove(n)
    for node in g.node:
        for i, inp in enumerate(node.input):
            if inp in remap:
                node.input[i] = remap[inp]

    # (b) clamp finfo.min / -inf mask values to -1e4 (softmax-equivalent, no overflow)
    def _clamp(t):
        if t.data_type != onnx.TensorProto.FLOAT:
            return 0
        arr = numpy_helper.to_array(t)
        if arr.size and arr.min() < -1e30:
            a = arr.copy(); a[a < -1e30] = -1e4
            t.CopyFrom(numpy_helper.from_array(a, t.name)); return 1
        return 0
    nclamp = 0
    for node in g.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    nclamp += _clamp(a.t)
    for t in g.initializer:
        nclamp += _clamp(t)

    # (c) compress calib masks (head-squeeze + optional fp16); bit-identical output
    _compress_masks(g, numpy_helper, onnx, fp16=fp16_mask)

    onnx.save(m, str(out_path), save_as_external_data=False)
    logger.info("slimmed: dedup %d large constants, clamp %d masks", len(rm), nclamp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=str(REPO / "configs/xlens_vits.yaml"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--constant_folding", action="store_true",
                    help="enable do_constant_folding (default off: folding the baked "
                         "ray_map/d_cam constants runs subgraphs and can OOM)")
    ap.add_argument("--no_fp16_mask", action="store_true",
                    help="keep calib masks in fp32 (default: fp16, bit-identical after "
                         "softmax). Head-squeeze is always applied.")
    ap.add_argument("--skip_verify", action="store_true")
    ap.add_argument("--verify_dir", default=None,
                    help="dump ONNX depth preview + fused point cloud here (default: <out_dir>/verify)")
    a = ap.parse_args()

    device = torch.device(a.device)

    # 1) Scene geometry (identical to inference.py)
    images, d_cams, cam_types, c2w, masks_np, names = load_scene(a.manifest)
    S = len(images)
    H, W = images[0].shape[:2]
    logger.info("scene: %d views %s  cam_types=%s  HxW=%dx%d",
                S, names, cam_types, H, W)
    if H % PATCH or W % PATCH:
        raise ValueError(f"H={H},W={W} must be multiples of {PATCH}")

    batch = assemble_batch(images, d_cams, cam_types, c2w=c2w, device=str(device))
    ray_map = batch["ray_map"].float()
    d_cam = fix_fully_out_patches(batch["d_cam"].float())
    cam_types_t = batch["cam_types"]

    # 2) Model (fp32, no autocast) via the released loader
    infer = XLensInference(a.ckpt, device=str(device), config=a.config)
    model = infer.model.float().eval().to(device)
    make_onnx_friendly(model)
    patch_static_calib_mask(device)

    # 3) Determine output keys present
    real_images = batch["images"].float()
    with torch.no_grad():
        probe = model(real_images, ray_feat=None, ray_map=ray_map,
                      d_cam=d_cam, cam_types=cam_types_t)
    canonical = ["depth", "depth_metric", "depth_conf", "metric_scaling_factor", "mask"]
    out_keys = [k for k in canonical if k in probe]
    logger.info("output keys: %s", out_keys)

    wrapper = OnnxWrapper(model, ray_map, d_cam, cam_types_t, out_keys).eval().to(device)

    # 4) Export (real images as the trace input) into an isolated temp dir to
    #    contain torch's per-constant external-data files, then merge + slim.
    out_path = Path(a.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / (out_path.stem + "__export_tmp")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    prev = os.getcwd()
    os.chdir(work)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper, (real_images,), "model.onnx",
                input_names=["images"], output_names=out_keys,
                opset_version=a.opset,
                do_constant_folding=a.constant_folding,
                dynamic_axes=None,      # everything fixed (geometry bound to rig/res)
                dynamo=False,
            )
    finally:
        os.chdir(prev)
    slim_onnx(work / "model.onnx", out_path, fp16_mask=not a.no_fp16_mask)
    shutil.rmtree(work)
    logger.info("exported -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    if a.skip_verify:
        return

    # 5) Verify on REAL images: PyTorch(wrapper, fp32) vs ONNXRuntime, then dump
    #    depth preview + fused point cloud from the ONNX output (correctness eyeball).
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping verification")
        return

    with torch.no_grad():
        ref = wrapper(real_images)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(out_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"images": real_images.cpu().numpy()})

    logger.info("--- PyTorch(fp32 wrapper) vs ONNXRuntime, REAL images ---")
    for name, r, o in zip(out_keys, ref, onnx_out):
        r_np = r.detach().cpu().numpy()
        diff = np.abs(r_np - o).max()
        rel = diff / (np.abs(r_np).max() + 1e-9)
        logger.info("  %-22s shape=%s  max|Δ|=%.3e  rel=%.3e",
                    name, tuple(r_np.shape), diff, rel)

    # dump ONNX depth preview + point cloud
    vdir = Path(a.verify_dir) if a.verify_dir else out_path.parent / "verify"
    vdir.mkdir(parents=True, exist_ok=True)
    depth = onnx_out[out_keys.index("depth_metric")][0]                # (S,H,W)
    conf = onnx_out[out_keys.index("depth_conf")][0] if "depth_conf" in out_keys else None
    d_cam_np = d_cam[0].cpu().numpy().transpose(0, 2, 3, 1)            # (S,H,W,3)
    rgb = np.stack(images)                                             # (S,H,W,3)
    np.save(vdir / "depth_metric_onnx.npy", depth)
    save_depth_preview(vdir / "depth_preview_onnx.png", depth, valid=masks_np)
    has_fisheye = any(t == CAM_TYPE_FISHEYE for t in cam_types)
    points, colors = fuse_point_cloud(
        depth, d_cam_np, c2w, rgb=rgb, conf=conf,
        conf_drop_pct=8.0 if has_fisheye else 0.0,
        fov_max_deg=85.0 if has_fisheye else None,
        masks=masks_np, max_depth=25.0)
    save_ply(vdir / "points_onnx.ply", points, colors)
    logger.info("verify artifacts -> %s (%d points)", vdir, len(points))


if __name__ == "__main__":
    main()
