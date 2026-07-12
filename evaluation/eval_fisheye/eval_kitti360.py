#!/usr/bin/env python3
"""KITTI-360 fisheye 2-view (motion-stereo) X-Lens depth evaluation.

Runs strictly per kitti360_eval_protocol.json (sequences/cameras/frames/resolution/
metrics/mask all fixed for comparability):
  * Data: fisheye_stereo_eval manifest. Each sample = [target (view0, has GT) + gap-frame
    support]; baseline comes from vehicle motion of a single fisheye camera (temporal
    stereo, not a rig). n_views=2, cam_type=0 (fisheye).
  * The model is given GT intrinsics + extrinsics (same inference/metrics as
    eval_heterogeneous.eval_calib_compare):
      - X-Lens: ray_map (= d_world + t, view0-canonical) + d_cam + cam_types -> depth_metric (z).
  * Depth convention (euclidean distance vs z-depth):
      - manifest raw GT = euclidean_range (distance along the ray to the camera center,
        uint16 png / scale=256).
      - KITTI360StereoDataset.load_gt already converts z = range * cos_theta
        (cos_theta = unit ray z component d_cam[2]) to z-depth, matching omni/WAI and
        model depth_metric output. So batch["depths"] is z-depth.
      - Both model outputs are also z-depth, so GT(z) vs pred(z) share the convention.
  * Mask: valid pixel = GT in (0.05,80] & finite & inside lens_mask (lens FoV). No sky mask.
  * Metrics (shared RelDepthAccumulator with eval_calib_compare, fisheye bucket):
      scale_absrel / depth_absrel (relative) / rmse / tau (d<1.03) / delta1 (d<1.25).
      Per-view median alignment.

Usage:
    python -m evaluation.eval_heterogeneous.eval_kitti360 \
        --stage3_ckpt /path/to/model.safetensors --config configs/xlens_vits.yaml \
        --out_dir ./eval_kitti360_out

    # smoke: first N frames per manifest
    python -m ...eval_kitti360 --stage3_ckpt .../best_model.pth --max_frames_per_manifest 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
    from evaluation._support.datasets.kitti360_stereo import (
        KITTI360StereoDataset, collate_kitti360_stereo,
    )
    from evaluation.eval_heterogeneous.eval_calib_compare import (
        build_stage3_model, infer_stage3, build_valid,
        ModelAccumulators, accumulate_frame, print_table, print_fps,
        parse_ckpt_specs, default_ckpt_label, dump_frame_set,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from evaluation._support.datasets.kitti360_stereo import (
        KITTI360StereoDataset, collate_kitti360_stereo,
    )
    from evaluation.eval_heterogeneous.eval_calib_compare import (
        build_stage3_model, infer_stage3, build_valid,
        ModelAccumulators, accumulate_frame, print_table, print_fps,
        parse_ckpt_specs, default_ckpt_label, dump_frame_set,
    )

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_kitti360")

DEFAULT_PROTOCOL = str(Path(__file__).resolve().parent / "kitti360_eval_protocol.json")


def _to_mono(batch):
    """Make a frame batch monocular: replace view1's image/geometry with copies of
    view0 (the model needs 2 views; feeding two identical images provides no extra
    temporal/stereo cue = monocular), and set view_mask[:,1]=False.

    Effect:
      - infer_stage3: still uses S=2 but both identical -> depth from a single image.
      - build_valid: view_mask ANDs into valid pixels -> view1 not scored, view0 only.
    Both KITTI-360 views are the same camera (d_cam/intrinsics shared); only
    images/ray_map/extrinsics vary per frame.
    """
    b = dict(batch)
    for k in ("images", "ray_map", "d_cam", "extrinsics_world", "intrinsics", "cam_types"):
        t = b.get(k)
        if torch.is_tensor(t) and t.dim() >= 2 and t.shape[1] >= 2:
            t = t.clone()
            t[:, 1] = t[:, 0]
            b[k] = t
    vm = b.get("view_mask")
    if torch.is_tensor(vm) and vm.dim() >= 2 and vm.shape[1] >= 2:
        vm = vm.clone()
        vm[:, 1:] = False
        b["view_mask"] = vm
    return b


def _resolve_root(protocol: Dict) -> str:
    """Prefer the override root if it exists, else fall back to the default root."""
    efs = protocol.get("kitti360_root_override_efs")
    if efs and Path(efs).exists():
        return efs
    return protocol["kitti360_root"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3_ckpt", type=Path, default=None,
                    help="single ckpt (legacy). For multiple, use --stage3_ckpts.")
    ap.add_argument("--config", type=Path, default=None,
                    help="model arch config yaml; overrides weight-inference. "
                         "For the released .safetensors pass configs/xlens_vits.yaml.")
    ap.add_argument("--stage3_ckpts", nargs="+", default=None,
                    help="evaluate several stage3 ckpts; each 'label=path' or bare 'path'. "
                         "Data loading runs once, shared across ckpts.")
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL, type=Path,
                    help="kitti360_eval_protocol.json (fixed sequences/frames/resolution/metrics)")
    ap.add_argument("--kitti360_root", default=None,
                    help="override protocol data root (default: override root if it exists, else default)")
    ap.add_argument("--num_workers", type=int, default=8,
                    help="DataLoader prefetch workers per manifest (0=synchronous main process)")
    ap.add_argument("--max_frames_per_manifest", type=int, default=0,
                    help="evaluate first N frames per (seq,cam) manifest, 0=all per protocol (smoke)")
    ap.add_argument("--mono", action="store_true",
                    help="monocular mode: view1 is a copy of view0 (model needs 2 views), view1 masked -> "
                         "metrics on view0 only. Evaluates pure monocular depth (no temporal/stereo cue).")
    ap.add_argument("--out_dir", default="./eval_kitti360_out", type=Path)
    ap.add_argument("--dump_dir", type=Path, default=None,
                    help="root dir for dumping depth maps (.npy+.png) + point clouds (.ply). "
                         "Omit to skip. In mono mode only view0 is dumped.")
    ap.add_argument("--dump_max_frames", type=int, default=0,
                    help="max frames dumped per manifest (0=all)")
    ap.add_argument("--dump_subsample", type=int, default=1, help="point-cloud pixel subsample stride")
    ap.add_argument("--dump_conf_drop_pct", type=float, default=20.0,
                    help="X-Lens point-cloud confidence cutoff: drop lowest-confidence percent per view "
                         "(default 20; 0=no cutoff).")
    ap.add_argument("--prestage_dir", type=str, default=None,
                    help="pre-stage large ckpts to this local fast-disk dir. Copies only large files "
                         "actually loaded (>300MB).")
    a = ap.parse_args()

    # ---- Optional arch config (overrides weight-inference) ----
    arch_cfg = None
    if a.config is not None:
        import yaml
        with open(a.config, "r") as f:
            arch_cfg = yaml.safe_load(f) or {}
        logger.info(f"[stage3] using arch config {a.config}")

    # ---- Resolve ckpt list (one or more) ----
    if a.stage3_ckpts:
        ckpt_specs = parse_ckpt_specs(a.stage3_ckpts)
    elif a.stage3_ckpt is not None:
        ckpt_specs = [(default_ckpt_label(a.stage3_ckpt), a.stage3_ckpt)]
    else:
        ap.error("--stage3_ckpt or --stage3_ckpts required")

    # ---- Pre-stage large ckpts to local fast disk (only large files actually loaded, >300MB) ----
    if a.prestage_dir:
        from evaluation.prestage_util import prestage as _prestage
        ckpt_specs = [(lbl, type(p)(_prestage(str(p), a.prestage_dir))) for lbl, p in ckpt_specs]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA unavailable, slow")

    protocol = json.loads(a.protocol.read_text())
    root = a.kitti360_root or _resolve_root(protocol)
    src_index = protocol["source_index"]
    input_hw = tuple(protocol["input_hw"])
    manifests = protocol["manifests"]
    logger.info(f"protocol {protocol['name']} | root={root} | input_hw={input_hw} | "
                f"{len(manifests)} manifests (seq x cam)")
    logger.info("depth convention: GT euclidean_range -> z-depth (load_gt: z=range*cos_theta); pred=z-depth.")

    # ---- Models (all resident; data read once) ----
    stage3_models = []  # list of (label, model, meta)
    for lbl, ckpt in ckpt_specs:
        logger.info(f"[stage3] loading {lbl} <- {ckpt}")
        m, meta = build_stage3_model(ckpt, device, config=arch_cfg)
        meta["ckpt"] = str(ckpt)
        stage3_models.append((lbl, m, meta))
    labels = [lbl for lbl, _, _ in stage3_models]

    acc_s3 = {lbl: ModelAccumulators() for lbl in labels}   # one per ckpt
    per_manifest = {}
    n_frames_done = 0
    timer = {lbl: [0.0, 0] for lbl in labels}      # [total sec, frames]
    warmup_left = 1
    t0 = time.time()

    def timed(fn):
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = time.time()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out, time.time() - ts

    for mi, m in enumerate(manifests):
        seq, cam = m["sequence"], m["camera"]
        sample_indices = list(m["sample_indices"])
        # Single (seq,cam): KITTI360StereoDataset.flat order = this manifest's sample order,
        # so flat[k] == samples[k] and protocol sample_indices index the Subset directly.
        try:
            ds = KITTI360StereoDataset(
                manifest_index=src_index, sequences=[seq], cameras=[cam],
                kitti360_root=root, target_hw=input_hw, cam_type=0,
                train=False, color_aug=False)
        except Exception as e:
            logger.warning(f"  [skip] {seq}/{cam}: dataset build failed {e}")
            continue
        idxs = [i for i in sample_indices if 0 <= i < len(ds)]
        if a.max_frames_per_manifest:
            idxs = idxs[:a.max_frames_per_manifest]
        if not idxs:
            logger.warning(f"  [skip] {seq}/{cam}: no evaluable samples (len(ds)={len(ds)})")
            continue

        s3_man = {lbl: ModelAccumulators() for lbl in labels}
        man_dumped = 0
        mtag = "mono" if a.mono else "2v"
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idxs),
            batch_size=1, shuffle=False, num_workers=a.num_workers,
            pin_memory=False, collate_fn=collate_kitti360_stereo,
        )
        for fi, batch in enumerate(loader):
            idx = idxs[fi]
            if a.mono:
                batch = _to_mono(batch)                           # view1<-view0 copy, mask view1
            valid = build_valid(batch, device)[0]                 # (S,H,W)
            gt = batch["depths"].numpy()[0]                       # (S,H,W) z-depth
            cam_types = batch["cam_types"].numpy()[0]             # (S,) all 0 fisheye

            warm = warmup_left > 0
            do_dump = a.dump_dir is not None and (a.dump_max_frames == 0 or man_dumped < a.dump_max_frames)
            frame_preds, frame_masks, frame_confs = {}, {}, {}
            for lbl, model, _ in stage3_models:
                if do_dump:
                    out, dt = timed(lambda mdl=model: infer_stage3(mdl, batch, device, return_mask=True))
                    pred_s3 = out[0][0]
                    frame_masks[lbl] = out[1][0] if out[1] is not None else None
                    frame_confs[lbl] = out[2][0] if out[2] is not None else None
                else:
                    pred_s3, dt = timed(lambda mdl=model: infer_stage3(mdl, batch, device)[0])
                if not warm:
                    timer[lbl][0] += dt; timer[lbl][1] += 1
                accumulate_frame(acc_s3[lbl], pred_s3, gt, valid, cam_types)
                accumulate_frame(s3_man[lbl], pred_s3, gt, valid, cam_types)
                frame_preds[lbl] = pred_s3

            if do_dump:
                try:
                    # KITTI-360 non_ambiguous_mask is pure lens (no sky); use it as the GT lens mask
                    lens_masks = None
                    if "non_ambiguous_mask" in batch:
                        lens_masks = batch["non_ambiguous_mask"][0].numpy().astype(bool)
                    dump_frame_set(a.dump_dir, f"{seq}_{cam}_{mtag}/frame{idx}", gt, frame_preds,
                                   batch, valid, a.dump_subsample,
                                   frame_masks=frame_masks,
                                   frame_confs=frame_confs, conf_drop_pct=a.dump_conf_drop_pct,
                                   lens_masks=lens_masks)
                    man_dumped += 1
                except Exception as e:
                    logger.warning(f"  [dump] {seq}/{cam} frame {idx} dump failed: {e}")
            if warmup_left > 0:
                warmup_left -= 1
            n_frames_done += 1

        key = f"{seq}/{cam}"
        per_manifest[key] = {"stage3": {lbl: s3_man[lbl].result() for lbl in labels},
                             "n_frames": len(idxs)}
        for lbl in labels:
            r = s3_man[lbl].result()["fisheye"]
            logger.info(f"  ({mi+1}/{len(manifests)}) {key} [{lbl}]: {len(idxs)} frames "
                        f"absrel={r['depth_absrel']:.4f} scale={r['scale_absrel']:.3f} "
                        f"d1={r['delta1_1.25']:.4f} | {n_frames_done/(time.time()-t0):.2f} f/s")

    fps = {lbl: (timer[lbl][1] / timer[lbl][0]) if timer[lbl][0] > 0 else float("nan")
           for lbl in labels}

    mode = "mono(view1=view0 copy, view0 only)" if a.mono else "stereo_2v(target+support)"
    stem = "summary_kitti360_mono" if a.mono else "summary_kitti360"
    summary = {
        "protocol": protocol["name"],
        "mode": mode,
        "stage3_ckpts": {lbl: meta["ckpt"] for lbl, _, meta in stage3_models},
        "stage3_models": {
            lbl: {"ckpt": meta["ckpt"], "global_step": meta["global_step"],
                  "result": acc_s3[lbl].result()}
            for lbl, _, meta in stage3_models
        },
        "kitti360_root": root,
        "n_manifests": len(per_manifest), "n_frames": n_frames_done,
        "depth_convention": "GT euclidean_range -> z-depth (z=range*cos_theta); pred z-depth",
        "fps_per_2cam_frame": fps,
        "per_manifest": per_manifest,
    }

    out_json = a.out_dir / f"{stem}.json"
    # Resume-friendly: merge with any existing summary; ckpts not loaded this run
    # (other X-Lens ckpts) keep their prior results.
    if out_json.exists():
        try:
            old = json.loads(out_json.read_text())
        except Exception:
            old = {}
        if old:
            for k in ("stage3_models", "stage3_ckpts", "fps_per_2cam_frame"):
                mm = dict(old.get(k, {})); mm.update(summary.get(k, {})); summary[k] = mm
            pm = {}
            for key in set(old.get("per_manifest", {})) | set(summary.get("per_manifest", {})):
                d = dict(old.get("per_manifest", {}).get(key, {}))
                d.update(summary.get("per_manifest", {}).get(key, {}))
                pm[key] = d
            summary["per_manifest"] = pm
            kept = [k for k in old if k not in summary]    # old top-level keys not produced this run
            for k in kept:
                summary[k] = old[k]
            if kept:
                logger.info(f"  merged existing summary, kept {len(kept)} un-rerun keys: {kept}")
    out_json.write_text(json.dumps(summary, indent=2))

    # Also write one file per ckpt (legacy single-ckpt schema)
    for lbl, _, meta in stage3_models:
        sub = {
            "protocol": protocol["name"],
            "mode": mode,
            "stage3_ckpt": meta["ckpt"],
            "stage3_global_step": meta["global_step"],
            "kitti360_root": root,
            "n_manifests": len(per_manifest), "n_frames": n_frames_done,
            "depth_convention": "GT euclidean_range -> z-depth (z=range*cos_theta); pred z-depth",
            "fps_per_2cam_frame": {"stage3": fps.get(lbl)},
            "stage3": acc_s3[lbl].result(),
            "per_manifest": {k: {"n_frames": v["n_frames"], "stage3": v["stage3"][lbl]}
                             for k, v in per_manifest.items()},
        }
        (a.out_dir / f"{stem}__{lbl}.json").write_text(json.dumps(sub, indent=2))

    tag = "KITTI-360 fisheye monocular" if a.mono else "KITTI-360 fisheye 2-view"
    for lbl in labels:
        print_table(f"stage3[{lbl}] ({tag})", acc_s3[lbl].result())
    print_fps(fps)
    logger.info(f"results written to {out_json}")


if __name__ == "__main__":
    main()
