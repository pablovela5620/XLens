"""
Pinhole depth evaluation on the WAI test set.

Datasets: eth3d / scannetppv2 (scene lists from
    scene_list_dir/<split>/<dataset>_scene_list_<split>.npy)
Each scene covisibility-samples num_views frames for the multi-view model, computes standard
per-view depth metrics (abs_rel / rmse / delta<1.25 ...), averages with equal per-frame weight,
then aggregates per dataset + overall.

Reuses eval_pinhole.simple_adapters (the X-Lens model adapter).
The model reconstructs ray_map from GT intrinsics/extrinsics (matching training) and does not predict pose.

Usage:
    python -m evaluation.eval_pinhole.run_eval \
        --config evaluation/eval_pinhole/eval_config.yaml

    # CLI overrides
    python -m evaluation.eval_pinhole.run_eval \
        --config .../eval_config.yaml --datasets eth3d --num_views 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

try:
    from evaluation.eval_pinhole.simple_adapters import build_simple_adapter
    from evaluation.eval_pinhole.wai_loader import list_scenes, load_scene_batch
    from evaluation.eval_pinhole.depth_metrics import (
        align_pred, compute_frame_depth_metrics, aggregate_frame_metrics,
        METRIC_KEYS, BUCKET_LABELS,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from evaluation.eval_pinhole.simple_adapters import build_simple_adapter
    from evaluation.eval_pinhole.wai_loader import list_scenes, load_scene_batch
    from evaluation.eval_pinhole.depth_metrics import (
        align_pred, compute_frame_depth_metrics, aggregate_frame_metrics,
        METRIC_KEYS, BUCKET_LABELS,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ============================================================================
# Single scene: inference + per-view depth metrics
# ============================================================================

def eval_scene(scene_batch: Dict, adapter, alignment: str,
               dump_ctx: Optional[Dict] = None) -> List[Dict[str, float]]:
    """Return per-frame metrics for all valid views of the scene (None = too few valid pixels).

    dump_ctx (optional): {dir, frame_key, model_name, subsample, max_depth, dump_gt}.
      If given, dumps the model's median-aligned depth maps + fused point cloud (multi-view -> one
      n-view cloud); with dump_gt=True also dumps GT (same geometry, for pred vs gt overlay).
    """
    out = adapter.infer(scene_batch)[0]
    pred_depth = out["pred_depth"]                         # (S, H, W) normalized space
    pred_sf = out.get("pred_metric_scale_factor", None)

    gt_depth = scene_batch["gt_depth"]                     # (S, H, W)
    gt_valid = scene_batch["gt_valid"]                     # (S, H, W) bool
    n_views = scene_batch["num_views"]
    pred_aligned = np.full_like(pred_depth, np.nan, dtype=np.float32)  # median-aligned (for dump)

    # Diagnostic: catches abnormal pred distributions (all 0 / NaN / huge)
    finite = np.isfinite(pred_depth)
    logger.info(
        f"[{scene_batch['dataset_name']}/{scene_batch['scene_name']}] "
        f"S={pred_depth.shape[0]} pred_depth: min={pred_depth[finite].min():.3f} "
        f"max={pred_depth[finite].max():.3f} mean={pred_depth[finite].mean():.3f} "
        f"sf={pred_sf}"
    )

    # Depth structure metrics (abs_rel/rmse/tau/delta) are always computed at relative scale =
    # median alignment; absolute scale is reported separately as scale_abs_rel = |sf - s*|/s*
    # (s* = median(gt/pred)), decoupled from structure.
    # (alignment param kept for compatibility, but structure metrics fixed to median.)
    frame_metrics: List[Dict[str, float]] = []
    for s in range(n_views):
        gt_s, v_s, pn_s = gt_depth[s], gt_valid[s], pred_depth[s]
        pred_m = align_pred(pn_s, gt_s, v_s, mode="median")   # relative scale
        if pred_m is None:
            frame_metrics.append(None)
            continue
        pred_aligned[s] = pred_m.astype(np.float32)
        # Absolute-scale error: predicted scale sf vs true median scale s*
        scale_abs_rel = None
        if pred_sf is not None and np.isfinite(pred_sf):
            pv, gv = pn_s[v_s], gt_s[v_s]
            kk = (pv > 1e-6) & (gv > 1e-6) & np.isfinite(pv) & np.isfinite(gv)
            if kk.sum() >= 10:
                s_star = float(np.median(gv[kk] / pv[kk]))
                if s_star > 1e-9:
                    scale_abs_rel = abs(float(pred_sf) - s_star) / s_star
        frame_metrics.append(
            compute_frame_depth_metrics(pred_m, gt_s, v_s, scale_abs_rel=scale_abs_rel)
        )

    if dump_ctx is not None:
        _dump_stage1_scene(scene_batch, pred_aligned, gt_depth, gt_valid, dump_ctx,
                           pred_mask=out.get("pred_valid_mask"),
                           pred_conf=out.get("pred_conf"))
    return frame_metrics


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _dump_stage1_scene(scene_batch, pred_aligned, gt_depth, gt_valid, dump_ctx,
                       pred_mask=None, pred_conf=None):
    """Dump block1 (pinhole multi-view): median-aligned depth maps + fused point cloud, geometry from GT cameras.
    pred cloud is sky-filtered by the model's own mask (pred_mask, 1=non-sky/valid), then confidence-truncated
    (drop lowest conf_drop_pct% of pred_conf). GT cloud uses gt_valid. Metrics are unaffected (dump only)."""
    from evaluation.dump_utils import dump_frame, d_cam_from_intrinsics
    K = _to_np(scene_batch["intrinsics"][0])               # (S,3,3)
    c2w = _to_np(scene_batch["extrinsics_world"][0])       # (S,4,4)
    imgs = _to_np(scene_batch["images"][0])                # (S,3,H,W)
    S, H, W = pred_aligned.shape
    dcam = np.stack([d_cam_from_intrinsics(K[s], H, W) for s in range(S)], 0)
    fd = dump_ctx["dir"] / dump_ctx["frame_key"]
    sub = dump_ctx.get("subsample", 1)
    md = dump_ctx.get("max_depth", 80.0)
    spcd = dump_ctx.get("save_pcd", True)
    dump_frame(fd, dump_ctx["model_name"], pred_aligned, dcam, c2w, imgs,
               pcd_mask_shw=pred_mask, conf_shw=pred_conf,
               conf_drop_pct=dump_ctx.get("conf_drop_pct"), max_depth=md, subsample=sub, save_pcd=spcd)
    if dump_ctx.get("dump_gt"):
        dump_frame(fd, "gt", gt_depth.astype(np.float32), dcam, c2w, imgs,
                   pcd_mask_shw=gt_valid, max_depth=md, subsample=sub, save_pcd=spcd)


# ============================================================================
# Config
# ============================================================================

def _load_yaml_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if getattr(args, "prestage_dir", None):
        cfg["prestage_dir"] = args.prestage_dir
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.wai_root is not None:
        cfg["wai_root"] = args.wai_root
    if args.omniocc_root is not None:
        cfg["omniocc_root"] = args.omniocc_root
    if args.scene_list_dir is not None:
        cfg["scene_list_dir"] = args.scene_list_dir
    if args.models is not None:
        cfg["models"] = args.models
    if args.datasets is not None:
        cfg["datasets"] = args.datasets
    if args.scenes is not None:
        cfg["scenes"] = args.scenes
    if args.num_views is not None:
        cfg["num_views"] = args.num_views
    if args.max_scenes is not None:
        cfg["max_scenes"] = args.max_scenes
    if args.skip_wai:
        cfg["skip_wai"] = True
    if args.target_h is not None:
        cfg["target_h"] = args.target_h
    if args.target_w is not None:
        cfg["target_w"] = args.target_w
    cfg.setdefault("xlens", {})
    if args.xlens_ckpt is not None:
        cfg["xlens"]["ckpt"] = args.xlens_ckpt
    if args.xlens_alignment is not None:
        cfg["xlens"]["alignment"] = args.xlens_alignment
    if args.xlens_config is not None:
        cfg["xlens"]["config"] = args.xlens_config
    if args.xlens_ckpts is not None:
        cfg["xlens_ckpts"] = args.xlens_ckpts
    if args.dump_dir is not None:
        cfg["dump_dir"] = args.dump_dir
    cfg["dump_max_frames"] = args.dump_max_frames
    cfg["dump_subsample"] = args.dump_subsample
    cfg["dump_conf_drop_pct"] = args.dump_conf_drop_pct
    cfg["dump_depth_only"] = args.dump_depth_only
    return cfg


def _parse_xlens_ckpts(items: List[str]) -> List:
    """['label=path', 'path', ...] -> [(label, path), ...], with deduplicated labels."""
    specs, seen = [], {}
    for it in items:
        if "=" in it:
            lbl, path = it.split("=", 1)
            lbl = lbl.strip()
        else:
            path = it
            p = Path(path)
            lbl = f"{p.parent.name}_{p.stem}"
        if lbl in seen:
            seen[lbl] += 1
            lbl = f"{lbl}#{seen[lbl]}"
        else:
            seen[lbl] = 0
        specs.append((lbl, path.strip()))
    return specs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--wai_root", type=str, default=None,
                    help="override config wai_root")
    ap.add_argument("--omniocc_root", type=str, default=None,
                    help="override config omniocc_root (OmniOCC data root)")
    ap.add_argument("--scene_list_dir", type=str, default=None,
                    help="override config scene_list_dir")
    ap.add_argument("--models", type=str, nargs="+", default=None,
                    choices=["xlens"])
    ap.add_argument("--datasets", type=str, nargs="+", default=None,
                    help="only run these datasets (eth3d / scannetppv2)")
    ap.add_argument("--scenes", type=str, nargs="*", default=None,
                    help="only run these scene names (matched across datasets)")
    ap.add_argument("--num_views", type=int, default=None)
    ap.add_argument("--max_scenes", type=int, default=None,
                    help="evaluate at most the first N scenes of WAI / OmniOCC each (0 or omit = all)")
    ap.add_argument("--skip_wai", action="store_true",
                    help="skip WAI test (do not read scene_list_dir), evaluate only OmniOCC")
    ap.add_argument("--target_h", type=int, default=None)
    ap.add_argument("--target_w", type=int, default=None)
    ap.add_argument("--xlens_ckpt", type=str, default=None)
    ap.add_argument("--xlens_ckpts", type=str, nargs="+", default=None,
                    help="evaluate multiple X-Lens ckpts at once, each 'label=path' or plain 'path'. "
                         "overrides --xlens_ckpt.")
    ap.add_argument("--xlens_config", type=str, default=None,
                    help="model arch config yaml (config embedded in a .pth ckpt takes "
                         "priority). For the released .safetensors pass configs/xlens_vits.yaml.")
    ap.add_argument("--xlens_alignment", type=str, default=None,
                    choices=["self_scale", "median", "none"])
    ap.add_argument("--dump_dir", type=str, default=None,
                    help="root dir for dumped depth maps (.npy+.png) + fused point clouds (.ply). unset = no dump. "
                         "depth is per-view median-aligned to meters; multi-view pinhole fused into one cloud.")
    ap.add_argument("--dump_max_frames", type=int, default=0,
                    help="dump at most N scenes for WAI/OmniOCC each (0 = all)")
    ap.add_argument("--dump_subsample", type=int, default=1, help="point cloud pixel subsample stride")
    ap.add_argument("--dump_conf_drop_pct", type=float, default=20.0,
                    help="X-Lens point cloud confidence truncation: drop lowest-confidence percent per view (default 20; 0 = no drop)")
    ap.add_argument("--dump_depth_only", action="store_true",
                    help="dump depth maps (.npy+.png) only, skip the fused point cloud (.ply)")
    ap.add_argument("--prestage_dir", type=str, default=None,
                    help="prestage large ckpts to this local fast-disk dir. Only copies actually-loaded large files (>300MB).")
    return ap.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        logger.error(f"config file not found: {cfg_path}")
        sys.exit(1)
    cfg = _apply_cli_overrides(_load_yaml_config(cfg_path), args)

    required = ["output_dir", "models", "wai_root", "scene_list_dir", "target_h", "target_w"]
    for k in required:
        if k not in cfg or cfg[k] is None:
            logger.error(f"missing config field: {k}")
            sys.exit(1)

    logger.info(f"sys.executable = {sys.executable}")
    logger.info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA unavailable, using CPU (slow). Run the training stack with isaac python.sh.")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Effective config:\n{yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)}")

    # Scene list
    split = cfg.get("split", "test")
    max_scenes = int(cfg.get("max_scenes") or 0)
    if cfg.get("skip_wai"):
        scene_pairs = []
        logger.info("--skip_wai: skipping WAI test, evaluating only OmniOCC")
    else:
        scene_pairs = list_scenes(cfg["scene_list_dir"], split, cfg.get("datasets"))
        if cfg.get("scenes"):
            want = set(cfg["scenes"])
            scene_pairs = [(d, s) for (d, s) in scene_pairs if s in want]
        if max_scenes > 0:
            scene_pairs = scene_pairs[:max_scenes]
        logger.info(f"evaluating {len(scene_pairs)} scenes: {scene_pairs[:5]}{'...' if len(scene_pairs) > 5 else ''}")

    target_hw = (int(cfg["target_h"]), int(cfg["target_w"]))
    num_views = int(cfg.get("num_views", 8))
    view_sampling = cfg.get("view_sampling", "covisibility")
    min_depth = float(cfg.get("min_depth", 0.05))
    max_depth = float(cfg.get("max_depth", 80.0))

    # OmniOCC 6-cam pinhole eval (output separately from WAI test).
    # On by default; omniocc_root overrides the data root.
    eval_omniocc = bool(cfg.get("eval_omniocc", True))
    omniocc_scenes: List[str] = []
    omni_calib = None
    if eval_omniocc:
        try:
            from evaluation.eval_pinhole.omniocc_depth_loader import (
                set_omniocc_root, list_omniocc_scenes, load_calib as _omni_load_calib,
                load_omniocc_scene_batch,
            )
            set_omniocc_root(cfg.get("omniocc_root"))
            omniocc_scenes = list_omniocc_scenes()
            if cfg.get("scenes"):
                want = set(cfg["scenes"])
                omniocc_scenes = [s for s in omniocc_scenes if s in want]
            if max_scenes > 0:
                omniocc_scenes = omniocc_scenes[:max_scenes]
            omni_calib = _omni_load_calib()
            logger.info(f"OmniOCC eval enabled: {len(omniocc_scenes)} scenes (6-cam pinhole)")
        except Exception as e:
            import traceback
            logger.warning(f"OmniOCC init failed, skipping OmniOCC this run: {e}\n{traceback.format_exc()}")
            eval_omniocc = False

    # Expand into a (result_name, model_type, ckpt_override) list:
    # if --xlens_ckpts is given, xlens expands into N entries (each labeled).
    # Data is re-read per ckpt serially.
    model_specs: List = []
    xlens_ckpt_specs = _parse_xlens_ckpts(cfg["xlens_ckpts"]) if cfg.get("xlens_ckpts") else None
    for model_name in cfg["models"]:
        if model_name == "xlens" and xlens_ckpt_specs:
            for lbl, path in xlens_ckpt_specs:
                model_specs.append((lbl, "xlens", path))
        else:
            model_specs.append((model_name, model_name, None))

    # Dump settings: dump_gt only for the first model (GT is model-independent, avoid rewriting)
    from pathlib import Path as _Path
    dump_root = _Path(cfg["dump_dir"]) if cfg.get("dump_dir") else None
    dump_cap = int(cfg.get("dump_max_frames") or 0)
    dump_sub = int(cfg.get("dump_subsample") or 1)
    dump_conf_pct = float(cfg.get("dump_conf_drop_pct") or 0)
    dump_save_pcd = not bool(cfg.get("dump_depth_only"))
    first_model_name = model_specs[0][0] if model_specs else None

    all_results: Dict[str, Dict] = {}
    omniocc_results: Dict[str, Dict] = {}
    for result_name, model_type, ckpt_override in model_specs:
        logger.info(f"\n========== {result_name} ({model_type}) ==========")
        t0 = time.time()
        model_cfg = dict(cfg.get(model_type, {}) or {})
        if ckpt_override is not None:
            model_cfg["ckpt"] = ckpt_override
        # Prestage large ckpt to local fast disk (only the model actually loaded, >300MB)
        if cfg.get("prestage_dir") and model_cfg.get("ckpt"):
            from evaluation.prestage_util import prestage as _prestage
            model_cfg["ckpt"] = _prestage(str(model_cfg["ckpt"]), cfg["prestage_dir"])
        amp_enabled = (device.type == "cuda")

        if model_type == "xlens":
            if not model_cfg.get("ckpt"):
                logger.error("xlens requires xlens.ckpt"); continue
            is_onnx = str(model_cfg["ckpt"]).lower().endswith(".onnx")
            xlens_cfg = None
            if not is_onnx:
                if not model_cfg.get("config"):
                    logger.error("xlens (.pth/.safetensors) requires xlens.config"); continue
                with open(model_cfg["config"]) as f:
                    xlens_cfg = yaml.safe_load(f)
            adapter = build_simple_adapter("xlens", device=device,
                                           checkpoint_path=model_cfg["ckpt"],
                                           config=xlens_cfg, amp_enabled=amp_enabled)
            alignment = model_cfg.get("alignment", "self_scale")
        else:
            logger.error(f"unknown model_type={model_type}"); continue
        logger.info(f"alignment={alignment}")

        # Run per scene, collect metrics per (dataset -> frame)
        per_dataset_frames: Dict[str, List[Dict]] = {}
        per_scene_dump: Dict[str, Dict] = {}
        wai_dumped: Dict[str, int] = {}
        for i, (ds_name, scene_name) in enumerate(scene_pairs):
            try:
                sb = load_scene_batch(
                    ds_name, scene_name, wai_root=cfg["wai_root"], target_hw=target_hw,
                    num_views=num_views, view_sampling=view_sampling,
                    min_depth=min_depth, max_depth=max_depth,
                )
            except Exception as e:
                logger.warning(f"[{ds_name}/{scene_name}] load failed, skip: {e}")
                continue
            dump_ctx = None
            if dump_root is not None and (dump_cap == 0 or wai_dumped.get(ds_name, 0) < dump_cap):
                dump_ctx = {"dir": dump_root / "wai", "frame_key": f"{ds_name}_{scene_name}",
                            "model_name": result_name, "subsample": dump_sub, "conf_drop_pct": dump_conf_pct,
                            "max_depth": max_depth, "dump_gt": (result_name == first_model_name),
                            "save_pcd": dump_save_pcd}
                wai_dumped[ds_name] = wai_dumped.get(ds_name, 0) + 1
            try:
                fms = eval_scene(sb, adapter, alignment, dump_ctx=dump_ctx)
            except Exception as e:
                import traceback
                logger.warning(f"[{ds_name}/{scene_name}] eval failed, skip: {e}\n{traceback.format_exc()}")
                continue
            per_dataset_frames.setdefault(ds_name, []).extend(fms)
            scene_agg = aggregate_frame_metrics(fms)
            per_scene_dump[f"{ds_name}/{scene_name}"] = scene_agg
            logger.info(
                f"  [{i+1}/{len(scene_pairs)}] {ds_name}/{scene_name:<14} "
                f"absrel={scene_agg['abs_rel']:.4f} rmse={scene_agg['rmse']:.4f} "
                f"d1={scene_agg['delta1']:.4f} (frames={scene_agg['n_frames']})"
            )

        # Aggregate: per dataset + overall (all frames)
        per_dataset_agg = {d: aggregate_frame_metrics(fr) for d, fr in per_dataset_frames.items()}
        all_frames = [f for fr in per_dataset_frames.values() for f in fr]
        overall_agg = aggregate_frame_metrics(all_frames)

        elapsed = time.time() - t0
        logger.info(f"{result_name} done, elapsed={elapsed:.1f}s")
        all_results[result_name] = {
            "alignment": alignment,
            "num_views": num_views,
            "per_dataset": per_dataset_agg,
            "overall": overall_agg,
            "per_scene": per_scene_dump,
            "elapsed_sec": elapsed,
        }

        # OmniOCC: same adapter, 6-cam pinhole, collected as a separate second result set
        if eval_omniocc and omniocc_scenes:
            t_omni = time.time()
            omni_frames: List[Dict] = []
            omni_scene_dump: Dict[str, Dict] = {}
            omni_dumped = 0
            for j, sc in enumerate(omniocc_scenes):
                try:
                    osb = load_omniocc_scene_batch(
                        sc, target_hw=target_hw, min_depth=min_depth,
                        max_depth=max_depth, calib=omni_calib)
                except Exception as e:
                    logger.warning(f"[omniocc/{sc}] load failed, skip: {e}")
                    continue
                dump_ctx = None
                if dump_root is not None and (dump_cap == 0 or omni_dumped < dump_cap):
                    dump_ctx = {"dir": dump_root / "omniocc", "frame_key": f"{sc}",
                                "model_name": result_name, "subsample": dump_sub, "conf_drop_pct": dump_conf_pct,
                                "max_depth": max_depth, "dump_gt": (result_name == first_model_name),
                                "save_pcd": dump_save_pcd}
                    omni_dumped += 1
                try:
                    ofms = eval_scene(osb, adapter, alignment, dump_ctx=dump_ctx)
                except Exception as e:
                    import traceback
                    logger.warning(f"[omniocc/{sc}] eval failed, skip: {e}\n{traceback.format_exc()}")
                    continue
                omni_frames.extend(ofms)
                osc_agg = aggregate_frame_metrics(ofms)
                omni_scene_dump[f"omniocc/{sc}"] = osc_agg
                logger.info(
                    f"  [omniocc {j+1}/{len(omniocc_scenes)}] {sc:<14} "
                    f"absrel={osc_agg['abs_rel']:.4f} rmse={osc_agg['rmse']:.4f} "
                    f"d1={osc_agg['delta1']:.4f} (frames={osc_agg['n_frames']})"
                )
            omni_overall = aggregate_frame_metrics(omni_frames)
            omniocc_results[result_name] = {
                "alignment": alignment,
                "num_views": 6,
                "per_dataset": {"omniocc": omni_overall},
                "overall": omni_overall,
                "per_scene": omni_scene_dump,
                "elapsed_sec": time.time() - t_omni,
            }
            logger.info(f"{result_name} OmniOCC done, "
                        f"absrel={omni_overall['abs_rel']:.4f} "
                        f"elapsed={time.time() - t_omni:.1f}s")

        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # WAI test and OmniOCC written to separate files (all X-Lens ckpts share one file).
    # Resume-friendly: if results exist on disk, merge with this run (this run's keys override,
    # others preserved), so backfilling only the missing models does not clobber prior results.
    def _merge_write(path: Path, new_results: Dict) -> Dict:
        merged: Dict = {}
        if path.exists():
            try:
                with open(path) as f:
                    merged = json.load(f)
            except Exception as e:
                logger.warning(f"failed to read existing results, overwriting {path}: {e}")
                merged = {}
        if merged:
            kept = [k for k in merged if k not in new_results]
            if kept:
                logger.info(f"  merged existing results (kept {len(kept)} models not rerun: {kept})")
        merged.update(new_results)
        with open(path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        return merged

    result_path = output_dir / "stage1_wai_depth_eval.json"
    all_results = _merge_write(result_path, all_results)
    logger.info(f"\nWAI test results -> {result_path}")
    _print_summary(all_results, title="Stage1 WAI test Depth Evaluation (per-frame mean)")

    if omniocc_results:
        omni_path = output_dir / "stage1_omniocc_depth_eval.json"
        omniocc_results = _merge_write(omni_path, omniocc_results)
        logger.info(f"\nOmniOCC results -> {omni_path}")
        _print_summary(omniocc_results, title="OmniOCC 6-cam Pinhole Depth Evaluation (per-frame mean)")

    # Also write a per-ckpt file for per-ckpt inspection when several X-Lens ckpts were evaluated
    xlens_labels = [rn for rn, mt, _ in model_specs if mt == "xlens"]
    if len(xlens_labels) > 1:
        for lbl in xlens_labels:
            for res_dict, stem in ((all_results, "stage1_wai_depth_eval"),
                                   (omniocc_results, "stage1_omniocc_depth_eval")):
                if lbl not in res_dict:
                    continue
                sub = {lbl: res_dict[lbl]}
                with open(output_dir / f"{stem}__{lbl}.json", "w") as f:
                    json.dump(sub, f, indent=2, default=str)


def _print_summary(all_results: Dict[str, Dict],
                   title: str = "Stage1 WAI Depth Evaluation (per-frame mean)") -> None:
    if not all_results:
        return
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)
    cols = ["abs_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]
    for model_name, res in all_results.items():
        print(f"\n  ## {model_name}  (alignment={res['alignment']}, num_views={res['num_views']})")
        header = f"    {'split':<16} | " + " | ".join(c.rjust(9) for c in cols) + " | frames"
        print(header)
        print(f"    {'-'*16}-+-" + "-+-".join("-"*9 for _ in cols) + "-+-------")
        rows = list(res["per_dataset"].items()) + [("OVERALL", res["overall"])]
        for split_name, m in rows:
            line = f"    {split_name:<16}"
            for c in cols:
                line += f" | {m.get(c, float('nan')):9.4f}"
            line += f" | {m.get('n_frames', 0):>6}"
            print(line)

        # MAE bucketed by GT distance (m)
        print(f"\n    [MAE by GT distance (m)]")
        print(f"    {'split':<16} | " + " | ".join(lb.rjust(9) for lb in BUCKET_LABELS))
        print(f"    {'-'*16}-+-" + "-+-".join("-"*9 for _ in BUCKET_LABELS))
        for split_name, m in rows:
            line = f"    {split_name:<16}"
            for lb in BUCKET_LABELS:
                line += f" | {m.get(f'mae@{lb}', float('nan')):9.4f}"
            print(line)
    print("\n" + "=" * 92)


if __name__ == "__main__":
    main()
