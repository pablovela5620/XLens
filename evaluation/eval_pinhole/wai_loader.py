"""
Eval data loading for the WAI datasets (eth3d / scannetppv2).
WAI data layout / loading follows MapAnything (https://github.com/facebookresearch/map-anything).

Reuses the exact training pipeline (no fork, so train/eval distributions match):
    WaiSceneDataset(covisibility sampling)
      -> [LoadWaiFrame, Resize(target, patch=14), Normalize]
      -> collate_multiview_frames
collate yields:
    - images        (1, S, 3, H, W)  ImageNet normalized
    - depths        (1, S, H, W)     GT z-depth (m, invalid pixels = 0)
    - intrinsics    (1, S, 3, 3)     scaled with Resize
    - extrinsics_world (1, S, 4, 4)  c2w, canonicalized to view0 (view0 = I)
    - non_ambiguous_mask / has_non_ambiguous_mask / view_mask / cam_types(=1 pinhole)

This module only adds:
    1. list_scenes: read the (dataset, scene) list from scene_list_dir/<split> npy
    2. load_scene_batch: sample + collate once for a (dataset, scene), then extract per-view
       GT depth + valid mask into the scene_batch run_eval expects.

Sampling determinism: the covisibility random walk uses np.random / random, reset to a fixed
seed per scene for reproducibility (default seed=777, same as map-anything BaseDataset).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from evaluation._support.datasets import (
    WaiSceneDataset,
    LoadWaiFrame,
    Resize,
    Normalize,
    collate_multiview_frames,
)
from evaluation._support.datasets.wai_dataset import load_scene_lists
from evaluation._support.data_configs import get_dataset_config

logger = logging.getLogger(__name__)


# ============================================================================
# Scene list
# ============================================================================

def list_scenes(
    scene_list_dir: str,
    split: str = "test",
    datasets: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Read scene_list_dir/<split>/*_scene_list_<split>.npy, return [(dataset_name, scene_name), ...].

    datasets: keep only these datasets (e.g. ["eth3d", "scannetppv2"]). None = all.
    """
    scene_map: Dict[str, set] = load_scene_lists(scene_list_dir, split)
    out: List[Tuple[str, str]] = []
    for ds_name in sorted(scene_map.keys()):
        if datasets is not None and ds_name not in datasets:
            continue
        for scene in sorted(scene_map[ds_name]):
            out.append((ds_name, scene))
    return out


# ============================================================================
# Single scene -> batch
# ============================================================================

def load_scene_batch(
    dataset_name: str,
    scene_name: str,
    wai_root: str,
    target_hw: Tuple[int, int],
    num_views: int = 8,
    view_sampling: str = "covisibility",
    min_depth: float = 0.05,
    max_depth: float = 80.0,
    seed: int = 777,
) -> Dict:
    """
    Return scene_batch:
        scene_name, dataset_name
        images / intrinsics / extrinsics_world / cam_types / view_mask  (collate as-is, batch dim = 1)
        gt_depth     (S, H, W)  float32  GT z-depth (m)
        gt_valid     (S, H, W)  bool     valid pixels (mask & depth in [min,max] & finite)
        target_hw    (H, W)
    """
    scene_dir = Path(wai_root) / dataset_name / scene_name
    if not (scene_dir / "scene_meta.json").exists():
        raise FileNotFoundError(f"scene_meta.json not found: {scene_dir}")

    # Declarative dataset config (depth_key / covisibility_threshold / mask strategy, etc.)
    ds_cfg = get_dataset_config(dataset_name)

    pipeline = [
        LoadWaiFrame(),
        Resize(target_size=tuple(target_hw), patch_size=14),
        Normalize(),
    ]
    ds = WaiSceneDataset(
        str(scene_dir),
        pipeline=pipeline,
        num_views=num_views,
        covisibility_threshold=ds_cfg.covisibility_threshold,
        view_sampling=view_sampling,
        dataset_config=ds_cfg,
    )

    # Determinism: reset to a fixed per-scene seed (covisibility uses np.random / random)
    np.random.seed(seed)
    random.seed(seed)
    sample = ds[0]                       # anchor=frame0, sample num_views covisible frames
    batch = collate_multiview_frames([sample])

    depths = batch["depths"][0].numpy()                       # (S, H, W) GT z-depth, 0=invalid
    S, H, W = depths.shape
    has_mask = batch["has_non_ambiguous_mask"][0].numpy()     # (S,) bool
    nam = batch["non_ambiguous_mask"][0].numpy()              # (S, H, W) bool

    finite = np.isfinite(depths)
    in_range = (depths > min_depth) & (depths < max_depth)
    gt_valid = finite & in_range
    # Views with a mask also AND the non-ambiguous mask; views without rely on depth range only
    for s in range(S):
        if has_mask[s]:
            gt_valid[s] &= nam[s]

    gt_depth = np.where(gt_valid, depths, 0.0).astype(np.float32)

    return {
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        # For the adapter (collate as-is, batch dim = 1)
        "images": batch["images"],
        "intrinsics": batch["intrinsics"],
        "extrinsics_world": batch["extrinsics_world"],
        "cam_types": batch["cam_types"],
        "view_mask": batch["view_mask"],
        # For metrics
        "gt_depth": gt_depth,
        "gt_valid": gt_valid,
        "target_hw": (H, W),
        "num_views": int(batch["view_mask"][0].sum().item()),
    }
