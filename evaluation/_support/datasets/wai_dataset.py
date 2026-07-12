"""WAI-format dataset (scene_meta.json + covisibility sampling).

Data layout:
    wai_root/
    +-- eth3d/
    |   +-- courtyard/
    |   |   +-- scene_meta.json
    |   |   +-- images/
    |   |   +-- depth/
    |   |   +-- covisibility/v0/pairwise_covisibility--NxN.npy
    +-- blendedmvs/
    +-- ...
"""

import bisect
import json
import os
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Callable, Union, List, Dict

import numpy as np
from torch.utils.data import Dataset, ConcatDataset

from .transforms import Compose
from ..data_configs import DatasetConfig, get_dataset_config

logger = logging.getLogger(__name__)

# Fallback thresholds, used only when a dataset lacks a data_configs/<name>.yaml.
# Per-dataset settings live under xlens/data_configs/.
DEFAULT_COVISIBILITY_THRESHOLDS = {
    "eth3d": 0.025,
    "mpsd": 0.15,
}
DEFAULT_COVISIBILITY_THRESHOLD = 0.25


class WaiSceneDataset(Dataset):
    """Dataset for a single WAI scene.

    Reads frame info from scene_meta.json and samples views via the covisibility matrix.

    Args:
        scene_dir: scene directory (contains scene_meta.json).
        pipeline: data transform pipeline.
        num_views: views per sample, int or [min, max].
        covisibility_threshold: covisibility threshold (0-1); frame pairs below it are disconnected.
        view_sampling: "covisibility" / "random" / "sequential".
    """

    def __init__(
        self,
        scene_dir: str,
        pipeline: Optional[Sequence[Callable]] = None,
        num_views: Union[int, List[int]] = 2,
        covisibility_threshold: float = 0.25,
        view_sampling: str = "covisibility",
        dataset_config: Optional[DatasetConfig] = None,
    ):
        self.scene_dir = Path(scene_dir)
        # Dataset config (declarative yaml under xlens/data_configs/). If not passed,
        # infer the dataset name from the parent directory and look it up in the registry.
        if dataset_config is None:
            dataset_name = self.scene_dir.parent.name
            dataset_config = get_dataset_config(dataset_name)
        self.dataset_config = dataset_config
        self._meta_path = self.scene_dir / "scene_meta.json"
        assert self._meta_path.exists(), f"scene_meta.json not found: {self._meta_path}"

        # Read only the frame count; do not keep the full meta resident.
        with open(self._meta_path, "r") as f:
            meta = json.load(f)
        self._num_frames = len(meta["frames"])
        assert self._num_frames > 0, f"Scene {scene_dir} has no frames"
        del meta

        # Load covisibility matrix (version from dataset_config, default v0). mmap maps the file
        # without physical memory; adj is computed lazily per row.
        self.covisibility = None
        self._adj = None
        self._covis_diag = None  # cached diagonal for per-row adj computation
        covis_dir = self.scene_dir / "covisibility" / self.dataset_config.covisibility_version
        if covis_dir.exists():
            covis_files = list(covis_dir.glob("pairwise_covisibility--*.npy"))
            if covis_files:
                self.covisibility = np.load(str(covis_files[0]), mmap_mode="r")
                # Diagonal only (N,), N*4 bytes, used for per-row adj.
                self._covis_diag = np.array(self.covisibility.diagonal()).copy()
                self._covis_diag = np.maximum(self._covis_diag, 1e-8)

        self.covisibility_threshold = covisibility_threshold
        self.view_sampling = view_sampling

        # Variable view count.
        if isinstance(num_views, (list, tuple)):
            assert len(num_views) == 2
            self.num_views_list = list(range(
                max(num_views[0], 2),
                num_views[1] + 1,
            ))
            if not self.num_views_list:
                self.num_views_list = [max(num_views[0], 2)]
            self.variable_num_views = True
        else:
            self.num_views_list = [num_views]
            self.variable_num_views = False

        self.pipeline = Compose(pipeline)

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, idx) -> dict:
        resolution_idx = None
        if isinstance(idx, tuple):
            if len(idx) == 2:
                idx, resolution_idx = idx
                num_views_idx = 0
            elif len(idx) == 3:
                idx, resolution_idx, num_views_idx = idx
            else:
                raise ValueError(f"Unsupported idx format: {idx}")
        else:
            num_views_idx = 0

        num_views_idx = min(num_views_idx, len(self.num_views_list) - 1)
        num_views = self.num_views_list[num_views_idx]
        num_views = min(num_views, self._num_frames)

        anchor_idx = idx % self._num_frames

        # Sample frame indices.
        if self.view_sampling == "covisibility" and self.covisibility is not None:
            frame_indices = self._covisibility_sampling(anchor_idx, num_views)
        elif self.view_sampling == "random":
            frame_indices = self._random_sampling(anchor_idx, num_views)
        else:
            frame_indices = self._sequential_sampling(anchor_idx, num_views)

        # Build sample_info. Pass dataset_config via to_dict() for collate/pickle friendliness,
        # and include only the sampled frames' metadata (not the whole scene_meta.json) to avoid
        # serializing hundreds of frames through DataLoader IPC.
        with open(self._meta_path, "r") as f:
            meta = json.load(f)
        all_frames = meta["frames"]
        sampled_frames = [all_frames[fi] for fi in frame_indices]
        del all_frames
        # shared_intrinsics: intrinsics live at scene top level; inject into sampled frames.
        if meta.get("shared_intrinsics", False):
            for fr in sampled_frames:
                for k in ("fl_x", "fl_y", "cx", "cy", "h", "w"):
                    if k not in fr and k in meta:
                        fr[k] = meta[k]
        del meta
        sample_info = {
            "scene_dir": str(self.scene_dir.absolute()),
            "dataset_format": "wai",
            "dataset_name": self.dataset_config.name,
            "dataset_config": self.dataset_config.to_dict(),
            "is_metric": self.dataset_config.is_metric,
            "participate_scale_factor": self.dataset_config.participate_scale_factor,
            "quality_tier": self.dataset_config.quality_tier,
            "sampling_weight": self.dataset_config.sampling_weight,
            "frame_indices": frame_indices,
            "sampled_frames": sampled_frames,
        }

        if resolution_idx is not None:
            sample_info["resolution_idx"] = resolution_idx

        if self.pipeline is not None:
            result = self.pipeline(sample_info)
            if result is None:
                # Load failed (e.g. missing file); retry with a random sample.
                retry_idx = random.randint(0, self._num_frames - 1)
                if resolution_idx is not None:
                    return self.__getitem__((retry_idx, resolution_idx, num_views_idx))
                return self.__getitem__(retry_idx)
            return result
        return sample_info

    def _covisibility_sampling(self, anchor_idx: int, num_views: int) -> List[int]:
        """Covisibility-based random-walk sampling."""
        n_frames = self._num_frames
        if num_views >= n_frames:
            return list(range(n_frames))

        covis = self.covisibility  # mmap, read row by row
        diag = self._covis_diag   # (N,) cached diagonal
        thresh = self.covisibility_threshold

        def _adj_row(row_idx):
            """Compute one adj row from a single mmap row (no full N x N cache)."""
            row = np.array(covis[row_idx])  # (N,)
            norm = np.maximum(diag[row_idx], diag)
            return (row / norm) >= thresh  # bool (N,)

        selected = [anchor_idx]
        visited = {anchor_idx}
        current = anchor_idx

        for _ in range(num_views - 1):
            # Read one covis row (mmap on demand) and compute adj per row.
            adj_row = _adj_row(current)
            covis_row = covis[current]
            neighbors = []
            weights = []
            for j in range(n_frames):
                if j not in visited and adj_row[j]:
                    neighbors.append(j)
                    weights.append(float(covis_row[j]))

            if not neighbors:
                for prev in selected:
                    adj_row_p = _adj_row(prev)
                    covis_row_p = covis[prev]
                    for j in range(n_frames):
                        if j not in visited and adj_row_p[j]:
                            neighbors.append(j)
                            weights.append(float(covis_row_p[j]))
                if not neighbors:
                    remaining = [j for j in range(n_frames) if j not in visited]
                    if not remaining:
                        break
                    neighbors = remaining
                    weights = [1.0] * len(remaining)

            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            chosen = np.random.choice(neighbors, p=probs)
            selected.append(chosen)
            visited.add(chosen)
            current = chosen

        return selected

    def _random_sampling(self, anchor_idx: int, num_views: int) -> List[int]:
        n_frames = self._num_frames
        if num_views >= n_frames:
            return list(range(n_frames))
        candidates = list(range(n_frames))
        candidates.remove(anchor_idx)
        others = random.sample(candidates, num_views - 1)
        return [anchor_idx] + others

    def _sequential_sampling(self, anchor_idx: int, num_views: int) -> List[int]:
        n_frames = self._num_frames
        if num_views >= n_frames:
            return list(range(n_frames))
        return [(anchor_idx + i) % n_frames for i in range(num_views)]


def _load_npy_compat(path: str) -> np.ndarray:
    """Load an npy file, handling numpy 2.x object arrays under numpy 1.x.

    numpy 2.x pickles object arrays with references to numpy._core, which numpy 1.x lacks
    (it only has numpy.core). Remap via a pickle compatibility layer.
    """
    try:
        return np.load(path, allow_pickle=True)
    except ModuleNotFoundError:
        import pickle
        import io

        class _NumpyCompat(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                # numpy._core.* -> numpy.core.*
                if module.startswith("numpy._core"):
                    module = module.replace("numpy._core", "numpy.core", 1)
                return super().find_class(module, name)

        with open(path, "rb") as f:
            # Skip the npy header.
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                np.lib.format.read_array_header_1_0(f)
            else:
                np.lib.format.read_array_header_2_0(f)
            return _NumpyCompat(f).load()


def load_scene_lists(scene_list_dir: str, split: str) -> Dict[str, set]:
    """Load the set of scene names for a split from the npy scene-list directory.

    Args:
        scene_list_dir: path containing train/val/test subdirectories.
        split: "train" / "val" / "test".

    Returns:
        dict: dataset_name -> set of scene names.
    """
    split_dir = Path(scene_list_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Scene-list directory not found: {split_dir}")
    result: Dict[str, set] = {}
    for npy_file in sorted(split_dir.glob(f"*_scene_list_{split}.npy")):
        # Filename format: <dataset_name>_scene_list_<split>.npy
        dataset_name = npy_file.name.replace(f"_scene_list_{split}.npy", "")
        scenes = _load_npy_compat(str(npy_file))
        result[dataset_name] = set(s if isinstance(s, str) else str(s) for s in scenes)
    logger.info(f"Loaded scene lists from {split_dir}: "
                + ", ".join(f"{k}({len(v)})" for k, v in result.items()))
    return result


def _scan_and_build_scene(
    scene_dir: Path,
    ds_cfg,
    covis_thresh: float,
    depth_key: Optional[str],
    pipeline,
    num_views,
    view_sampling: str,
):
    """Scan and construct a single scene, for use with ThreadPoolExecutor.

    Returns (WaiSceneDataset|None, no_depth_flag):
        - (ds, False)   : success
        - (None, True)  : depth_key missing, count as skipped_no_depth
        - (None, False) : other exception, already logged
    """
    try:
        if depth_key:
            with open(scene_dir / "scene_meta.json", "r") as f:
                _meta = json.load(f)
            _frames = _meta["frames"]
            _sample_frame = (
                _frames[0] if isinstance(_frames, list) else next(iter(_frames.values()))
            )
            _flat_key = depth_key.replace("/", "_")
            if (
                _flat_key not in _sample_frame
                and depth_key not in _sample_frame
                and "depth" not in _sample_frame
            ):
                return None, True

        scene_cfg = DatasetConfig(
            **{**ds_cfg.to_dict(), "covisibility_threshold": covis_thresh}
        )
        ds = WaiSceneDataset(
            str(scene_dir),
            pipeline=pipeline,
            num_views=num_views,
            covisibility_threshold=covis_thresh,
            view_sampling=view_sampling,
            dataset_config=scene_cfg,
        )
        return ds, False
    except Exception as e:
        logger.warning(f"Skipping scene {scene_dir}: {e}")
        return None, False


class WaiDataset(ConcatDataset):
    """Multi-scene WAI-format dataset.

    Supports selecting specific datasets or using all. Each dataset uses its own covisibility
    threshold.

    Args:
        wai_root: WAI data root (contains eth3d, blendedmvs, ... subdirectories).
        datasets: list of dataset names to use, e.g. ["eth3d", "blendedmvs"]. None uses all.
        pipeline: data transform pipeline.
        num_views: views per sample, int or [min, max].
        covisibility_thresholds: per-dataset threshold dict, e.g. {"eth3d": 0.025}. Datasets not
                                 listed use default_covisibility_threshold.
        default_covisibility_threshold: default threshold.
        view_sampling: sampling strategy.
        scene_list: per-dataset allowed scene-name sets, e.g. {"eth3d": {"courtyard", ...}}.
                    None disables filtering (all scenes).
    """

    def __init__(
        self,
        wai_root: str,
        datasets: Optional[List[str]] = None,
        pipeline: Optional[Sequence[Callable]] = None,
        num_views: Union[int, List[int]] = 2,
        covisibility_thresholds: Optional[Dict[str, float]] = None,
        default_covisibility_threshold: float = DEFAULT_COVISIBILITY_THRESHOLD,
        view_sampling: str = "covisibility",
        # Legacy single-threshold parameter.
        covisibility_threshold: Optional[float] = None,
        # Predefined scene-list filter.
        scene_list: Optional[Dict[str, set]] = None,
    ):
        wai_root = Path(wai_root)
        assert wai_root.exists(), f"WAI data root not found: {wai_root}"

        # Merge thresholds: built-in defaults overridden by user input.
        thresholds = dict(DEFAULT_COVISIBILITY_THRESHOLDS)
        if covisibility_thresholds is not None:
            thresholds.update(covisibility_thresholds)

        # Legacy: a single covisibility_threshold becomes the default for all datasets.
        if covisibility_threshold is not None:
            default_covisibility_threshold = covisibility_threshold

        # Enumerate dataset directories.
        if datasets is not None:
            dataset_dirs = [wai_root / name for name in datasets]
            for d in dataset_dirs:
                assert d.exists(), f"Specified dataset not found: {d}"
        elif scene_list is not None:
            # scene_list specifies which datasets participate; enumerate the existing ones.
            dataset_dirs = sorted([
                wai_root / name for name in scene_list
                if (wai_root / name).is_dir()
            ])
        else:
            dataset_dirs = sorted([
                d for d in wai_root.iterdir() if d.is_dir()
            ])

        # Enumerate all scenes.
        # Threshold priority: data_configs/<name>.yaml > covisibility_thresholds
        #                     > DEFAULT_COVISIBILITY_THRESHOLDS > default_covisibility_threshold.
        scene_datasets = []
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            ds_cfg = get_dataset_config(dataset_name)
            # User-passed thresholds still override the yaml (for ablation).
            if dataset_name in thresholds:
                covis_thresh = thresholds[dataset_name]
            else:
                covis_thresh = ds_cfg.covisibility_threshold

            allowed_scenes = scene_list.get(dataset_name) if scene_list is not None else None

            scene_dirs = sorted([
                d for d in dataset_dir.iterdir()
                if d.is_dir() and (d / "scene_meta.json").exists()
                   and (allowed_scenes is None or d.name in allowed_scenes)
            ])
            depth_key = ds_cfg.depth_key
            # Run IO-heavy per-scene work in a thread pool (large speedup on network storage):
            #   1) open scene_meta.json to check depth_key
            #   2) WaiSceneDataset.__init__ opens meta + mmaps covisibility + copies the diagonal
            # json.load / np.load / open release the GIL, so a high thread count is fine.
            _scan_one = partial(
                _scan_and_build_scene,
                ds_cfg=ds_cfg,
                covis_thresh=covis_thresh,
                depth_key=depth_key,
                pipeline=pipeline,
                num_views=num_views,
                view_sampling=view_sampling,
            )
            max_workers = min(64, max(len(scene_dirs), 1))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_scan_one, scene_dirs))
            skipped_no_depth = 0
            for ds, no_depth in results:
                if ds is not None:
                    scene_datasets.append(ds)
                elif no_depth:
                    skipped_no_depth += 1
            if skipped_no_depth > 0:
                logger.warning(
                    f"  {dataset_name}: skipped {skipped_no_depth} scenes missing depth_key='{depth_key}'"
                )

            if scene_dirs or allowed_scenes is not None:
                n_loaded = len([d for d in scene_datasets
                                if hasattr(d, 'dataset_config') and d.dataset_config.name == dataset_name])
                logger.info(
                    f"  {dataset_name}: {n_loaded} scenes "
                    f"[is_metric={ds_cfg.is_metric}, "
                    f"covisibility_threshold={covis_thresh}, "
                    f"covis_version={ds_cfg.covisibility_version}, "
                    f"depth_key={ds_cfg.depth_key}, "
                    f"moge_mask={ds_cfg.apply_moge_mask}, "
                    f"sky_mask={ds_cfg.apply_sky_mask}, "
                    f"clip_p95={ds_cfg.clip_depth_p95}]"
                )

        assert len(scene_datasets) > 0, f"No valid WAI scenes found in {wai_root}"

        total_samples = sum(len(d) for d in scene_datasets)
        dataset_names = [d.name for d in dataset_dirs]
        logger.info(
            f"WAI dataset: {dataset_names}, "
            f"{len(scene_datasets)} scenes, {total_samples} samples"
        )

        super().__init__(scene_datasets)

    def get_dataset_groups(self) -> Dict[str, List[int]]:
        """Return dict: dataset_name -> global sample indices in the ConcatDataset index space.

        Used for weighted multi-dataset sampling: the sampler takes one group's indices and
        samples uniformly from it to build a batch.
        """
        groups: Dict[str, List[int]] = {}
        for ds_idx, scene_ds in enumerate(self.datasets):
            name = scene_ds.dataset_config.name
            start = self.cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0
            end = self.cumulative_sizes[ds_idx]
            groups.setdefault(name, []).extend(range(start, end))
        return groups

    def get_dataset_weights(self) -> Dict[str, float]:
        """Return dict: dataset_name -> sampling_weight (from data_configs/<name>.yaml).

        All scenes of a dataset share one weight.
        """
        weights: Dict[str, float] = {}
        for scene_ds in self.datasets:
            name = scene_ds.dataset_config.name
            weights[name] = float(scene_ds.dataset_config.sampling_weight)
        return weights

    def __getitem__(self, idx):
        """Support tuple idx: (sample_idx, ar_idx, nv_idx)."""
        if isinstance(idx, tuple):
            sample_idx = idx[0]
            rest = idx[1:]
            # Use ConcatDataset's bisect logic to find the sub-dataset and local idx.
            if sample_idx < 0:
                if -sample_idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                sample_idx = len(self) + sample_idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, sample_idx)
            if dataset_idx == 0:
                local_idx = sample_idx
            else:
                local_idx = sample_idx - self.cumulative_sizes[dataset_idx - 1]
            return self.datasets[dataset_idx][(local_idx, *rest)]
        return super().__getitem__(idx)
