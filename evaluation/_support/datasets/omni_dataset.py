"""Omni fisheye dataset (Isaac Sim, 4 cameras, LUT-based).

Data layout:
    data_root/
    +-- home000/home000/                # doubly nested
    |   +-- common/{path:04d}_{frame:04d}.npy   # per-camera dict (intrinsics + extrinsics_world + extrinsics_camera)
    |   +-- rgb/CAM_{A,B,C,D}/{path:04d}_{frame:04d}.jpg
    |   +-- depth/CAM_{A,B,C,D}/{path:04d}_{frame:04d}.npy   (inf = infinite range)
    |   +-- mask/CAM_{X}_mask.png       # static per-camera lens valid region
    +-- home001/home001/
    +-- ...

LUT files (shared by the camera rig, outside data_root):
    <lut_dir>/CAM_{A,B,C,D}_rayEnterDirection.exr   # (H, W, 3) camera-frame unit ray dir

Each sample defaults to all 4 cameras of a single frame (S=4). Set num_views=8/12/16 with
view_sampling="cross_frame_within_path" to sample multiple frames within one path.
"""

import bisect
import os
import random
import logging
from pathlib import Path
from typing import Optional, Sequence, Callable, Iterable, Set

from torch.utils.data import Dataset, ConcatDataset

from .transforms import Compose

logger = logging.getLogger(__name__)


DEFAULT_LUT_DIR = "/path/to/lut"


class OmniSceneDataset(Dataset):
    """A single Omni fisheye scene (one home_xxx)."""

    def __init__(
        self,
        scene_dir: str,
        lut_dir: str = DEFAULT_LUT_DIR,
        pipeline: Optional[Sequence[Callable]] = None,
        image_format: str = "jpg",
        view_sampling: str = "single_frame",  # "single_frame" / "cross_frame_within_path" / "cross_frame_any_view"
        num_views: int = 4,
        use_cleaned_valid_mask: bool = True,   # auto-detect and use valid_mask/ if present
        use_sky_mask: bool = True,             # auto-detect sky_mask/; if present fold sky into non_ambiguous_mask
        cameras_filter: Optional[Iterable[str]] = None,  # keep only these cameras
        path_filter: Optional[Iterable[str]] = None,    # keep only these path_ids
        pinhole_cams: Optional[Iterable[str]] = None,   # treat these as pinhole (d_cam from K, no LUT, cam_type=1)
        cross_frame_window: Optional[int] = None,       # any_view sampling time window (+/-N frames); None = whole path
        next_frames_window: int = 5,                    # dynamic mode draws extra views from the next N frames
    ):
        self.scene_dir = Path(scene_dir)
        assert self.scene_dir.exists(), f"Scene directory not found: {self.scene_dir}"

        # Handle home000/home000/ double nesting.
        inner = self.scene_dir / self.scene_dir.name
        if inner.exists() and (inner / "common").exists():
            self.scene_root = inner
        elif (self.scene_dir / "common").exists():
            self.scene_root = self.scene_dir
        else:
            raise ValueError(f"common/ directory not found: {self.scene_dir}")

        self.rgb_dir = self.scene_root / "rgb"
        self.depth_dir = self.scene_root / "depth"
        self.common_dir = self.scene_root / "common"
        self.mask_dir = self.scene_root / "mask"
        # per-frame valid_mask: enabled if present. ANDed with the static lens mask, additionally
        # removing sky / inf / far-range pixels above threshold.
        self.valid_mask_dir = self.scene_root / "valid_mask"
        self.use_cleaned_valid_mask = (
            use_cleaned_valid_mask and self.valid_mask_dir.exists()
        )
        # per-frame sky_mask (depth>thr): enabled if present. Folded into non_ambiguous_mask,
        # removing sky from depth supervision and serving as mask-loss GT.
        self.sky_mask_root = self.scene_root / "sky_mask"
        self.use_sky_mask = use_sky_mask and self.sky_mask_root.exists()

        for d in (self.rgb_dir, self.depth_dir, self.common_dir):
            assert d.exists(), f"Directory not found: {d}"

        all_cams = sorted([
            d.name for d in self.rgb_dir.iterdir() if d.is_dir()
        ])
        if cameras_filter is not None:
            keep_cams: Set[str] = set(cameras_filter)
            missing = keep_cams - set(all_cams)
            if missing:
                logger.warning(
                    f"[{self.scene_dir.name}] cameras_filter entries {sorted(missing)} not under rgb/, ignored"
                )
            self.cameras_name = [c for c in all_cams if c in keep_cams]
        else:
            self.cameras_name = all_cams
        self.num_cameras = len(self.cameras_name)
        assert self.num_cameras >= 2, (
            f"Need at least 2 cameras, {self.num_cameras} left after cameras_filter (from {all_cams})"
        )

        # Pinhole cameras: no LUT, d_cam computed from K in LoadOmniFrame.
        self.pinhole_cams: Set[str] = set(pinhole_cams) if pinhole_cams else set()

        self.lut_dir = Path(lut_dir)
        for cam in self.cameras_name:
            if cam in self.pinhole_cams:
                continue   # pinhole needs no LUT
            lut_path = self.lut_dir / f"{cam}_rayEnterDirection.exr"
            assert lut_path.exists(), f"LUT file not found: {lut_path}"

        self.image_format = image_format
        self.depth_format = "npy"
        self.common_format = "npy"

        all_frames = sorted([
            s.replace(f".{self.common_format}", "")
            for s in os.listdir(self.common_dir)
            if s.endswith(f".{self.common_format}")
        ])
        if path_filter is not None:
            keep_paths: Set[str] = set(path_filter)
            self.frame_names = [n for n in all_frames if n.split("_")[0] in keep_paths]
            if not self.frame_names:
                raise AssertionError(
                    f"No samples after path_filter={sorted(keep_paths)}, all paths: "
                    f"{sorted(set(n.split('_')[0] for n in all_frames))}"
                )
            logger.info(
                f"[{self.scene_dir.name}] path_filter applied: "
                f"{len(all_frames)} -> {len(self.frame_names)} frames (paths={sorted(keep_paths)})"
            )
        else:
            self.frame_names = all_frames
        assert len(self.frame_names) > 0, f"No samples: {self.common_dir}"

        # path_id → list of frame_names within that path (for cross-frame sampling)
        self.path_to_frames: dict[str, list[str]] = {}
        for name in self.frame_names:
            path_id = name.split("_")[0]
            self.path_to_frames.setdefault(path_id, []).append(name)
        for fs in self.path_to_frames.values():
            fs.sort()

        self.view_sampling = view_sampling
        # Two non-full-frame configurations:
        #  - any_view (cross_frame_any_view): flatten all (frame, cam) in a path into a view pool
        #    and draw N (need not be a multiple of num_cameras). Geometric alignment is handled by
        #    the collate's _canonicalize_to_view0.
        #  - dynamic (dynamic_anchor_plus_next): the anchor frame's full view set is always the base;
        #    extra views are drawn from the next next_frames_window frames. Use with num_views=[min,max].
        self.any_view_mode = (view_sampling == "cross_frame_any_view")
        self.dynamic_mode = (view_sampling == "dynamic_anchor_plus_next")
        self.cross_frame_window = cross_frame_window
        self.next_frames_window = next_frames_window

        # num_views: int or [min, max]. The latter enables variable_num_views, with the count chosen
        # per batch by DynamicBatchSampler via nv_idx (same protocol as WaiDataset).
        if isinstance(num_views, (list, tuple)):
            assert len(num_views) == 2, f"num_views range must be [min,max], got {num_views}"
            lo, hi = int(num_views[0]), int(num_views[1])
            self.num_views_list = list(range(max(lo, 2), hi + 1)) or [max(lo, 2)]
            self.variable_num_views = True
        else:
            self.num_views_list = [int(num_views)]
            self.variable_num_views = False
        self.num_views = self.num_views_list[0]

        if self.dynamic_mode:
            # base = anchor's full num_cameras views; extra = num_views - num_cameras >= 0
            assert min(self.num_views_list) >= self.num_cameras, (
                f"dynamic mode num_views min {min(self.num_views_list)} must be >= num_cameras {self.num_cameras}"
            )
            self.frames_per_sample = None
        elif self.any_view_mode:
            assert min(self.num_views_list) >= 2, f"any_view mode requires num_views >= 2"
            self.frames_per_sample = None
        else:
            # Full-frame mode: no variable count, num_views must be a multiple of num_cameras.
            assert not self.variable_num_views, (
                f"Full-frame mode ({view_sampling}) does not support num_views range; use dynamic_anchor_plus_next"
            )
            assert self.num_views % self.num_cameras == 0, (
                f"num_views={self.num_views} must be a multiple of num_cameras={self.num_cameras}"
            )
            self.frames_per_sample = self.num_views // self.num_cameras
        self.views_per_frame = self.num_cameras

        self.pipeline = Compose(pipeline)

    def __len__(self) -> int:
        return len(self.frame_names)

    def __getitem__(self, idx) -> dict:
        # Bad-sample fallback: occasional missing rgb/depth files or I/O jitter. A single bad
        # sample must not crash the worker (would stall the DDP barrier). Retry 5 times, bumping
        # sample_idx by N each time. idx may be an int or a tuple (sample_idx, ar_idx[, nv_idx]);
        # only sample_idx is bumped, the rest forwarded, to avoid tuple+int TypeError.
        last_exc = None
        for retry in range(5):
            if isinstance(idx, tuple):
                bumped = (idx[0] + retry, *idx[1:])
            else:
                bumped = idx + retry
            try:
                return self._getitem_impl(bumped)
            except (AssertionError, FileNotFoundError, OSError, ValueError) as e:
                last_exc = e
                logger.warning(
                    f"[{self.scene_dir.name}] sample idx={bumped} load failed (retry {retry}): "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )
        raise RuntimeError(
            f"5 consecutive samples failed to load, scene={self.scene_dir.name}, "
            f"start_idx={idx}, last error: {last_exc}"
        )

    def _getitem_impl(self, idx) -> dict:
        # Parse the sampler's tuple idx. Third element is nv_idx: the per-batch view-count index.
        resolution_idx = None
        num_views_idx = 0
        if isinstance(idx, tuple):
            if len(idx) == 2:
                idx, resolution_idx = idx
            elif len(idx) == 3:
                idx, resolution_idx, num_views_idx = idx
            else:
                raise ValueError(f"Unsupported idx: {idx}")
        num_views_idx = min(num_views_idx, len(self.num_views_list) - 1)
        cur_num_views = self.num_views_list[num_views_idx]

        anchor = self.frame_names[idx % len(self.frame_names)]
        path_id = anchor.split("_")[0]
        siblings = self.path_to_frames[path_id]

        if self.dynamic_mode:
            # Anchor full-view base + random extras from the next N frames.
            views = self._sample_dynamic_views(anchor, siblings, cur_num_views)
        elif self.any_view_mode:
            # Arbitrary view count: sample (frame, cam) pairs directly.
            views = self._sample_any_views(anchor, siblings, cur_num_views)
        elif self.frames_per_sample == 1:
            frames = [anchor]
            views = [(f, cam) for f in frames for cam in self.cameras_name]
        else:
            anchor_local = siblings.index(anchor)
            if self.view_sampling == "cross_frame_within_path":
                pool = [f for f in siblings if f != anchor]
                if len(pool) >= self.frames_per_sample - 1:
                    extras = random.sample(pool, self.frames_per_sample - 1)
                else:
                    # Pool too small: sample with repetition.
                    extras = (pool * (self.frames_per_sample // max(len(pool), 1) + 1))[
                        : self.frames_per_sample - 1
                    ]
                frames = [anchor] + extras
            else:
                # single_frame but multiple frames requested: take anchor's following frames in order.
                frames = [
                    siblings[(anchor_local + i) % len(siblings)]
                    for i in range(self.frames_per_sample)
                ]
            # views: [(frame_name, cam_name), ...], outer frame, inner camera.
            views = [(f, cam) for f in frames for cam in self.cameras_name]

        sample_info = {
            "scene_dir": str(self.scene_root.absolute()),
            "views": views,
            "image_dir": "rgb",
            "depth_dir": "depth",
            "common_dir": "common",
            "mask_dir": "mask",
            "valid_mask_dir": "valid_mask" if self.use_cleaned_valid_mask else None,
            "sky_mask_dir": "sky_mask" if self.use_sky_mask else None,
            "lut_dir": str(self.lut_dir.absolute()),
            "pinhole_cams": sorted(self.pinhole_cams),   # LoadOmniFrame uses K instead of LUT for these
            "image_format": self.image_format,
            "depth_format": self.depth_format,
            "common_format": self.common_format,
            "dataset_name": "omni_isaacsim",
            "is_metric": True,
            "quality_tier": "A",
        }
        if resolution_idx is not None:
            sample_info["resolution_idx"] = resolution_idx

        if self.pipeline is not None:
            return self.pipeline(sample_info)
        return sample_info

    def _sample_dynamic_views(self, anchor: str, siblings: list, num_views: int) -> list:
        """Dynamic view count: anchor full-view base plus random extras from the next N frames.

        - base = anchor's full num_cameras views (order = cameras_name), always present;
          view 0 = anchor's first camera, the world origin for _canonicalize_to_view0.
        - extra num_views - num_cameras views: sampled without replacement from the (frame, cam)
          pool of the next next_frames_window frames (cross-frame, with translation baseline).
        - Falls back to preceding frames at the trajectory tail; repeats base for single-frame paths.
        """
        base = [(anchor, cam) for cam in self.cameras_name]
        extra_n = num_views - self.num_cameras
        if extra_n <= 0:
            return base[:num_views]

        anchor_local = siblings.index(anchor)
        W = self.next_frames_window
        next_frames = [
            siblings[anchor_local + i]
            for i in range(1, W + 1)
            if anchor_local + i < len(siblings)
        ]
        if not next_frames:
            # Trajectory tail: no following frames, fall back to the preceding W frames.
            next_frames = [
                siblings[anchor_local - i]
                for i in range(1, W + 1)
                if anchor_local - i >= 0
            ]
        pool = [(f, c) for f in next_frames for c in self.cameras_name]
        if not pool:
            # Single-frame path: nothing else to sample, repeat base.
            pool = list(base)

        if len(pool) >= extra_n:
            extras = random.sample(pool, extra_n)
        else:
            extras = (pool * (extra_n // max(len(pool), 1) + 1))[:extra_n]
        return base + extras

    def _sample_any_views(self, anchor: str, siblings: list, num_views: int) -> list:
        """Arbitrary view count: draw num_views views from the (frame, cam) pool of one path.

        - view 0 is anchored to (anchor, random camera) so the indexed frame is always present and
          serves as a stable world origin for _canonicalize_to_view0.
        - the remaining num_views-1 are sampled without replacement from all (frame, cam) pairs in
          the time window; falls back to sampling with replacement if the pool is too small.
        - mixes same-frame views (baseline 0) and cross-frame views (translation baseline).
        """
        anchor_local = siblings.index(anchor)
        w = self.cross_frame_window
        if w is not None and w > 0:
            lo = max(0, anchor_local - w)
            hi = min(len(siblings), anchor_local + w + 1)
            cand_frames = siblings[lo:hi]
        else:
            cand_frames = siblings

        view0_cam = random.choice(self.cameras_name)
        views = [(anchor, view0_cam)]

        need = num_views - 1
        pool = [
            (f, c)
            for f in cand_frames
            for c in self.cameras_name
            if not (f == anchor and c == view0_cam)
        ]
        if len(pool) >= need:
            extras = random.sample(pool, need)
        else:
            extras = (pool * (need // max(len(pool), 1) + 1))[:need]
        views.extend(extras)
        return views


class OmniDataset(ConcatDataset):
    """Omni multi-scene concatenation (scans all home_xxx under data_root)."""

    def __init__(
        self,
        data_root: str = "/path/to/omni",
        lut_dir: str = DEFAULT_LUT_DIR,
        pipeline: Optional[Sequence[Callable]] = None,
        image_format: str = "jpg",
        view_sampling: str = "single_frame",
        num_views: int = 4,
        scene_filter: Optional[Iterable[str]] = None,
        exclude_scenes: Optional[Iterable[str]] = None,
        use_cleaned_valid_mask: bool = True,
        use_sky_mask: bool = True,
        cameras_filter: Optional[Iterable[str]] = None,
        path_filter: Optional[Iterable[str]] = None,
        pinhole_cams: Optional[Iterable[str]] = None,
        cross_frame_window: Optional[int] = None,
        next_frames_window: int = 5,
    ):
        data_root = Path(data_root)
        assert data_root.exists(), f"data_root not found: {data_root}"

        # Scene directories: nested or flat.
        scene_dirs = sorted([
            d for d in data_root.iterdir()
            if d.is_dir() and (
                (d / d.name / "common").exists() or (d / "common").exists()
            )
        ])
        assert len(scene_dirs) > 0, f"No valid omni scenes found: {data_root}"

        # scene_filter: keep only these scenes (by directory name); exclude_scenes: drop these.
        if scene_filter is not None:
            keep: Set[str] = set(scene_filter)
            missing = keep - {d.name for d in scene_dirs}
            if missing:
                logger.warning(f"scene_filter: {len(missing)} scenes not found under data_root: {sorted(missing)[:5]}...")
            scene_dirs = [d for d in scene_dirs if d.name in keep]
            assert len(scene_dirs) > 0, f"No scenes left after scene_filter"
        elif exclude_scenes is not None:
            drop: Set[str] = set(exclude_scenes)
            scene_dirs = [d for d in scene_dirs if d.name not in drop]
            assert len(scene_dirs) > 0, f"No scenes left after exclude_scenes"

        datasets = []
        for sd in scene_dirs:
            try:
                ds = OmniSceneDataset(
                    str(sd),
                    lut_dir=lut_dir,
                    pipeline=pipeline,
                    image_format=image_format,
                    view_sampling=view_sampling,
                    num_views=num_views,
                    use_cleaned_valid_mask=use_cleaned_valid_mask,
                    use_sky_mask=use_sky_mask,
                    cameras_filter=cameras_filter,
                    path_filter=path_filter,
                    pinhole_cams=pinhole_cams,
                    cross_frame_window=cross_frame_window,
                    next_frames_window=next_frames_window,
                )
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Skipping {sd}: {e}")

        assert len(datasets) > 0, f"No valid scenes: {data_root}"

        total = sum(len(d) for d in datasets)
        logger.info(
            f"Omni dataset: {len(datasets)} scenes, {total} samples, S={num_views}, "
            f"sampling={view_sampling}"
        )
        for sd, ds in zip(scene_dirs, datasets):
            logger.info(f"  {sd.name}: {len(ds)} samples, cams={ds.cameras_name}")

        super().__init__(datasets)

    def __getitem__(self, idx):
        """Support tuple idx (sample_idx, ar_idx, nv_idx).

        DynamicBatchSampler yields tuple idx in weighted multi-dataset mode, but
        torch.ConcatDataset.__getitem__ tests `if idx < 0` first, raising TypeError on a tuple.
        Same pattern as WaiDataset.__getitem__: bisect the sub-dataset and forward (local_idx, *rest).
        """
        if isinstance(idx, tuple):
            sample_idx = idx[0]
            rest = idx[1:]
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
