"""WAI-format data loading transform.

Loads RGB (png) and depth (exr) from scene_meta.json frame info, builds the 3x3 intrinsics from
fl_x/fl_y/cx/cy and the 4x4 c2w extrinsics from transform_matrix.

Cleaning:
1. Depth: NaN/inf -> 0, negatives -> 0
2. Intrinsics: validate 3x3 pinhole
3. Extrinsics: validate 4x4 c2w, all finite
4. MoGe mask: when moge2_mask is present, keep only pixels MoGe deems reliable
"""

import logging
import os.path as osp

import cv2
import numpy as np
import OpenEXR
import Imath

from .transforms import BaseTransform

logger = logging.getLogger(__name__)


def _rebase(scene_dir: str, rel: str) -> str:
    """Rebase a frame path from scene_meta.json onto the current scene_dir.

    Some subsets store absolute paths from an old mount point; osp.join would drop scene_dir and
    return a nonexistent path. For absolute paths, take the part after "<scene_name>/" and rejoin
    onto scene_dir (falling back to "<parent>/<file>"). Relative paths are joined as-is.
    """
    if osp.isabs(rel):
        scene_name = osp.basename(scene_dir.rstrip("/"))
        marker = f"/{scene_name}/"
        if marker in rel:
            rel = rel.split(marker, 1)[1]
        else:
            rel = osp.join(osp.basename(osp.dirname(rel)), osp.basename(rel))
    return osp.join(scene_dir, rel)


def _load_exr_depth(path: str) -> np.ndarray:
    """Load an EXR depth map, returning an (H, W) float32 array."""
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    channels = list(header["channels"].keys())
    raw = exr_file.channel(channels[0], Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
    return depth


def _load_moge_mask(path: str) -> np.ndarray:
    """Load a MoGe confidence mask (PNG), returning an (H, W) bool array."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    # MoGe mask: nonzero = reliable region.
    return mask > 0


class LoadWaiFrame(BaseTransform):
    """Load WAI-format multi-view frame data.

    Input sample_info must contain:
        - scene_dir: scene absolute path
        - frame_indices: frame index list
        - meta: scene_meta.json contents

    Output:
        - cameras_name: view IDs
        - cameras_rgb: {view_id: np.array(H, W, 3, uint8)}
        - cameras_depth: {view_id: np.array(H, W, float32)}
        - cameras_intrinsics_extrinsics: {view_id: {"intrinsics": (3,3), "extrinsics_world": (4,4)}}

    Cleaning:
        - transform_matrix is a c2w matrix
        - intrinsics built and validated from fl_x, fl_y, cx, cy
        - depth: NaN/inf/negatives -> 0
        - MoGe mask: when moge2_mask is present, keep only reliable pixels
    """

    def transform(self, sample_info: dict) -> dict:
        scene_dir = sample_info["scene_dir"]
        frame_indices = sample_info["frame_indices"]
        # Prefer sampled_frames (sampled frames only) to reduce IPC serialization.
        sampled_frames = sample_info.get("sampled_frames", None)
        if sampled_frames is not None:
            frames = None
        else:
            # Legacy meta path.
            meta = sample_info["meta"]
            frames = meta["frames"]

        # Dataset config injected by WaiSceneDataset (see data_configs/). Defaults to an empty
        # config (all cleaning off).
        ds_cfg = sample_info.get("dataset_config", {})
        depth_key = ds_cfg.get("depth_key", "depth")
        # If the depth_key field is absent for some frames, fall back to fallback_depth_key
        # rather than dropping the sample.
        fallback_depth_key = ds_cfg.get("fallback_depth_key", None)
        apply_moge_mask = ds_cfg.get("apply_moge_mask", True)
        apply_sky_mask = ds_cfg.get("apply_sky_mask", False)
        clip_depth_p95 = ds_cfg.get("clip_depth_p95", False)
        nan_inf_to_zero = ds_cfg.get("nan_inf_to_zero", True)
        negative_to_zero = ds_cfg.get("negative_to_zero", True)
        confidence_key = ds_cfg.get("confidence_key", None)
        confidence_thres = float(ds_cfg.get("confidence_thres", 0.0))

        cameras_name = []
        cameras_rgb = {}
        cameras_depth = {}
        cameras_intrinsics_extrinsics = {}
        # Non-ambiguous mask GT, three cases:
        #   1. MoGe mask present: use MoGe directly (no intersection with depth>0). These are
        #      SfM/MVS/reconstruction datasets where depth=0 is mostly MVS failure, not ambiguity.
        #   2. Synthetic/rendered (scale_type=synthetic): mask = (depth>0) & finite; depth=0 = sky.
        #   3. Otherwise (laser/SfM without MoGe): mask ~ all 1 (fallback); blind spots are not
        #      semantic ambiguity.
        cameras_mask = {}
        scale_type = ds_cfg.get("scale_type", "unknown")

        for i, fi in enumerate(frame_indices):
            frame = sampled_frames[i] if sampled_frames is not None else frames[fi]
            frame_name = frame["frame_name"]
            view_id = frame_name
            cameras_name.append(view_id)

            # Load RGB (returns None on missing file so the dataset retries another sample).
            image_rel = frame.get("image", frame.get("file_path"))
            image_path = _rebase(scene_dir, image_rel)
            image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image_data is None:
                logger.warning(f"Image missing or unreadable, skipping: {image_path}")
                return None
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            cameras_rgb[view_id] = image_data

            # Load depth + cleaning. depth_key may be "depth" / "rendered_depth" / "pred_depth/..".
            depth_rel = frame.get(depth_key.replace("/", "_"))
            if depth_rel is None:
                # WAI frame fields are flat (no slash); retry with the full key.
                depth_rel = frame.get(depth_key)
            # Primary key missing -> fallback_depth_key -> "depth".
            if depth_rel is None and fallback_depth_key:
                depth_rel = frame.get(fallback_depth_key.replace("/", "_"))
                if depth_rel is None:
                    depth_rel = frame.get(fallback_depth_key)
            if depth_rel is None:
                depth_rel = frame.get("depth")
            if depth_rel is None:
                logger.warning(
                    f"No depth field (key={depth_key}, fallback={fallback_depth_key}), skipping: {scene_dir}/{view_id}"
                )
                return None
            depth_path = _rebase(scene_dir, depth_rel)
            try:
                if depth_path.endswith(".exr"):
                    depth_data = _load_exr_depth(depth_path)
                else:
                    depth_data = np.load(depth_path, allow_pickle=True).astype(np.float32)
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"Depth missing or unreadable, skipping: {depth_path}")
                return None

            # Step 1: NaN/inf/negatives -> 0
            if nan_inf_to_zero:
                depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
            if negative_to_zero:
                depth_data[depth_data < 0] = 0.0

            # Step 2: confidence filtering (mvsanywhere depth_confidence)
            if confidence_key and confidence_key in frame:
                conf_path = _rebase(scene_dir, frame[confidence_key])
                try:
                    if conf_path.endswith(".exr"):
                        conf = _load_exr_depth(conf_path)
                    else:
                        conf = np.load(conf_path, allow_pickle=True).astype(np.float32)
                    if conf.shape != depth_data.shape:
                        conf = cv2.resize(
                            conf.astype(np.float32),
                            (depth_data.shape[1], depth_data.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    depth_data = np.where(conf > confidence_thres, depth_data, 0.0)
                except (FileNotFoundError, OSError):
                    pass  # skip this step if the conf file is missing

            # Step 3: sky mask (zero out sky pixel depth)
            if apply_sky_mask and "skymask" in frame:
                sky_path = _rebase(scene_dir, frame["skymask"])
                sky = cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)
                if sky is not None:
                    if sky.shape != depth_data.shape:
                        sky = cv2.resize(
                            sky, (depth_data.shape[1], depth_data.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    depth_data = np.where(sky > 0, 0.0, depth_data)

            # Step 4: MoGe mask (keep only MoGe-approved pixels).
            # moge_mask_applied drives the mask-GT branch selection below.
            moge_mask_applied = False
            moge_mask_resized = None
            if apply_moge_mask and "moge2_mask" in frame:
                mask_rel = frame["moge2_mask"]
                mask_path = _rebase(scene_dir, mask_rel)
                moge_mask = _load_moge_mask(mask_path)
                if moge_mask is not None:
                    if moge_mask.shape != depth_data.shape:
                        moge_mask = cv2.resize(
                            moge_mask.astype(np.uint8),
                            (depth_data.shape[1], depth_data.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ) > 0
                    depth_data = np.where(moge_mask, depth_data, 0.0)
                    moge_mask_applied = True
                    moge_mask_resized = moge_mask.astype(np.bool_)

            # Step 5: far-point clipping (drop the top 5% to stabilize the metric-loss tail).
            if clip_depth_p95:
                valid = depth_data[depth_data > 0]
                if valid.size > 100:
                    p95 = float(np.percentile(valid, 95))
                    depth_data = np.where(depth_data > p95, 0.0, depth_data)

            cameras_depth[view_id] = depth_data

            # Non-ambiguous mask GT: three cases (see the note above).
            if moge_mask_applied and moge_mask_resized is not None:
                # Case 1: pure MoGe (no intersection with depth>0).
                non_ambig_mask_view = moge_mask_resized
            elif scale_type == "synthetic":
                # Case 2: depth>0 as mask; synthetic data has no blind spots, depth=0 = sky.
                non_ambig_mask_view = (depth_data > 0) & np.isfinite(depth_data)
            else:
                # Case 3 fallback: nearly all 1; laser/SfM depth=0 is blind spot, not ambiguity.
                non_ambig_mask_view = np.ones_like(depth_data, dtype=np.bool_)
            cameras_mask[view_id] = non_ambig_mask_view.astype(np.bool_)

            # Build and validate intrinsics K (3x3).
            fl_x = float(frame["fl_x"])
            fl_y = float(frame["fl_y"])
            cx = float(frame["cx"])
            cy = float(frame["cy"])
            K = np.array([
                [fl_x, 0.0, cx],
                [0.0, fl_y, cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)

            if fl_x <= 0 or fl_y <= 0 or not (np.isfinite(fl_x) and np.isfinite(fl_y)):
                logger.warning(f"Invalid intrinsics (fl_x={fl_x}, fl_y={fl_y}), skipping: {scene_dir}/{view_id}")
                return None

            # Extrinsics: transform_matrix is c2w.
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)

            if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                logger.warning(f"Invalid extrinsics (shape={c2w.shape}, finite={np.isfinite(c2w).all()}), skipping: {scene_dir}/{view_id}")
                return None

            cameras_intrinsics_extrinsics[view_id] = {
                "intrinsics": K,
                "extrinsics_world": c2w,
            }

        # Keep only the metadata the collate needs; avoid passing large objects to the main process.
        frame_info = {
            "dataset_name": sample_info.get("dataset_name", ""),
            "quality_tier": sample_info.get("quality_tier", "A"),
            "is_metric": sample_info.get("is_metric", False),
        }
        result = {
            "frame_info": frame_info,
            "cameras_name": cameras_name,
            "cameras_rgb": cameras_rgb,
            "cameras_depth": cameras_depth,
            "cameras_non_ambiguous_mask": cameras_mask,
            "cameras_intrinsics_extrinsics": cameras_intrinsics_extrinsics,
            # Expose dataset scale info to the batch for loss routing:
            #   is_metric=True  -> metric scale-factor + scale-invariant supervision
            #   is_metric=False -> scale-invariant / relative depth only
            "dataset_name": sample_info.get("dataset_name", ""),
            "is_metric": bool(sample_info.get("is_metric", False)),
            "participate_scale_factor": bool(
                sample_info.get("participate_scale_factor",
                                sample_info.get("is_metric", False))
            ),
        }

        if "resolution_idx" in sample_info:
            result["resolution_idx"] = sample_info["resolution_idx"]

        return result
