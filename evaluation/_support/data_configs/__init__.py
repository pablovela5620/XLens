"""
xlens / data_configs

One yaml config per WAI dataset, centralizing:

- metric depth flag (controls scale-factor estimation / metric loss participation)
- covisibility threshold and version
- modality presence (moge mask / sky mask / dynamic mask / confidence map / ...)
- training-time GT cleaning (moge mask, sky mask, p95 far-point clipping)
- camera convention / zoom

Moves per-dataset cleaning logic out of the dataloader into declarative yaml
for easier auditing, adding datasets, and cross-task reuse.

Usage:
    from evaluation._support.data_configs import get_dataset_config, list_dataset_configs

    cfg = get_dataset_config("megadepth")     # -> DatasetConfig
    cfg.is_metric                             # False
    cfg.covisibility_threshold                # 0.25
    cfg.apply_moge_mask                       # True
    cfg.participate_scale_factor              # False

Unconfigured datasets return a conservative default (metric=False, no mask,
covis=0.25) with a warning. Add a dataset by dropping a yaml into this directory;
no Python changes required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent


@dataclass
class DatasetConfig:
    # ---------- identity ----------
    name: str
    display_name: str = ""
    source: str = ""

    # ---------- depth scale ----------
    # metric depth (meters). False means arbitrary units (colmap-scale / synthetic-scale).
    is_metric: bool = False
    # participate in metric scale-factor estimation.
    # Usually = is_metric; non-metric datasets only do scale-invariant / relative depth regression.
    participate_scale_factor: bool = False
    # metric / colmap / synthetic / stereo (informational)
    scale_type: str = "unknown"

    # ---------- data mixing strategy (aligned with MoGe-v2) ----------
    # Quality tier controlling which loss terms participate:
    #   A = synthetic rendered GT (highest quality, full loss incl. normal and fine-grained patch)
    #   B = laser-scan / high-quality reconstruction GT (normal disabled due to surface noise)
    #   C = SfM pseudo-GT / self-distilled pseudo-GT (normal + grad + local disabled;
    #       fine geometry unreliable, only global L1 + camera/scale supervision)
    quality_tier: str = "A"
    # Weight for multi-dataset weighted sampling (MoGe-v2 v2.json style, hand-tuned by quality).
    # Tuned by quality x inverse size, not by dataset size ratio.
    sampling_weight: float = 1.0

    # ---------- modalities present in WAI ----------
    depth_key: str = "depth"        # frame[depth_key] points to .exr / .npy
    # Used when depth_key is absent (e.g. frames where teacher_depth alignment failed).
    # Typical config:
    #   depth_key: teacher_depth
    #   fallback_depth_key: mvsanywhere_depth    # dl3dv
    #   fallback_depth_key: depth                # blendedmvs / megadepth
    fallback_depth_key: Optional[str] = None
    has_depth: bool = True
    has_moge_mask: bool = False     # frame["moge2_mask"]
    has_moge_depth: bool = False    # frame["moge2_depth"]
    has_sky_mask: bool = False      # frame["skymask"]
    has_dynamic_mask: bool = False
    has_confidence: bool = False    # frame["depth_confidence/<predictor>"]
    confidence_key: Optional[str] = None
    confidence_thres: float = 0.0

    # ---------- loss participation ----------
    # Participate in mask-head BCE supervision. Sources without sky/semantic masks
    # (e.g. KITTI-360 has only a lens FOV mask) should set False, otherwise a
    # depth-valid / FOV mask gets fed as the mask target and pollutes the mask head.
    contributes_mask_loss: bool = True

    # ---------- view sampling ----------
    view_sampling: str = "covisibility"     # covisibility / random / sequential
    covisibility_version: str = "v0"        # subdir under covisibility/
    covisibility_threshold: float = 0.25

    # ---------- depth cleaning (training-time) ----------
    apply_moge_mask: bool = False           # mask GT depth with moge2_mask
    apply_sky_mask: bool = False            # set sky pixel depth to 0 via skymask
    apply_dynamic_mask: bool = False
    clip_depth_p95: bool = False            # zero out depth > p95 (far outdoor points)
    nan_inf_to_zero: bool = True
    negative_to_zero: bool = True

    # ---------- camera ----------
    camera_convention: str = "opencv"       # transform_matrix convention
    varying_intrinsics: bool = False

    # ---------- misc ----------
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if "name" not in data:
            data["name"] = path.stem
        # `participate_scale_factor` defaults to is_metric
        if "participate_scale_factor" not in data:
            data["participate_scale_factor"] = bool(data.get("is_metric", False))
        # Drop unknown fields to avoid crashing on extra yaml keys
        known = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - known
        if unknown:
            logger.warning(
                f"data_configs/{path.name}: ignoring unknown fields: {sorted(unknown)}"
            )
            for k in unknown:
                data.pop(k)
        return cls(**data)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------- registry ----------
_REGISTRY: Dict[str, DatasetConfig] = {}
_LOADED = False


def _load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    for yml in sorted(CONFIGS_DIR.glob("*.yaml")):
        if yml.stem.startswith("_"):
            continue  # _template.yaml, etc.
        try:
            cfg = DatasetConfig.from_yaml(yml)
        except Exception as e:
            logger.error(f"failed to load data_configs/{yml.name}: {e}")
            continue
        _REGISTRY[cfg.name] = cfg
    _LOADED = True
    logger.info(
        f"data_configs: loaded {len(_REGISTRY)} dataset configs: "
        f"{sorted(_REGISTRY.keys())}"
    )


def get_dataset_config(name: str, strict: bool = False) -> DatasetConfig:
    """Get config by dataset (directory) name. Returns conservative default if unregistered; raises if strict=True."""
    _load_all()
    if name in _REGISTRY:
        return _REGISTRY[name]
    if strict:
        raise KeyError(
            f"dataset '{name}' has no data_configs/{name}.yaml config. "
            f"registered: {sorted(_REGISTRY.keys())}"
        )
    logger.warning(
        f"dataset '{name}' not registered in data_configs/, using conservative default "
        f"(is_metric=False, no masks, covis=0.25)"
    )
    return DatasetConfig(name=name)


def list_dataset_configs() -> List[str]:
    _load_all()
    return sorted(_REGISTRY.keys())


def iter_dataset_configs() -> Iterator[DatasetConfig]:
    _load_all()
    for name in sorted(_REGISTRY.keys()):
        yield _REGISTRY[name]


__all__ = [
    "DatasetConfig",
    "get_dataset_config",
    "list_dataset_configs",
    "iter_dataset_configs",
]
