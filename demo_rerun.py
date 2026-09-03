"""Run X-Lens on the bundled heterogeneous rig or an ETH3D stereo pair.

The model path is unchanged upstream inference: X-Lens preprocessing helpers,
``XLensInference`` with the released ViT-S YAML, and BF16 CUDA autocast. Rerun
adds calibrated pinhole rigs, 2D LUT-fisheye views, and fused scene geometry.

    pixi run demo
    pixi run demo-eth3d
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypeAlias, TypedDict, cast

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import tyro
from einops import rearrange
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray
from PIL import Image
from safetensors.torch import load_file
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.rerun_rig_logger import SCHEMA_VERSION, log_rig_static
from simplecv.rig import CameraSensor, Rig, RigCalibration
from torch import Tensor

from xlens.inference import XLensInference
from xlens.inference.geometry import fuse_point_cloud
from xlens.inference.preprocess import (
    CAM_TYPE_FISHEYE,
    CAM_TYPE_PINHOLE,
    assemble_batch,
    load_fisheye_lut,
    pinhole_d_cam,
)

SceneChoice: TypeAlias = Literal["hetero_loft", "eth3d"]
Batch: TypeAlias = dict[str, Tensor | None]


class ViewManifest(TypedDict):
    """One view from an upstream X-Lens scene manifest."""

    image: str
    c2w: list[list[float]]
    camera: NotRequired[str]
    mask: NotRequired[str]
    dcam: NotRequired[str]
    lut: NotRequired[str]
    K: NotRequired[list[list[float]]]
    cam_type: NotRequired[int]


class SceneManifest(TypedDict):
    """Fields consumed from an upstream X-Lens scene manifest."""

    image_size: NotRequired[list[int]]
    views: list[ViewManifest]


@dataclass(frozen=True, slots=True)
class SceneData:
    """One calibrated multi-view scene ready for X-Lens preprocessing."""

    names: list[str]
    """Human-readable camera names in model view order."""
    images: list[UInt8[ndarray, "h w 3"]]
    """RGB images with a shared spatial shape."""
    d_cams: list[Float32[ndarray, "h w 3"]]
    """Per-view OpenCV camera-frame unit rays."""
    cam_types: list[int]
    """Camera type IDs: zero for fisheye and one for pinhole."""
    c2w: Float32[ndarray, "views 4 4"]
    """Camera-to-world poses in the view-0 camera frame."""
    masks: list[Bool[ndarray, "h w"]]
    """Per-view masks used only for logging and point-cloud fusion."""
    intrinsics: list[Float32[ndarray, "3 3"] | None]
    """Pinhole matrices, with ``None`` for LUT fisheye views."""


@dataclass(frozen=True, slots=True)
class MiddleburyCalibration:
    """Calibration fields needed from an ETH3D Middlebury-v3 file."""

    left_k: Float64[ndarray, "3 3"]
    """Left pinhole intrinsic matrix in pixels."""
    right_k: Float64[ndarray, "3 3"]
    """Right pinhole intrinsic matrix in pixels."""
    baseline_m: float
    """Camera-centre separation converted from millimetres to metres."""
    doffs_px: float
    """Disparity offset from the Middlebury calibration."""


@dataclass(frozen=True, slots=True)
class Eth3dMetrics:
    """Single-scene ETH3D baseline metrics."""

    epe_px: float
    """Mean absolute disparity error on the protocol mask."""
    bad1_percent: float
    """Percentage of protocol-mask pixels above one-pixel error."""
    abs_rel: float
    """Mean relative metric-depth error against depth derived from GT."""
    negative_sign_epe_px: float
    """EPE of the rejected negative-disparity sign convention."""
    valid_pixels: int
    """Number of pixels in the evaluation mask."""


@dataclass
class Config:
    """X-Lens Rerun demo configuration."""

    rr_config: RerunTyroConfig
    """Rerun spawn, connect, save, and headless options."""
    scene: SceneChoice = "hetero_loft"
    """Scene to run: the bundled heterogeneous rig or ETH3D pair."""
    checkpoint: Path = Path("weights/x-lens/model.safetensors")
    """Authors' gated safetensors checkpoint."""
    model_config: Path = Path("configs/xlens_vits.yaml")
    """Required ViT-S architecture YAML for the bare safetensors."""
    hetero_manifest: Path = Path("examples/hetero_loft/scene.json")
    """Bundled six-view X-Lens manifest."""
    eth3d_scene_dir: Path = Path("data/datasets/ETH3D/two_view_training/playground_1l")
    """ETH3D directory containing the images and ``calib.txt``."""
    eth3d_gt_dir: Path = Path("data/datasets/ETH3D/two_view_training_gt/playground_1l")
    """ETH3D directory containing disparity and non-occlusion GT."""
    max_depth_m: float = 25.0
    """Maximum euclidean point range and displayed depth in metres."""
    max_disparity_px: float = 192.0
    """ETH3D ground-truth disparity cutoff."""
    fov_max_deg: float = 85.0
    """Maximum fisheye ray angle used only for cloud fusion."""
    conf_drop_percent: float = 8.0
    """Global confidence percentile dropped only during cloud fusion."""
    benchmark_warmups: int = 10
    """Untimed heterogeneous-scene CUDA forwards."""
    benchmark_runs: int = 50
    """Timed heterogeneous-scene CUDA forwards."""


def read_rgb(path: Path, image_hw: tuple[int, int] | None = None) -> UInt8[ndarray, "h w 3"]:
    """Read one RGB image and optionally resize as the upstream CLI does.

    Args:
        path: Input image path.
        image_hw: Optional target ``(height, width)``.

    Returns:
        RGB uint8 image with shape ``(height, width, 3)``.
    """
    image: Image.Image = Image.open(path).convert("RGB")
    if image_hw is not None:
        image = image.resize((image_hw[1], image_hw[0]), Image.Resampling.BILINEAR)
    rgb: UInt8[ndarray, "h w 3"] = np.asarray(image, dtype=np.uint8).copy()
    return rgb


def read_mask(path: Path, image_hw: tuple[int, int]) -> Bool[ndarray, "h w"]:
    """Read a mask with upstream nearest-neighbour resizing.

    Args:
        path: Grayscale mask path.
        image_hw: Target ``(height, width)``.

    Returns:
        Boolean valid-pixel mask with shape ``(height, width)``.
    """
    mask_image: Image.Image = Image.open(path).convert("L")
    if mask_image.size != (image_hw[1], image_hw[0]):
        mask_image = mask_image.resize((image_hw[1], image_hw[0]), Image.Resampling.NEAREST)
    mask: Bool[ndarray, "h w"] = np.asarray(mask_image) > 127
    return mask


def load_manifest_scene(path: Path) -> SceneData:
    """Parse the tiny JSON boundary and use upstream helpers for camera rays.

    Args:
        path: Upstream X-Lens scene manifest.

    Returns:
        Images, rays, camera types, masks, poses, and optional intrinsics.
    """
    manifest: SceneManifest = cast(SceneManifest, json.loads(path.read_text()))
    root: Path = path.parent
    size_values: list[int] | None = manifest.get("image_size")
    image_hw: tuple[int, int] | None = None if size_values is None else (size_values[0], size_values[1])
    names: list[str] = []
    images: list[UInt8[ndarray, "h w 3"]] = []
    d_cams: list[Float32[ndarray, "h w 3"]] = []
    cam_types: list[int] = []
    poses: list[Float32[ndarray, "4 4"]] = []
    masks: list[Bool[ndarray, "h w"]] = []
    intrinsics: list[Float32[ndarray, "3 3"] | None] = []

    for index, view in enumerate(manifest["views"]):
        image: UInt8[ndarray, "h w 3"] = read_rgb(root / view["image"], image_hw)
        height: int = image.shape[0]
        width: int = image.shape[1]
        if "dcam" in view or "lut" in view:
            lut_name: str = view["dcam"] if "dcam" in view else view["lut"]
            d_cam: Float32[ndarray, "h w 3"] = load_fisheye_lut(str(root / lut_name), height, width)
            cam_type: int = int(view.get("cam_type", CAM_TYPE_FISHEYE))
            intrinsic: Float32[ndarray, "3 3"] | None = None
        elif "K" in view:
            intrinsic = np.asarray(view["K"], dtype=np.float32)
            d_cam = pinhole_d_cam(intrinsic, height, width)
            cam_type = int(view.get("cam_type", CAM_TYPE_PINHOLE))
        else:
            raise ValueError(f"Manifest view {index} needs K, lut, or dcam calibration")

        pose: Float32[ndarray, "4 4"] = np.asarray(view["c2w"], dtype=np.float32)
        mask: Bool[ndarray, "h w"] = (
            read_mask(root / view["mask"], (height, width))
            if "mask" in view
            else np.ones((height, width), dtype=np.bool_)
        )
        names.append(view.get("camera", Path(view["image"]).stem))
        images.append(image)
        d_cams.append(d_cam)
        cam_types.append(cam_type)
        poses.append(pose)
        masks.append(mask)
        intrinsics.append(intrinsic)

    c2w: Float32[ndarray, "views 4 4"] = np.stack(poses).astype(np.float32, copy=False)
    return SceneData(
        names=names,
        images=images,
        d_cams=d_cams,
        cam_types=cam_types,
        c2w=c2w,
        masks=masks,
        intrinsics=intrinsics,
    )


def required_match(pattern: str, text: str, field_name: str) -> str:
    """Return one required regex capture from calibration text."""
    match: re.Match[str] | None = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Missing {field_name} in calibration file")
    return match.group(1)


def parse_matrix(text: str) -> Float64[ndarray, "3 3"]:
    """Parse a semicolon-delimited Middlebury 3x3 matrix."""
    matrix: Float64[ndarray, "3 3"] = np.asarray(
        [[float(value) for value in row.split()] for row in text.split(";")],
        dtype=np.float64,
    )
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 calibration matrix, got {matrix.shape}")
    return matrix


def read_middlebury_calibration(path: Path) -> MiddleburyCalibration:
    """Read camera matrices, baseline, and offset from Middlebury-v3 text."""
    text: str = path.read_text()
    left_k: Float64[ndarray, "3 3"] = parse_matrix(required_match(r"cam0=\[(.*?)\]", text, "cam0"))
    right_k: Float64[ndarray, "3 3"] = parse_matrix(required_match(r"cam1=\[(.*?)\]", text, "cam1"))
    baseline_mm: float = float(required_match(r"baseline=([\d.]+)", text, "baseline"))
    doffs_px: float = float(required_match(r"doffs=([-\d.]+)", text, "doffs"))
    return MiddleburyCalibration(
        left_k=left_k,
        right_k=right_k,
        baseline_m=baseline_mm / 1000.0,
        doffs_px=doffs_px,
    )


def read_pfm(path: Path) -> Float32[ndarray, "h w"]:
    """Read a grayscale PFM with its bottom-to-top row order corrected."""
    with path.open("rb") as file:
        header: bytes = file.readline().strip()
        if header != b"Pf":
            raise ValueError(f"Expected grayscale PFM at {path}, got {header!r}")
        dimensions: list[bytes] = file.readline().split()
        if len(dimensions) != 2:
            raise ValueError(f"Malformed PFM dimensions in {path}")
        width: int = int(dimensions[0])
        height: int = int(dimensions[1])
        scale: float = float(file.readline())
        values: Float32[ndarray, "pixels"] = np.fromfile(
            file,
            dtype="<f4" if scale < 0.0 else ">f4",
            count=width * height,
        )
    if values.size != height * width:
        raise ValueError(f"PFM payload has {values.size} values; expected {height * width}")
    disparity: Float32[ndarray, "h w"] = np.ascontiguousarray(
        np.flipud(rearrange(values, "(h w) -> h w", h=height, w=width))
    ).astype(np.float32)
    return disparity


def load_eth3d_scene(scene_dir: Path) -> tuple[SceneData, MiddleburyCalibration]:
    """Load and top-left crop ETH3D to a patch-compatible pinhole pair."""
    left_full: UInt8[ndarray, "h w 3"] = read_rgb(scene_dir / "im0.png")
    right_full: UInt8[ndarray, "h w 3"] = read_rgb(scene_dir / "im1.png")
    if left_full.shape != right_full.shape:
        raise ValueError(f"ETH3D image shapes differ: {left_full.shape} and {right_full.shape}")
    crop_height: int = left_full.shape[0] - left_full.shape[0] % 14
    crop_width: int = left_full.shape[1] - left_full.shape[1] % 14
    if crop_height < 28 or crop_width < 28:
        raise ValueError(f"ETH3D crop is below X-Lens minimum: {(crop_height, crop_width)}")
    left: UInt8[ndarray, "h w 3"] = left_full[:crop_height, :crop_width].copy()
    right: UInt8[ndarray, "h w 3"] = right_full[:crop_height, :crop_width].copy()
    calibration: MiddleburyCalibration = read_middlebury_calibration(scene_dir / "calib.txt")
    left_k: Float32[ndarray, "3 3"] = calibration.left_k.astype(np.float32)
    right_k: Float32[ndarray, "3 3"] = calibration.right_k.astype(np.float32)
    left_pose: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    right_pose: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    right_pose[0, 3] = calibration.baseline_m
    valid_mask: Bool[ndarray, "h w"] = np.ones((crop_height, crop_width), dtype=np.bool_)
    scene: SceneData = SceneData(
        names=["left", "right"],
        images=[left, right],
        d_cams=[
            pinhole_d_cam(left_k, crop_height, crop_width),
            pinhole_d_cam(right_k, crop_height, crop_width),
        ],
        cam_types=[CAM_TYPE_PINHOLE, CAM_TYPE_PINHOLE],
        c2w=np.stack([left_pose, right_pose]),
        masks=[valid_mask, valid_mask.copy()],
        intrinsics=[left_k, right_k],
    )
    print(f"ETH3D top-left crop: {left_full.shape[0]}x{left_full.shape[1]} -> {crop_height}x{crop_width}")
    print(f"ETH3D calibration: baseline={calibration.baseline_m:.7f} m doffs={calibration.doffs_px:.6f} px")
    return scene, calibration


def load_verified_model(checkpoint: Path, model_config: Path) -> XLensInference:
    """Load upstream inference and prove the released state dict is complete."""
    if not torch.cuda.is_available():
        raise RuntimeError("X-Lens Gate 1 requires CUDA, but torch.cuda.is_available() is false")
    inference: XLensInference = XLensInference(
        str(checkpoint),
        device="cuda",
        amp_dtype="bf16",
        config=str(model_config),
    )
    state: dict[str, Tensor] = load_file(str(checkpoint), device="cpu")
    normalized_state: dict[str, Tensor] = {key.replace("module.", "", 1): value for key, value in state.items()}
    load_result: tuple[list[str], list[str]] = inference.model.load_state_dict(normalized_state, strict=False)
    missing_keys: list[str] = list(load_result[0])
    unexpected_keys: list[str] = list(load_result[1])
    print(f"state_dict missing keys: {missing_keys}")
    print(f"state_dict unexpected keys: {unexpected_keys}")
    if missing_keys or unexpected_keys:
        raise RuntimeError("Released checkpoint did not load completely; see key lists above")
    return inference


def infer_scene(inference: XLensInference, scene: SceneData) -> tuple[Batch, Float32[ndarray, "views h w"], Float32[ndarray, "views h w"], float]:
    """Assemble one scene with upstream helpers and run upstream inference."""
    batch: Batch = assemble_batch(
        scene.images,
        scene.d_cams,
        scene.cam_types,
        c2w=scene.c2w,
        device="cuda",
    )
    output: dict[str, Tensor] = inference(batch)
    depth: Float32[ndarray, "views h w"] = output["depth_metric"][0].numpy().astype(np.float32, copy=False)
    confidence: Float32[ndarray, "views h w"] = output["depth_conf"][0].numpy().astype(np.float32, copy=False)
    scale: float = float(output["metric_scaling_factor"][0].item())
    print(f"metric_scaling_factor: {scale:.9f}")
    for index, name in enumerate(scene.names):
        view_depth: Float32[ndarray, "h w"] = depth[index]
        finite_depth: Float32[ndarray, "valid"] = view_depth[np.isfinite(view_depth) & (view_depth > 0.0)]
        if finite_depth.size == 0:
            raise RuntimeError(f"View {index} ({name}) has no finite positive depth")
        print(
            f"depth[{index}] {name}: min={float(finite_depth.min()):.6f} m "
            f"median={float(np.median(finite_depth)):.6f} m max={float(finite_depth.max()):.6f} m"
        )
    return batch, depth, confidence, scale


def benchmark_model(inference: XLensInference, batch: Batch, warmups: int, runs: int) -> float:
    """Measure synchronized warm CUDA model-forward latency for one scene."""
    if warmups < 0 or runs <= 0:
        raise ValueError("Benchmark warmups must be non-negative and runs must be positive")
    images: Tensor = cast(Tensor, batch["images"])
    ray_map: Tensor | None = batch.get("ray_map")
    d_cam: Tensor | None = batch.get("d_cam")
    cam_types: Tensor | None = batch.get("cam_types")
    with torch.inference_mode(), torch.autocast("cuda", dtype=inference.amp_dtype):
        for _ in range(warmups):
            inference.model(images, ray_map=ray_map, d_cam=d_cam, cam_types=cam_types)
        torch.cuda.synchronize()
        start_s: float = time.perf_counter()
        for _ in range(runs):
            inference.model(images, ray_map=ray_map, d_cam=d_cam, cam_types=cam_types)
        torch.cuda.synchronize()
    elapsed_ms: float = (time.perf_counter() - start_s) * 1000.0
    per_scene_ms: float = elapsed_ms / runs
    print(f"warm timing: {per_scene_ms:.3f} ms per 6-view scene ({warmups} warm-ups, {runs} forwards)")
    return per_scene_ms


def make_pinhole_sensor(scene: SceneData, index: int) -> CameraSensor:
    """Build one typed simplecv sensor from a scene's pinhole calibration."""
    intrinsic_matrix: Float32[ndarray, "3 3"] | None = scene.intrinsics[index]
    if intrinsic_matrix is None:
        raise ValueError(f"View {index} is not a pinhole camera")
    height: int = scene.images[index].shape[0]
    width: int = scene.images[index].shape[1]
    intrinsics: Intrinsics = Intrinsics.from_k_matrix(
        camera_conventions="RDF",
        k_matrix=intrinsic_matrix,
        height=height,
        width=width,
    )
    pose: Float32[ndarray, "4 4"] = scene.c2w[index]
    extrinsics: Extrinsics = Extrinsics(cam_R_world=pose[:3, :3], cam_t_world=pose[:3, 3])
    pinhole: PinholeParameters = PinholeParameters(
        name=scene.names[index],
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
    return CameraSensor(index=index, name=scene.names[index], kind="rgb", pinhole=pinhole)


def log_scene_rig(scene: SceneData, depth: Float32[ndarray, "views h w"], max_depth_m: float) -> list[str]:
    """Log simplecv pinholes and the justified LUT-fisheye fallback."""
    pinhole_indices: list[int] = [
        index for index, cam_type in enumerate(scene.cam_types) if cam_type == CAM_TYPE_PINHOLE
    ]
    sensors: list[CameraSensor] = [make_pinhole_sensor(scene, index) for index in pinhole_indices]
    rig: Rig = Rig(
        index=0,
        calibration=RigCalibration(cameras=sensors, reference_index=0),
        image_plane_distance=0.5,
    )
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    log_rig_static(rig)
    if len(sensors) != len(scene.images):
        rr.log(
            "world/rig_00",
            rr.AnyValues(schema_version=SCHEMA_VERSION, reference="cam_00", num_cameras=len(scene.images)),
            static=True,
        )

    depth_paths: list[str] = []
    for index, cam_type in enumerate(scene.cam_types):
        camera_path: str = f"world/rig_00/cam_{index:02d}"
        display_depth: Float32[ndarray, "h w"] = np.where(
            scene.masks[index] & np.isfinite(depth[index]) & (depth[index] > 0.0) & (depth[index] <= max_depth_m),
            depth[index],
            0.0,
        ).astype(np.float32)
        if cam_type == CAM_TYPE_PINHOLE:
            projection_path: str = f"{camera_path}/pinhole"
        else:
            pose: Float32[ndarray, "4 4"] = scene.c2w[index]
            rr.log(
                camera_path,
                rr.Transform3D(
                    translation=pose[:3, 3],
                    mat3x3=pose[:3, :3],
                    relation=rr.TransformRelation.ChildFromParent,
                ),
                static=True,
            )
            rr.log(camera_path, rr.AnyValues(name=scene.names[index], kind="rgb", projection="per-pixel LUT"), static=True)
            projection_path = f"{camera_path}/fisheye"
        rr.log(f"{projection_path}/image", rr.Image(scene.images[index]).compress(jpeg_quality=90), static=True)
        depth_path: str = f"{projection_path}/depth"
        rr.log(
            depth_path,
            rr.DepthImage(display_depth, meter=1.0, depth_range=(0.0, max_depth_m)),
            static=True,
        )
        depth_paths.append(depth_path)
    return depth_paths


def hetero_blueprint(scene: SceneData, depth_paths: list[str]) -> rrb.Blueprint:
    """Build a 3D view beside a six-camera image/depth grid."""
    exclusions: list[str] = []
    camera_pairs: list[rrb.Vertical] = []
    for index, cam_type in enumerate(scene.cam_types):
        projection_name: str = "pinhole" if cam_type == CAM_TYPE_PINHOLE else "fisheye"
        projection_path: str = f"world/rig_00/cam_{index:02d}/{projection_name}"
        camera_pairs.append(
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{projection_path}/image", name=f"{scene.names[index]} image"),
                rrb.Spatial2DView(origin=depth_paths[index], name=f"{scene.names[index]} depth (m)"),
                name=scene.names[index],
            )
        )
        if cam_type == CAM_TYPE_FISHEYE:
            exclusions.extend([f"- {projection_path}/image", f"- {projection_path}/depth"])
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                contents=["$origin/**", *exclusions],
                name="view-0 frame + fused cloud",
            ),
            rrb.Grid(*camera_pairs, grid_columns=3, name="six images + depths"),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


def run_hetero(cfg: Config) -> None:
    """Run, benchmark, fuse, and log the bundled six-camera scene."""
    scene: SceneData = load_manifest_scene(cfg.hetero_manifest)
    inference: XLensInference = load_verified_model(cfg.checkpoint, cfg.model_config)
    batch: Batch
    depth: Float32[ndarray, "views h w"]
    confidence: Float32[ndarray, "views h w"]
    scale: float
    batch, depth, confidence, scale = infer_scene(inference, scene)
    benchmark_model(inference, batch, cfg.benchmark_warmups, cfg.benchmark_runs)
    d_cam: Float32[ndarray, "views h w 3"] = cast(Tensor, batch["d_cam"])[0].detach().cpu().numpy().transpose(0, 2, 3, 1)
    rgb: UInt8[ndarray, "views h w 3"] = np.stack(scene.images)
    masks: Bool[ndarray, "views h w"] = np.stack(scene.masks)
    points: Float32[ndarray, "points 3"]
    colors: UInt8[ndarray, "points 3"] | None
    points, colors = fuse_point_cloud(
        depth,
        d_cam,
        scene.c2w,
        rgb=rgb,
        conf=confidence,
        conf_drop_pct=cfg.conf_drop_percent,
        masks=masks,
        max_depth=cfg.max_depth_m,
        fov_max_deg=cfg.fov_max_deg,
    )
    if colors is None:
        raise RuntimeError("Upstream point-cloud fusion did not return RGB colors")
    print(f"fused points: {len(points)}")
    depth_paths: list[str] = log_scene_rig(scene, depth, cfg.max_depth_m)
    rr.log("world/points", rr.Points3D(points, colors=colors), static=True)
    rr.log(
        "metrics",
        rr.TextDocument(
            f"# X-Lens heterogeneous scene\n\nScale: {scale:.9f}  \nFused points: {len(points)}",
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )
    rr.send_blueprint(hetero_blueprint(scene, depth_paths))


def compute_eth3d_metrics(
    depth: Float32[ndarray, "views h w"],
    calibration: MiddleburyCalibration,
    gt_dir: Path,
    crop_hw: tuple[int, int],
    max_disparity_px: float,
) -> tuple[Eth3dMetrics, Float32[ndarray, "h w"], Float32[ndarray, "h w"], Bool[ndarray, "h w"]]:
    """Derive disparity from left depth and score the stereo-port protocol."""
    left_depth: Float32[ndarray, "h w"] = depth[0]
    numerator_m_px: float = float(calibration.left_k[0, 0]) * calibration.baseline_m
    disparity: Float32[ndarray, "h w"] = np.full_like(left_depth, np.nan)
    predicted_valid: Bool[ndarray, "h w"] = np.isfinite(left_depth) & (left_depth > 0.0)
    disparity[predicted_valid] = numerator_m_px / left_depth[predicted_valid] + calibration.doffs_px
    gt_full: Float32[ndarray, "h w"] = read_pfm(gt_dir / "disp0GT.pfm")
    nocc_full: UInt8[ndarray, "h w"] = np.asarray(Image.open(gt_dir / "mask0nocc.png"), dtype=np.uint8)
    crop_height: int = crop_hw[0]
    crop_width: int = crop_hw[1]
    gt: Float32[ndarray, "h w"] = gt_full[:crop_height, :crop_width].copy()
    nocc: UInt8[ndarray, "h w"] = nocc_full[:crop_height, :crop_width].copy()
    valid: Bool[ndarray, "h w"] = np.isfinite(gt) & (gt < max_disparity_px) & (nocc == 255)
    error: Float32[ndarray, "h w"] = np.abs(disparity - gt)
    epe_px: float = float(error[valid].mean())
    bad1_percent: float = 100.0 * float((error[valid] > 1.0).mean())
    negative_disparity: Float32[ndarray, "h w"] = np.full_like(left_depth, np.nan)
    negative_disparity[predicted_valid] = -numerator_m_px / left_depth[predicted_valid] + calibration.doffs_px
    negative_error: Float32[ndarray, "h w"] = np.abs(negative_disparity - gt)
    negative_epe_px: float = float(negative_error[valid].mean())
    if not epe_px < negative_epe_px:
        raise RuntimeError(
            f"Positive disparity sign check failed: {epe_px:.6f} px versus negative-sign {negative_epe_px:.6f} px"
        )
    gt_depth: Float32[ndarray, "h w"] = np.full_like(gt, np.nan)
    depth_valid: Bool[ndarray, "h w"] = valid & (gt > 0.0) & predicted_valid
    gt_depth[depth_valid] = numerator_m_px / gt[depth_valid]
    relative_error: Float32[ndarray, "valid"] = np.abs(left_depth[depth_valid] - gt_depth[depth_valid]) / gt_depth[depth_valid]
    abs_rel: float = float(relative_error.mean())
    metrics: Eth3dMetrics = Eth3dMetrics(
        epe_px=epe_px,
        bad1_percent=bad1_percent,
        abs_rel=abs_rel,
        negative_sign_epe_px=negative_epe_px,
        valid_pixels=int(valid.sum()),
    )
    return metrics, disparity, gt, valid


def eth3d_blueprint(depth_paths: list[str]) -> rrb.Blueprint:
    """Build calibrated pair, depth, GT, and error views."""
    left_path: str = "world/rig_00/cam_00"
    right_path: str = "world/rig_00/cam_01"
    disparity_path: str = f"{left_path}/disparity"
    gt_path: str = f"{left_path}/disparity_gt"
    error_path: str = f"{left_path}/disparity_error"
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                contents=["$origin/**", f"- {disparity_path}", f"- {gt_path}", f"- {error_path}"],
                name="ETH3D pinhole rig + predicted depths",
            ),
            rrb.Grid(
                rrb.Spatial2DView(origin=f"{left_path}/pinhole/image", name="left image"),
                rrb.Spatial2DView(origin=f"{right_path}/pinhole/image", name="right image"),
                rrb.Spatial2DView(origin=depth_paths[0], name="left depth (m)"),
                rrb.Spatial2DView(origin=depth_paths[1], name="right depth (m)"),
                rrb.Spatial2DView(origin=disparity_path, name="derived disparity"),
                rrb.Spatial2DView(origin=gt_path, name="GT disparity"),
                rrb.Spatial2DView(origin=error_path, name="absolute error (px)"),
                grid_columns=2,
                name="images, depths, and score inputs",
            ),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


def run_eth3d(cfg: Config) -> None:
    """Run X-Lens on ETH3D, compute the baseline, and log the pair."""
    scene: SceneData
    calibration: MiddleburyCalibration
    scene, calibration = load_eth3d_scene(cfg.eth3d_scene_dir)
    inference: XLensInference = load_verified_model(cfg.checkpoint, cfg.model_config)
    batch: Batch
    depth: Float32[ndarray, "views h w"]
    confidence: Float32[ndarray, "views h w"]
    scale: float
    batch, depth, confidence, scale = infer_scene(inference, scene)
    metrics: Eth3dMetrics
    disparity: Float32[ndarray, "h w"]
    gt: Float32[ndarray, "h w"]
    valid: Bool[ndarray, "h w"]
    metrics, disparity, gt, valid = compute_eth3d_metrics(
        depth,
        calibration,
        cfg.eth3d_gt_dir,
        (scene.images[0].shape[0], scene.images[0].shape[1]),
        cfg.max_disparity_px,
    )
    print(
        f"BASELINE playground_1l X-Lens: EPE {metrics.epe_px:.6f} px "
        f"bad1 {metrics.bad1_percent:.6f}% abs-rel {metrics.abs_rel:.6f} "
        f"(non-occluded, gt < {cfg.max_disparity_px:g}, n={metrics.valid_pixels})"
    )
    print(f"disparity sign check: positive EPE={metrics.epe_px:.6f} px negative EPE={metrics.negative_sign_epe_px:.6f} px")
    depth_paths: list[str] = log_scene_rig(scene, depth, cfg.max_depth_m)
    left_path: str = "world/rig_00/cam_00"
    error: Float32[ndarray, "h w"] = np.abs(disparity - gt)
    disparity_display: Float32[ndarray, "h w"] = np.where(np.isfinite(disparity), disparity, 0.0).astype(np.float32)
    gt_display: Float32[ndarray, "h w"] = np.where(np.isfinite(gt), gt, 0.0).astype(np.float32)
    error_display: Float32[ndarray, "h w"] = np.where(valid, error, 0.0).astype(np.float32)
    rr.log(f"{left_path}/disparity", rr.DepthImage(disparity_display), static=True)
    rr.log(f"{left_path}/disparity_gt", rr.DepthImage(gt_display), static=True)
    rr.log(
        f"{left_path}/disparity_error",
        rr.DepthImage(error_display, depth_range=(0.0, 5.0)),
        static=True,
    )
    rr.log(
        "metrics",
        rr.TextDocument(
            "# X-Lens ETH3D baseline\n\n"
            f"EPE: {metrics.epe_px:.6f} px  \n"
            f"bad1: {metrics.bad1_percent:.6f}%  \n"
            f"depth abs-rel: {metrics.abs_rel:.6f}  \n"
            f"scale: {scale:.9f}",
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )
    rr.send_blueprint(eth3d_blueprint(depth_paths))


def main(cfg: Config) -> None:
    """Run the selected scene."""
    if cfg.scene == "hetero_loft":
        run_hetero(cfg)
    else:
        run_eth3d(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
