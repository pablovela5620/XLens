"""
Simplified evaluation adapters. Return only pred_depth (+ pred_metric_scale_factor
+ pred_cam_centers). This eval pipeline uses GT intrinsics/extrinsics, so ray
outputs and pose recovery are ignored.

All adapters return:
    {
        "pred_depth":                (S, H, W)  float32, raw model depth (may be normalized)
        "pred_metric_scale_factor":  Optional[float] - None means no scale head
        "pred_valid_mask":           Optional[np.ndarray (S, H, W) bool] - None if absent
        "pred_cam_centers_c2w":      Optional[np.ndarray (S, 3)] - predicted camera
                                     centers (c2w translation, any frame/unit). None if
                                     no pose. Used for pose_scale alignment.
    }
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# XLensSimpleAdapter - the X-Lens multiview model

class XLensONNXAdapter:
    """ONNX inference adapter for X-Lens.

    Consumes model.onnx from tools/export_onnx.py (5 outputs):
        depth (B,S,H,W) / depth_metric (B,S,H,W) / metric_scaling_factor (B,) /
        pred_translation (B,S,3) / pred_quaternion (B,S,4)

    Returns the same dict as XLensSimpleAdapter, so run_eval / save_pointclouds work
    unchanged. ONNX is image-only (no intrinsics/extrinsics).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.session = None
        self.input_name = "images"

    def load(self, onnx_path: str) -> None:
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"[XLensONNX] onnx not found: {onnx_path}")
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime not installed. pip install onnxruntime-gpu (CUDA) or "
                "onnxruntime (CPU)"
            ) from e

        if self.device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Record the provider actually used (ort silently falls back to CPU).
        used = self.session.get_providers()
        inputs = [(i.name, i.shape) for i in self.session.get_inputs()]
        outputs = [(o.name, o.shape) for o in self.session.get_outputs()]
        logger.info(f"[XLensONNX] onnx loaded: {onnx_path}")
        logger.info(f"[XLensONNX] providers = {used}")
        logger.info(f"[XLensONNX] inputs = {inputs}")
        logger.info(f"[XLensONNX] outputs = {outputs}")
        # Cache the input name (dynamically; nominally 'images').
        if inputs:
            self.input_name = inputs[0][0]

    def infer(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        # ONNX takes numpy: convert (B, S, 3, H, W) torch tensor to float32.
        images_np = batch["images"].detach().cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: images_np})
        # Order matches export_onnx.py output_names:
        #   depth / depth_metric / metric_scaling_factor / pred_translation / pred_quaternion
        depth, _depth_metric, sf, pred_trans, _pred_quat = outputs
        B = depth.shape[0]
        results: List[Dict[str, Any]] = []
        for b in range(B):
            results.append({
                "pred_depth": depth[b].astype(np.float32),
                "pred_metric_scale_factor": float(sf[b]),
                "pred_valid_mask": None,
                "pred_cam_centers_c2w": pred_trans[b].astype(np.float32),
            })
        return results


class XLensSimpleAdapter:
    def __init__(self, device: torch.device, amp_enabled: bool = True,
                 amp_dtype: str = "bf16"):
        self.device = device
        # No AMP on CPU (avoids autocast('cuda') warning).
        self.amp_enabled = amp_enabled and (device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.model = None

    def load(self, checkpoint_path: str, config: Dict[str, Any]) -> None:
        from xlens.models import XLensNet

        # Prefer the config saved in the ckpt (matches the weights); YAML config is
        # only a fallback.
        ckpt = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            if str(checkpoint_path).endswith(".safetensors"):
                # released weights-only checkpoint: arch comes from the yaml config
                # (configs/xlens_vits.yaml), passed in via --da3_config.
                from safetensors.torch import load_file
                ckpt = {"model": load_file(str(checkpoint_path))}
            else:
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
            if isinstance(ckpt_cfg, dict):
                logger.info(f"[XLensSimple] using config saved in ckpt (overrides yaml config)")
                config = ckpt_cfg

        # Pass through every parameter XLensNet actually accepts.
        import inspect
        sig = inspect.signature(XLensNet.__init__)
        accepted = {k for k in sig.parameters if k != "self"}
        kwargs = {k: v for k, v in config.items() if k in accepted}
        # Alias: yaml/ckpt uses 'backbone', constructor uses 'backbone_name'.
        if "backbone" in config and "backbone_name" not in kwargs:
            kwargs["backbone_name"] = config["backbone"]
        # Force unfrozen for eval (immediately .eval()'d).
        kwargs["freeze_backbone"] = False
        # No need to load DINOv2 pretrained weights; ckpt carries the full backbone.
        logger.info(f"[XLensSimple] building XLensNet with: "
                    f"{ {k: kwargs[k] for k in sorted(kwargs) if k != 'checkpoint_dir'} }")

        model = XLensNet(**kwargs).to(self.device)

        if ckpt is not None:
            state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            info = model.load_state_dict(cleaned, strict=False)
            logger.info(f"[XLensSimple] ckpt loaded: {checkpoint_path}")
            if info.missing_keys:
                logger.warning(f"[XLensSimple] missing {len(info.missing_keys)} keys")
            if info.unexpected_keys:
                logger.warning(f"[XLensSimple] unexpected {len(info.unexpected_keys)} keys")
        else:
            logger.warning(f"[XLensSimple] ckpt not found: {checkpoint_path}, random weights")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[XLensSimple] params: {total_params / 1e6:.1f}M")
        model.eval()
        self.model = model

    @torch.no_grad()
    def infer(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        images = batch["images"].to(self.device)
        # Training feeds ray_map most of the time; feed it in eval to match the
        # training distribution. OmniOCC is pinhole, so derive d_cam + ray_map from
        # (intrinsics, extrinsics_world).
        ray_map = d_cam = None
        if "intrinsics" in batch and "extrinsics_world" in batch:
            from evaluation._support.geom import RayMapHelper
            H, W = images.shape[-2:]
            d_cam, ray_map, _pose_scale = RayMapHelper._compute_pinhole_ray_map(
                batch["intrinsics"].to(self.device),
                batch["extrinsics_world"].to(self.device),
                H, W, device=self.device,
                view_mask=batch.get("view_mask"),  # None -> all True
            )

        # cam_types: pinhole batches default to 1 in training; pass it in eval too,
        # otherwise cam_type_embed=None diverges (for n_cam_types>0 ckpts).
        cam_types = batch.get("cam_types")
        if cam_types is not None:
            cam_types = cam_types.to(self.device)

        with torch.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
            output = self.model(images, ray_map=ray_map, d_cam=d_cam, cam_types=cam_types)

        depth = output["depth"].cpu().float().numpy()                        # (B, S, H, W)
        sf = output["metric_scaling_factor"].cpu().float().numpy()           # (B,)
        # cam_head c2w translation (B, S, 3), in normalized space (unit matches depth).
        cam_centers = None
        if "pred_translation" in output:
            cam_centers = output["pred_translation"].cpu().float().numpy()   # (B, S, 3)
        # Predicted non-ambiguous mask (sigmoid(mask_logits)>0.5). Used only for
        # dumped-cloud sky filtering; metrics use the GT mask.
        pred_mask = None
        if "mask_logits" in output:
            pred_mask = (torch.sigmoid(output["mask_logits"].float()) > 0.5).cpu().numpy()  # (B,S,H,W)
        # Depth confidence (B,S,H,W): used for dumped-cloud clipping (metrics ignore it).
        pred_conf = None
        if "depth_conf" in output:
            pred_conf = output["depth_conf"].cpu().float().numpy()
        B = depth.shape[0]
        results: List[Dict[str, Any]] = []
        for b in range(B):
            results.append({
                "pred_depth": depth[b].astype(np.float32),
                "pred_metric_scale_factor": float(sf[b]),
                "pred_valid_mask": (pred_mask[b] if pred_mask is not None else None),
                "pred_conf": (pred_conf[b] if pred_conf is not None else None),
                "pred_cam_centers_c2w": (cam_centers[b].astype(np.float32)
                                         if cam_centers is not None else None),
            })
        return results


def build_simple_adapter(
    model_type: str,
    device: torch.device,
    *,
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    amp_enabled: bool = True,
) -> Any:
    mt = model_type.lower()
    if mt == "xlens":
        # Dispatch by extension: .onnx -> ONNX runtime, else (.pth / .pt / .safetensors / dir) -> PyTorch
        if checkpoint_path and str(checkpoint_path).lower().endswith(".onnx"):
            adapter = XLensONNXAdapter(device)
            adapter.load(checkpoint_path)
        else:
            adapter = XLensSimpleAdapter(device, amp_enabled=amp_enabled, amp_dtype="bf16")
            adapter.load(checkpoint_path, config or {})
    else:
        raise ValueError(f"unknown model_type={model_type}")
    return adapter
