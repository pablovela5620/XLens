<div align="center">

# X-Lens：Real-Time Metric Depth Estimation with Heterogeneous Cameras

<!-- TODO(authors): add homepage links to names if desired, e.g. [**Heng Zhou**](https://...) -->
**Heng Zhou**<sup>1&ast;</sup> · [**Shuhong Liu**](https://shuhongll.github.io/)<sup>1,2&ast;</sup> · **Yonghao He**<sup>1</sup> · **Bohao Zhang**<sup>1</sup> · **Fa Fu**<sup>1</sup> · **Chenhui Hou**<sup>1</sup> · **Xianbao Hou**<sup>1,3</sup> · **Lijun Han**<sup>1</sup> · **Wei Sui**<sup>1&dagger;&#9993;</sup>

<sup>1</sup>D-Robotics&emsp;<sup>2</sup>The University of Tokyo&emsp;<sup>3</sup>Soochow University

<sup>&ast;</sup>Equal Contribution&emsp;<sup>&dagger;</sup>Project Lead&emsp;<sup>&#9993;</sup>Corresponding Author

###
<!-- TODO(links): fill in once the paper / project page / model weights are public -->
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-1f8ceb.svg)](https://your-project-page.github.io)
[![Weights](https://img.shields.io/badge/🤗%20HuggingFace-Weights-ffce44.svg)](https://huggingface.co/henryzhou998/X-Lens)
[![License](https://img.shields.io/badge/License-Apache%202.0-4c9a2a.svg)](LICENSE)

<!-- TODO(teaser): drop a demo gif / point-cloud render in assets/ and point here -->
<img src="assets/teaser.gif" width="80%" alt="X-Lens teaser"/>

</div>

> **X-Lens** recovers dense **metric depth** and a **fused 3D point cloud** from an
> arbitrary set of views (`S ≥ 2`) captured by an arbitrary mix of cameras — perspective
> **pinhole**, wide-FoV **fisheye**, or a **heterogeneous** rig that combines both in a single
> pass. Built on an alternating (frame ⇄ global) cross-view attention backbone, it
> reasons jointly across views and camera models, so a fisheye surround-view scene and a set of
> pinhole cameras are handled by **one model, one forward**.

💎 **One model, any camera.** Fisheye and pinhole views are consumed side-by-side through a
per-pixel ray representation + camera-type conditioning — no rectification, no per-camera retraining.

✨ **Metric & geometry-aware.** Predicts scale factor, depth, and a confidence map, unprojected to a single consistent point cloud.

---

## 📰 News

<!-- TODO(news): fill dates/links as you release. Newest first. -->
- **[XX-XX-2026]** 📦 **OmniScene dataset released** on [Hugging Face](https://huggingface.co/datasets/henryzhou998/OmniScene/tree/main)
- **[XX-XX-2026]** 🎉 Initial release: inference code + XLens-S checkpoints for all three camera modes.
- **[XX-XX-2026]** 📄 Paper released on arXiv.

---

## ✨ Highlights

- **Heterogeneous multi-view.** Mix fisheye and pinhole views in the same scene; camera type is
  injected per view via calibration tokens + a geometric distortion bias.
- **Native fisheye.** Wide-FoV cameras are handled directly through a per-pixel
  unit-ray LUT and ray-angle rotary position encoding — no undistortion to a lossy pinhole crop.
- **Metric depth + point cloud.** A learned scale head produces metric depth; every view unprojects
  into one fused, view-0-canonical world point cloud.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/zhouhengamerica/XLens.git xlens
cd xlens
conda create -n xlens python=3.10 -y
conda activate xlens
pip install -r requirements.txt
pip install -e .          # exposes the `xlens` package
```

### 2. Get the weights

Download a checkpoint into `checkpoints/` (see the [Model Zoo](#-model-zoo)):

```bash
mkdir -p checkpoints
hf download henryzhou998/X-Lens --local-dir checkpoints
```

> The DINOv2 backbone weights are bundled inside the checkpoint — no separate download needed for inference.

### 3. Run

A scene is a small JSON manifest listing, per view, an image and its calibration
(pinhole `K`, per-pixel `dcam`, or fisheye `lut`) plus an optional `c2w` pose. Three
ready-to-run demo scenes live in [`examples/`](examples/).

```bash
python inference.py \
    --ckpt checkpoints/xlens.pth \
    --manifest examples/hetero_loft/scene.json \
    --out out/hetero --max-depth 25 --conf-drop-pct 10
# -> out/hetero/depth_metric.npy   (6, 504, 798) metric depth
# -> out/hetero/depth_preview.png  per-view depth mosaic
# -> out/hetero/points.ply         fused world-frame point cloud
```

---

## 🎥 The three inference modes

All three modes use the **same checkpoint**. The mode is selected purely by what each view
carries in the manifest — the CLI assigns the camera type automatically. Each demo below is a
**self-contained, ready-to-run** bundle (RGB images + intrinsics/extrinsics + masks).

<table>
<tr><th>Mode</th><th>Per-view intrinsics</th><th>Per-view extrinsics</th><th>Bundled demo</th></tr>
<tr>
<td><b>Pure pinhole</b> (6-cam)</td>
<td>3x3 intrinsics <code>K</code></td>
<td rowspan="3">4x4 camera world pose <code>c2w</code><br>(relative, view-0 canonical)</td>
<td><a href="examples/pinhole_4f_ba1/"><code>pinhole_4f_ba1/</code></a> - OmniOCC real capture</td>
</tr>
<tr>
<td><b>Pure fisheye</b> (4-cam)</td>
<td>per-pixel unit-ray <code>dcam</code></td>
<td><a href="examples/fisheye_fuelbar/"><code>fisheye_fuelbar/</code></a> - FS3 FuelBar</td>
</tr>
<tr>
<td><b>Heterogeneous</b> (6-cam)</td>
<td>mix of <code>K</code> and <code>dcam</code></td>
<td><a href="examples/hetero_loft/"><code>hetero_loft/</code></a> - 4 fisheye + 2 pinhole</td>
</tr>
</table>

```bash
# Pure pinhole — 6-camera OmniOCC rig
python inference.py --ckpt checkpoints/xlens.pth --manifest examples/pinhole_4f_ba1/scene.json --out out/pinhole --max-depth 15

# Pure fisheye — 4 fisheye cameras   (fisheye cleanup applied automatically)
python inference.py --ckpt checkpoints/xlens.pth --manifest examples/fisheye_fuelbar/scene.json --out out/fisheye --max-depth 25 --fov-max 85 --conf-drop-pct 8

# Heterogeneous — 4 fisheye + 2 pinhole in one scene   (fisheye cleanup applied automatically)
python inference.py --ckpt checkpoints/xlens.pth --manifest examples/hetero_loft/scene.json --out out/hetero --max-depth 25 --fov-max 85 --conf-drop-pct 8
```

**Manifest format.** Each view carries an `image`, a `c2w` pose, an optional `mask`, and its
calibration — one of: `K` (3×3 pinhole intrinsics), `dcam` (a `.npy` of per-pixel unit rays, for
any camera model), or `lut` (a fisheye ray LUT, see below). `cam_type` (0 fisheye / 1 pinhole) is
inferred from the calibration or set explicitly.

### 💡 Recommended point-cloud cleanup for fisheye

> **Near a fisheye lens's edge, distortion is extreme and per-pixel depth becomes unreliable — those
> peripheral rays unproject into stray points that shoot *outside* the scene.** We strongly recommend
> cleaning them up. When a scene contains **any fisheye view**, `inference.py` does this **automatically**:
> it trims rays more than **85° off the optical axis** (`--fov-max 85`) and drops the **lowest 8% of points
> by confidence** (`--conf-drop-pct 8`). In our tests this collapses the outlier tail (e.g. max point
> range 25 m → ~14 m on pure fisheye, and 25 m → ~5 m on a mixed rig) while keeping ~90% of points.

Override or disable:
```bash
--fov-max 80 --conf-drop-pct 15     # stronger cleanup
--no-fisheye-clean                  # turn the automatic cleanup off
```
Other point-cloud filters (any mode): `--max-depth` clips by euclidean range (m); `--conf-thresh`
sets an absolute confidence cutoff.

<details>
<summary><b>Python API</b> (build inputs yourself)</summary>

```python
import numpy as np
from xlens.inference import XLensInference
from xlens.inference.preprocess import pinhole_d_cam, load_fisheye_lut, assemble_batch
from xlens.inference.geometry import fuse_point_cloud, save_ply

model = XLensInference("checkpoints/xlens.pth", device="cuda")

images    = [img0, img1]                 # list of (H, W, 3) uint8 RGB
d_cams    = [pinhole_d_cam(K0, H, W),    # pinhole view  -> unit rays from K
             load_fisheye_lut("cam1_lut.npy", H, W)]   # fisheye view -> unit rays from LUT
cam_types = [1, 0]                       # 1 = pinhole, 0 = fisheye
c2w       = np.stack([c2w0, c2w1])       # (S, 4, 4), optional (enables point-cloud fusion)

batch = assemble_batch(images, d_cams, cam_types, c2w=c2w, device="cuda")
out   = model(batch)

depth = out["depth_metric"][0].numpy()   # (S, H, W) metric depth
pts, col = fuse_point_cloud(depth, batch["d_cam"][0].cpu().numpy().transpose(0,2,3,1),
                            c2w, rgb=np.stack(images), conf=out["depth_conf"][0].numpy())
save_ply("points.ply", pts, col)
```

</details>

<details>
<summary><b>Fisheye calibration → LUT</b></summary>

A fisheye view is fed to the model through a **look-up table (LUT)**: a `(H, W, 3)` array giving,
for every pixel, the OpenCV camera-frame **unit** viewing direction (X right, Y down, Z forward).
Precompute one LUT per physical camera from your calibration (OpenCV `cv2.fisheye`, Kannala–Brandt,
Mei, …) and save it as `.npy`:

```python
import cv2, numpy as np
H, W = 504, 798
u, v = np.meshgrid(np.arange(W), np.arange(H))
pts = np.stack([u, v], -1).astype(np.float32).reshape(-1, 1, 2)
undist = cv2.fisheye.undistortPoints(pts, K, D).reshape(H, W, 2)   # normalized image plane
d = np.concatenate([undist, np.ones((H, W, 1), np.float32)], -1)
d /= np.linalg.norm(d, axis=-1, keepdims=True)
np.save("cam0_lut.npy", d.astype(np.float32))
```

The LUT resolution must match the images you feed the model.
</details>

<details>
<summary><b>ONNX export</b> (deploy on a fixed camera rig, full geometry baked)</summary>

Once your rig is final (view count, resolution, and the fixed per-camera calibration), export to ONNX
with **all geometry baked in** and run it with ONNX Runtime / TensorRT. `S`, `H`, `W` and the whole
calibration are frozen at export time — re-export when the rig changes.

`xlens/tools/export_onnx_geo.py` takes the same `scene.json` manifest as `inference.py` and bakes the
network's entire geometric input — `ray_map` (as a precomputed `ray_feat` constant), per-pixel `d_cam`
(ray-angle RoPE), `cam_types`, and the calibration-token attention masks — into the graph as constants. This
covers **pinhole, fisheye, and heterogeneous** rigs identically. The **only runtime input is `images`**;
camera extrinsics are used only for optional point-cloud fusion, outside ONNX.

```bash
python -m xlens.tools.export_onnx_geo \
    --ckpt checkpoints/xlens_vits.safetensors --config configs/xlens_vits.yaml \
    --manifest examples/hetero_loft/scene.json \
    --out onnx/xlens_hetero_loft.onnx --device cuda
# bakes geometry -> single-file ONNX, then runs a PyTorch-vs-ONNXRuntime check on the real images
```

Run the exported model (mirrors `inference.py`, but only feeds `images` to the graph):

```bash
python -m xlens.tools.infer_onnx \
    --onnx onnx/xlens_hetero_loft.onnx \
    --manifest examples/hetero_loft/scene.json \
    --out out/hetero_onnx --provider cuda --max-depth 25
# writes depth_metric.npy + depth_preview.png + points.ply. Use --provider cpu on a
# small GPU: 6-view global attention (L≈12k) needs a large-memory GPU under ONNX Runtime.
```

Or drive it from your own code:

```python
import numpy as np, onnxruntime as ort
from xlens.inference.preprocess import assemble_batch
from xlens.inference.geometry import fuse_point_cloud, save_ply

sess  = ort.InferenceSession("onnx/xlens_hetero_loft.onnx", providers=["CUDAExecutionProvider"])
# only `images` is needed at runtime (ImageNet-normalized, fixed view order + resolution)
batch = assemble_batch(images, d_cams, cam_types, c2w=c2w)
depth, depth_metric, depth_conf, scale, mask = sess.run(
    None, {"images": batch["images"].cpu().numpy().astype(np.float32)})
# depth maps are the model output; fusing to a world point cloud still needs d_cam + c2w (outside ONNX)
pts, col = fuse_point_cloud(depth_metric[0], batch["d_cam"][0].cpu().numpy().transpose(0, 2, 3, 1), c2w)
save_ply("points.ply", pts, col)
```

Outputs: `depth`, `depth_metric`, `depth_conf`, `metric_scaling_factor`, `mask`. Options: `--opset 17`
· `--no_fp16_mask` keeps the baked attention masks in fp32 (default is fp16 — bit-identical after
softmax; masks are also head-squeezed, keeping the file well under the 2 GiB ONNX limit). ImageNet
normalization and depth→point-cloud unprojection stay outside the graph. Camera **extrinsics are never
part of the network** — they are used only for point-cloud fusion.
</details>

---


## 🗂️ Model Zoo

<!-- TODO(zoo): fill parameter counts, training data, download links, metrics -->
| Model | Backbone | Params | Pinhole | Fisheye | Heterogeneous | Download |
|-------|----------|:------:|:-------:|:-------:|:-------------:|----------|
| `xlens_vits` | ViT-S/14 | 0.04B | ✅ | ✅ | ✅ | [🤗 HF](https://huggingface.co/henryzhou998/X-Lens) |
| `xlens_vitl` | ViT-L/14 | 0.54B | ✅ | ✅ | ✅ | 🔜 **upcoming** |

> The checkpoint is trained through a pinhole → fisheye → mixed curriculum, so a **single** model
> serves all three modes. Mode is chosen at inference time by the inputs, not the weights.
>
> **Two backbones, one trade-off.** `xlens_vits` is the fast, real-time-friendly default.
> `xlens_vitl` is an **upcoming** higher-accuracy variant for users who prioritize
> reconstruction quality over latency — same interface and checkpoint format, just a larger backbone.

---

## 📁 Repository layout

```
xlens/
├── inference.py                 # CLI: manifest -> depth + point cloud
├── configs/xlens_vits.yaml      # architecture reference (loader also reads arch from the ckpt)
├── examples/                    # scene manifests for the three modes
├── xlens/
│   ├── models/                  # the model (self-contained: backbone + heads)
│   ├── inference/               # preprocess (d_cam / ray_map / cam_types), geometry, pipeline
│   └── tools/                    # export_onnx_geo.py (bake geometry -> ONNX) + infer_onnx.py (run it)
├── evaluation/                  # depth eval: eval_pinhole / eval_fisheye / eval_heterogeneous
└── examples/                    # ready-to-run demo scenes (pinhole / fisheye / heterogeneous)
```

---

## 📦 OmniScene Dataset

We open-source **OmniScene**, the photorealistic synthetic dataset (rendered in NVIDIA Isaac Sim)
used to train X-Lens. Each scene is captured by a **6-camera omnidirectional rig** with metric
ground-truth depth, so a single sample already contains the pinhole / fisheye / heterogeneous mix
the model is designed for.

🤗 **Download:** [huggingface.co/datasets/henryzhou998/OmniScene](https://huggingface.co/datasets/henryzhou998/OmniScene/tree/main)

### The camera rig (6 views, all 1920×1200)

| View | Model | Notes |
|------|-------|-------|
| `CAM_A` `CAM_B` `CAM_C` `CAM_D` | **fisheye** — omnidirectional (Mei / unified-sphere: `xi` + `radtan` distortion) | 4 side fisheye cameras |
| `CAM_Front` `CAM_Back` | **pinhole** — perspective (3×3 `K`) | forward / backward views |

This is exactly the **4 fisheye + 2 pinhole** heterogeneous rig X-Lens is built for.


### Directory structure

On the Hub, **each sequence is packaged as a single `.tar`** (691 `train` + 62 `valid` + 23 `test`),
which keeps the repo well under the Hub's file-count limit:

```
OmniScene/                          # Hugging Face repo
├── README.md
├── DEPTH_FORMAT.txt                # depth decoding note (see below)
├── train/<scene>.tar               # e.g. taobao02_AIUE_V01_001_100_10_60.tar
├── valid/<scene>.tar
├── test/<scene>.tar
└── texture/                        # shared fisheye ray LUTs (CAM_*_rayEnterDirection.exr) — used by --lut_dir
```

Each tar extracts to the per-scene layout:

```
<scene>/
├── rgb/     CAM_*/<frame>.jpg     # 8-bit RGB
├── depth/   CAM_*/<frame>.png     # 16-bit metric depth (see decoding)
├── mask/     CAM_*_mask.png       # static per-camera valid-lens mask
├── sky_mask/ _meta.json (+ per-frame masks when sky is present)
└── common/  <frame>.npy           # per-frame camera parameters (all 6 views)
```

`common/<frame>.npy` is a pickled dict keyed by camera name; each entry provides:
- `intrinsics` — 3×3 pinhole `K` (the calibration used for the `CAM_Front/Back` pinhole views)
- `extrinsics_world` — 4×4 camera-to-world (OpenCV convention: X right, Y down, Z forward)
- `intrinsics_full` — the full source calibration: the `omni` fisheye model (`xi`, `fx/fy/cx/cy`,
  `radtan` coeffs) for `CAM_A/B/C/D`, plus the camera-to-camera extrinsics to the other views.
  (`CAM_Front/Back` also carry their Isaac Sim source metadata here, but are consumed as pinhole via `K`.)

### Decoding depth

Depth is stored as **16-bit PNG**; recover metres by dividing by `256`. A `0` value means
**invalid / sky** (pair it with `sky_mask`). Because the raw 16-bit values are small, the PNGs look
almost black in a normal image viewer — that is expected, load them numerically:

```python
import cv2, numpy as np

png     = cv2.imread("depth/CAM_A/0000_0000.png", cv2.IMREAD_UNCHANGED)  # MUST be UNCHANGED (keep 16-bit)
depth_m = png.astype(np.float32) / 256.0   # metres
valid   = png > 0                          # 0 = invalid / sky
```

(`depth_meters = png_value / 256`, `0 = invalid/sky`, max range 256 m — also recorded in `DEPTH_FORMAT.txt`.)

### Getting the data

```python
from huggingface_hub import hf_hub_download, snapshot_download
import tarfile

# one scene
p = hf_hub_download("henryzhou998/OmniScene", "test/<scene>.tar", repo_type="dataset")
tarfile.open(p).extractall("omniscene/")     # -> omniscene/<scene>/{rgb,depth,mask,...}

# a whole split (downloads the tars; extract each as above)
snapshot_download("henryzhou998/OmniScene", repo_type="dataset",
                  allow_patterns="test/*", local_dir="omniscene_tars")
```

Or grab the tars with the CLI, then extract:

```bash
hf download henryzhou998/OmniScene --repo-type dataset --include "test/*" --local-dir ./omniscene_tars
for f in ./omniscene_tars/test/*.tar; do tar -xf "$f" -C ./omniscene/; done
```

### License

OmniScene is released for **non-commercial research use** under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
<!-- TODO: confirm dataset license -->

---

## 📊 Evaluation

Reproduce the X-Lens depth metrics across all three camera regimes. Each track loads the released
checkpoint with the shipped arch config (`configs/xlens_vits.yaml`) and runs **self-contained** —
no training code, no external model repos.

> **Baselines are not included.** The open-source evaluation ships only the X-Lens model. The
> baseline comparisons in the paper (DAC · UniDAC · MapAnything · VGGT · …) are third-party and are
> not redistributed here.

| Track | Data | Entry point |
|-------|------|-------------|
| **Pinhole** | OmniOCC (6-camera) + WAI (ETH3D / ScanNet++)[^wai] | `evaluation.eval_pinhole.run_eval` |
| **Fisheye** | OmniScene 4-camera fisheye · KITTI-360 (monocular fisheye) | `evaluation.eval_fisheye.{eval_fisheye, eval_kitti360}` |
| **Heterogeneous** | OmniScene 6-camera (4 fisheye + 2 pinhole) | `evaluation.eval_heterogeneous.eval_calib_compare` |

Each track reads its paths/options from a small YAML — edit the `/path/to/...` placeholders first.

```bash
# Pinhole — OmniOCC + WAI depth
python -m evaluation.eval_pinhole.run_eval \
    --config evaluation/eval_pinhole/eval_config.yaml

# Fisheye — 4-camera omni
python -m evaluation.eval_fisheye.eval_fisheye \
    --config evaluation/eval_fisheye/eval_fisheye_config.yaml

# Fisheye — KITTI-360 (monocular fisheye, motion-stereo)
python -m evaluation.eval_fisheye.eval_kitti360 \
    --stage3_ckpt checkpoints/model.safetensors \
    --config configs/xlens_vits.yaml \
    --data_root /path/to/KITTI-360

# Heterogeneous — 6-camera (4 fisheye + 2 pinhole)
python -m evaluation.eval_heterogeneous.eval_calib_compare \
    --stage3_ckpt checkpoints/model.safetensors \
    --config configs/xlens_vits.yaml \
    --data_root /path/to/foundationstage3/test \
    --lut_dir   /path/to/OmniScene/texture      # fisheye ray LUTs, shipped with the OmniScene dataset
```

> For the released **`.safetensors`** (weights only) always pass `--config configs/xlens_vits.yaml`.
> A `.pth` checkpoint embeds its own arch config, which then takes priority.

**Reading the metrics.** Fisheye and pinhole views are reported separately and pooled (`overall`):
- **`scale_absrel`** — unaligned metric-scale error `|mean(pred) − mean(gt)| / mean(gt)` (absolute metric accuracy).
- **`depth_absrel` · `rmse` · `δ1@1.25`** — per-view median-aligned, scale-invariant relative-depth quality.

[^wai]: WAI data layout and loading follow [MapAnything](https://github.com/facebookresearch/map-anything).

---

## ❓ FAQ

<!-- TODO(faq): expand as questions come in -->

**Can I run a single pinhole image?** The model is multi-view; one view should duplicate the view.

---

## 🙏 Acknowledgements

Built on [DINOv2](https://github.com/facebookresearch/dinov2). WAI dataset handling for the pinhole
evaluation follows [MapAnything](https://github.com/facebookresearch/map-anything). <!-- TODO: add any other credits -->

---

## 📜 Citation

<!-- TODO(citation): fill in on release -->
```bibtex
@article{xlens,
  title   = {X-Lens: Real-Time Metric Depth Estimation with Heterogeneous Cameras},
  author  = {TODO},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Released under the [Apache-2.0](LICENSE) license. <!-- TODO: confirm license choice -->
