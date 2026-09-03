# NOTES — fork + pixify log

## Decisions

- `main` is frozen at upstream commit `e6bdf2f66b26a9e4ef1663feaaf7e4618e1e8f7d` (2026-07-16), the last Apache-2.0 snapshot. Upstream commit `7f63191` changed the repository to CC-BY-NC 4.0, so the fork's `main` was reset and force-pushed before `pixi` was created.
- The code at the frozen commit is Apache-2.0. The released weights in gated Hugging Face repository `henryzhou998/X-Lens` are CC-BY-NC 4.0. They are never mirrored; the guarded task downloads them from the authors with the user's Hugging Face token.
- The checkpoint is the 154,981,136-byte `model.safetensors` at revision `1d0c96353b69464addad12389fadbb816e3978ae`. Its SHA-256 is `266a0340b53e5cb996cc613a1b0c5966b5bcaeee1ec7c4431e4fc6e7d1e58a0c`. A bare safetensors has no architecture metadata, so every loader call supplies `configs/xlens_vits.yaml`; otherwise upstream silently defaults to ViT-L.
- The demo reloads the state dict with `strict=False`, prints both key lists, and fails unless missing and unexpected keys are empty. The construction warning for absent `checkpoints/dinov2_vits14_reg4_pretrain.pth` is expected because the X-Lens state dict embeds the DINOv2 backbone.
- The Phase 2 inference subset is `xlens/inference/{pipeline,preprocess,geometry}.py`, `xlens/models/{net,dpt_head,ray_map_encoder}.py`, `xlens/models/utils/head_utils.py`, and `xlens/models/dinov2/**`. Evaluation and ONNX tools are out of scope.
- Runtime imports need PyTorch, NumPy, einops, safetensors, PyYAML, and Pillow. There are no custom kernels or required xformers/flash-attn paths. Matplotlib and imageio are omitted because this demo neither writes previews nor reads EXR LUTs. `py-opencv` is omitted because ETH3D is pinhole-only and the bundled fisheye scene already contains ray LUTs.
- Pins copy the template/monorepo contract: Python 3.12, CUDA 13.0.*, conda `pytorch-gpu >=2.12,<2.13`, Rerun SDK 0.36.2 with DataFusion, einops 0.8.x, tyro 0.9.x, and `typing-extensions>=4.1,<4.16`.
- Upstream evaluation requires large ETH3D multi-view, ScanNet++, and KITTI-360 downloads. The hosted ETH3D `playground_1l` two-view sample is therefore a baseline using the stereo-port protocol (non-occluded, GT disparity below 192 px), not a reproduction of an upstream X-Lens number.
- Upstream input is RGB `/255` plus ImageNet normalization. `pinhole_d_cam` uses `+0.5` pixel centres. `assemble_batch` consumes images, camera rays/types, and view-0-canonical `c2w`. CUDA inference uses BF16 autocast. Outputs are per-view camera-frame metric z-depth, uncalibrated confidence above 1, mask, and one scene scale. At least two views and dimensions divisible by 14 (minimum 28) are required; global attention scales quadratically in total tokens. The bundled scenes use 504×798.
- `fuse_point_cloud`, FoV 85° filtering, and the 8% confidence drop are post-processing only. Heterogeneous fusion also caps euclidean range at 25 m.
- ETH3D images and ground truth are cropped from the top-left to multiples of 14 rather than resized. This preserves the calibration matrix. The right camera centre is at `(+baseline_m, 0, 0)` in the left-camera frame.

## Commands (fork, download, run)

- Fork: `GH_TOKEN=$(gh auth token --user pablovela5620) gh repo fork zhouhengamerica/XLens --clone=false`
- Clone: `git clone git@github.com:pablovela5620/XLens.git ~/0Dev/forks/XLens`
- Freeze Apache branch: `git reset --hard e6bdf2f66b26a9e4ef1663feaaf7e4618e1e8f7d && git push --force origin main`
- Checkpoint: `hf download henryzhou998/X-Lens model.safetensors --revision 1d0c96353b69464addad12389fadbb816e3978ae --local-dir weights/x-lens`
- Sample: `hf download pablovela5620/monoprior-example --include 'stereo/eth3d/*' --repo-type dataset --local-dir data/hf`
- Heterogeneous demo: `pixi run demo --rr-config.headless --rr-config.save /tmp/grill/jobs/xlens-fork/rrd/hetero_loft.rrd`
- ETH3D demo: `pixi run demo-eth3d --rr-config.headless --rr-config.save /tmp/grill/jobs/xlens-fork/rrd/eth3d.rrd`
- Upstream CLI: `pixi run demo-upstream`

## Upstream edits

- No existing upstream Python file is edited. `git diff main -- '*.py'` contains only new `demo_rerun.py`.
- Hand-roll: `simplecv` has no camera type for a per-pixel ray LUT. The four fisheye cameras therefore use manual `rr.Transform3D` nodes and 2D image/depth entities. The two pinhole cameras use `Rig`, `CameraSensor`, `PinholeParameters`, and `log_rig_static` unchanged.
- The demo contains only the manifest parsing and grayscale PFM reader needed at its CLI boundary. Model preprocessing, camera-ray construction, batching, inference, and point-cloud fusion use upstream X-Lens helpers.

## Gotchas found

- A bare X-Lens safetensors silently selects incompatible ViT-L defaults unless `configs/xlens_vits.yaml` is supplied.
- The loader deliberately uses `strict=False`; an explicit empty missing/unexpected-key assertion is needed to turn a partial load into a Gate 1 failure.
- `simplecv` does not declare all runtime packages supplied by the monorepo common feature, so `av`, `pyarrow`, `einops`, and `typing-extensions<4.16` are explicit.
- Although X-Lens itself does not need OpenCV or Matplotlib for these demos, the current `simplecv` Git dependency pulls `opencv-python` and Matplotlib transitively. They are not direct dependencies in `pixi.toml`.
- LUT fisheye images cannot be backprojected by Rerun without a supported camera projection. Their depth maps remain 2D; the upstream fused point cloud supplies their 3D geometry.
- ETH3D `playground_1l` has `doffs=0`, so it cannot distinguish `+doffs` from `-doffs`. The demo still uses the requested `fx * baseline_m / depth + doffs` formula and checks the positive-disparity sign against its negative alternative.
- `RerunTyroConfig` opens the save sink in its dataclass initializer, before `main()` can create a parent directory. The first fresh-clone gate therefore failed at `out/hetero_loft.rrd`; both public demo tasks now run `mkdir -p out` before Python.

## Reproduction table

ETH3D `playground_1l`, non-occluded pixels with GT disparity `<192` px. X-Lens is a one-scene baseline, not an upstream benchmark reproduction.

| model | EPE px | bad1 % | depth abs-rel | context |
|---|---:|---:|---:|---|
| X-Lens | 4.558876 | 88.851027 | 0.772609 | this Phase 1 baseline; 216,576 valid pixels |
| LiteAnyStereo V2 M | — | 2.24 | — | prior stereo-port baseline |
| LiteAnyStereo V2 H | — | 1.12 | — | prior stereo-port baseline |
| Fast-FoundationStereo | — | 0.48 | — | prior stereo-port baseline |

## Fresh-clone timing

A fresh remote clone started without `.pixi/`, weights, or data and used a warm shared Pixi package cache. `pixi run demo` materialized the environment, downloaded the gated checkpoint, ran inference and the 10+50 benchmark, fused/logged the scene, and serialized the Rerun recording in **62.60 s** wall time. Its synchronized model-only result was **883.114 ms per six-view scene**. A second fresh run took 69.50 s and measured 983.129 ms; an earlier complete run in the working clone took 60.61 s and measured 943.793 ms. All produced scale factor 3.391602755 and 1,990,286 fused points. The ETH3D scale factor was 12.725567818. The canonical heterogeneous demo does not need or download the ETH3D data.

## Skill discrepancies

- `fork-and-pixify.md` requires a Rerun screenshot in Gate 1, while the job explicitly says not to try screenshots because there is no display and says the reviewer will validate pixels from the `.rrd` files. This run follows the job-specific instruction and records `.rrd` evidence only.
- The generic Phase 1 task says `demo` depends on both checkpoint and sample downloads, but this job says the canonical bundled `hetero_loft` demo needs no data download. `demo` depends only on the checkpoint; `demo-eth3d` depends on both.
- The generic task calls `playground_1l` a stereo fallback. X-Lens predicts per-view metric depth rather than rectified disparity, so the job specifies the derived-disparity formula and adds depth abs-rel; these are not defined in the generic skill.
- The job requires `+doffs` in predicted disparity, while common Middlebury depth conversion is often written with an offset in the denominator. This sample has `doffs=0`, so the requested offset convention cannot be empirically disambiguated here.
- Gate 1 asks for a fresh-clone `pixi run demo`, but the environment has no display and Rerun defaults to spawning a viewer. The fork's public demo tasks therefore bake in headless `.rrd` output; later explicit `--rr-config.save` arguments override the default save path.
- Neither `fork-and-pixify.md` nor its template notes that `RerunTyroConfig` opens a save path before demo code runs and does not create its parent. A headless task with a relative default save path must create that directory before launching Python.
- The generic skill says `simplecv` does not carry its runtime dependencies. The current Git package now declares many, including `opencv-python`, but still omits `av` and `pyarrow`; the wording should identify the remaining undeclared packages. The job's instruction to omit direct `py-opencv` therefore still results in a transitive PyPI OpenCV wheel.
