# Pure fisheye — FS3 `FuelBar_100` frame 0 (4 cameras)

Four **fisheye** cameras (`CAM_A`–`CAM_D`) of a synthetic FuelBar scene. Each view ships a
precomputed per-pixel unit-ray field `calib/<cam>_dcam.npy` (OpenCV convention, float16) plus a
static valid mask — so no fisheye calibration model is needed to run it.

```
scene.json          image_size + per-view image / dcam / mask / c2w / cam_type
imgs/  calib/  mask/
```

### Run
```bash
python inference.py --ckpt checkpoints/xlens.pth \
    --manifest examples/fisheye_fuelbar/scene.json --out out/fisheye --max-depth 25
```
The recommended fisheye cleanup (FoV ≤ 85° + drop lowest 8% confidence) is applied automatically
to the fused point cloud — high lens distortion at the edges otherwise produces stray points that
extend outside the scene. Disable with `--no-fisheye-clean`.
