# Heterogeneous — FS3 `LoftOffice_kichen` frame 340 (6 cameras)

A mixed rig in one pass: **4 fisheye** (`CAM_A`–`CAM_D`, per-pixel `dcam` .npy) + **2 pinhole**
(`CAM_Front`, `CAM_Back`, intrinsics `K`). `cam_type` is set per view (0 = fisheye, 1 = pinhole).

```
scene.json          image_size + per-view image / (dcam|K) / mask / c2w / cam_type
imgs/  calib/  mask/
```

### Run
```bash
python inference.py --ckpt checkpoints/xlens.pth \
    --manifest examples/hetero_loft/scene.json --out out/hetero --max-depth 25
```
Because the scene contains fisheye views, the recommended fisheye cleanup (FoV ≤ 85° + drop
lowest 8% confidence) is applied to the fused point cloud automatically — see the main README.
Disable with `--no-fisheye-clean`.
