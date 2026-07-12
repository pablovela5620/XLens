# Pure pinhole — OmniOCC `4f_ba1` (6 cameras)

A real 6-camera **pinhole** capture (3 stereo pairs: cam0/1/2 × left/right) from the
OmniOCC dataset. Every view ships a 3×3 intrinsics `K`; `d_cam` is derived from `K` at run time.

```
scene.json     image_size + per-view image / K / c2w
imgs/          6 RGB images (504×798)
```

### Run
```bash
python inference.py --ckpt checkpoints/xlens.pth \
    --manifest examples/pinhole_4f_ba1/scene.json --out out/pinhole --max-depth 15
```
