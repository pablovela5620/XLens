"""Fisheye depth evaluation for the X-Lens model.

Entry points:
    eval_fisheye.py  - 4-camera omni fisheye: compute MAE and other depth metrics on the
                       test set and save 3-panel figures (GT depth | pred depth | error map).
    eval_kitti360.py - KITTI-360 monocular fisheye (motion-stereo) depth evaluation.
"""
