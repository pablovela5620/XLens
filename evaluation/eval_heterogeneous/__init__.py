"""Heterogeneous multi-camera depth evaluation for the X-Lens model.

Entry point:
    eval_calib_compare.py - 6-camera mixed-domain (4 fisheye + 2 pinhole) calibrated
                            depth evaluation on foundationstage3/test.

The model is rebuilt from an arch config (configs/xlens_vits.yaml) or the config
embedded in a .pth checkpoint, else inferred from the weights. It uses n_cam_types=2:
CAM_Front/CAM_Back are pinhole (cam_type=1), the rest fisheye (cam_type=0).
"""
