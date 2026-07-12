"""KITTI-360 fisheye temporal stereo / multi-view training dataset (dataset_mode="kitti360_stereo").

Monocular temporal multi-view (SfM, non-rig): a single fisheye camera's target frame
(view 0, with GT) plus gap-spaced support frames, with baseline from ego motion. Reads the
fisheye_stereo_eval manifest and produces batch fields compatible with trainer/criterion
(images/depths/intrinsics/extrinsics_world/d_cam/ray_map/cam_types/view_mask/non_ambiguous_mask).

Output convention (matches omni, fed to the same _train_step):
  images(S,3,H,W) ImageNet-normalized; depths(S,H,W) z-depth in meters, 0=invalid
    (manifest GT is euclidean_range; load_gt multiplies by cos(theta) to convert to z-depth);
  d_cam(S,3,H,W) inverse-MEI unit rays; ray_map(S,6,H,W) [d_world, t_norm] canonicalized to view 0;
  extrinsics_world(S,4,4) c2w canonical (view0=I); intrinsics(S,3,3) placeholder K;
  cam_types(S,); view_mask(S,); non_ambiguous_mask(S,H,W)=lens; has_non_ambiguous_mask(S,).
"""
import os, re, json, glob
import numpy as np
import cv2
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset

from ..data_configs import get_dataset_config

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _build_dcam(mei, Wt, Ht):
    """Inverse MEI: pixels -> camera-frame unit rays (3,Ht,Wt) plus lens (theta<=92.5) bool."""
    sx, sy = Wt / mei["width"], Ht / mei["height"]
    g1, g2 = mei["gamma1"] * sx, mei["gamma2"] * sy
    u0, v0 = mei["u0"] * sx, mei["v0"] * sy
    xi, k1, k2 = mei["xi"], mei["k1"], mei["k2"]
    uu, vv = np.meshgrid(np.arange(Wt), np.arange(Ht))
    mx = (uu - u0) / g1; my = (vv - v0) / g2
    dx, dy = mx.copy(), my.copy()
    for _ in range(20):
        r2 = dx * dx + dy * dy; f = 1 + k1 * r2 + k2 * r2 * r2
        dx = mx / f; dy = my / f
    m2 = dx * dx + dy * dy
    disc = np.clip(xi * xi - (m2 + 1) * (xi * xi - 1), 0, None)
    t = (xi + np.sqrt(disc)) / (m2 + 1)
    ray = np.stack([dx * t, dy * t, t - xi], 0).astype(np.float32)
    ray /= np.linalg.norm(ray, axis=0, keepdims=True) + 1e-9
    rz = ray[2]
    lens = np.degrees(np.arccos(np.clip(rz, -1, 1))) <= 92.5
    K = np.array([[g1, 0, u0], [0, g2, v0], [0, 0, 1]], np.float32)
    return ray, lens, K


class _Manifest:
    """Lightweight handle for one (seq, cam) manifest: caches d_cam/lens/K and serves samples."""
    def __init__(self, path, root, Ht, Wt, lens_fov_deg=92.5, lens_mask_dir=None):
        self.m = json.load(open(path)); self.meta = self.m["meta"]
        self.root = root or self.meta["kitti360_root"]
        self.Ht, self.Wt = Ht, Wt
        self.seq = self.meta["sequence"]; self.cam = self.meta["camera"]
        self.frames = self.m["frames"]; self.samples = self.m["samples"]
        self.scale = self.meta["depth"]["scale"]
        self.dcam, lens_geom, self.K = _build_dcam(self.meta["intrinsics"], Wt, Ht)
        # lens mask: prefer the pregenerated real lens FOV mask (includes vignetting/occlusion);
        # fall back to the geometric theta<=92.5 circle. Used to (1) gate d_cam/ray_map to zero
        # outside the FOV and (2) serve as non_ambiguous_mask (loss only inside FOV and depth>0).
        lm = self._load_lens_mask(lens_mask_dir)
        self.lens = lm if lm is not None else lens_geom.astype(bool)

    def _load_lens_mask(self, lens_mask_dir):
        if not lens_mask_dir:
            return None
        p = os.path.join(lens_mask_dir, f"{self.cam}_lens_mask.png")
        if not os.path.isfile(p):
            return None
        m = imageio.imread(p)
        if m.ndim == 3:
            m = m[..., 0]
        m = cv2.resize(m.astype(np.uint8), (self.Wt, self.Ht), interpolation=cv2.INTER_NEAREST)
        return m > 0

    def _img_path(self, f):
        return os.path.join(self.root, self.meta["image"]["rel_dir"],
                            self.meta["image"]["subpath"].format(seq=self.seq, cam=self.cam, frame=f))

    def _gt_path(self, f):
        return os.path.join(self.root, self.meta["depth"]["rel_dir"],
                            self.meta["depth"]["subpath"].format(seq=self.seq, cam=self.cam, frame=f))

    def load_img(self, f, color_aug=False):
        bgr = cv2.imread(self._img_path(f), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self._img_path(f))
        bgr = cv2.resize(bgr, (self.Wt, self.Ht), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if color_aug:
            g = np.random.uniform(0.8, 1.2)
            rgb = np.clip(rgb * g, 0, 1)
        return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)

    def load_gt(self, f):
        p = self._gt_path(f)
        if not os.path.isfile(p):
            return np.zeros((self.Ht, self.Wt), np.float32)
        d = imageio.imread(p).astype(np.float32) / self.scale
        d = cv2.resize(d, (self.Wt, self.Ht), interpolation=cv2.INTER_NEAREST)  # nearest for sparse GT
        # manifest GT is euclidean_range (distance along ray to camera center); convert to z-depth
        # to match the criterion's convention: z = range * cos(theta) = range * ray_z,
        # where self.dcam[2] is the unit ray's z component (cos theta).
        d = d * self.dcam[2]
        d[d < 0] = 0.0    # rz<0 near the 90-92.5 deg edge -> invalid (dropped by lens / depth>0)
        return d.astype(np.float32)


class KITTI360StereoDataset(Dataset):
    def __init__(self, manifest_index, sequences, kitti360_root=None, cameras=None,
                 target_hw=(504, 504), cam_type=0, train=True, color_aug=True):
        idx = json.load(open(manifest_index))
        base = os.path.dirname(manifest_index)
        root = kitti360_root or idx.get("kitti360_root")
        if isinstance(target_hw[0], (list, tuple)):
            target_hw = tuple(target_hw[0])
        Ht, Wt = int(target_hw[0]), int(target_hw[1])
        seqs = set(sequences); cams = set(cameras) if cameras else None
        self.cam_type = int(cam_type)
        self.train = train; self.color_aug = train and color_aug
        self.Ht, self.Wt = Ht, Wt
        # Declarative flag (data_configs/kitti360.yaml): KITTI has no sky/semantic mask,
        # so it does not contribute to the mask head loss.
        self.contributes_mask_loss = bool(get_dataset_config("kitti360").contributes_mask_loss)
        # Pregenerated lens FOV mask directory (sibling of manifests/).
        lens_mask_dir = os.path.join(base, "lens_masks")
        if not os.path.isdir(lens_mask_dir):
            lens_mask_dir = None
        # For DynamicBatchSampler / mixed sampling: KITTI temporal stereo is fixed at 2 views.
        self.n_views = 2
        self.num_views_list = [2]
        self.variable_num_views = False
        self.views_per_frame = 2
        self.flat = []  # (Manifest, sample_idx)
        for e in idx["manifests"]:
            if e["sequence"] not in seqs:
                continue
            if cams is not None and e["camera"] not in cams:
                continue
            man = _Manifest(os.path.join(base, e["file"]), root, Ht, Wt,
                            lens_mask_dir=lens_mask_dir)
            for si in range(len(man.samples)):
                self.flat.append((man, si))

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, i):
        # sampler may pass a tuple idx (sample_idx, ar_idx, nv_idx); KITTI is fixed at 2 views.
        if isinstance(i, tuple):
            i = i[0]
        # Bad-sample fallback: occasional truncated/corrupt PNGs or I/O jitter. A single bad
        # sample must not crash the worker (would stall the DDP barrier). Retry 5 times,
        # advancing to the next sample.
        import logging
        last = None
        n = len(self.flat)
        for r in range(5):
            try:
                return self._getitem_impl((i + r) % n)
            except (FileNotFoundError, OSError, ValueError, cv2.error) as e:
                last = e
                logging.getLogger(__name__).warning(
                    f"[kitti360_stereo] sample {(i + r) % n} load failed (retry {r}): "
                    f"{type(e).__name__}: {str(e)[:160]}")
        raise RuntimeError(f"5 consecutive samples failed to load, start={i}, last error: {last}")

    def _getitem_impl(self, i):
        man, si = self.flat[i]
        s = man.samples[si]
        vfr = [s["target"]] + s["support"]            # view0 = target frame (has GT)
        S, Ht, Wt = len(vfr), man.Ht, man.Wt
        imgs = np.stack([man.load_img(f, self.color_aug) for f in vfr], 0)   # (S,3,H,W)
        depths = np.stack([man.load_gt(f) for f in vfr], 0)                  # (S,H,W)
        # canonicalize to view 0
        c2w = [np.array(man.frames[str(f)]["cam2world"], np.float64) for f in vfr]
        inv0 = np.linalg.inv(c2w[0]); rel = [inv0 @ c for c in c2w]
        ts = [c[:3, 3] for c in rel]
        norms = [np.linalg.norm(t) for t in ts[1:]]
        ps = float(np.mean(norms)) if norms and np.mean(norms) > 1e-6 else 1.0
        dcam = man.dcam                                                      # (3,H,W)
        rays = []
        for c in rel:
            R = c[:3, :3].astype(np.float32)
            dw = (R @ dcam.reshape(3, -1)).reshape(3, Ht, Wt)
            tn = (c[:3, 3] / ps).astype(np.float32).reshape(3, 1, 1) * np.ones((3, Ht, Wt), np.float32)
            rays.append(np.concatenate([dw, tn], 0))
        ray_map = np.stack(rays, 0).astype(np.float32)                      # (S,6,H,W)
        lens = man.lens.astype(bool)
        # Zero d_cam / ray_map outside the lens FOV (MEI inverse projection is ill-conditioned
        # at square corners, theta>92.5 deg), matching omni's convention.
        lensf = lens.astype(np.float32)                                     # (H,W)
        dcam_g = (dcam * lensf[None]).astype(np.float32)                    # (3,H,W)
        ray_map = ray_map * lensf[None, None]                              # (S,6,H,W) broadcast
        out = {
            "images": torch.from_numpy(imgs.astype(np.float32)),
            "depths": torch.from_numpy(depths.astype(np.float32)),
            "d_cam": torch.from_numpy(np.stack([dcam_g] * S, 0).astype(np.float32)),
            "ray_map": torch.from_numpy(ray_map.astype(np.float32)),
            "extrinsics_world": torch.from_numpy(np.stack(rel, 0).astype(np.float32)),
            "intrinsics": torch.from_numpy(np.stack([man.K] * S, 0).astype(np.float32)),
            "cam_types": torch.full((S,), self.cam_type, dtype=torch.long),
            "view_mask": torch.ones(S, dtype=torch.bool),
            "non_ambiguous_mask": torch.from_numpy(np.stack([lens] * S, 0)),
            "has_non_ambiguous_mask": torch.ones(S, dtype=torch.bool),
            "cameras_name": [f"{man.cam}_{f}" for f in vfr],
            "dataset_format": "kitti360",   # mixed collate dispatches on this to collate_kitti360_stereo
            "contributes_mask_loss": self.contributes_mask_loss,  # False -> trainer skips mask loss
        }
        return out


def collate_kitti360_stereo(batch):
    """Stack per-sample (S,...) into (B,S,...). All samples share S/H/W, so tensors stack
    directly; non-tensor fields (cameras_name / dataset_format) are collected into lists."""
    out = {}
    for k in batch[0].keys():
        if torch.is_tensor(batch[0][k]):
            out[k] = torch.stack([b[k] for b in batch], 0)
        else:
            out[k] = [b[k] for b in batch]
    return out
