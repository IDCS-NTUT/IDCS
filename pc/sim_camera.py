# pc/sim_camera.py
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np

from pc.renderers import Renderer, get_renderer

# Simple 3D-ish scene: ground grid + a few boxes ("buildings").
# Camera yaws/pitches over time. Outputs BGR frames.

def _rotz(a):  # yaw
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, -sa, 0.0],
                     [ sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def _rotx(a):  # pitch
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  ca, -sa],
                     [0.0,  sa,  ca]], dtype=np.float32)




class SimCamera:
    def __init__(
        self,
        width=1280,
        height=720,
        fov_deg=70.0,
        cam_height=1.6,
        yaw0_deg=0.0,
        pitch0_deg=-10.0,
        yaw_speed_dps=15.0,
        pitch_speed_dps=8.0,
        pitch_limits_deg=(-25, 10),
        seed=42,
        renderer_name: str = "cpu",
        renderer_opts: dict | None = None,
    ):
        self.W, self.H = width, height
        self.aspect = width / height
        self.fov = math.radians(fov_deg)
        fx = (0.5 * width) / math.tan(self.fov * 0.5)
        fy = fx  # square pixels
        self.K = np.array([[fx, 0, width/2],
                           [0, fy, height/2],
                           [0,  0, 1]], dtype=np.float32)

        self.cam_h = cam_height
        self.yaw = math.radians(yaw0_deg)
        self.pitch = math.radians(pitch0_deg)
        self.yaw_w = math.radians(yaw_speed_dps)     # rad/s
        self.pitch_w = math.radians(pitch_speed_dps) # rad/s
        self.pitch_lo, self.pitch_hi = map(math.radians, pitch_limits_deg)
        self.pitch_phase = 0.0

        rng = np.random.default_rng(seed)
        # Random “buildings” on ground plane (X-Z), Y is up
        self.boxes = []
        for _ in range(8):
            x = rng.uniform(-15, 15)
            z = rng.uniform(8, 40)
            w = rng.uniform(1.0, 3.0)
            d = rng.uniform(1.0, 3.0)
            h = rng.uniform(2.0, 7.0)
            color = tuple(int(c) for c in rng.integers(80, 220, size=3))
            self.boxes.append(((x, 0.0, z, w, d, h), color))

        # Pre-generate ground grid points
        self.grid_lines = []
        grid_extent = 60
        step = 2
        for x in range(-grid_extent, grid_extent+1, step):
            self.grid_lines.append(((x, 0, 2), (x, 0, grid_extent)))
        for z in range(2, grid_extent+1, step):
            self.grid_lines.append(((-grid_extent, 0, z), (grid_extent, 0, z)))

        self.t_last = time.monotonic()

        context = SimpleNamespace(
            width=self.W,
            height=self.H,
            proj_masked=self._proj_masked,
            grid_lines=self.grid_lines,
            boxes=self.boxes,
            intrinsics=self.K.copy(),
            fov=self.fov,
            aspect=self.aspect,
        )
        opts = dict(renderer_opts or {})
        opts.setdefault("context", context)
        self._renderer: Renderer = get_renderer(renderer_name, **opts)

    def _pose(self, t_now):
        # time delta
        dt = t_now - self.t_last
        self.t_last = t_now
        # yaw: steady sweep
        self.yaw += self.yaw_w * dt
        # pitch: bounded sinusoid
        self.pitch_phase += self.pitch_w * dt
        mid = 0.5 * (self.pitch_hi + self.pitch_lo)
        amp = 0.5 * (self.pitch_hi - self.pitch_lo)
        self.pitch = mid + amp * math.sin(self.pitch_phase)
        # Rotation: yaw then pitch (R = Rz * Rx)
        R = _rotx(self.pitch) @ _rotz(self.yaw)
        # Camera at (0, cam_h, 0) in world
        t = np.array([[0.0], [self.cam_h], [0.0]], dtype=np.float32)
        # World → Camera: Xc = R^T (Xw - C). For cv2.projectPoints, use rvec/tvec of camera wrt world.
        rvec, _ = cv2.Rodrigues(R.T)   # camera rotation
        tvec = -R.T @ t                # camera translation
        return rvec.astype(np.float32), tvec.astype(np.float32)

    def _proj(self, pts3d_world, rvec, tvec):
        # Pinhole projection
        dist = np.zeros((5,1), dtype=np.float32)  # no distortion
        pts2d, _ = cv2.projectPoints(pts3d_world.astype(np.float32), rvec, tvec, self.K, dist)
        return pts2d.reshape(-1, 2)

    def next_frame(self):
        now = time.monotonic()
        rvec, tvec = self._pose(now)
        img = np.empty((self.H, self.W, 3), dtype=np.uint8)
        self._renderer.render(img, rvec=rvec, tvec=tvec)
        # simple horizon/crosshair
        cv2.circle(img, (self.W//2, self.H//2), 4, (0,0,0), -1, cv2.LINE_AA)
        return True, img

    def _proj_masked(self, Xw, rvec, tvec):
        # Returns (pts2d, mask_visible) where mask excludes points behind camera
        R, _ = cv2.Rodrigues(rvec)
        C = -R.T @ tvec
        Xc = (R.T @ (Xw.T - C)).T  # world->camera
        mask = Xc[:, 2] > 1e-6     # keep only Zc > 0
        if not np.any(mask):
            return None, mask
        dist = np.zeros((5,1), np.float32)
        pts2d, _ = cv2.projectPoints(Xw[mask].astype(np.float32), rvec, tvec, self.K, dist)
        out = np.empty((Xw.shape[0], 2), np.float32)
        out[:] = np.nan
        out[mask] = pts2d.reshape(-1,2)
        return out, mask

