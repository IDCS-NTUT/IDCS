# pc/sim_camera.py
import math, time
from types import SimpleNamespace
import numpy as np
import cv2

# pulls the renderer factory (CPU now, GL later)
from pc.renderers import Renderer, get_renderer


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
    """
    Same public API as before, but rendering is delegated to a backend:
      - renderer_name: "cpu" (default) or "gl"
      - renderer_opts: dict forwarded to the backend (optional)
    """
    def __init__(self, width=1280, height=720, fov_deg=70.0,
                 cam_height=1.6, yaw0_deg=0.0, pitch0_deg=-10.0,
                 yaw_speed_dps=15.0, pitch_speed_dps=8.0,
                 pitch_limits_deg=(-25, 10), seed=42,
                 renderer_name: str = "cpu",
                 renderer_opts: dict | None = None):
        # --- camera intrinsics (unchanged)
        self.W, self.H = int(width), int(height)
        self.aspect = float(width) / float(height)
        self.fov = math.radians(fov_deg)
        fx = (0.5 * width) / math.tan(self.fov * 0.5)
        fy = fx
        self.K = np.array([[fx, 0, width/2],
                           [0, fy, height/2],
                           [0,  0, 1]], dtype=np.float32)

        # --- camera motion (unchanged)
        self.cam_h = cam_height
        self.yaw = math.radians(yaw0_deg)
        self.pitch = math.radians(pitch0_deg)
        self.yaw_w = math.radians(yaw_speed_dps)     # rad/s
        self.pitch_w = math.radians(pitch_speed_dps) # rad/s
        self.pitch_lo, self.pitch_hi = map(math.radians, pitch_limits_deg)
        self.pitch_phase = 0.0

        # --- scene (unchanged)
        rng = np.random.default_rng(seed)
        self.boxes = []
        for _ in range(8):
            x = rng.uniform(-15, 15)
            z = rng.uniform(8, 40)
            w = rng.uniform(1.0, 3.0)
            d = rng.uniform(1.0, 3.0)
            h = rng.uniform(2.0, 7.0)
            color = tuple(int(c) for c in rng.integers(80, 220, size=3))
            self.boxes.append(((x, 0.0, z, w, d, h), color))

        self.grid_lines = []
        grid_extent = 60
        step = 2
        for x in range(-grid_extent, grid_extent+1, step):
            self.grid_lines.append(((x, 0, 2), (x, 0, grid_extent)))
        for z in range(2, grid_extent+1, step):
            self.grid_lines.append(((-grid_extent, 0, z), (grid_extent, 0, z)))

        # timebase (unchanged)
        self.t_last = time.monotonic()

        # --- renderer context & backend
        # Keep this tiny and stable so backends can evolve independently.
        context = SimpleNamespace(
            width=self.W,
            height=self.H,
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
        # identical motion update to your original
        dt = t_now - self.t_last
        self.t_last = t_now
        self.yaw += self.yaw_w * dt
        self.pitch_phase += self.pitch_w * dt
        mid = 0.5 * (self.pitch_hi + self.pitch_lo)
        amp = 0.5 * (self.pitch_hi - self.pitch_lo)
        self.pitch = mid + amp * math.sin(self.pitch_phase)

        # Rotation: yaw then pitch
        R = _rotx(self.pitch) @ _rotz(self.yaw)
        t = np.array([[0.0], [self.cam_h], [0.0]], dtype=np.float32)
        rvec, _ = cv2.Rodrigues(R.T)
        tvec = -R.T @ t
        return rvec.astype(np.float32), tvec.astype(np.float32)

    def next_frame(self):
        # same external contract: returns (ok, BGR frame)
        now = time.monotonic()
        rvec, tvec = self._pose(now)
        img = np.empty((self.H, self.W, 3), dtype=np.uint8)

        # delegate the drawing
        self._renderer.render(img, rvec=rvec, tvec=tvec)

        # simple horizon/crosshair (unchanged)
        cv2.circle(img, (self.W//2, self.H//2), 4, (0,0,0), -1, cv2.LINE_AA)
        return True, img
