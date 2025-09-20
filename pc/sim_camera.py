# pc/sim_camera.py
from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Optional
import numpy as np
import cv2

Vec3 = Tuple[float, float, float]
GridLine = Tuple[Vec3, Vec3]
BoxSpec = Tuple[Tuple[float, float, float, float, float, float], Tuple[int, int, int]]

def _rotz(a: float) -> np.ndarray:  # yaw
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, -sa, 0.0],
                     [ sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def _rotx(a: float) -> np.ndarray:  # pitch
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  ca, -sa],
                     [0.0,  sa,  ca]], dtype=np.float32)

@dataclass
class ActorSpec:
    kind: str
    path: str
    scale: float = 1.0
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    # optional initial pose
    x: float = 0.0
    y: float = 0.0
    z: float = -10.0
    yaw_deg: float = 0.0

class SimCamera:
    """CPU scene generator that now *exposes* data for renderers (CPU or GL)."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fov_deg: float = 70.0,
        cam_height: float = 1.6,
        yaw0_deg: float = 0.0,
        pitch0_deg: float = -10.0,
        yaw_speed_dps: float = 15.0,
        pitch_speed_dps: float = 8.0,
        pitch_limits_deg: Tuple[float, float] = (-25, 10),
        seed: int = 42,
        *,
        renderer_name: str = "cpu", renderer_opts: dict | None = None, actors: Optional[Sequence[dict]] = None
    ) -> None:
        self.width, self.height = int(width), int(height)
        self.aspect = self.width / self.height
        self.fov = math.radians(float(fov_deg))

        fx = (0.5 * self.width) / math.tan(self.fov * 0.5)
        fy = fx
        self.K = np.array([[fx, 0, self.width/2],
                           [0, fy, self.height/2],
                           [0,  0, 1]], dtype=np.float32)

        self.cam_h = float(cam_height)
        self.yaw   = math.radians(yaw0_deg)
        self.pitch = math.radians(pitch0_deg)
        self.yaw_w   = math.radians(yaw_speed_dps)
        self.pitch_w = math.radians(pitch_speed_dps)
        self.pitch_lo, self.pitch_hi = map(math.radians, pitch_limits_deg)
        self.pitch_phase = 0.0

        rng = np.random.default_rng(seed)
        # Random ¡§buildings¡¨
        self.boxes: List[BoxSpec] = []
        for _ in range(8):
            x = float(rng.uniform(-15, 15))
            z = float(rng.uniform(8, 40))
            w = float(rng.uniform(1.0, 3.0))
            d = float(rng.uniform(1.0, 3.0))
            h = float(rng.uniform(2.0, 7.0))
            color = tuple(int(c) for c in rng.integers(80, 220, size=3))
            self.boxes.append(((x, 0.0, z, w, d, h), color))

        # Ground grid (world units)
        self.grid_lines: List[GridLine] = []
        grid_extent = 60
        step = 2
        for x in range(-grid_extent, grid_extent+1, step):
            self.grid_lines.append(((x, 0, 2), (x, 0, grid_extent)))
        for z in range(2, grid_extent+1, step):
            self.grid_lines.append(((-grid_extent, 0, z), (grid_extent, 0, z)))

        # --- Actors (for GL renderer) ---
        self.actor_meshes: List[dict] = []
        self._actor_poses: List[List[float]] = []
        if actors:
            # Normalize into ActorSpec; also store initial poses if provided
            for spec in actors:
                a = ActorSpec(
                    kind = spec.get("kind", "obj"),
                    path = spec["path"],
                    scale = float(spec.get("scale", 1.0)),
                    color = tuple(map(float, spec.get("color", (0.8,0.8,0.8)))),
                    x = float(spec.get("x", 0.0)),
                    y = float(spec.get("y", 0.0)),
                    z = float(spec.get("z", -10.0)),
                    yaw_deg = float(spec.get("yaw_deg", 0.0)),
                )
                self.actor_meshes.append({
                    "kind": a.kind, "path": a.path, "scale": a.scale, "color": a.color
                })
                self._actor_poses.append([a.x, a.y, a.z, a.yaw_deg])

        self.t_last = time.monotonic()

        # Renderer (plug-in)
        from .renderers import get_renderer
        renderer_opts = renderer_opts or {}
        self._renderer = get_renderer(renderer_name, context=self, **renderer_opts)

    # --- Pose helpers ---------------------------------------------------------
    def _pose(self, t_now: float) -> Tuple[np.ndarray, np.ndarray]:
        dt = t_now - self.t_last
        self.t_last = t_now
        # yaw continuous
        self.yaw += self.yaw_w * dt
        # pitch oscillate in bounds
        self.pitch_phase += self.pitch_w * dt
        mid = 0.5 * (self.pitch_hi + self.pitch_lo)
        amp = 0.5 * (self.pitch_hi - self.pitch_lo)
        self.pitch = mid + amp * math.sin(self.pitch_phase)

        R = _rotx(self.pitch) @ _rotz(self.yaw)
        t = np.array([[0.0], [self.cam_h], [0.0]], dtype=np.float32)
        rvec, _ = cv2.Rodrigues(R.T)   # camera rotation wrt world
        tvec = -R.T @ t                # camera translation
        return rvec.astype(np.float32), tvec.astype(np.float32)

    # Optional per-frame actor transforms provider (GL renderer will use it if present)
    def get_actor_transforms(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        t = time.monotonic()
        for i, (x, y, z, yaw_deg) in enumerate(self._actor_poses):
            yaw = math.radians(yaw_deg + 20.0 * math.sin(0.4 * t + i))
            c, s = math.cos(yaw), math.sin(yaw)
            M = np.eye(4, dtype=np.float32)
            M[0, 0] =  c; M[0, 2] =  s
            M[2, 0] = -s; M[2, 2] =  c
            M[1, 1] = 1.0
            M[:3, 3] = np.array([x, y, z], dtype=np.float32)
            out.append(M)
        return out

    # --- Legacy CPU drawing (kept for CPU renderer) ---------------------------
    def _proj(self, pts3d_world: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        dist = np.zeros((5,1), dtype=np.float32)
        pts2d, _ = cv2.projectPoints(pts3d_world.astype(np.float32), rvec, tvec, self.K, dist)
        return pts2d.reshape(-1, 2)

    def _draw_boxes(self, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
        for (x, y, z, w, d, h), color in self.boxes:
            X = np.array([
                [x-0.5*w, y,       z-0.5*d],
                [x+0.5*w, y,       z-0.5*d],
                [x+0.5*w, y,       z+0.5*d],
                [x-0.5*w, y,       z+0.5*d],
                [x-0.5*w, y+h,     z-0.5*d],
                [x+0.5*w, y+h,     z-0.5*d],
                [x+0.5*w, y+h,     z+0.5*d],
                [x-0.5*w, y+h,     z+0.5*d],
            ], dtype=np.float32)
            pts = self._proj(X, rvec, tvec).astype(int)
            edges = [(0,1),(1,2),(2,3),(3,0),
                     (4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]
            for a,b in edges:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)

    def _draw_ground(self, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
        img[:] = (180, 180, 210)
        cv2.rectangle(img, (0, self.height//2), (self.width, self.height), (170, 190, 170), -1)
        for (x1,y1,z1), (x2,y2,z2) in self.grid_lines:
            X = np.array([[x1,y1,z1], [x2,y2,z2]], dtype=np.float32)
            pts = self._proj(X, rvec, tvec).astype(int)
            cv2.line(img, tuple(pts[0]), tuple(pts[1]), (120,120,120), 1, cv2.LINE_AA)

    # --- Public frame generator ----------------------------------------------
    def next_frame(self) -> Tuple[bool, np.ndarray]:
        rvec, tvec = self._pose(time.monotonic())
        img = np.empty((self.height, self.width, 3), dtype=np.uint8)
        # Delegate to current renderer
        self._renderer.render(img, rvec=rvec, tvec=tvec)
        # simple crosshair
        cv2.circle(img, (self.width//2, self.height//2), 4, (0,0,0), -1, cv2.LINE_AA)
        return True, img
