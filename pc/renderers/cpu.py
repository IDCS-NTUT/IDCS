# pc/renderers/cpu.py
from __future__ import annotations
import numpy as np
import cv2

from . import register_renderer  # comes from pc/renderers/__init__.py
from typing import Any, Sequence, Tuple, Optional, List, Union  # expand types

# New: billboard spec
Billboard = Tuple[
    Tuple[float, float, float],  # world position (x,y,z)
    str,                         # size mode: "meters" or "pixels"
    float,                       # size value
    Tuple[int,int,int,int],      # RGBA color (0..255) or ignored if image used
    Optional[str],               # image path (optional)
]


GridLine = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
BoxSpec  = Tuple[Tuple[float, float, float, float, float, float], Tuple[int, int, int]]
def _project_points_KRt(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, Pw: np.ndarray):
    pts, _ = cv2.projectPoints(Pw.astype(np.float32), rvec, tvec, K, np.zeros((5,1), np.float32))
    uv = pts.reshape(-1, 2)
    R, _ = cv2.Rodrigues(rvec.astype(np.float32))
    Xc = (R @ Pw.T + tvec.reshape(3,1)).T
    Zc = Xc[:, 2]
    ok = Zc > 1e-6
    return uv.astype(np.float32), Zc.astype(np.float32), ok

def _draw_sprite_bgr(frame, center_uv, size_px, rgba=None, sprite_bgr=None):
    h, w = frame.shape[:2]
    u, v = float(center_uv[0]), float(center_uv[1])
    s = max(1, int(round(size_px)))
    x0, y0 = int(round(u - s/2)), int(round(v - s/2))
    x1, y1 = x0 + s, y0 + s
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)

    if sprite_bgr is None:
        # (existing solid RGBA path unchanged)
        ...
        return

    # --- image sprite with alpha ---
    # Resize once to s¡Ñs
    if sprite_bgr.shape[0] != s or sprite_bgr.shape[1] != s:
        sprite_bgr = cv2.resize(sprite_bgr, (s, s), interpolation=cv2.INTER_AREA)

    # Crop overlapping ROI
    patch = frame[y0c:y1c, x0c:x1c]
    sx0, sy0 = x0c - x0, y0c - y0
    sx1, sy1 = sx0 + (x1c - x0c), sy0 + (y1c - y0c)
    sub = sprite_bgr[sy0:sy1, sx0:sx1]

    if sub.shape[:2] != patch.shape[:2]:
        return

    if sub.shape[2] == 4:

    fov_h = math.radians(fov) if fov > math.pi else fov

    # SimCamera defines a horizontal FOV and mirrors fx to fy so the CPU
    # renderer must follow the same relationship to preserve visuals.
    expected_fx = (0.5 * W) / math.tan(0.5 * fov_h)
    expected_fy = expected_fx
    cx, cy = 0.5 * W, 0.5 * H

    if hasattr(ctx, "intrinsics") and getattr(ctx, "intrinsics") is not None:
        K = np.array(getattr(ctx, "intrinsics"), dtype=np.float32)
    elif hasattr(ctx, "K") and getattr(ctx, "K") is not None:
        K = np.array(getattr(ctx, "K"), dtype=np.float32)
    else:
        K = np.array([[expected_fx, 0.0, cx],
                      [0.0, expected_fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    tol = dict(rel_tol=1e-4, abs_tol=1e-3)
    if not (math.isfinite(fx) and math.isfinite(fy)):
        raise ValueError("Camera intrinsics must be finite")
    if not (math.isclose(fx, expected_fx, **tol) and math.isclose(fy, expected_fy, **tol)):
        raise AssertionError(
            f"CPU renderer expects fx=fy from SimCamera (got fx={fx}, fy={fy}, expected {expected_fx})"
        )

    ctx.intrinsics = K
                               r.astype(np.float32)])
        blended = a3 * sub_bgr_f + inva3 * patch_f
        patch[:] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        # No alpha ¡÷ normal copy
        patch[:] = sub[:, :, :3]


def _ensure_intrinsics_on_context(ctx):
    import math
    if hasattr(ctx, "intrinsics") and ctx.intrinsics is not None:
        return
    W = int(getattr(ctx, "width"))
    H = int(getattr(ctx, "height"))
    aspect = float(getattr(ctx, "aspect", W / H))
    fov = float(getattr(ctx, "fov", math.radians(60.0)))  # accept deg or rad
    fov_y = math.radians(fov) if fov > math.pi else fov
    fy = 0.5 * H / math.tan(0.5 * fov_y)
    fx = fy * aspect
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    import numpy as np
    ctx.intrinsics = np.array([[fx, 0,  cx],
                               [0,  fy, cy],
                               [0,  0,   1]], dtype=np.float32)

def _build_grid_lines_default():
    # Same parameters your GL backend uses
    x_extent = 40.0
    z_near   = -2.0
    z_far    = -40.0
    step     = 2.0
    y = 0.0
    lines = []
    z = z_far
    while z <= z_near + 1e-6:
        lines.append(((-x_extent, y, z), ( x_extent, y, z)))
        z += step
    x = -x_extent
    while x <= x_extent + 1e-6:
        lines.append(((x, y, z_far), (x, y, z_near)))
        x += step
    return lines



class CPURenderer:
    """
    Drop-in CPU backend that reproduces your current SimCamera drawing with OpenCV.
    It fills the provided BGR frame using the camera pose (rvec/tvec).
    """

    def __init__(self, *, context: Any):
        _ensure_intrinsics_on_context(context)  # <-- new

        required = ("width", "height", "intrinsics")
        missing = [n for n in required if not hasattr(context, n)]
        if missing:
            raise AttributeError("CPU renderer context missing: " + ", ".join(missing))

        self.W = int(getattr(context, "width"))
        self.H = int(getattr(context, "height"))

        # Grid: use provided lines if present; otherwise auto-build (like GL)
        if hasattr(context, "grid_lines") and getattr(context, "grid_lines"):
            self.grid_lines = tuple(getattr(context, "grid_lines"))
        else:
            self.grid_lines = tuple(_build_grid_lines_default())  # <-- new

        # Boxes (same schema you already use)
        self.boxes = tuple(getattr(context, "boxes", []) or [])

        self.K = np.array(getattr(context, "intrinsics"), dtype=np.float32)

        # Colors (BGR)
        self.sky  = (180, 180, 210)
        self.ground = (170, 190, 170)
        self.grid = (120, 120, 120)

        self._dist = np.zeros((5, 1), dtype=np.float32)  # no distortion

        # Predeclare box edges
        self._edges = (
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        )

        self.billboards: List[Billboard] = []
        ctx_bbs = getattr(context, "billboards", None)
        if ctx_bbs:
            # Normalize and optionally preload images (BGR)
            for bb in ctx_bbs:
                pos, mode, size_val, rgba, img_path = bb
                img_bgr = None
                if isinstance(img_path, str) and len(img_path) > 0:
                    try:
                        img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    except Exception:
                        img_bgr = None
                self.billboards.append((tuple(map(float, pos)), str(mode), float(size_val), tuple(map(int, rgba)), img_path))
            # Cache of loaded images to avoid re-reading every frame
            self._bb_cache = {}  # path -> BGR np.ndarray
        else:
            self._bb_cache = {}

    def _proj(self, X: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        pts, _ = cv2.projectPoints(X.astype(np.float32), rvec, tvec, self.K, self._dist)
        return pts.reshape(-1, 2).astype(int)

    def render(self, frame: np.ndarray, /, *, rvec: np.ndarray, tvec: np.ndarray) -> None:
        # background
        frame[:] = self.sky
        cv2.rectangle(frame, (0, self.H // 2), (self.W, self.H), self.ground, -1)

        # grid
        for (x1, y1, z1), (x2, y2, z2) in self.grid_lines:
            X = np.array([[x1, y1, z1], [x2, y2, z2]], dtype=np.float32)
            p = self._proj(X, rvec, tvec)
            cv2.line(frame, tuple(p[0]), tuple(p[1]), self.grid, 1, cv2.LINE_AA)

        # boxes
        for (x, y, z, w, d, h), color in self.boxes:
            X = np.array([
                [x-0.5*w, y,   z-0.5*d],
                [x+0.5*w, y,   z-0.5*d],
                [x+0.5*w, y,   z+0.5*d],
                [x-0.5*w, y,   z+0.5*d],
                [x-0.5*w, y+h, z-0.5*d],
                [x+0.5*w, y+h, z-0.5*d],
                [x+0.5*w, y+h, z+0.5*d],
                [x-0.5*w, y+h, z+0.5*d],
            ], dtype=np.float32)
            p = self._proj(X, rvec, tvec)
            for a, b in self._edges:
                cv2.line(frame, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)

        # --- BILLBOARDS (CPU) ---
        if self.billboards:
            # Collect positions
            Pw = np.array([bb[0] for bb in self.billboards], dtype=np.float32)
            uv, Zc, ok = _project_points_KRt(self.K, rvec, tvec, Pw)

            # Painter¡¦s algo: draw far -> near so closer billboards overwrite farther ones
            order = np.argsort(Zc)  # ascending Z (farther negative? Our Zc>0 forward; larger Z is farther)
            # We want far-to-near: largest Z first
            order = order[::-1]

            fx = float(self.K[0, 0])
            diag_max = 0.35 * float(max(self.W, self.H))  # clamp to avoid huge sprites

            for idx in order:
                if not ok[idx]:
                    continue
                (pos, mode, size_val, rgba, img_path) = self.billboards[idx]
                u, v = uv[idx]
                z = float(Zc[idx])

                # Convert size to pixels
                if mode == "meters":
                    size_px = fx * (float(size_val) / max(z, 1e-6))
                else:
                    size_px = float(size_val)

                # Clamp to sane range
                size_px = max(2.0, min(size_px, diag_max))

                # Load sprite lazily if needed
                sprite_bgr = None
                if isinstance(img_path, str) and len(img_path) > 0:
                    if img_path in self._bb_cache:
                        sprite_bgr = self._bb_cache[img_path]
                    else:
                        try:
                            sprite_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        except Exception:
                            sprite_bgr = None
                        self._bb_cache[img_path] = sprite_bgr

                _draw_sprite_bgr(frame, (u, v), size_px, rgba=rgba, sprite_bgr=sprite_bgr)


    
# Register backend under the name "cpu"
register_renderer("cpu", lambda **kw: CPURenderer(**kw))
__all__ = ["CPURenderer"]
