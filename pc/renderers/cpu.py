# pc/renderers/cpu.py
from __future__ import annotations
from typing import Any, Sequence, Tuple
import numpy as np
import cv2

from . import register_renderer  # comes from pc/renderers/__init__.py


GridLine = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
BoxSpec  = Tuple[Tuple[float, float, float, float, float, float], Tuple[int, int, int]]


class CPURenderer:
    """
    Drop-in CPU backend that reproduces your current SimCamera drawing with OpenCV.
    It fills the provided BGR frame using the camera pose (rvec/tvec).
    """

    def __init__(self, *, context: Any):
        # Expect the minimal context SimCamera provides
        required = ("width", "height", "grid_lines", "boxes", "intrinsics")
        missing = [n for n in required if not hasattr(context, n)]
        if missing:
            raise AttributeError("CPU renderer context missing: " + ", ".join(missing))

        self.W: int = int(getattr(context, "width"))
        self.H: int = int(getattr(context, "height"))
        self.grid_lines: Sequence[GridLine] = tuple(getattr(context, "grid_lines"))
        self.boxes: Sequence[BoxSpec] = tuple(getattr(context, "boxes"))
        self.K: np.ndarray = np.array(getattr(context, "intrinsics"), dtype=np.float32)

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


# Register backend under the name "cpu"
register_renderer("cpu", lambda **kw: CPURenderer(**kw))
__all__ = ["CPURenderer"]
