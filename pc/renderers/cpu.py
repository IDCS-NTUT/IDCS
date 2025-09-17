"""CPU renderer backend implementation."""

from __future__ import annotations

from typing import Any, Iterable, Sequence, Tuple

import cv2
import numpy as np

from . import register_renderer


BoxSpec = Tuple[Tuple[float, float, float, float, float, float], Tuple[int, int, int]]
GridLine = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


class CPURenderer:
    """Renderer that rasterises the simulation scene on the CPU."""

    def __init__(self, *, context: Any) -> None:
        """Initialise the renderer with the shared simulation context."""

        required = ("width", "height", "proj_masked", "grid_lines", "boxes")
        missing = [name for name in required if not hasattr(context, name)]
        if missing:
            raise AttributeError(
                "CPU renderer context is missing required attributes: "
                + ", ".join(sorted(missing))
            )

        self._width = int(getattr(context, "width"))
        self._height = int(getattr(context, "height"))
        self._proj_masked = getattr(context, "proj_masked")
        self._grid_lines: Sequence[GridLine] = tuple(getattr(context, "grid_lines"))
        self._boxes: Sequence[BoxSpec] = tuple(getattr(context, "boxes"))

    def render(
        self,
        frame: np.ndarray,
        /,
        *,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> None:
        """Render the ground plane and scene geometry into ``frame``."""

        self._draw_ground(frame, rvec, tvec)
        self._draw_boxes(frame, rvec, tvec)

    def _draw_ground(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """Render the sky gradient and ground grid."""

        frame[:] = (180, 180, 210)
        cv2.rectangle(
            frame,
            (0, self._height // 2),
            (self._width, self._height),
            (170, 190, 170),
            -1,
        )

        grid_color = (150, 150, 150)
        for (x1, y1, z1), (x2, y2, z2) in self._grid_lines:
            segment = np.array([[x1, y1, z1], [x2, y2, z2]], dtype=np.float32)
            pts, _ = self._proj_masked(segment, rvec, tvec)
            if pts is None:
                continue

            p0, p1 = pts
            if not (
                np.isfinite(p0).all()
                and np.isfinite(p1).all()
            ):
                continue

            cv2.line(
                frame,
                tuple(np.round(p0).astype(int)),
                tuple(np.round(p1).astype(int)),
                grid_color,
                1,
                cv2.LINE_AA,
            )

    def _draw_boxes(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """Render the collection of simple box obstacles."""

        edges: Iterable[Tuple[int, int]] = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )

        for (x, y, z, w, d, h), color in self._boxes:
            corners = np.array(
                [
                    [x - 0.5 * w, y, z - 0.5 * d],
                    [x + 0.5 * w, y, z - 0.5 * d],
                    [x + 0.5 * w, y, z + 0.5 * d],
                    [x - 0.5 * w, y, z + 0.5 * d],
                    [x - 0.5 * w, y + h, z - 0.5 * d],
                    [x + 0.5 * w, y + h, z - 0.5 * d],
                    [x + 0.5 * w, y + h, z + 0.5 * d],
                    [x - 0.5 * w, y + h, z + 0.5 * d],
                ],
                dtype=np.float32,
            )

            pts, mask = self._proj_masked(corners, rvec, tvec)
            if pts is None:
                continue

            for idx_a, idx_b in edges:
                pa, pb = pts[idx_a], pts[idx_b]
                if not (
                    mask[idx_a]
                    and mask[idx_b]
                    and np.isfinite(pa).all()
                    and np.isfinite(pb).all()
                ):
                    continue

                cv2.line(
                    frame,
                    tuple(pa.astype(int)),
                    tuple(pb.astype(int)),
                    color,
                    2,
                    cv2.LINE_AA,
                )


register_renderer("cpu", CPURenderer)


__all__ = ["CPURenderer"]

