"""Minimal CPU renderer used by :mod:`pc.sim_camera`.

This module deliberately keeps the renderer extremely small so that future
work can rebuild the scene graph from scratch.  The renderer only generates a
simple background along with a moving marker to make it obvious that new
frames are produced.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import cv2
import numpy as np

from . import register_renderer


class CPURenderer:
    """Trivial placeholder renderer.

    The renderer fills the frame with a flat background colour, draws a simple
    crosshair in the middle of the screen, and animates a small dot so that the
    output visibly changes from frame to frame.  This keeps the simulation
    source usable while the real rendering stack is rebuilt.
    """

    def __init__(self, *, context: Any) -> None:
        try:
            self.width = int(getattr(context, "width"))
            self.height = int(getattr(context, "height"))
        except AttributeError as exc:  # pragma: no cover - defensive only
            raise AttributeError("SimCamera context must expose width/height") from exc

        self._crosshair_radius = max(2, min(self.width, self.height) // 32)
        self._dot_radius = max(4, min(self.width, self.height) // 24)

    def render(self, frame: np.ndarray, /, *, frame_id: Optional[int] = None) -> None:
        """Render a single frame into ``frame``.

        Parameters
        ----------
        frame:
            Destination image buffer in BGR format.
        frame_id:
            Optional monotonically increasing identifier used to animate the
            moving dot.  When omitted the renderer falls back to ``0``.
        """

        if frame_id is None:
            frame_id = 0

        frame[:] = (48, 60, 76)

        # Diagonal gradient so the background isn't completely flat.
        grad = np.linspace(0.0, 32.0, self.width, dtype=np.float32)
        channel = frame[:, :, 2].astype(np.float32)
        channel = np.clip(channel + grad, 0.0, 255.0)
        frame[:, :, 2] = channel.astype(np.uint8)

        centre = (self.width // 2, self.height // 2)
        cv2.drawMarker(
            frame,
            centre,
            (220, 220, 220),
            markerType=cv2.MARKER_CROSS,
            markerSize=self._crosshair_radius * 2,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        # Animate a small dot around the crosshair using a slow circular motion.
        angle = (frame_id % 360) * math.pi / 180.0
        orbit = min(self.width, self.height) * 0.25
        offset = (int(math.cos(angle) * orbit), int(math.sin(angle) * orbit))
        dot_pos = (centre[0] + offset[0], centre[1] + offset[1])
        cv2.circle(frame, dot_pos, self._dot_radius, (64, 180, 250), -1, cv2.LINE_AA)

        # Subtle border to make the frame edges easy to see.
        cv2.rectangle(frame, (0, 0), (self.width - 1, self.height - 1), (90, 110, 130), 1)


register_renderer("cpu", lambda **kwargs: CPURenderer(**kwargs))

__all__ = ["CPURenderer"]
