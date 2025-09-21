"""Minimal simulation frame generator.

The previous simulation camera provided a fairly involved scene description and
multiple renderer back-ends.  Those pieces are being rebuilt, so the camera now
serves purely as a lightweight source that returns placeholder frames from the
CPU renderer.  The public ``next_frame`` API is preserved so the rest of the
streaming pipeline keeps working while the renderer stack is redesigned.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .renderers import get_renderer


class SimCamera:
    """Tiny frame generator used while the real renderer is rebuilt."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        *,
        renderer_name: str | None = None,
        renderer_opts: Dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._frame_id = 0

        opts = renderer_opts or {}
        self._renderer = get_renderer(renderer_name, context=self, **opts)

    def next_frame(self) -> Tuple[bool, np.ndarray]:
        """Return the next simulated frame.

        The method maintains a monotonically increasing frame identifier so the
        renderer can animate simple placeholder elements.  A fresh NumPy buffer
        is allocated for each call to keep the implementation straightforward.
        """

        self._frame_id += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._renderer.render(frame, frame_id=self._frame_id)
        return True, frame
