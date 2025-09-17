"""CPU renderer backend placeholder."""

from __future__ import annotations

import numpy as np

from . import register_renderer


class CPURenderer:
    """Stub renderer that will host the CPU drawing routines."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - trivial init
        """Initialise the CPU renderer backend."""

    def next_frame(self) -> tuple[bool, np.ndarray]:
        """Produce the next frame.

        The concrete implementation will be populated once the drawing
        routines are migrated from :mod:`pc.sim_camera`.
        """

        raise NotImplementedError("CPU renderer not yet implemented")


register_renderer("cpu", CPURenderer)


__all__ = ["CPURenderer"]

