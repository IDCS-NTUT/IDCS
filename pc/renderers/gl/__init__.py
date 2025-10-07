"""Minimal OpenGL renderer scaffold.

Step 2 of the renderer rebuild focuses on standing up a headless OpenGL context
so future tasks can focus on mesh loading and actual rendering.  The renderer
exposes the same ``render`` signature as the CPU backend so ``SimCamera`` can
switch between them without changing call sites.  For now the frame content is a
simple placeholder gradient so the GPU path can be verified while the true draw
pipeline is implemented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .. import Renderer, register_renderer
from .context import HeadlessContextError, HeadlessGLContext, create_headless_context

__all__ = ["GLRenderer"]


_LOGGER = logging.getLogger(__name__)


@dataclass
class _RendererState:
    context: HeadlessGLContext
    frame_shape: tuple[int, int, int]
    clear_colour: tuple[int, int, int]
    last_frame_id: Optional[int] = None


class GLRenderer:
    """Skeleton renderer that only sets up the GL context for now."""

    def __init__(
        self,
        *,
        context: Any,
        backend: str = "auto",
        msaa_samples: int = 0,
        lighting: Optional[Dict[str, Any]] = None,
        laser: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            gl_context = create_headless_context(
                context.width,
                context.height,
                backend=backend,
                msaa_samples=msaa_samples,
                logger=_LOGGER,
            )
        except HeadlessContextError as exc:
            raise RuntimeError(f"OpenGL initialisation failed: {exc}") from exc

        self._state = _RendererState(
            context=gl_context,
            frame_shape=(context.height, context.width, 3),
            clear_colour=(20, 20, 20),
        )

        self._lighting = lighting or {}
        self._laser = laser or {}

        gl_context.make_current()
        _LOGGER.debug(
            "GL renderer initialised (%sx%s, backend=%s, api=%s)",
            context.width,
            context.height,
            gl_context.backend,
            gl_context.api,
        )

    # ------------------------------------------------------------------ public
    def render(self, frame: np.ndarray, /, *, frame_id: Optional[int] = None) -> None:
        """Populate ``frame`` with a placeholder gradient.

        The draw path intentionally remains CPU based until later tasks add GL
        drawing and read-back.  Keeping a predictable output lets the streaming
        stack continue to validate frame cadence and downstream plumbing while
        the GPU implementation is built.
        """

        if frame.shape != self._state.frame_shape:
            raise ValueError(
                f"Frame shape mismatch: expected {self._state.frame_shape}, got {frame.shape}"
            )

        if frame_id is None:
            frame_id = 0
        self._state.last_frame_id = frame_id

        height, width, _ = self._state.frame_shape
        grad_x = np.linspace(0.0, 35.0, width, dtype=np.float32)
        grad_y = np.linspace(0.0, 35.0, height, dtype=np.float32)
        gradient = grad_y[:, None] + grad_x[None, :]
        base = np.empty_like(frame, dtype=np.float32)
        base[...] = np.array(self._state.clear_colour, dtype=np.float32)
        blended = base + gradient[..., None]
        np.clip(blended, 0.0, 255.0, out=blended)
        frame[:] = blended.astype(np.uint8)

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._state.context.close()


def _factory(**kwargs: Any) -> Renderer:
    return GLRenderer(**kwargs)


register_renderer("gl", _factory)
