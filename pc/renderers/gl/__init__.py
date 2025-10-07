"""OpenGL renderer scaffold with resource loading.

The renderer now initialises a headless OpenGL context, inspects the device,
and preloads meshes/textures from a scene manifest so the upcoming draw
pipeline can focus on camera projection and shading.  Frame output remains a
deterministic gradient placeholder while the GPU read-back path is wired up in
later steps, keeping SimCamera compatible with the legacy CPU renderer API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .. import Renderer, register_renderer
from .assets import AssetStore
from .context import HeadlessContextError, HeadlessGLContext, create_headless_context
from .gpu import GLBindings, GL_RENDERER, GL_VENDOR, GL_VERSION

__all__ = ["GLRenderer"]


_LOGGER = logging.getLogger(__name__)


@dataclass
class _RendererState:
    context: HeadlessGLContext
    gl: GLBindings
    assets: Optional[AssetStore]
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
        assets: Optional[Dict[str, Any]] = None,
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

        self._lighting = lighting or {}
        self._laser = laser or {}

        gl_context.make_current()
        gl = GLBindings(gl_context, logger=_LOGGER)
        vendor = gl.get_string(GL_VENDOR) or "unknown"
        renderer_name = gl.get_string(GL_RENDERER) or "unknown"
        version = gl.get_string(GL_VERSION) or "unknown"

        assets_cfg = assets if isinstance(assets, dict) else None
        asset_store: Optional[AssetStore]
        try:
            asset_store = AssetStore(gl, assets_cfg, logger=_LOGGER)
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("Asset initialisation failed: %s", exc)
            asset_store = None

        self._state = _RendererState(
            context=gl_context,
            gl=gl,
            assets=asset_store,
            frame_shape=(context.height, context.width, 3),
            clear_colour=(20, 20, 20),
        )
        self._gl = gl
        self._assets = asset_store

        _LOGGER.debug(
            "GL renderer initialised (%sx%s, backend=%s, api=%s)",
            context.width,
            context.height,
            gl_context.backend,
            gl_context.api,
        )
        _LOGGER.info("GL device: %s (vendor=%s, version=%s)", renderer_name, vendor, version)

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
        if self._assets is not None:
            try:
                self._assets.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            self._assets = None
        self._state.context.close()


def _factory(**kwargs: Any) -> Renderer:
    return GLRenderer(**kwargs)


register_renderer("gl", _factory)
