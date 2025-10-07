"""OpenGL renderer package placeholder."""

from __future__ import annotations

from typing import Any, Mapping

from .. import register_renderer
from .context import ContextConfig, GLContext, GLContextError, create_gl_context


class GLRenderer:
    """Placeholder OpenGL renderer stub.

    The actual implementation will be provided in subsequent steps.  The class
    is registered with the renderer plug-in system so that configuration wiring
    can be exercised ahead of the full renderer bring-up.
    """

    def __init__(
        self,
        *,
        context: GLContext | Any | None = None,
        context_config: ContextConfig | Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        if context is None:
            try:
                self._context = create_gl_context(context_config)
            except GLContextError as exc:  # pragma: no cover - passthrough
                raise RuntimeError("Unable to create OpenGL context") from exc
        else:
            self._context = context

    def render(self, frame: Any, /, **_: Any) -> None:  # pragma: no cover - stub
        raise NotImplementedError("OpenGL renderer is not implemented yet")


register_renderer("gl", lambda **kwargs: GLRenderer(**kwargs))

__all__ = [
    "GLRenderer",
    "ContextConfig",
    "GLContext",
    "GLContextError",
    "create_gl_context",
]
