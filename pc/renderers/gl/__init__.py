"""OpenGL renderer package."""

from __future__ import annotations

from typing import Any

from .. import register_renderer
from .context import ContextConfig, GLContext, GLContextError, create_gl_context
from .renderer import GLRenderer


def _factory(**kwargs: Any) -> GLRenderer:
    return GLRenderer(**kwargs)


register_renderer("gl", _factory)

__all__ = [
    "GLRenderer",
    "ContextConfig",
    "GLContext",
    "GLContextError",
    "create_gl_context",
]

