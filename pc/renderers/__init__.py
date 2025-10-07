"""Renderer plug-in registry.

The registry keeps renderer construction logic decoupled from :mod:`pc.sim_camera`.
The classic CPU renderer and the new headless OpenGL scaffold are both registered
on import so configuration can switch between them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, runtime_checkable




@runtime_checkable
class Renderer(Protocol):
    def render(self, frame: Any, /, **kwargs: Any) -> None: ...


RendererFactory = Callable[..., Renderer]
_RENDERERS: Dict[str, RendererFactory] = {}


def register_renderer(name: str, factory: RendererFactory) -> None:
    if name in _RENDERERS:
        raise ValueError(f"renderer '{name}' is already registered")
    _RENDERERS[name] = factory


def get_renderer(name: str | None = None, /, **kwargs: Any) -> Renderer:
    resolved = (name or "cpu").strip().lower()
    try:
        factory = _RENDERERS[resolved]
    except KeyError as exc:
        available = ", ".join(sorted(_RENDERERS)) or "<none>"
        raise KeyError(f"Unknown renderer '{resolved}'. Available: {available}") from exc
    return factory(**kwargs)


from .cpu import CPURenderer  # noqa: E402  (registers "cpu")
from .gl import GLRenderer  # noqa: E402  (registers "gl")

__all__ = [
    "Renderer",
    "RendererFactory",
    "register_renderer",
    "get_renderer",
    "CPURenderer",
    "GLRenderer",
]
