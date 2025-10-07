"""Renderer plug-in registry.

Only the minimal CPU renderer is currently available.  The registry helpers are
kept so that new back-ends can be added easily during the upcoming renderer
rebuild.
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
