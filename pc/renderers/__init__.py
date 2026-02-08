"""Renderer plug-in registry.

Only the minimal CPU renderer is currently available.  The registry helpers are
kept so that new back-ends can be added easily during the upcoming renderer
rebuild.
"""

from __future__ import annotations

import logging
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
    logger = logging.getLogger(__name__)
    resolved = (name or "cpu").strip().lower()
    try:
        factory = _RENDERERS[resolved]
    except KeyError:
        # Attempt to import a renderer module lazily (e.g. pc.renderers.opengl)
        try:
            import importlib

            importlib.import_module(f"pc.renderers.{resolved}")
        except Exception:
            available = ", ".join(sorted(_RENDERERS)) or "<none>"
            raise KeyError(f"Unknown renderer '{resolved}'. Available: {available}")

        try:
            factory = _RENDERERS[resolved]
        except KeyError as exc:
            available = ", ".join(sorted(_RENDERERS)) or "<none>"
            raise KeyError(f"Unknown renderer '{resolved}'. Available: {available}") from exc

    try:
        return factory(**kwargs)
    except Exception:
        if resolved != "cpu":
            logger.warning("Renderer '%s' failed to initialize; falling back to CPU", resolved)
            cpu_factory = _RENDERERS.get("cpu")
            if cpu_factory is not None:
                return cpu_factory(**kwargs)
        raise


from .cpu import CPURenderer  # noqa: E402  (registers "cpu")

__all__ = ["Renderer", "RendererFactory", "register_renderer", "get_renderer", "CPURenderer"]
