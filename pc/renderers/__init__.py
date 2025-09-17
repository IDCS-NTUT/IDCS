"""Renderer backends for the simulation camera."""

from __future__ import annotations

from typing import Any, Callable, Dict, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Renderer(Protocol):
    """Protocol describing a renderer backend."""

    def render(
        self,
        frame: np.ndarray,
        /,
        *,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> None:
        """Fill ``frame`` with the rendered scene for the supplied pose."""


RendererFactory = Callable[..., Renderer]

_RENDERERS: Dict[str, RendererFactory] = {}


def register_renderer(name: str, factory: RendererFactory) -> None:
    """Register a renderer backend under ``name``."""

    if name in _RENDERERS:
        raise ValueError(f"renderer '{name}' is already registered")
    _RENDERERS[name] = factory


def get_renderer(name: str = "cpu", /, **kwargs: Any) -> Renderer:
    """Return an instance of the requested renderer backend.

    Parameters
    ----------
    name:
        Identifier of the renderer backend. Defaults to ``"cpu"``.
    **kwargs:
        Keyword arguments forwarded to the backend factory.
    """

    try:
        factory = _RENDERERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        available = ", ".join(sorted(_RENDERERS)) or "<none>"
        raise KeyError(
            f"Unknown renderer '{name}'. Available renderers: {available}"
        ) from exc
    return factory(**kwargs)


__all__ = [
    "Renderer",
    "RendererFactory",
    "register_renderer",
    "get_renderer",
]


from .cpu import CPURenderer  # noqa: E402  (import side-effects register backend)

__all__.append("CPURenderer")

try:  # noqa: E402 - optional backend import
    from .gl import GLRenderer
except Exception:  # pragma: no cover - optional dependency may be missing
    GLRenderer = None  # type: ignore[assignment]
else:
    __all__.append("GLRenderer")

