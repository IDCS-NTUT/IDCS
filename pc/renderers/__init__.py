# pc/renderers/__init__.py
from __future__ import annotations
from typing import Any, Callable, Dict, Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Renderer(Protocol):
    def render(self, frame: np.ndarray, /, *, rvec: np.ndarray, tvec: np.ndarray) -> None: ...

RendererFactory = Callable[..., Renderer]
_RENDERERS: Dict[str, RendererFactory] = {}

def register_renderer(name: str, factory: RendererFactory) -> None:
    if name in _RENDERERS:
        raise ValueError(f"renderer '{name}' is already registered")
    _RENDERERS[name] = factory

def get_renderer(name: str = "gl", /, **kwargs: Any) -> Renderer:
    try:
        factory = _RENDERERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_RENDERERS)) or "<none>"
        raise KeyError(f"Unknown renderer '{name}'. Available: {available}") from exc
    return factory(**kwargs)

# --- Register available backends ---
# GL is required for your current setup
from .gl import GLRenderer  # registers "gl"

# CPU backend is optional; leave this as a try/except if you add it back later
try:
    from .cpu import CPURenderer  # registers "cpu"
except Exception:
    pass

__all__ = ["Renderer", "RendererFactory", "register_renderer", "get_renderer"]
