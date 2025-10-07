"""Shader source helpers for the OpenGL renderer."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class ShaderSource:
    """Container for a vertex/fragment shader pair."""

    vertex: str
    fragment: str


def load_shader(name: str) -> ShaderSource:
    """Load ``name`` from the package shader directory."""

    package = __name__
    vertex_path = resources.files(package).joinpath(f"{name}.vert")
    fragment_path = resources.files(package).joinpath(f"{name}.frag")
    vertex_source = vertex_path.read_text(encoding="utf8")
    fragment_source = fragment_path.read_text(encoding="utf8")
    return ShaderSource(vertex=vertex_source, fragment=fragment_source)


__all__ = ["ShaderSource", "load_shader"]

