"""Shader source helpers for the OpenGL renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShaderSource:
    """Container for a vertex/fragment shader pair."""

    vertex: str
    fragment: str


_REPO_ROOT = Path(__file__).resolve().parents[4]
_ASSETS_DIR = _REPO_ROOT / "assets" / "shaders"


def _read_shader_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"shader '{path.name}' was not found in '{path.parent}'")
    return path.read_text(encoding="utf8")


def load_shader(name: str) -> ShaderSource:
    """Load ``name`` from the shared ``assets/shaders`` directory."""

    vertex_path = _ASSETS_DIR / f"{name}.vert"
    fragment_path = _ASSETS_DIR / f"{name}.frag"
    vertex_source = _read_shader_source(vertex_path)
    fragment_source = _read_shader_source(fragment_path)
    return ShaderSource(vertex=vertex_source, fragment=fragment_source)


__all__ = ["ShaderSource", "load_shader"]
