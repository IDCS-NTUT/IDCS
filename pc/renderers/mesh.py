"""Mesh loading utilities for renderers (ASCII-safe).

Provides a small wrapper around ``trimesh`` to load OBJ/GLTF assets, ensure
triangulation, recenter to origin, normalize scale, and guarantee normals. The
results are cached to avoid repeated CPU¡÷GPU uploads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import trimesh
except Exception:  # pragma: no cover - optional dependency
    trimesh = None


@dataclass(frozen=True)
class MeshBuffers:
    vertices: np.ndarray  # (N,3) float32
    normals: np.ndarray   # (N,3) float32
    uvs: Optional[np.ndarray]  # (N,2) float32 or None
    indices: np.ndarray   # (M,) uint32 (tri list)


@lru_cache(maxsize=32)
def load_mesh(path: str) -> MeshBuffers:
    """Load and preprocess a mesh file using ``trimesh``.

    This function opens the asset in binary mode and asks trimesh to interpret
    it according to the file extension. Opening as binary avoids any implicit
    text-decoding that can fail on non-UTF8 bytes (binary STL or malformed OBJ).
    """
    if trimesh is None:
        raise ImportError("trimesh is required for mesh loading; install with the pc extras")

    p = Path(path)
    suffix = p.suffix.lower().lstrip(".")
    if not suffix:
        raise ValueError(f"cannot determine file type for mesh '{path}'")

    # Always load from a binary file handle and provide file_type explicitly.
    # This avoids unicode/text decoding issues when trimesh tries to read text.
    try:
        with open(path, "rb") as fh:
            mesh = trimesh.load(fh, file_type=suffix)
    except Exception as exc:
        # Provide helpful context for failures.
        raise RuntimeError(f"failed to load mesh '{path}': {exc}") from exc

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    mesh.rezero()
    if mesh.scale > 0:
        mesh.apply_scale(1.0 / mesh.scale)

    mesh.fix_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if not np.all(np.isfinite(normals)) or normals.shape != vertices.shape:
        normals = np.zeros_like(vertices, dtype=np.float32)

    uvs = None
    if mesh.visual and getattr(mesh.visual, "uv", None) is not None:
        uv = np.asarray(mesh.visual.uv, dtype=np.float32)
        if uv.size >= 2:
            uvs = uv[:, :2].copy()

    indices = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)

    return MeshBuffers(vertices=vertices, normals=normals, uvs=uvs, indices=indices)


__all__ = ["MeshBuffers", "load_mesh"]
