"""Mesh loading utilities for renderers (ASCII-safe).

Provides a small wrapper around ``trimesh`` to load OBJ/GLTF assets, ensure
triangulation, recenter to origin, normalize scale, and guarantee normals. The
results are cached to avoid repeated CPU¡÷GPU uploads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
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
    """Load and preprocess a mesh file using ``trimesh``."""

    if trimesh is None:
        raise ImportError("trimesh is required for mesh loading; install with the pc extras")

    mesh = trimesh.load(path)

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
