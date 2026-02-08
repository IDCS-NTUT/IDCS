"""Mesh loading utilities for 3D renderers.

This module provides a simple interface for loading OBJ and GLTF mesh files
using trimesh, with automatic triangulation, normalization, and normal
computation. Loaded meshes are cached per file path for efficient reuse.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None


# Global cache: filepath -> (vertices, normals, indices)
_MESH_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def load_mesh(
    path: str, normalize: bool = True, scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a mesh from an OBJ or GLTF file.
    
    The mesh is triangulated, optionally normalized to fit in a unit cube,
    and normals are computed if not present. Results are cached per file path.
    
    Parameters
    ----------
    path : str
        Path to the mesh file (OBJ, GLTF, GLB, etc.).
    normalize : bool, optional
        If True, center the mesh at origin and scale to fit in a unit cube.
        Default is True.
    scale : float, optional
        If provided, scale the mesh by this factor after normalization.
        Only used if normalize=True.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - vertices: (N, 3) float32 array of vertex positions
        - normals: (N, 3) float32 array of vertex normals
        - indices: (M, 3) uint32 array of triangle indices
    
    Raises
    ------
    RuntimeError
        If trimesh is not available or if mesh loading fails.
    """
    if trimesh is None:
        raise RuntimeError(
            "trimesh is not installed. Install it with: pip install trimesh"
        )

    # Check cache first
    if path in _MESH_CACHE:
        return _MESH_CACHE[path]

    # Load mesh
    try:
        loaded = trimesh.load(path, force="mesh")
    except Exception as exc:
        raise RuntimeError(f"Failed to load mesh from {path}: {exc}") from exc

    # Handle Scene vs Mesh
    if isinstance(loaded, trimesh.Scene):
        # Concatenate all geometries in the scene
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise RuntimeError(f"No geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    # Triangulate if needed
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Loaded object is not a Trimesh: {type(mesh)}")
    
    if mesh.faces.shape[1] != 3:
        try:
            mesh = mesh.triangulate()
        except Exception as exc:
            raise RuntimeError(f"Failed to triangulate mesh: {exc}") from exc

    # Normalize mesh to unit cube centered at origin
    if normalize:
        try:
            # Center at origin
            mesh.rezero()
            # Scale to fit in unit cube (max extent = 1.0)
            current_scale = mesh.scale
            if current_scale > 1e-6:
                mesh.apply_scale(1.0 / current_scale)
            # Apply additional scale if requested
            if scale is not None and scale > 1e-6:
                mesh.apply_scale(scale)
        except Exception as exc:
            raise RuntimeError(f"Failed to normalize mesh: {exc}") from exc

    # Ensure normals are computed
    try:
        if not hasattr(mesh, "vertex_normals") or mesh.vertex_normals is None:
            # No normals exist, compute them
            mesh.compute_vertex_normals()
        # Note: We don't call fix_normals() automatically as it can be expensive
        # for large meshes. Users can call it separately if needed.
    except Exception as exc:
        # If normal computation fails, generate simple normals
        try:
            mesh.vertex_normals = np.zeros_like(mesh.vertices)
        except Exception:
            raise RuntimeError(f"Failed to compute normals: {exc}") from exc

    # Extract data as numpy arrays
    vertices = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)
    indices = mesh.faces.astype(np.uint32)

    # Validate output
    if vertices.shape[1] != 3:
        raise RuntimeError(f"Invalid vertex shape: {vertices.shape}")
    if normals.shape[1] != 3:
        raise RuntimeError(f"Invalid normal shape: {normals.shape}")
    if indices.shape[1] != 3:
        raise RuntimeError(f"Invalid face shape: {indices.shape}")

    # Cache the result
    result = (vertices, normals, indices)
    _MESH_CACHE[path] = result

    return result


def clear_cache() -> None:
    """Clear the mesh cache."""
    global _MESH_CACHE
    _MESH_CACHE.clear()


def get_cache_size() -> int:
    """Return the number of cached meshes."""
    return len(_MESH_CACHE)


__all__ = ["load_mesh", "clear_cache", "get_cache_size"]
