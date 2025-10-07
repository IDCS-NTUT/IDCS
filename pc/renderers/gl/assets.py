"""Mesh utilities for the OpenGL renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency during import
    import moderngl
except ModuleNotFoundError:  # pragma: no cover - allows type checking without GL
    moderngl = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MeshGeometry:
    """Raw vertex data for a mesh."""

    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray | None = None
    mode: str = "triangles"

    def interleaved(self) -> np.ndarray:
        """Return a tightly packed vertex array."""

        count = int(self.positions.shape[0])
        normals = self.normals
        if normals.shape[0] != count:
            normals = np.zeros((count, 3), dtype=np.float32)
        uvs = self.uvs
        if uvs.shape[0] != count:
            uvs = np.zeros((count, 2), dtype=np.float32)
        return np.hstack((self.positions, normals, uvs)).astype(np.float32, copy=False)


class MeshAsset:
    """GPU resources backing a mesh geometry."""

    def __init__(self, ctx: Any, geometry: MeshGeometry) -> None:
        self._ctx = ctx
        self._geometry = geometry
        self._vbo = ctx.buffer(geometry.interleaved().tobytes())
        self._ibo = None
        self._mode = geometry.mode
        if geometry.indices is not None:
            self._ibo = ctx.buffer(np.asarray(geometry.indices, dtype=np.uint32).tobytes())

        if moderngl is not None:
            if geometry.mode == "lines":
                self._mode_enum = getattr(ctx, "LINES", None)
            else:
                self._mode_enum = getattr(ctx, "TRIANGLES", None)
        else:  # pragma: no cover - type checking path
            self._mode_enum = None

        self._vao_cache: Dict[int, Any] = {}

    def vertex_array(self, program: Any) -> Any:
        """Return a VAO configured for ``program``."""

        key = int(getattr(program, "glo", id(program)))
        vao = self._vao_cache.get(key)
        if vao is None:
            vao = self._ctx.vertex_array(
                program,
                [(self._vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
                index_buffer=self._ibo,
            )
            self._vao_cache[key] = vao
        return vao

    def render(self, program: Any) -> None:
        vao = self.vertex_array(program)
        if self._mode_enum is not None:
            vao.render(mode=self._mode_enum)
        else:  # pragma: no cover - fallback path without moderngl constants
            vao.render()

    def release(self) -> None:
        for vao in self._vao_cache.values():
            release = getattr(vao, "release", None)
            if callable(release):
                release()
        self._vao_cache.clear()

        for resource in (self._vbo, self._ibo):
            if resource is not None:
                release = getattr(resource, "release", None)
                if callable(release):
                    release()


class MeshCache:
    """Lazy mesh factory that caches GPU buffers per geometry signature."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._meshes: Dict[Tuple[Any, ...], MeshAsset] = {}

    def get_box(self, half_extents: Iterable[float]) -> MeshAsset:
        dims = tuple(float(abs(v)) for v in half_extents)
        key = ("box",) + tuple(round(v, 5) for v in dims)
        mesh = self._meshes.get(key)
        if mesh is None:
            mesh = MeshAsset(self._ctx, build_box_geometry(dims))
            self._meshes[key] = mesh
        return mesh

    def get_billboard(self, width: float, height: float) -> MeshAsset:
        dims = (float(abs(width)), float(abs(height)))
        key = ("billboard",) + tuple(round(v, 5) for v in dims)
        mesh = self._meshes.get(key)
        if mesh is None:
            mesh = MeshAsset(self._ctx, build_billboard_geometry(*dims))
            self._meshes[key] = mesh
        return mesh

    def release(self) -> None:
        for mesh in self._meshes.values():
            mesh.release()
        self._meshes.clear()


def build_box_geometry(half_extents: Iterable[float]) -> MeshGeometry:
    hx, hy, hz = (float(abs(v)) for v in half_extents)
    positions = np.array(
        [
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
            (-hx, -hy, -hz),
            (-hx, hy, -hz),
            (hx, hy, -hz),
            (hx, -hy, -hz),
            (-hx, hy, -hz),
            (-hx, hy, hz),
            (hx, hy, hz),
            (hx, hy, -hz),
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, -hy, hz),
            (-hx, -hy, hz),
            (-hx, -hy, -hz),
            (-hx, -hy, hz),
            (-hx, hy, hz),
            (-hx, hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (hx, hy, hz),
            (hx, -hy, hz),
        ],
        dtype=np.float32,
    )
    normals = np.array(
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, -1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ],
        dtype=np.float32,
    )
    uvs = np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        * 6,
        dtype=np.float32,
    )
    indices = np.array(
        [
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
            8, 9, 10, 8, 10, 11,
            12, 13, 14, 12, 14, 15,
            16, 17, 18, 16, 18, 19,
            20, 21, 22, 20, 22, 23,
        ],
        dtype=np.uint32,
    )
    return MeshGeometry(positions=positions, normals=normals, uvs=uvs, indices=indices)


def build_billboard_geometry(width: float, height: float) -> MeshGeometry:
    hw = float(width) * 0.5
    hh = float(height) * 0.5
    positions = np.array(
        [
            (-hw, -hh, 0.0),
            (hw, -hh, 0.0),
            (hw, hh, 0.0),
            (-hw, hh, 0.0),
        ],
        dtype=np.float32,
    )
    normals = np.tile(np.array(((0.0, 0.0, 1.0),), dtype=np.float32), (4, 1))
    uvs = np.array(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ],
        dtype=np.float32,
    )
    indices = np.array((0, 1, 2, 0, 2, 3), dtype=np.uint32)
    return MeshGeometry(positions=positions, normals=normals, uvs=uvs, indices=indices)


__all__ = [
    "MeshGeometry",
    "MeshAsset",
    "MeshCache",
    "build_box_geometry",
    "build_billboard_geometry",
]

