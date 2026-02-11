"""Hardcoded OpenGL renderer based on tools/gl_progress.py.

This renderer intentionally ignores the simulator world/config and always
renders the fixed preview scene used by the standalone OpenGL progress tool.
"""

from __future__ import annotations
from PIL import Image
import logging
import math
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from . import register_renderer
from .mesh import load_mesh

try:
    import moderngl
except Exception:  # pragma: no cover - defensive import
    moderngl = None

logger = logging.getLogger(__name__)

_VERT_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_normal;
uniform mat4 MVP;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_normal = in_normal;
    v_pos = in_position;
}
"""


_FRAG_SHADER = """
#version 330
in vec3 v_normal;
in vec3 v_pos;
out vec4 f_color;
uniform float u_grid;
uniform vec3 u_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 ldir = normalize(vec3(0.3, 1.0, 0.2));
    float l = max(dot(n, ldir), 0.0);
    float ambient = 0.35;
    float diffuse = 0.65;
    vec3 base = u_color;
    vec3 col = base * (ambient + diffuse * l);

    float gx = abs(fract(v_pos.x * 0.1) - 0.5);
    float gz = abs(fract(v_pos.z * 0.1) - 0.5);
    float grid = step(0.48, 0.5 - min(gx, gz)) * u_grid;
    col = mix(col, col * 0.5, grid);

    f_color = vec4(col, 1.0);
}
"""


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    nf = 1.0 / (near - far)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) * nf
    m[2, 3] = (2.0 * far * near) * nf
    m[3, 2] = -1.0
    return m


def _look_at(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> np.ndarray:
    eye_v = np.array(eye, dtype=np.float32)
    target_v = np.array(target, dtype=np.float32)
    up_v = np.array(up, dtype=np.float32)

    f = target_v - eye_v
    f /= np.linalg.norm(f) + 1e-12
    u = up_v / (np.linalg.norm(up_v) + 1e-12)
    s = np.cross(f, u)
    s /= np.linalg.norm(s) + 1e-12
    u2 = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u2
    m[2, 0:3] = -f
    t = np.eye(4, dtype=np.float32)
    t[0:3, 3] = -eye_v
    return m @ t


def _build_ground_plane(size: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    half = size * 0.5
    positions = np.array(
        [
            (-half, 0.0, -half),
            (half, 0.0, -half),
            (half, 0.0, half),
            (-half, 0.0, half),
        ],
        dtype=np.float32,
    )
    normals = np.array([(0.0, 1.0, 0.0)] * 4, dtype=np.float32)
    vertices = np.hstack([positions, normals])
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return vertices, indices


def _build_unit_box() -> Tuple[np.ndarray, np.ndarray]:
    positions = [
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, 0.5, -0.5),
        (0.5, -0.5, 0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
    ]
    normals = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
    ]
    vertices = np.hstack(
        [np.array(positions, dtype=np.float32), np.array(normals, dtype=np.float32)]
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
    return vertices, indices


def _load_mesh_buffers(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        fallback = path.with_suffix(".stl")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Mesh asset not found: {path}")

    buffers = load_mesh(str(path))
    vertices = np.hstack([buffers.vertices, buffers.normals]).astype("f4")
    indices = buffers.indices.astype("u4", copy=False)
    return vertices, indices


class OpenGLRenderer:
    """Fixed OpenGL renderer that outputs the gl_progress scene."""

    def __init__(self, *, context: Any) -> None:
        try:
            self.width = int(getattr(context, "width"))
            self.height = int(getattr(context, "height"))
        except AttributeError as exc:  # pragma: no cover - defensive only
            raise AttributeError("SimCamera context must expose width/height") from exc

        self._gl = None
        self._fbo = None
        self._prog = None
        self._ground_vao = None
        self._mesh_vao = None
        self._building_vao = None

        self._proj = None
        self._model_ground = np.eye(4, dtype=np.float32)

        self._orbit_radius = 8.0
        self._orbit_height = 5.0
        self._orbit_speed = 0.35
        self._mesh_distance = 4.0
        self._mesh_scale = 3.0
        self._building_specs = [
            {"base_centre": (0.0, -18.0), "footprint": (8.0, 6.0), "height": 12.0},
            {"base_centre": (-10.0, -26.0), "footprint": (10.0, 8.0), "height": 18.0},
            {"base_centre": (12.0, -28.0), "footprint": (12.0, 7.0), "height": 15.0},
        ]
        self._frame_time = 1.0 / 30.0

        if moderngl is None:
            logger.error("moderngl is not available; OpenGL renderer will fall back to CPU")
            return

        self._gl = self._init_context()
        if self._gl is None:
            logger.error("Failed to create OpenGL context; renderer will fall back to CPU")
            return

        self._fbo = self._gl.simple_framebuffer((self.width, self.height))
        self._prog = self._gl.program(vertex_shader=_VERT_SHADER, fragment_shader=_FRAG_SHADER)

        ground_vertices, ground_indices = _build_ground_plane(1000.0)
        ground_vbo = self._gl.buffer(ground_vertices.tobytes())
        ground_ibo = self._gl.buffer(ground_indices.tobytes())

        assets_root = Path(__file__).resolve().parents[2] / "assets"
        mesh_vertices, mesh_indices = _load_mesh_buffers(assets_root / "drone.stl")
        mesh_vbo = self._gl.buffer(mesh_vertices.tobytes())
        mesh_ibo = self._gl.buffer(mesh_indices.tobytes())

        building_vertices, building_indices = _build_unit_box()
        building_vbo = self._gl.buffer(building_vertices.tobytes())
        building_ibo = self._gl.buffer(building_indices.tobytes())

        self._ground_vao = self._gl.vertex_array(
            self._prog,
            [(ground_vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=ground_ibo,
        )
        self._mesh_vao = self._gl.vertex_array(
            self._prog,
            [(mesh_vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=mesh_ibo,
        )
        self._building_vao = self._gl.vertex_array(
            self._prog,
            [(building_vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=building_ibo,
        )

        self._proj = _perspective(60.0, self.width / self.height, 0.1, 100.0)

    def _init_context(self):
        attempts = []
        if hasattr(moderngl, "create_standalone_context"):
            attempts.append(("moderngl-egl", lambda: moderngl.create_standalone_context(backend="egl")))
            attempts.append(("moderngl-auto", lambda: moderngl.create_standalone_context()))

        for name, factory in attempts:
            try:
                ctx = factory()
            except Exception as exc:  # pragma: no cover - platform specific
                logger.warning("OpenGL context creation failed (%s): %s", name, exc)
                continue
            logger.info("OpenGL context created via %s", name)
            return ctx
        return None

    def render(self, frame: np.ndarray, /, *, frame_id: Optional[int] = None) -> None:
        if frame_id is None:
            frame_id = 0

        if self._gl is None or self._fbo is None or self._prog is None:
            frame[:] = np.full((self.height, self.width, 3), 120, dtype=np.uint8)
            return

        elapsed = float(frame_id) * self._frame_time
        angle = elapsed * self._orbit_speed
        eye = (
            math.cos(angle) * self._orbit_radius,
            self._orbit_height,
            math.sin(angle) * self._orbit_radius,
        )
        target = (0.0, 2.0, 0.0)
        view = _look_at(eye, target, (0.0, 1.0, 0.0))

        self._fbo.use()
        self._gl.viewport = (0, 0, self.width, self.height)
        self._gl.enable(moderngl.DEPTH_TEST)

        mvp_ground = self._proj @ view @ self._model_ground
        self._prog["MVP"].write(mvp_ground.T.astype("f4").tobytes())
        self._prog["u_grid"].value = 1.0
        self._prog["u_color"].value = (0.45, 0.7, 0.85)

        self._fbo.clear(1.0, 1.0, 1.0, 1.0, depth=1.0)
        self._ground_vao.render()

        self._prog["u_grid"].value = 0.0
        self._prog["u_color"].value = (0.7, 0.7, 0.82)
        for spec in self._building_specs:
            base_x, base_z = spec["base_centre"]
            width, depth = spec["footprint"]
            height = spec["height"]
            model_building = np.eye(4, dtype=np.float32)
            model_building[0, 0] = float(width)
            model_building[1, 1] = float(height)
            model_building[2, 2] = float(depth)
            model_building[0:3, 3] = (
                float(base_x),
                float(height) * 0.5,
                float(base_z),
            )
            mvp_building = self._proj @ view @ model_building
            self._prog["MVP"].write(mvp_building.T.astype("f4").tobytes())
            self._building_vao.render()

        forward = np.array(target, dtype=np.float32) - np.array(eye, dtype=np.float32)
        forward /= np.linalg.norm(forward) + 1e-12
        mesh_pos = np.array(eye, dtype=np.float32) + forward * self._mesh_distance
        model_mesh = np.eye(4, dtype=np.float32)
        model_mesh[0, 0] = self._mesh_scale
        model_mesh[1, 1] = self._mesh_scale
        model_mesh[2, 2] = self._mesh_scale
        model_mesh[0:3, 3] = mesh_pos
        mvp_mesh = self._proj @ view @ model_mesh
        self._prog["MVP"].write(mvp_mesh.T.astype("f4").tobytes())
        self._prog["u_color"].value = (0.6, 0.7, 0.8)
        self._mesh_vao.render()

        data = self._fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8)
        img = img.reshape((self.height, self.width, 3))
        img = img[::-1, :, ::-1]
        frame[:] = img


register_renderer("opengl", lambda **kwargs: OpenGLRenderer(**kwargs))

__all__ = ["OpenGLRenderer"]
