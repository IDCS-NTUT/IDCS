"""OpenGL renderer that mirrors the SimCamera world description."""

from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np

from . import register_renderer
from ._common import NEAR_CLIP, build_camera, projection_matrix, view_matrix
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
    """OpenGL renderer that consumes the SimCamera world description."""

    def __init__(self, *, context: Any) -> None:
        try:
            self.width = int(getattr(context, "width"))
            self.height = int(getattr(context, "height"))
        except AttributeError as exc:  # pragma: no cover - defensive only
            raise AttributeError("SimCamera context must expose width/height") from exc

        self._context = context
        self._gl = None
        self._fbo = None
        self._prog = None
        self._ground_vao = None
        self._box_vao = None
        self._mesh_cache: dict[str, dict[str, Any]] = {}

        self._proj = None
        self._model_ground = np.eye(4, dtype=np.float32)

        self._orbit_radius = 8.0
        self._orbit_height = 5.0
        self._orbit_speed = 0.35
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

        building_vertices, building_indices = _build_unit_box()
        building_vbo = self._gl.buffer(building_vertices.tobytes())
        building_ibo = self._gl.buffer(building_indices.tobytes())

        self._ground_vao = self._gl.vertex_array(
            self._prog,
            [(ground_vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=ground_ibo,
        )
        self._box_vao = self._gl.vertex_array(
            self._prog,
            [(building_vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=building_ibo,
        )
        self._proj = projection_matrix(60.0, self.width / self.height, NEAR_CLIP, 100.0)

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

    def _fetch_world(self, frame_id: int) -> Optional[dict[str, Any]]:
        describe = getattr(self._context, "describe_world", None)
        if not callable(describe):
            return None
        try:
            world = describe(frame_id)
        except Exception:
            return None
        if not isinstance(world, dict):
            return None
        return world

    def _iter_objects(self, world: Optional[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        if world is None:
            return ()
        objects = world.get("objects", ())
        if isinstance(objects, dict):
            return (objects,)
        if isinstance(objects, Iterable):
            return objects
        return ()

    def _color_to_vec(self, color: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if color is None:
            return default
        try:
            values = np.asarray(color, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return default
        if values.size < 3:
            return default
        rgb = np.clip(values[:3] / 255.0, 0.0, 1.0)
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))

    def _model_matrix(
        self,
        centre: Tuple[float, float, float],
        scale: Tuple[float, float, float],
        rotation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        model = np.eye(4, dtype=np.float32)
        if rotation is not None:
            if rotation.shape == (3, 3):
                model[0:3, 0:3] = rotation.astype(np.float32)
            elif rotation.shape == (4, 4):
                model[:, :] = rotation.astype(np.float32)
        scale_vec = np.array(scale, dtype=np.float32)
        model[0:3, 0:3] = model[0:3, 0:3] @ np.diag(scale_vec)
        model[0:3, 3] = np.array(centre, dtype=np.float32)
        return model


    def _resolve_asset_path(self, asset: str) -> Path:
        assets_root = Path(__file__).resolve().parents[2] / "assets"
        asset_path = Path(asset)
        if not asset_path.is_absolute():
            if asset_path.parts and asset_path.parts[0] == "assets":
                asset_path = assets_root.joinpath(*asset_path.parts[1:])
            else:
                asset_path = assets_root / asset_path
        return asset_path

    def _get_mesh_entry(self, asset: str) -> Optional[dict[str, Any]]:
        if self._gl is None or self._prog is None:
            return None
        asset_path = self._resolve_asset_path(asset)
        key = str(asset_path)
        entry = self._mesh_cache.get(key)
        if entry is not None:
            return entry

        try:
            mesh_vertices, mesh_indices = _load_mesh_buffers(asset_path)
        except FileNotFoundError as exc:
            logger.warning("Mesh asset not found: %s", exc)
            return None

        mesh_vbo = self._gl.buffer(mesh_vertices.tobytes())
        mesh_ibo = self._gl.buffer(mesh_indices.tobytes())
        mesh_vao = self._gl.vertex_array(
            self._prog,
            [(mesh_vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=mesh_ibo,
        )
        entry = {"vao": mesh_vao, "vbo": mesh_vbo, "ibo": mesh_ibo}
        self._mesh_cache[key] = entry
        return entry

    def render(self, frame: np.ndarray, /, *, frame_id: Optional[int] = None) -> None:
        if frame_id is None:
            frame_id = 0

        if self._gl is None or self._fbo is None or self._prog is None:
            frame[:] = np.full((self.height, self.width, 3), 120, dtype=np.uint8)
            return

        world = self._fetch_world(frame_id)
        camera = None
        if world is not None:
            camera_state = world.get("camera")
            if isinstance(camera_state, dict):
                camera = build_camera(
                    camera_state,
                    context=self._context,
                    width=self.width,
                    height=self.height,
                )

        if camera is None:
            elapsed = float(frame_id) * self._frame_time
            angle = elapsed * self._orbit_speed
            eye = (
                math.cos(angle) * self._orbit_radius,
                self._orbit_height,
                math.sin(angle) * self._orbit_radius,
            )
            target = (0.0, 2.0, 0.0)
            forward = np.asarray(target, dtype=np.float32) - np.asarray(eye, dtype=np.float32)
            camera = {
                "position": np.asarray(eye, dtype=np.float32),
                "forward": forward,
                "up": np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
                "fov_y": 60.0,
                "aspect": float(self.width) / float(self.height),
            }

        view = view_matrix(camera["position"], camera["forward"], camera["up"])
        self._proj = projection_matrix(camera["fov_y"], camera["aspect"], NEAR_CLIP, 100.0)

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

        for obj in self._iter_objects(world):
            if not isinstance(obj, dict):
                continue
            obj_type = obj.get("type")
            if obj_type == "building":
                if self._box_vao is None:
                    continue
                base = obj.get("base_centre")
                footprint = obj.get("footprint")
                height = obj.get("height")
                if base is None or footprint is None or height is None:
                    continue
                try:
                    base_vals = np.asarray(base, dtype=np.float32).reshape(-1)
                    foot_vals = np.asarray(footprint, dtype=np.float32).reshape(-1)
                    height_val = float(height)
                except (TypeError, ValueError):
                    continue
                if base_vals.size < 2 or foot_vals.size < 2:
                    continue
                if not math.isfinite(height_val) or height_val <= 0.0:
                    continue
                centre = (float(base_vals[0]), float(height_val) * 0.5, float(base_vals[1]))
                scale = (float(abs(foot_vals[0])), float(height_val), float(abs(foot_vals[1])))
                model = self._model_matrix(centre, scale)
                mvp = self._proj @ view @ model
                self._prog["MVP"].write(mvp.T.astype("f4").tobytes())
                self._prog["u_color"].value = self._color_to_vec(
                    obj.get("color") or obj.get("colour"),
                    (0.7, 0.7, 0.82),
                )
                self._box_vao.render()
            elif obj_type == "cube":
                if self._box_vao is None:
                    continue
                centre = obj.get("centre") or obj.get("center") or (0.0, 0.0, 0.0)
                half = obj.get("half_extents") or (0.5, 0.5, 0.5)
                try:
                    centre_vals = np.asarray(centre, dtype=np.float32).reshape(-1)
                    half_vals = np.asarray(half, dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if centre_vals.size < 3 or half_vals.size < 3:
                    continue
                scale = tuple(float(abs(v)) * 2.0 for v in half_vals[:3])
                rotation = None
                if "rotation" in obj:
                    try:
                        rotation_vals = np.asarray(obj["rotation"], dtype=np.float32)
                    except (TypeError, ValueError):
                        rotation_vals = None
                    if rotation_vals is not None and rotation_vals.shape in ((3, 3), (4, 4)):
                        rotation = rotation_vals
                model = self._model_matrix(
                    (float(centre_vals[0]), float(centre_vals[1]), float(centre_vals[2])),
                    scale,
                    rotation=rotation,
                )
                mvp = self._proj @ view @ model
                self._prog["MVP"].write(mvp.T.astype("f4").tobytes())
                self._prog["u_color"].value = self._color_to_vec(
                    obj.get("color") or obj.get("colour"),
                    (0.6, 0.7, 0.8),
                )
                self._box_vao.render()
            elif obj_type == "target":
                sprite = str(obj.get("sprite", "")).lower()
                asset = obj.get("asset") or obj.get("path")
                if not asset:
                    if "person" in sprite:
                        asset = "person.obj"
                    elif "drone" in sprite:
                        asset = "drone.stl"
                if asset is None:
                    continue
                entry = self._get_mesh_entry(str(asset))
                if entry is None:
                    continue
                try:
                    centre_vals = np.asarray(obj.get("centre"), dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if centre_vals.size < 3:
                    continue
                size_spec = obj.get("size")
                if size_spec is None:
                    continue
                try:
                    size_vals = np.asarray(size_spec, dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if size_vals.size == 0:
                    continue
                if size_vals.size == 1:
                    width = float(size_vals[0])
                    height = float(size_vals[0])
                else:
                    width = float(size_vals[0])
                    height = float(size_vals[1])
                if not math.isfinite(width) or not math.isfinite(height):
                    continue
                width = abs(width)
                height = abs(height)
                if width <= 0.0 or height <= 0.0:
                    continue
                scale = (width, height, width)
                rotation = None
                if "person" in sprite:
                    cos_a = math.cos(-math.pi * 0.5)
                    sin_a = math.sin(-math.pi * 0.5)
                    rotation = np.array(
                        (
                            (cos_a, 0.0, sin_a),
                            (0.0, 1.0, 0.0),
                            (-sin_a, 0.0, cos_a),
                        ),
                        dtype=np.float32,
                    )
                model = self._model_matrix(
                    (float(centre_vals[0]), float(centre_vals[1]), float(centre_vals[2])),
                    scale,
                    rotation=rotation,
                )
                mvp = self._proj @ view @ model
                self._prog["MVP"].write(mvp.T.astype("f4").tobytes())
                self._prog["u_color"].value = self._color_to_vec(
                    obj.get("color") or obj.get("colour"),
                    (0.1, 0.1, 0.1),
                )
                entry["vao"].render()

        data = self._fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8)
        img = img.reshape((self.height, self.width, 3))
        img = img[::-1, :, ::-1]
        frame[:] = img


register_renderer("opengl", lambda **kwargs: OpenGLRenderer(**kwargs))

__all__ = ["OpenGLRenderer"]
