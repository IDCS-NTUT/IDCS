"""Headless OpenGL renderer with basic mesh drawing and lighting."""

from __future__ import annotations

import ctypes
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import Renderer, register_renderer
from .._camera import CameraDescription, build_camera_description
from .assets import AssetStore
from .context import HeadlessContextError, HeadlessGLContext, create_headless_context
from .gpu import (
    GL_BACK,
    GLBindings,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_FRONT,
    GL_CCW,
    GL_LEQUAL,
    GL_PACK_ALIGNMENT,
    GL_RENDERER,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_INT,
    GL_VENDOR,
    GL_VERSION,
    GL_RGBA,
    ShaderProgram,
)

__all__ = ["GLRenderer"]


_LOGGER = logging.getLogger(__name__)


@dataclass
class _DrawCall:
    mesh: Any
    model_matrix: np.ndarray
    normal_matrix: np.ndarray
    material_color: np.ndarray
    texture: Optional[Any]
    double_sided: bool


@dataclass
class _RendererState:
    camera_context: Any
    context: HeadlessGLContext
    gl: GLBindings
    assets: Optional[AssetStore]
    program: Optional[ShaderProgram]
    uniforms: Dict[str, int]
    frame_shape: tuple[int, int, int]
    scratch_rgba: np.ndarray
    clear_colour: tuple[int, int, int]
    frame_interval_s: float
    light_direction: np.ndarray
    light_color: np.ndarray
    ambient_color: np.ndarray
    last_frame_id: Optional[int] = None
    last_camera: Optional[CameraDescription] = None
    last_world: Optional[Dict[str, Any]] = None


class GLRenderer:
    """Render SimCamera frames using an offscreen OpenGL pipeline."""

    def __init__(
        self,
        *,
        context: Any,
        backend: str = "auto",
        msaa_samples: int = 0,
        lighting: Optional[Dict[str, Any]] = None,
        laser: Optional[Dict[str, Any]] = None,
        assets: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            gl_context = create_headless_context(
                context.width,
                context.height,
                backend=backend,
                msaa_samples=msaa_samples,
                logger=_LOGGER,
            )
        except HeadlessContextError as exc:
            raise RuntimeError(f"OpenGL initialisation failed: {exc}") from exc

        lighting_cfg = lighting or {}
        self._laser = laser or {}

        gl_context.make_current()
        gl = GLBindings(gl_context, logger=_LOGGER)
        vendor = gl.get_string(GL_VENDOR) or "unknown"
        renderer_name = gl.get_string(GL_RENDERER) or "unknown"
        version = gl.get_string(GL_VERSION) or "unknown"

        assets_cfg = assets if isinstance(assets, dict) else None
        try:
            asset_store = AssetStore(gl, assets_cfg, logger=_LOGGER)
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("Asset initialisation failed: %s", exc)
            asset_store = None

        try:
            vertex_src, fragment_src = self._build_shader_sources(gl_context.api)
            program = gl.create_program(vertex_src, fragment_src)
            uniforms = self._resolve_uniforms(program)
        except Exception as exc:
            _LOGGER.warning("Shader program initialisation failed: %s", exc)
            program = None
            uniforms = {}

        clear_colour = (20, 20, 20)
        frame_shape = (context.height, context.width, 3)
        scratch = np.empty((context.height, context.width, 4), dtype=np.uint8)
        frame_interval = self._resolve_frame_interval(context)

        light_direction = self._normalise_vector(
            np.asarray(lighting_cfg.get("direction", (-0.35, -1.0, -0.3)), dtype=np.float32)
        )
        intensity = self._coerce_float(lighting_cfg.get("intensity", 1.0), default=1.0)
        if not bool(lighting_cfg.get("enabled", True)):
            intensity = 0.0
        ambient_level = self._coerce_float(lighting_cfg.get("ambient", 0.25), default=0.25)
        light_color = np.clip(np.array((intensity, intensity, intensity), dtype=np.float32), 0.0, None)
        ambient_color = np.clip(np.full(3, ambient_level, dtype=np.float32), 0.0, None)

        self._state = _RendererState(
            camera_context=context,
            context=gl_context,
            gl=gl,
            assets=asset_store,
            program=program,
            uniforms=uniforms,
            frame_shape=frame_shape,
            scratch_rgba=scratch,
            clear_colour=clear_colour,
            frame_interval_s=frame_interval,
            light_direction=light_direction,
            light_color=light_color,
            ambient_color=ambient_color,
        )
        self._gl = gl
        self._assets = asset_store

        gl.glViewport(0, 0, context.width, context.height)
        gl.glEnable(GL_DEPTH_TEST)
        gl.glDepthFunc(GL_LEQUAL)
        gl.glDepthMask(1)
        gl.glEnable(GL_CULL_FACE)
        gl.glCullFace(GL_BACK)
        gl.glFrontFace(GL_CCW)

        _LOGGER.debug(
            "GL renderer initialised (%sx%s, backend=%s, api=%s)",
            context.width,
            context.height,
            gl_context.backend,
            gl_context.api,
        )
        _LOGGER.info("GL device: %s (vendor=%s, version=%s)", renderer_name, vendor, version)

    # ------------------------------------------------------------------ public
    def render(self, frame: np.ndarray, /, *, frame_id: Optional[int] = None) -> None:
        if frame.shape != self._state.frame_shape:
            raise ValueError(
                f"Frame shape mismatch: expected {self._state.frame_shape}, got {frame.shape}"
            )

        if frame_id is None:
            frame_id = 0
        self._state.last_frame_id = frame_id

        camera = self._resolve_camera_description(frame_id)
        self._state.last_camera = camera
        world = self._state.last_world

        if camera is None or self._state.program is None or self._assets is None:
            self._fill_gradient(frame)
            return

        self._state.context.make_current()
        gl = self._gl
        height, width, _ = self._state.frame_shape
        gl.glViewport(0, 0, width, height)

        clear_rgb = np.asarray(self._state.clear_colour, dtype=np.float32) / 255.0
        gl.glClearColor(float(clear_rgb[0]), float(clear_rgb[1]), float(clear_rgb[2]), 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        program = self._state.program
        uniforms = self._state.uniforms
        program.use()
        self._upload_matrix4(uniforms.get("u_view_proj", -1), camera.gl_view_projection_matrix)
        self._upload_vec3(uniforms.get("u_camera_pos", -1), camera.position)
        self._upload_vec3(uniforms.get("u_light_dir", -1), self._state.light_direction)
        self._upload_vec3(uniforms.get("u_light_color", -1), self._state.light_color)
        self._upload_vec3(uniforms.get("u_ambient_color", -1), self._state.ambient_color)
        self._set_int(uniforms.get("u_texture", -1), 0)

        draw_calls = self._build_draw_calls(frame_id, world)
        for call in draw_calls:
            self._submit_draw_call(call)

        gl.glBindVertexArray(0)
        gl.glBindTexture(GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)

        self._readback_frame(frame)

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._state.program is not None:
            try:
                self._state.program.release()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            self._state.program = None
        self._state.uniforms.clear()
        if self._assets is not None:
            try:
                self._assets.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            self._assets = None
        self._state.context.close()

    def get_last_camera(self) -> Optional[CameraDescription]:
        """Return the most recently evaluated camera description."""

        return self._state.last_camera

    # ------------------------------------------------------------ internals
    def _resolve_camera_description(
        self, frame_id: int
    ) -> Optional[CameraDescription]:
        context = self._state.camera_context
        describe = getattr(context, "describe_world", None)
        if not callable(describe):
            self._state.last_world = None
            return None

        try:
            world = describe(frame_id)
        except Exception:  # pragma: no cover - defensive fallback
            self._state.last_world = None
            return None
        if not isinstance(world, dict):
            self._state.last_world = None
            return None

        camera_state = world.get("camera")
        if not isinstance(camera_state, dict):
            self._state.last_world = world
            return None

        width = getattr(context, "width", self._state.frame_shape[1])
        height = getattr(context, "height", self._state.frame_shape[0])
        world_up = getattr(context, "world_up", (0.0, 1.0, 0.0))

        description = build_camera_description(
            camera_state,
            int(width),
            int(height),
            default_up=world_up,
        )
        self._state.last_world = world
        return description

    def _resolve_frame_interval(self, context: Any) -> float:
        for attr in ("frame_interval_s", "frame_interval", "frame_dt", "dt"):
            value = getattr(context, attr, None)
            if value is None:
                continue
            try:
                interval = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(interval) and interval > 0.0:
                return interval

        fps = getattr(context, "fps", None)
        if fps is not None:
            try:
                rate = float(fps)
            except (TypeError, ValueError):
                rate = 0.0
            if math.isfinite(rate) and rate > 0.0:
                return 1.0 / rate

        return 1.0 / 60.0

    def _build_draw_calls(
        self,
        frame_id: int,
        world: Optional[Dict[str, Any]],
    ) -> List[_DrawCall]:
        assets = self._assets
        if assets is None:
            return []

        frame_seconds = frame_id * self._state.frame_interval_s
        overrides = self._extract_world_overrides(world)

        draw_calls: List[_DrawCall] = []
        for node in assets.manifest.static_nodes:
            call = self._build_node_draw_call(node, frame_seconds, None)
            if call is not None:
                draw_calls.append(call)

        target_nodes = assets.manifest.target_nodes
        for idx, node in enumerate(target_nodes):
            override = overrides[idx] if idx < len(overrides) else None
            call = self._build_node_draw_call(node, frame_seconds, override)
            if call is not None:
                draw_calls.append(call)

        return draw_calls

    def _extract_world_overrides(
        self, world: Optional[Dict[str, Any]]
    ) -> List[Dict[str, np.ndarray]]:
        results: List[Dict[str, np.ndarray]] = []
        if not isinstance(world, dict):
            return results

        objects = world.get("objects")
        if isinstance(objects, dict):
            objects = [objects]
        if not isinstance(objects, Sequence):
            return results

        for entry in objects:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "billboard":
                continue
            centre = entry.get("centre")
            if centre is None:
                continue
            try:
                translate = np.asarray(centre, dtype=np.float32).reshape(3)
            except Exception:
                continue
            if not np.all(np.isfinite(translate)):
                continue
            results.append({"translate": translate})

        return results

    def _build_node_draw_call(
        self,
        node: Any,
        frame_seconds: float,
        override: Optional[Dict[str, np.ndarray]],
    ) -> Optional[_DrawCall]:
        assets = self._assets
        if assets is None:
            return None

        mesh = assets.meshes.get(node.mesh)
        if mesh is None:
            return None

        material_color = np.array((0.7, 0.7, 0.7), dtype=np.float32)
        texture = None
        double_sided = False

        if mesh.material is not None:
            material = assets.materials.get(mesh.material)
            if material is not None:
                material_color = np.array(material.color, dtype=np.float32)
                if material.texture:
                    texture = assets.textures.get(material.texture)
                double_sided = bool(material.double_sided)

        translate = np.array(node.transform.translate, dtype=np.float32)
        scale = np.array(node.transform.scale, dtype=np.float32)
        rotate = np.array(node.transform.rotate_deg, dtype=np.float32)

        if override is not None and "translate" in override:
            translate = override["translate"]
            animation_offset = np.zeros(3, dtype=np.float32)
        else:
            animation_offset = self._evaluate_animation(node.animation, frame_seconds)

        translate = translate + animation_offset

        model_matrix = self._compose_transform(translate, rotate, scale)
        normal_matrix = self._compute_normal_matrix(model_matrix)

        return _DrawCall(
            mesh=mesh,
            model_matrix=model_matrix,
            normal_matrix=normal_matrix,
            material_color=material_color,
            texture=texture,
            double_sided=double_sided,
        )

    def _evaluate_animation(
        self, animation: Optional[Dict[str, Any]], frame_seconds: float
    ) -> np.ndarray:
        if not isinstance(animation, dict):
            return np.zeros(3, dtype=np.float32)

        anim_type = str(animation.get("type", animation.get("kind", ""))).strip().lower()
        if anim_type != "circle":
            return np.zeros(3, dtype=np.float32)

        try:
            radius = float(animation.get("radius", 0.0))
        except (TypeError, ValueError):
            radius = 0.0
        if not math.isfinite(radius) or radius <= 1e-6:
            return np.zeros(3, dtype=np.float32)

        try:
            speed_deg = float(animation.get("speed_deg_per_s", 45.0))
        except (TypeError, ValueError):
            speed_deg = 45.0
        if not math.isfinite(speed_deg):
            speed_deg = 45.0

        try:
            phase = float(animation.get("phase_deg", 0.0))
        except (TypeError, ValueError):
            phase = 0.0

        axis = str(animation.get("axis", "y")).strip().lower()

        angle = math.radians(phase) + math.radians(speed_deg) * frame_seconds
        offset = np.zeros(3, dtype=np.float32)

        if axis == "x":
            offset[1] = math.cos(angle) * radius
            offset[2] = math.sin(angle) * radius
        elif axis == "z":
            offset[0] = math.cos(angle) * radius
            offset[1] = math.sin(angle) * radius
        else:
            offset[0] = math.cos(angle) * radius
            offset[2] = math.sin(angle) * radius

        return offset.astype(np.float32)

    def _compose_transform(
        self, translate: np.ndarray, rotate_deg: np.ndarray, scale: np.ndarray
    ) -> np.ndarray:
        tx, ty, tz = translate.astype(np.float32)
        sx, sy, sz = scale.astype(np.float32)
        rx, ry, rz = np.radians(rotate_deg.astype(np.float32))

        scale_matrix = np.array(
            [[sx, 0.0, 0.0, 0.0], [0.0, sy, 0.0, 0.0], [0.0, 0.0, sz, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        cx, sx_sin = math.cos(rx), math.sin(rx)
        cy, sy_sin = math.cos(ry), math.sin(ry)
        cz, sz_sin = math.cos(rz), math.sin(rz)

        rot_x = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, cx, -sx_sin, 0.0], [0.0, sx_sin, cx, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        rot_y = np.array(
            [[cy, 0.0, sy_sin, 0.0], [0.0, 1.0, 0.0, 0.0], [-sy_sin, 0.0, cy, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        rot_z = np.array(
            [[cz, -sz_sin, 0.0, 0.0], [sz_sin, cz, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        translation = np.eye(4, dtype=np.float32)
        translation[0, 3] = tx
        translation[1, 3] = ty
        translation[2, 3] = tz

        model = translation @ rot_z @ rot_y @ rot_x @ scale_matrix
        return model.astype(np.float32)

    def _compute_normal_matrix(self, model_matrix: np.ndarray) -> np.ndarray:
        basis = model_matrix[:3, :3]
        try:
            normal = np.linalg.inv(basis).T
        except np.linalg.LinAlgError:
            normal = np.eye(3, dtype=np.float32)
        return normal.astype(np.float32)

    def _submit_draw_call(self, call: _DrawCall) -> None:
        gl = self._gl
        uniforms = self._state.uniforms

        if call.double_sided:
            gl.glDisable(GL_CULL_FACE)
        else:
            gl.glEnable(GL_CULL_FACE)
            gl.glCullFace(GL_BACK)

        call.mesh.vertex_array.bind()
        self._upload_matrix4(uniforms.get("u_model", -1), call.model_matrix)
        self._upload_matrix3(uniforms.get("u_normal_matrix", -1), call.normal_matrix)
        self._upload_vec3(uniforms.get("u_material_color", -1), call.material_color)

        if call.texture is not None:
            if self._gl.glActiveTexture is not None:
                self._gl.glActiveTexture(GL_TEXTURE0)
            gl.glBindTexture(GL_TEXTURE_2D, call.texture.handle)
            self._set_int(uniforms.get("u_use_texture", -1), 1)
        else:
            gl.glBindTexture(GL_TEXTURE_2D, 0)
            self._set_int(uniforms.get("u_use_texture", -1), 0)

        if call.mesh.index_count > 0:
            gl.glDrawElements(
                GL_TRIANGLES,
                call.mesh.index_count,
                GL_UNSIGNED_INT,
                ctypes.c_void_p(0),
            )
        else:
            gl.glDrawArrays(GL_TRIANGLES, 0, call.mesh.vertex_count)

    def _readback_frame(self, frame: np.ndarray) -> None:
        gl = self._gl
        scratch = self._state.scratch_rgba
        height, width, _ = self._state.frame_shape
        gl.glPixelStorei(GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(
            0,
            0,
            width,
            height,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            ctypes.c_void_p(scratch.ctypes.data),
        )

        np.copyto(frame, scratch[::-1, :, :3][:, :, ::-1])

    def _fill_gradient(self, frame: np.ndarray) -> None:
        height, width, _ = self._state.frame_shape
        grad_x = np.linspace(0.0, 35.0, width, dtype=np.float32)
        grad_y = np.linspace(0.0, 35.0, height, dtype=np.float32)
        gradient = grad_y[:, None] + grad_x[None, :]
        base = np.empty_like(frame, dtype=np.float32)
        base[...] = np.array(self._state.clear_colour, dtype=np.float32)
        blended = base + gradient[..., None]
        np.clip(blended, 0.0, 255.0, out=blended)
        frame[:] = blended.astype(np.uint8)

    def _resolve_uniforms(self, program: ShaderProgram) -> Dict[str, int]:
        names = (
            "u_view_proj",
            "u_model",
            "u_normal_matrix",
            "u_camera_pos",
            "u_light_dir",
            "u_light_color",
            "u_ambient_color",
            "u_material_color",
            "u_use_texture",
            "u_texture",
        )
        return {name: program.get_uniform_location(name) for name in names}

    def _build_shader_sources(self, api: str) -> Tuple[str, str]:
        if (api or "").strip().lower() == "opengles":
            vertex_header = "#version 300 es\n"
            fragment_header = "#version 300 es\nprecision mediump float;\n"
        else:
            vertex_header = "#version 330 core\n"
            fragment_header = "#version 330 core\n"

        vertex_src = vertex_header + (
            "layout(location = 0) in vec3 in_position;\n"
            "layout(location = 1) in vec3 in_normal;\n"
            "layout(location = 2) in vec2 in_uv;\n"
            "uniform mat4 u_view_proj;\n"
            "uniform mat4 u_model;\n"
            "uniform mat3 u_normal_matrix;\n"
            "out vec3 v_world_pos;\n"
            "out vec3 v_normal;\n"
            "out vec2 v_uv;\n"
            "void main() {\n"
            "    vec4 world_pos = u_model * vec4(in_position, 1.0);\n"
            "    v_world_pos = world_pos.xyz;\n"
            "    v_normal = normalize(u_normal_matrix * in_normal);\n"
            "    v_uv = in_uv;\n"
            "    gl_Position = u_view_proj * world_pos;\n"
            "}\n"
        )

        fragment_src = fragment_header + (
            "in vec3 v_world_pos;\n"
            "in vec3 v_normal;\n"
            "in vec2 v_uv;\n"
            "uniform vec3 u_camera_pos;\n"
            "uniform vec3 u_light_dir;\n"
            "uniform vec3 u_light_color;\n"
            "uniform vec3 u_ambient_color;\n"
            "uniform vec3 u_material_color;\n"
            "uniform bool u_use_texture;\n"
            "uniform sampler2D u_texture;\n"
            "out vec4 frag_color;\n"
            "void main() {\n"
            "    vec3 base_color = u_material_color;\n"
            "    if (u_use_texture) {\n"
            "        base_color *= texture(u_texture, v_uv).rgb;\n"
            "    }\n"
            "    vec3 normal = normalize(v_normal);\n"
            "    float diffuse = max(dot(normal, -u_light_dir), 0.0);\n"
            "    vec3 color = base_color * (u_ambient_color + u_light_color * diffuse);\n"
            "    frag_color = vec4(color, 1.0);\n"
            "}\n"
        )

        return vertex_src, fragment_src

    def _upload_matrix4(self, location: int, matrix: np.ndarray) -> None:
        if location < 0:
            return
        data = np.ascontiguousarray(matrix, dtype=np.float32)
        self._gl.glUniformMatrix4fv(location, 1, 1, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def _upload_matrix3(self, location: int, matrix: np.ndarray) -> None:
        if location < 0:
            return
        data = np.ascontiguousarray(matrix, dtype=np.float32)
        self._gl.glUniformMatrix3fv(location, 1, 1, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def _upload_vec3(self, location: int, vector: np.ndarray) -> None:
        if location < 0:
            return
        data = np.ascontiguousarray(vector, dtype=np.float32)
        self._gl.glUniform3fv(location, 1, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def _set_int(self, location: int, value: int) -> None:
        if location < 0:
            return
        self._gl.glUniform1i(location, int(value))

    def _normalise_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6 or not math.isfinite(norm):
            return np.array((0.0, -1.0, 0.0), dtype=np.float32)
        return (vector / norm).astype(np.float32)

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(result):
            return default
        return result


def _factory(**kwargs: Any) -> Renderer:
    return GLRenderer(**kwargs)


register_renderer("gl", _factory)
