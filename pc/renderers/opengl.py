"""Minimal OpenGL renderer used by :mod:`pc.sim_camera`.

This renderer implements the same public contract as the CPU renderer: it is
constructed with a ``context`` and exposes ``render(frame, frame_id=None)``.
The initial implementation is intentionally small: it creates an offscreen
moderngl context if available (or falls back to a hidden GLFW-backed context),
renders a simple ground plane and a shaded cube, then reads the pixels back to
an OpenCV-compatible BGR ``numpy`` array and writes them into ``frame``.

The file registers the renderer under the name ``opengl`` so the simulator
can select it via ``sim.renderer: opengl`` in the config. This is a starting
point for replacing the CPU billboard rendering with real mesh rendering from
``assets/`` in a follow-up iteration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import register_renderer

try:
    import moderngl
except Exception:  # pragma: no cover - defensive import
    moderngl = None

_VERT_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_normal;
uniform mat4 MVP;
out vec3 v_normal;
void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_normal = in_normal;
}
"""

_FRAG_SHADER = """
#version 330
in vec3 v_normal;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 ldir = normalize(vec3(-0.4, 0.9, 0.3));
    float l = max(dot(n, ldir), 0.0);
    float ambient = 0.35;
    float diffuse = 0.65;
    vec3 base = vec3(0.4, 0.7, 0.9);
    vec3 col = base * (ambient + diffuse * l);
    f_color = vec4(col, 1.0);
}
"""


class OpenGLRenderer:
    """Simple OpenGL-backed renderer that produces BGR frames.

    The implementation focuses on a correct and compact readback path and a
    matching camera/projection convention to make comparison with the CPU
    renderer straightforward.
    """

    def __init__(self, *, context: Any) -> None:
        try:
            self.width = int(getattr(context, "width"))
            self.height = int(getattr(context, "height"))
        except AttributeError as exc:  # pragma: no cover - defensive only
            raise AttributeError("SimCamera context must expose width/height") from exc

        self._context = context
        self._near_clip = 0.05

        self._gl = None
        self._prog = None
        self._vao = None
        self._fbo = None

        if moderngl is None:
            # moderngl not available; renderer will draw a fallback in render()
            return

        # Try to create a headless context; fall back to a visible GLFW window
        # if platform headless support is not available (tests in repo use
        # glfw/moderngl in a windowed mode).
        try:
            self._gl = moderngl.create_standalone_context()
        except Exception:
            try:
                # lazily import glfw only if needed
                import glfw

                if not glfw.init():
                    raise RuntimeError("GLFW init failed")
                glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
                win = glfw.create_window(self.width, self.height, "", None, None)
                if not win:
                    raise RuntimeError("GLFW window creation failed")
                glfw.make_context_current(win)
                self._gl = moderngl.create_context()
            except Exception:
                self._gl = None

        if self._gl is None:
            return

        # build shaders + framebuffer
        self._prog = self._gl.program(vertex_shader=_VERT_SHADER, fragment_shader=_FRAG_SHADER)

        # Create a framebuffer with a color texture and depth renderbuffer
        color_tex = self._gl.texture((self.width, self.height), components=3)
        depth_rb = self._gl.depth_renderbuffer((self.width, self.height))
        self._fbo = self._gl.framebuffer(color_attachments=[color_tex], depth_attachment=depth_rb)

        # create ground plane and cube vertex/index buffers
        vdata, idata = self._build_scene_geometry()
        vbo = self._gl.buffer(vdata.tobytes())
        ibo = self._gl.buffer(idata.tobytes())

        self._vao = self._gl.vertex_array(
            self._prog, [(vbo, '3f 3f', 'in_position', 'in_normal')], index_buffer=ibo
        )

    # ----------------------------- public API -----------------------------
    def render(self, frame: np.ndarray, /, *, frame_id: Optional[int] = None) -> None:
        """Render a single frame into ``frame`` (BGR numpy array).

        If GL initialization failed the renderer writes a fallback background
        so the pipeline can continue.
        """

        if frame_id is None:
            frame_id = 0

        if self._gl is None or self._fbo is None:
            # fallback: simple flat background to avoid breaking callers
            frame[:] = np.full((self.height, self.width, 3), 120, dtype=np.uint8)
            return

        # Prepare camera from world description when available
        world = self._fetch_world(frame_id)
        camera = None
        if world is not None:
            cam_state = world.get('camera')
            if isinstance(cam_state, dict):
                camera = self._build_camera(cam_state)

        if camera is None:
            # default camera looking at origin from +Z
            camera = {
                'position': np.array((3.0, 3.0, 3.0), dtype=np.float32),
                'forward': self._normalise(np.array(( -3.0, -3.0, -3.0), dtype=np.float32)),
                'right': np.array((1.0, 0.0, 0.0), dtype=np.float32),
                'up': np.array((0.0, 1.0, 0.0), dtype=np.float32),
                'fov_y': 60.0,
                'aspect': float(self.width) / float(self.height),
            }

        # Build projection and view matrices
        proj = self._projection_matrix(camera['fov_y'], camera['aspect'], self._near_clip, 100.0)
        view = self._view_matrix(camera['position'], camera['forward'], camera['up'])

        # Model matrices for ground and cube
        model_ground = np.eye(4, dtype=np.float32)
        model_cube = np.eye(4, dtype=np.float32)
        angle = (frame_id % 360) * math.pi / 180.0
        rot = self._rotation_y(angle)
        model_cube = rot

        # Render to FBO
        self._fbo.use()
        self._gl.enable(moderngl.DEPTH_TEST)
        self._gl.clear(0.78, 0.78, 0.78)

        # draw ground
        mvp = proj @ view @ model_ground
        self._prog['MVP'].write(mvp.astype('f4').tobytes())
        # first half of indices are ground (drawn as triangles)
        self._vao.render()

        # draw rotating cube
        mvp = proj @ view @ model_cube
        self._prog['MVP'].write(mvp.astype('f4').tobytes())
        self._vao.render()

        # read pixels (returns RGB bytes, bottom->top)
        data = self._fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        # flip vertically and convert RGB->BGR for OpenCV
        img = np.ascontiguousarray(img[::-1, :, ::-1])

        if img.shape[0] != self.height or img.shape[1] != self.width:
            # defensive fallback
            frame[:] = np.full((self.height, self.width, 3), 100, dtype=np.uint8)
        else:
            frame[:] = img

    # --------------------------- helpers / geometry -----------------------
    def _fetch_world(self, frame_id: int) -> Optional[Dict[str, Any]]:
        describe = getattr(self._context, 'describe_world', None)
        if not callable(describe):
            return None
        try:
            world = describe(frame_id)
        except Exception:
            return None
        if not isinstance(world, dict):
            return None
        return world

    def _build_camera(self, camera_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            position = np.asarray(camera_state['position'], dtype=np.float32)
        except Exception:
            return None

        orientation = camera_state.get('orientation')
        if orientation is not None:
            # minimal parsing: fallback to CPU convention via yaw/pitch/roll if present
            yaw = float(getattr(orientation, 'get', lambda k, d: 0.0)('yaw', 0.0)) if isinstance(orientation, dict) else 0.0
            pitch = float(getattr(orientation, 'get', lambda k, d: 0.0)('pitch', 0.0)) if isinstance(orientation, dict) else 0.0
            # convert to forward/right/up simplistically
            yaw_rad = math.radians(yaw)
            pitch_rad = math.radians(pitch)
            forward = np.array((math.cos(pitch_rad) * math.sin(yaw_rad), math.sin(pitch_rad), -math.cos(pitch_rad) * math.cos(yaw_rad)), dtype=np.float32)
            forward = self._normalise(forward)
            up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
            right = self._normalise(np.cross(up, forward))
            true_up = self._normalise(np.cross(forward, right))
        else:
            try:
                target = np.asarray(camera_state['target'], dtype=np.float32)
            except Exception:
                return None
            up_vec = camera_state.get('up', (0.0, 1.0, 0.0))
            up = self._normalise(np.asarray(up_vec, dtype=np.float32))
            forward = self._normalise(target - position)
            if self._vector_length(forward) < 1e-6:
                return None
            right = self._normalise(np.cross(up, forward))
            true_up = self._normalise(np.cross(forward, right))

        fov_y = float(camera_state.get('fov_y', 60.0))
        return {
            'position': position,
            'forward': forward,
            'right': right,
            'up': true_up,
            'fov_y': fov_y,
            'aspect': float(self.width) / float(self.height),
        }

    def _build_scene_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build interleaved vertex buffer (pos, normal) and index buffer.

        The simple layout encodes a ground plane and a cube. The returned arrays
        interleave positions and normals as float32 and use uint32 indices.
        """
        # Ground: a large square centered at origin (y=0)
        gsize = 50.0
        ground_positions = np.array([
            (-gsize, 0.0, -gsize),
            (gsize, 0.0, -gsize),
            (gsize, 0.0, gsize),
            (-gsize, 0.0, gsize),
        ], dtype=np.float32)
        ground_normals = np.tile(np.array((0.0, 1.0, 0.0), dtype=np.float32), (4, 1))
        ground_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # Cube centered at origin (size 1.0)
        offsets = np.array([
            (-0.5, -0.5, -0.5),
            (0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
        ], dtype=np.float32)

        # per-vertex normals (for a simple cube we duplicate normals per face is fine)
        # We'll set normals as normalized position for a shaded look.
        cube_positions = offsets
        cube_normals = np.array([self._normalise(v) for v in offsets], dtype=np.float32)
        cube_indices = np.array([
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
            0, 4, 7, 0, 7, 3,
            1, 5, 6, 1, 6, 2,
            3, 7, 6, 3, 6, 2,
        ], dtype=np.uint32)

        # Concatenate ground then cube into one big buffer with adjusted indices
        verts = np.vstack([ground_positions, cube_positions]).astype(np.float32)
        norms = np.vstack([ground_normals, cube_normals]).astype(np.float32)
        vbuf = np.hstack([verts, norms]).astype(np.float32)

        # indices: ground indices already correct, cube indices need an offset of 4
        cube_offset = 4
        combined_indices = np.concatenate([ground_indices, cube_indices + cube_offset]).astype(np.uint32)

        return vbuf, combined_indices

    @staticmethod
    def _projection_matrix(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
        nf = 1.0 / (near - far)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) * nf
        m[2, 3] = (2.0 * far * near) * nf
        m[3, 2] = -1.0
        return m

    @staticmethod
    def _view_matrix(eye: Sequence[float], forward: Sequence[float], up: Sequence[float]) -> np.ndarray:
        f = np.asarray(forward, dtype=np.float32)
        f = f / (np.linalg.norm(f) + 1e-12)
        u = np.asarray(up, dtype=np.float32)
        u = u / (np.linalg.norm(u) + 1e-12)
        s = np.cross(u, f)
        s = s / (np.linalg.norm(s) + 1e-12)
        u2 = np.cross(f, s)

        m = np.eye(4, dtype=np.float32)
        m[0, 0:3] = s
        m[1, 0:3] = u2
        m[2, 0:3] = -f
        t = np.eye(4, dtype=np.float32)
        t[0:3, 3] = -np.asarray(eye, dtype=np.float32)
        return m @ t

    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return m

    @staticmethod
    def _vector_length(vec: np.ndarray) -> float:
        return float(np.linalg.norm(vec))

    def _normalise(self, vec: np.ndarray) -> np.ndarray:
        length = self._vector_length(vec)
        if length <= 1e-6:
            return vec
        return vec / length


register_renderer("opengl", lambda **kwargs: OpenGLRenderer(**kwargs))

__all__ = ["OpenGLRenderer"]
