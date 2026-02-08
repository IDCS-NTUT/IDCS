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

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import register_renderer
from ._common import (
    NEAR_CLIP,
    build_camera,
    normalise,
    projection_matrix,
    view_matrix,
)

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
        self._near_clip = NEAR_CLIP

        self._gl = None
        self._prog = None
        self._vao = None
        self._fbo = None

        if moderngl is None:
            logger.error("moderngl is not available; OpenGL renderer will fall back to CPU")
            return

        self._gl = self._init_context()
        if self._gl is None:
            logger.error("Failed to create any OpenGL context; renderer will fall back to CPU")
            return

        # build shaders + framebuffer
        self._prog = self._gl.program(vertex_shader=_VERT_SHADER, fragment_shader=_FRAG_SHADER)

        # Create a framebuffer with a color texture and depth renderbuffer
        color_tex = self._gl.texture(
            (self.width, self.height), components=3, dtype="u1", alignment=1
        )
        depth_rb = self._gl.depth_renderbuffer((self.width, self.height))
        self._fbo = self._gl.framebuffer(color_attachments=[color_tex], depth_attachment=depth_rb)

        # create ground plane and cube vertex/index buffers
        vdata, idata = self._build_scene_geometry()
        vbo = self._gl.buffer(vdata.tobytes())
        ibo = self._gl.buffer(idata.tobytes())

        self._vao = self._gl.vertex_array(
            self._prog, [(vbo, '3f 3f', 'in_position', 'in_normal')], index_buffer=ibo
        )

    def _init_context(self):
        attempts = []
        if hasattr(moderngl, "create_standalone_context"):
            attempts.append(("moderngl-egl", lambda: moderngl.create_standalone_context(backend="egl")))
            attempts.append(("moderngl-auto", lambda: moderngl.create_standalone_context()))
        attempts.append(("glfw", self._create_glfw_context))

        for name, factory in attempts:
            try:
                ctx = factory()
            except Exception as exc:  # pragma: no cover - platform specific
                logger.warning("OpenGL context creation failed (%s): %s", name, exc)
                continue
            logger.info("OpenGL context created via %s", name)
            return ctx
        return None

    def _create_glfw_context(self):  # pragma: no cover - platform/windowed fallback
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
        return moderngl.create_context()

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
                camera = build_camera(
                    cam_state,
                    context=self._context,
                    width=self.width,
                    height=self.height,
                )

        if camera is None:
            # default camera looking at origin from +Z
            camera = {
                'position': np.array((3.0, 3.0, 3.0), dtype=np.float32),
                'forward': normalise(np.array((-3.0, -3.0, -3.0), dtype=np.float32)),
                'right': np.array((1.0, 0.0, 0.0), dtype=np.float32),
                'up': np.array((0.0, 1.0, 0.0), dtype=np.float32),
                'fov_y': 60.0,
                'aspect': float(self.width) / float(self.height),
            }

        # Build projection and view matrices
        proj = projection_matrix(camera['fov_y'], camera['aspect'], self._near_clip, 100.0)
        view = view_matrix(camera['position'], camera['forward'], camera['up'])

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
        data = self._fbo.read(components=3, alignment=1, dtype="u1")
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
        cube_normals = np.array([normalise(v) for v in offsets], dtype=np.float32)
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
    def _rotation_y(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return m


register_renderer("opengl", lambda **kwargs: OpenGLRenderer(**kwargs))

__all__ = ["OpenGLRenderer"]
