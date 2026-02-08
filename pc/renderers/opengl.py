"""Minimal OpenGL renderer used by :mod:`pc.sim_camera`.

This renderer implements the same public contract as the CPU renderer: it is
constructed with a ``context`` and exposes ``render(frame, frame_id=None)``.

It creates a headless EGL/GLES context when available (Jetson-friendly), falls
back to moderngl standalone, or uses a hidden GLFW-backed context for development.
Renders a simple ground plane and a shaded cube, then reads the pixels back to
an OpenCV-compatible BGR ``numpy`` array with proper alignment and flipping.

The file registers the renderer under the name ``opengl`` so the simulator
can select it via ``sim.renderer: opengl`` in the config.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import register_renderer
from ._common import _build_camera, _projection_matrix, _view_matrix, _near_clip

try:
    import moderngl
except Exception:  # pragma: no cover - defensive import
    moderngl = None

_logger = logging.getLogger(__name__)

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
        self._near_clip = _near_clip()

        self._gl = None
        self._prog = None
        self._vao = None
        self._fbo = None

        if moderngl is None:
            _logger.warning("moderngl not available; renderer will use fallback")
            return

        # Try to create a context in priority order:
        # 1. EGL/GLES headless (Jetson-friendly)
        # 2. moderngl standalone (X11/desktop headless)
        # 3. GLFW hidden window fallback
        self._gl = self._create_gl_context()

        if self._gl is None:
            _logger.warning("Failed to create OpenGL context; using fallback rendering")
            return

        # build shaders + framebuffer
        try:
            self._prog = self._gl.program(vertex_shader=_VERT_SHADER, fragment_shader=_FRAG_SHADER)
        except Exception as exc:
            _logger.error(f"Failed to compile shaders: {exc}")
            self._gl = None
            return

        # Create a framebuffer with proper alignment
        # RGB texture with alignment=1 for correct readback
        try:
            color_tex = self._gl.texture((self.width, self.height), components=3)
            color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            depth_rb = self._gl.depth_renderbuffer((self.width, self.height))
            self._fbo = self._gl.framebuffer(color_attachments=[color_tex], depth_attachment=depth_rb)
        except Exception as exc:
            _logger.error(f"Failed to create framebuffer: {exc}")
            self._gl = None
            return

        # create ground plane and cube vertex/index buffers
        try:
            vdata, idata = self._build_scene_geometry()
            vbo = self._gl.buffer(vdata.tobytes())
            ibo = self._gl.buffer(idata.tobytes())

            self._vao = self._gl.vertex_array(
                self._prog, [(vbo, '3f 3f', 'in_position', 'in_normal')], index_buffer=ibo
            )
        except Exception as exc:
            _logger.error(f"Failed to create geometry buffers: {exc}")
            self._gl = None
            return

    def _create_gl_context(self) -> Optional[Any]:
        """Create OpenGL context with EGL → standalone → GLFW fallback."""
        
        # Try EGL/GLES first (Jetson-friendly headless)
        try:
            ctx = moderngl.create_context(standalone=True, backend='egl')
            _logger.info("Created EGL headless context")
            return ctx
        except Exception as exc:
            _logger.debug(f"EGL context creation failed: {exc}")

        # Try moderngl standalone (X11 headless)
        try:
            ctx = moderngl.create_standalone_context()
            _logger.info("Created moderngl standalone context")
            return ctx
        except Exception as exc:
            _logger.debug(f"Standalone context creation failed: {exc}")

        # Fall back to GLFW hidden window
        try:
            import glfw

            if not glfw.init():
                _logger.error("GLFW init failed")
                return None
            
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            win = glfw.create_window(self.width, self.height, "IDCS Hidden Renderer", None, None)
            if not win:
                _logger.error("GLFW window creation failed")
                return None
            
            glfw.make_context_current(win)
            ctx = moderngl.create_context()
            _logger.info("Created GLFW-backed context")
            return ctx
        except Exception as exc:
            _logger.error(f"GLFW context creation failed: {exc}")
            return None

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
                camera = _build_camera(cam_state, self.width, self.height, self._context)

        if camera is None:
            # default camera looking at origin from +Z
            from ._common import _normalise
            camera = {
                'position': np.array((3.0, 3.0, 3.0), dtype=np.float32),
                'forward': _normalise(np.array((-3.0, -3.0, -3.0), dtype=np.float32)),
                'right': np.array((1.0, 0.0, 0.0), dtype=np.float32),
                'up': np.array((0.0, 1.0, 0.0), dtype=np.float32),
                'fov_y': 60.0,
                'aspect': float(self.width) / float(self.height),
            }

        # Build projection and view matrices using shared functions
        proj = _projection_matrix(camera['fov_y'], camera['aspect'], self._near_clip, 100.0)
        view = _view_matrix(camera['position'], camera['forward'], camera['up'])

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

        # Read pixels with proper alignment (RGB bytes, bottom->top)
        # alignment=1 ensures no row padding
        data = self._fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        
        # Flip vertically (OpenGL bottom-left origin → image top-left origin)
        # and convert RGB→BGR for OpenCV
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
        from ._common import _normalise
        cube_positions = offsets
        cube_normals = np.array([_normalise(v) for v in offsets], dtype=np.float32)
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
        """Build a rotation matrix around the Y axis."""
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
