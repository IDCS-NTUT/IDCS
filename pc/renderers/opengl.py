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
uniform vec4 u_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 ldir = normalize(vec3(-0.4, 0.9, 0.3));
    float l = max(dot(n, ldir), 0.0);
    float ambient = 0.35;
    float diffuse = 0.65;
    vec3 base = u_color.rgb;
    float alpha = u_color.a;
    vec3 col = base * (ambient + diffuse * l);
    f_color = vec4(col, alpha);
}
"""

_VERT_TEX = """
#version 330
in vec3 in_position;
in vec2 in_uv;
uniform mat4 MVP;
out vec2 v_uv;
void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    v_uv = in_uv;
}
"""

_FRAG_TEX = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D tex;
void main() {
    f_color = texture(tex, v_uv);
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
        self._prog_tex = None
        self._vao = None
        self._fbo = None
        self._mesh_vaos: Dict[str, Any] = {}
        self._mesh_vbos: Dict[str, Any] = {}
        self._mesh_ibos: Dict[str, Any] = {}
        self._sprite_textures: Dict[str, Any] = {}

        if moderngl is None:
            logger.error("moderngl is not available; OpenGL renderer will fall back to CPU")
            return

        self._gl = self._init_context()
        if self._gl is None:
            logger.error("Failed to create any OpenGL context; renderer will fall back to CPU")
            return

        # build shaders + framebuffer
        self._prog = self._gl.program(vertex_shader=_VERT_SHADER, fragment_shader=_FRAG_SHADER)
        self._prog_tex = self._gl.program(vertex_shader=_VERT_TEX, fragment_shader=_FRAG_TEX)

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
        self._gl.viewport = (0, 0, self.width, self.height)
        self._gl.enable(moderngl.DEPTH_TEST)
        self._gl.enable(moderngl.CULL_FACE)
        self._gl.front_face = 'ccw'
        self._gl.cull_face = 'back'
        self._gl.disable(moderngl.BLEND)
        self._gl.clear(0.78, 0.78, 0.78)

        # draw ground
        mvp = proj @ view @ model_ground
        self._prog['MVP'].write(mvp.astype('f4').tobytes())
        self._prog['u_color'].value = (0.4, 0.7, 0.9, 1.0)
        self._vao.render()

        # draw rotating cube
        mvp = proj @ view @ model_cube
        self._prog['MVP'].write(mvp.astype('f4').tobytes())
        self._prog['u_color'].value = (0.6, 0.3, 0.1, 1.0)
        self._vao.render()

        # draw world objects (meshes, billboards)
        if world is not None:
            self._draw_world_objects(world, proj, view)

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

    def _draw_world_objects(self, world: Dict[str, Any], proj: np.ndarray, view: np.ndarray) -> None:
        objects = world.get('objects', ())
        if isinstance(objects, dict):
            objects = (objects,)
        if not isinstance(objects, (list, tuple)):
            return

        opaque = []
        alpha_blended = []

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            alpha = float(obj.get('alpha', 1.0)) if 'alpha' in obj else 1.0
            if alpha >= 0.999:
                opaque.append(obj)
            else:
                alpha_blended.append(obj)

        # opaque first
        self._gl.disable(moderngl.BLEND)
        for obj in opaque:
            self._draw_object(obj, proj, view)

        # transparent sorted back-to-front
        if alpha_blended:
            def depth_key(o: Dict[str, Any]):
                centre = np.asarray(o.get('centre', (0.0, 0.0, 0.0)), dtype=np.float32)
                rel = centre - world.get('camera', {}).get('position', np.zeros(3, dtype=np.float32))
                return -float(np.dot(rel, rel))

            alpha_blended.sort(key=depth_key)
            self._gl.enable(moderngl.BLEND)
            self._gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            for obj in alpha_blended:
                self._draw_object(obj, proj, view)

    def _draw_object(self, obj: Dict[str, Any], proj: np.ndarray, view: np.ndarray) -> None:
        obj_type = obj.get('type')

        if obj_type == 'billboard':
            sprite = obj.get('sprite')
            if sprite in {'drone', 'person'}:
                asset_map = {
                    'drone': 'assets/drone.stl',
                    'person': 'assets/person.obj',
                }
                mesh_obj = dict(obj)
                mesh_obj['type'] = 'mesh'
                mesh_obj.setdefault('asset', asset_map[sprite])
                size = obj.get('size')
                if isinstance(size, (list, tuple)) and len(size) >= 2:
                    try:
                        mesh_obj.setdefault('scale', float(size[1]))
                    except (TypeError, ValueError):
                        mesh_obj.setdefault('scale', 1.0)
                self._draw_mesh(mesh_obj, proj, view)
                return

        if obj_type in {'drone', 'person'}:
            asset_map = {
                'drone': 'assets/drone.stl',
                'person': 'assets/person.obj',
            }
            mesh_obj = dict(obj)
            mesh_obj['type'] = 'mesh'
            mesh_obj.setdefault('asset', asset_map[obj_type])
            self._draw_mesh(mesh_obj, proj, view)
        elif obj_type == 'mesh':
            self._draw_mesh(obj, proj, view)
        elif obj_type == 'billboard':
            self._draw_billboard(obj, proj, view)

    def _draw_mesh(self, obj: Dict[str, Any], proj: np.ndarray, view: np.ndarray) -> None:
        asset = obj.get('asset') or obj.get('path')
        if not asset:
            return
        try:
            from .mesh import load_mesh

            buffers = load_mesh(asset)
        except Exception as exc:
            logger.warning("Failed to load mesh %s: %s", asset, exc)
            return

        vao = self._mesh_vaos.get(asset)
        if vao is None:
            vdata = np.hstack([buffers.vertices, buffers.normals]).astype('f4')
            vbo = self._gl.buffer(vdata.tobytes())
            ibo = self._gl.buffer(buffers.indices.tobytes())
            vao = self._gl.vertex_array(
                self._prog,
                [(vbo, '3f 3f', 'in_position', 'in_normal')],
                index_buffer=ibo,
            )
            self._mesh_vaos[asset] = vao
            self._mesh_vbos[asset] = vbo
            self._mesh_ibos[asset] = ibo

        centre = np.asarray(obj.get('centre', (0.0, 0.0, 0.0)), dtype=np.float32)
        scale = obj.get('scale', 1.0)
        if isinstance(scale, (list, tuple, np.ndarray)):
            svec = np.asarray(scale, dtype=np.float32)
            if svec.size < 3:
                svec = np.array((float(scale), float(scale), float(scale)), dtype=np.float32)
        else:
            svec = np.array((float(scale), float(scale), float(scale)), dtype=np.float32)

        rotation = obj.get('rotation')
        rot_yaw = rot_pitch = rot_roll = 0.0
        if rotation is not None:
            try:
                rvals = np.asarray(rotation, dtype=np.float32).reshape(-1)
                if rvals.size >= 3:
                    rot_yaw, rot_pitch, rot_roll = map(float, rvals[:3])
            except Exception:
                rot_yaw = rot_pitch = rot_roll = 0.0

        model = self._compose_transform(centre, svec, rot_yaw, rot_pitch, rot_roll)
        mvp = proj @ view @ model
        self._prog['MVP'].write(mvp.astype('f4').tobytes())

        colour_spec = obj.get('color', obj.get('colour', (0.6, 0.7, 0.8)))
        alpha = float(obj.get('alpha', 1.0)) if 'alpha' in obj else 1.0
        try:
            col = np.asarray(colour_spec, dtype=np.float32).reshape(-1)
            base_color = (
                float(col[0] if col.size > 0 else 0.6),
                float(col[1] if col.size > 1 else 0.7),
                float(col[2] if col.size > 2 else 0.8),
                float(alpha),
            )
        except Exception:
            base_color = (0.6, 0.7, 0.8, float(alpha))
        self._prog['u_color'].value = base_color

        if alpha < 0.999:
            self._gl.enable(moderngl.BLEND)
            self._gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        vao.render()

    @staticmethod
    def _compose_transform(centre: np.ndarray, scale: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
        cy, syaw = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))

        # Rotation order Y (yaw) then X (pitch) then Z (roll)
        rot_y = np.array(
            [
                [cy, 0.0, syaw, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-syaw, 0.0, cy, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rot_x = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cp, -sp, 0.0],
                [0.0, sp, cp, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rot_z = np.array(
            [
                [cr, -sr, 0.0, 0.0],
                [sr, cr, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        rot = rot_y @ rot_x @ rot_z
        scale_m = np.eye(4, dtype=np.float32)
        scale_m[0, 0] = sx
        scale_m[1, 1] = sy
        scale_m[2, 2] = sz
        trans = np.eye(4, dtype=np.float32)
        trans[0:3, 3] = centre[0:3]
        return trans @ rot @ scale_m

    def _draw_billboard(self, obj: Dict[str, Any], proj: np.ndarray, view: np.ndarray) -> None:
        """Draw a camera-facing billboard sprite.

        The sprite loader in ``pc._sprites`` returns a (bgr, alpha) tuple.
        This function unpacks that tuple, constructs an RGBA image with the
        correct byte order for OpenGL, uploads it once to a texture and renders
        a quad facing the camera.
        """
        sprite = obj.get('sprite')
        size = obj.get('size')
        centre = obj.get('centre')
        if sprite is None or size is None or centre is None:
            return

        tex = self._sprite_textures.get(sprite)
        if tex is None:
            try:
                # load_sprite_image returns (bgr, alpha)
                from .._sprites import load_sprite_image  # lazy import to avoid optional-deps at module import
                bgr, alpha = load_sprite_image(sprite)
            except Exception as exc:
                logger.warning("Failed to load sprite '%s': %s", sprite, exc)
                return

            # Validate returned arrays
            if bgr is None or alpha is None:
                return
            if not isinstance(bgr, np.ndarray) or not isinstance(alpha, np.ndarray):
                return
            if bgr.size == 0 or alpha.size == 0:
                return

            # Ensure alpha is single-channel and matches image spatial dims
            h, w = bgr.shape[:2]
            if alpha.shape != (h, w):
                # Try to coerce common shapes to single-channel alpha
                if alpha.ndim == 3 and alpha.shape[2] >= 1:
                    # If an RGBA-like array was accidentally returned as the second element,
                    # pick the last channel as alpha or convert to grayscale.
                    alpha = alpha[..., -1]
                else:
                    # Fallback to opaque if shapes mismatch
                    alpha = np.full((h, w), 255, dtype=np.uint8)

            # Build RGBA: convert BGR->RGB then append alpha channel
            rgb = bgr[..., ::-1]  # BGR->RGB
            rgba = np.dstack([rgb, alpha])
            rgba = np.ascontiguousarray(rgba)

            # Create texture (width,height) reversed order for moderngl
            tex = self._gl.texture(rgba.shape[1::-1], 4, rgba.tobytes())
            tex.build_mipmaps()
            tex.repeat_x = False
            tex.repeat_y = False
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._sprite_textures[sprite] = tex

        centre_vec = np.asarray(centre, dtype=np.float32)
        width, height = float(size[0]), float(size[1])

        # Build billboard quad facing the camera (use camera basis from view matrix)
        view_inv = np.linalg.inv(view)
        right = view_inv[0:3, 0]
        up_vec = view_inv[0:3, 1]
        half_w = width * 0.5
        half_h = height * 0.5
        corners = np.array(
            [
                centre_vec - right * half_w - up_vec * half_h,
                centre_vec + right * half_w - up_vec * half_h,
                centre_vec + right * half_w + up_vec * half_h,
                centre_vec - right * half_w + up_vec * half_h,
            ],
            dtype=np.float32,
        )

        verts = np.hstack(
            [
                corners,
                np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        vbo = self._gl.buffer(verts.astype('f4').tobytes())
        ibo = self._gl.buffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32).tobytes())
        vao = self._gl.vertex_array(
            self._prog_tex,
            [(vbo, '3f 2f', 'in_position', 'in_uv')],
            index_buffer=ibo,
        )

        model = np.eye(4, dtype=np.float32)
        mvp = proj @ view @ model
        self._prog_tex['MVP'].write(mvp.astype('f4').tobytes())
        self._prog_tex['tex'].value = 0
        tex.use(location=0)
        self._gl.enable(moderngl.BLEND)
        self._gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        vao.render()
        

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
        cube_indices = np.array(
            [
                0,
                1,
                2,
                0,
                2,
                3,
                4,
                5,
                6,
                4,
                6,
                7,
                0,
                4,
                7,
                0,
                7,
                3,
                1,
                5,
                6,
                1,
                6,
                2,
                3,
                7,
                6,
                3,
                6,
                2,
            ],
            dtype=np.uint32,
        )

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
