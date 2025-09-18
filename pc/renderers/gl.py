"""ModernGL renderer backend implementation."""

from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

import cv2
import numpy as np

from . import register_renderer


GridLine = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
BoxSpec = Tuple[Tuple[float, float, float, float, float, float], Tuple[int, int, int]]


class GLRenderer:
    """Renderer that draws the simulation scene using ModernGL."""

    def __init__(
        self,
        *,
        context: Any,
        samples: int = 0,
        near: float = 0.1,
        far: float = 200.0,
        finish_before_read: bool = True,
    ) -> None:
        try:
            import moderngl  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ModernGL is required for the 'gl' renderer. Install the 'pc' extra."
            ) from exc
        try:
            import glfw  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "GLFW is required for the 'gl' renderer. Install the 'pc' extra."
            ) from exc

        required = (
            "width",
            "height",
            "grid_lines",
            "boxes",
            "fov",
            "aspect",
        )
        missing = [name for name in required if not hasattr(context, name)]
        if missing:
            raise AttributeError(
                "GL renderer context is missing required attributes: "
                + ", ".join(sorted(missing))
            )

        self._mgl = moderngl
        self._glfw = glfw
        self._window: Any | None = None
        self._glfw_initialized = False
        self._closed = False
        self._ctx: Any | None = None

        samples = int(samples)
        if samples < 0:
            raise ValueError("samples must be >= 0")

        self._width = int(getattr(context, "width"))
        self._height = int(getattr(context, "height"))
        self._grid_lines: Sequence[GridLine] = tuple(getattr(context, "grid_lines"))
        self._boxes: Sequence[BoxSpec] = tuple(getattr(context, "boxes"))
        self._fov = float(getattr(context, "fov"))
        self._aspect = float(getattr(context, "aspect"))
        self._near = float(near)
        self._far = float(far)
        self._finish_before_read = bool(finish_before_read)

        self._create_window_context()

        if self._ctx is None:  # pragma: no cover - defensive guard
            raise RuntimeError("ModernGL context was not created")

        if samples > 0:
            self._fbo = self._ctx.simple_framebuffer(
                (self._width, self._height), components=3, samples=samples
            )
            self._resolve_fbo = self._ctx.simple_framebuffer(
                (self._width, self._height), components=3
            )
        else:
            self._fbo = self._ctx.simple_framebuffer(
                (self._width, self._height), components=3
            )
            self._resolve_fbo = None

        self._proj = self._build_projection()
        self._view_fix = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

        self._bg_program = self._ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            """,
        )
        self._bg_ground = self._create_ground_vao()

        self._grid_vbo = None
        self._grid_program = self._ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_pos;
                void main() {
                    gl_Position = mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            """,
        )
        self._grid_vao, self._grid_vertex_count = self._create_grid_vao()

        self._box_vbo = None
        self._box_program = self._ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_pos;
                in vec3 in_color;
                out vec3 v_color;
                void main() {
                    v_color = in_color;
                    gl_Position = mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(v_color, 1.0);
                }
            """,
        )
        self._box_vao, self._box_vertex_count = self._create_box_vao()

        # Precompute static colours
        self._sky_color = tuple(c / 255.0 for c in (180, 180, 210))
        self._ground_color = tuple(c / 255.0 for c in (170, 190, 170))
        self._grid_color = tuple(c / 255.0 for c in (150, 150, 150))

    def render(
        self,
        frame: np.ndarray,
        /,
        *,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> None:
        """Render the scene into ``frame`` using the supplied camera pose."""

        if self._closed:
            raise RuntimeError("Cannot render with a closed GLRenderer")

        self._ensure_context_current()

        self._fbo.use()
        self._ctx.clear(
            *self._sky_color,
            1.0,
            depth=1.0,
            viewport=(0, 0, self._width, self._height),
        )

        self._bg_program["color"].value = self._ground_color
        self._bg_ground.render(mode=self._mgl.TRIANGLE_STRIP)

        mvp = self._compute_mvp(rvec, tvec)
        mvp_bytes = mvp.tobytes()

        if self._grid_vertex_count and self._grid_vao is not None:
            self._grid_program["mvp"].write(mvp_bytes)
            self._grid_program["color"].value = self._grid_color
            self._grid_vao.render(mode=self._mgl.LINES, vertices=self._grid_vertex_count)

        if self._box_vertex_count and self._box_vao is not None:
            self._box_program["mvp"].write(mvp_bytes)
            self._box_vao.render(mode=self._mgl.LINES, vertices=self._box_vertex_count)

        if self._resolve_fbo is not None:
            self._ctx.copy_framebuffer(dst=self._resolve_fbo, src=self._fbo)
            target = self._resolve_fbo
        else:
            target = self._fbo

        if self._finish_before_read:
            self._ctx.finish()
        data = target.read(components=3, dtype="u1")
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(self._height, self._width, 3)
        rgb = np.flip(rgb, axis=0)
        frame[:] = rgb[:, :, ::-1]

    def close(self) -> None:
        """Release OpenGL resources and destroy the hidden window."""

        if self._closed:
            return
        self._closed = True

        try:
            ctx = self._ctx
            if ctx is not None:
                self._ensure_context_current()

                for attr in (
                    "_bg_ground",
                    "_grid_vao",
                    "_box_vao",
                ):
                    obj = getattr(self, attr, None)
                    if obj is not None:
                        try:
                            obj.release()
                        except Exception:  # pragma: no cover - best effort cleanup
                            pass
                        setattr(self, attr, None)

                for attr in (
                    "_grid_vbo",
                    "_box_vbo",
                    "_fbo",
                    "_resolve_fbo",
                    "_bg_program",
                    "_grid_program",
                    "_box_program",
                ):
                    obj = getattr(self, attr, None)
                    if obj is not None:
                        try:
                            obj.release()
                        except Exception:  # pragma: no cover - best effort cleanup
                            pass
                        setattr(self, attr, None)

                try:
                    ctx.release()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
                self._ctx = None
        finally:
            glfw = getattr(self, "_glfw", None)
            window = self._window

            if glfw is not None and window is not None:
                try:
                    current = glfw.get_current_context()
                except Exception:  # pragma: no cover - best effort cleanup
                    current = None
                if current == window:
                    try:
                        glfw.make_context_current(None)
                    except Exception:  # pragma: no cover - best effort cleanup
                        pass
                try:
                    glfw.destroy_window(window)
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
                self._window = None

            if glfw is not None and self._glfw_initialized:
                try:
                    glfw.terminate()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
                self._glfw_initialized = False

    def __del__(self) -> None:  # pragma: no cover - destructor best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _create_window_context(self) -> None:
        """Initialise a hidden GLFW window and ModernGL context."""

        glfw = self._glfw
        if glfw is None:  # pragma: no cover - defensive
            raise RuntimeError("GLFW is not available")

        if not glfw.init():  # pragma: no cover - system-dependent failure
            raise RuntimeError("Failed to initialise GLFW for the 'gl' renderer")

        self._glfw_initialized = True

        try:
            glfw.default_window_hints()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            if hasattr(glfw, "OPENGL_FORWARD_COMPAT"):
                glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

            window = glfw.create_window(
                int(self._width),
                int(self._height),
                "IDCS GLRenderer",
                None,
                None,
            )
            if window is None:
                raise RuntimeError("Failed to create hidden GLFW window for GLRenderer")

            self._window = window
            glfw.make_context_current(window)
            glfw.swap_interval(0)

            self._ctx = self._mgl.create_context(require=330)
            self._ctx.enable(self._mgl.DEPTH_TEST)
            self._ctx.viewport = (0, 0, self._width, self._height)
        except Exception:
            self.close()
            raise

    def _ensure_context_current(self) -> None:
        """Ensure our hidden window's context is current on the thread."""

        glfw = getattr(self, "_glfw", None)
        window = getattr(self, "_window", None)
        if glfw is None or window is None:
            return

        try:
            current = glfw.get_current_context()
        except Exception:  # pragma: no cover - best effort cleanup
            current = None

        if current != window:
            glfw.make_context_current(window)

    def _build_projection(self) -> np.ndarray:
        fovy = 2.0 * math.atan(math.tan(self._fov * 0.5) / max(self._aspect, 1e-6))
        f = 1.0 / math.tan(fovy * 0.5)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self._aspect
        proj[1, 1] = f
        proj[2, 2] = (self._far + self._near) / (self._near - self._far)
        proj[2, 3] = (2.0 * self._far * self._near) / (self._near - self._far)
        proj[3, 2] = -1.0
        return proj

    def _compute_mvp(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(rvec.astype(np.float32))
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R
        view[:3, 3] = tvec.reshape(3)
        view = self._view_fix @ view
        return self._proj @ view

    def _create_ground_vao(self):
        verts = np.array(
            [
                (-1.0, -1.0),
                (1.0, -1.0),
                (-1.0, 0.0),
                (1.0, 0.0),
            ],
            dtype=np.float32,
        )
        vbo = self._ctx.buffer(verts.tobytes())
        vao = self._ctx.vertex_array(
            self._bg_program,
            [(vbo, "2f", "in_pos")],
        )
        return vao

    def _create_grid_vao(self):
        if not self._grid_lines:
            return None, 0

        data = []
        for start, end in self._grid_lines:
            data.extend(start)
            data.extend(end)
        arr = np.array(data, dtype=np.float32)
        vbo = self._ctx.buffer(arr.tobytes())
        vao = self._ctx.vertex_array(
            self._grid_program,
            [(vbo, "3f", "in_pos")],
        )
        self._grid_vbo = vbo
        return vao, arr.size // 3

    def _create_box_vao(self):
        if not self._boxes:
            return None, 0

        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        data = []
        for (x, y, z, w, d, h), color_bgr in self._boxes:
            corners = np.array(
                [
                    [x - 0.5 * w, y, z - 0.5 * d],
                    [x + 0.5 * w, y, z - 0.5 * d],
                    [x + 0.5 * w, y, z + 0.5 * d],
                    [x - 0.5 * w, y, z + 0.5 * d],
                    [x - 0.5 * w, y + h, z - 0.5 * d],
                    [x + 0.5 * w, y + h, z - 0.5 * d],
                    [x + 0.5 * w, y + h, z + 0.5 * d],
                    [x - 0.5 * w, y + h, z + 0.5 * d],
                ],
                dtype=np.float32,
            )
            color = tuple(c / 255.0 for c in color_bgr[::-1])
            for ia, ib in edges:
                data.extend((*corners[ia], *color))
                data.extend((*corners[ib], *color))

        arr = np.array(data, dtype=np.float32)
        vbo = self._ctx.buffer(arr.tobytes())
        vao = self._ctx.vertex_array(
            self._box_program,
            [(vbo, "3f 3f", "in_pos", "in_color")],
        )
        self._box_vbo = vbo
        return vao, arr.size // 6


register_renderer("gl", GLRenderer)


__all__ = ["GLRenderer"]
