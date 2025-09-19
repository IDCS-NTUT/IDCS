# pc/renderers/gl.py — STEP 1a diagnostic
from __future__ import annotations
from typing import Any
import numpy as np
from . import register_renderer

class GLRenderer:
    def __init__(self, *, context: Any, finish_before_read: bool = True) -> None:
        import moderngl, glfw
        self._mgl, self._glfw = moderngl, glfw
        self._W = int(getattr(context, "width"))
        self._H = int(getattr(context, "height"))
        self._finish = bool(finish_before_read)

        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if hasattr(glfw, "OPENGL_FORWARD_COMPAT"):
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        self._win = glfw.create_window(self._W, self._H, "IDCS-GL", None, None)
        if not self._win:
            glfw.terminate()
            raise RuntimeError("Failed to create hidden GLFW window")

        glfw.make_context_current(self._win)
        glfw.swap_interval(0)

        # Create ModernGL context *after* the GL context is current
        # swap just the context creation part in __init__
        self._ctx = moderngl.create_standalone_context(require=330)
        self._ctx.viewport = (0, 0, self._W, self._H)
        self._fbo = self._ctx.simple_framebuffer((self._W, self._H), components=3)
        # no window; in render() remove calls to make_context_current


        print(f"[gl] step1a context OK {self._W}x{self._H}")

    def _read_to_frame(self, fb, frame: np.ndarray):
        if self._finish:
            self._ctx.finish()
        data = fb.read(components=3, dtype="u1")
        rgb  = np.frombuffer(data, np.uint8).reshape(self._H, self._W, 3)
        rgb  = np.flip(rgb, axis=0)
        frame[:] = rgb[:, :, ::-1]  # RGB->BGR

    def render(self, frame: np.ndarray, /, *, rvec: np.ndarray, tvec: np.ndarray) -> None:
        # Make sure our context is current each frame (defensive)
        self._glfw.make_context_current(self._win)
        self._ctx.viewport = (0, 0, self._W, self._H)

        # ---- Test A: clear default framebuffer (screen) and read it
        self._ctx.screen.use()
        self._ctx.clear(1.0, 0.0, 1.0, 1.0)  # purple
        self._read_to_frame(self._ctx.screen, frame)
        if frame.max() > 0:
            # Screen path works → GL & readback OK; stop here (purple should show)
            return

        # ---- Test B: clear our offscreen FBO and read it
        self._fbo.use()
        self._ctx.clear(0.0, 1.0, 0.0, 1.0)  # green
        self._read_to_frame(self._fbo, frame)
        # if still black, both paths are failing in your GL stack

    def __del__(self):
        try:
            if getattr(self, "_fbo", None): self._fbo.release()
            if getattr(self, "_ctx", None): self._ctx.release()
            if getattr(self, "_win", None): self._glfw.destroy_window(self._win)
            self._glfw.terminate()
        except Exception:
            pass

register_renderer("gl", GLRenderer)
__all__ = ["GLRenderer"]
