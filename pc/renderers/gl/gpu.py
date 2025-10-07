"""OpenGL bindings and GPU resource helpers for the headless renderer."""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .context import HeadlessContextError, HeadlessGLContext

__all__ = [
    "GLBindings",
    "GPUBuffer",
    "Texture2D",
    "GL_ARRAY_BUFFER",
    "GL_ELEMENT_ARRAY_BUFFER",
    "GL_STATIC_DRAW",
]


_LOGGER = logging.getLogger(__name__)


# Common GL enum values -------------------------------------------------------
GLenum = ctypes.c_uint
GLuint = ctypes.c_uint
GLint = ctypes.c_int
GLsizei = ctypes.c_int
GLsizeiptr = ctypes.c_size_t

GL_ARRAY_BUFFER = 0x8892
GL_ELEMENT_ARRAY_BUFFER = 0x8893
GL_STATIC_DRAW = 0x88E4

GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_CLAMP_TO_EDGE = 0x812F
GL_REPEAT = 0x2901
GL_LINEAR = 0x2601
GL_NEAREST = 0x2600
GL_LINEAR_MIPMAP_LINEAR = 0x2703

GL_RGBA = 0x1908
GL_RGBA8 = 0x8058
GL_SRGB8_ALPHA8 = 0x8C43
GL_UNSIGNED_BYTE = 0x1401

GL_UNPACK_ALIGNMENT = 0x0CF5

GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02

GL_TEXTURE0 = 0x84C0

GL_NO_ERROR = 0


class GLBindings:
    """Resolve OpenGL entry points for the active headless context."""

    def __init__(self, context: HeadlessGLContext, *, logger: Optional[logging.Logger] = None) -> None:
        self._context = context
        self._logger = logger or _LOGGER

        self.glGetError = self._load_function("glGetError", ctypes.c_uint, [])
        self.glGetString = self._load_function("glGetString", ctypes.c_void_p, [GLenum])

        self.glGenBuffers = self._load_function("glGenBuffers", None, [GLsizei, ctypes.POINTER(GLuint)])
        self.glBindBuffer = self._load_function("glBindBuffer", None, [GLenum, GLuint])
        self.glBufferData = self._load_function(
            "glBufferData",
            None,
            [GLenum, GLsizeiptr, ctypes.c_void_p, GLenum],
        )
        self.glDeleteBuffers = self._load_function(
            "glDeleteBuffers",
            None,
            [GLsizei, ctypes.POINTER(GLuint)],
        )

        self.glGenTextures = self._load_function("glGenTextures", None, [GLsizei, ctypes.POINTER(GLuint)])
        self.glBindTexture = self._load_function("glBindTexture", None, [GLenum, GLuint])
        self.glTexParameteri = self._load_function(
            "glTexParameteri",
            None,
            [GLenum, GLenum, GLint],
        )
        self.glTexImage2D = self._load_function(
            "glTexImage2D",
            None,
            [GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, ctypes.c_void_p],
        )
        self.glDeleteTextures = self._load_function(
            "glDeleteTextures",
            None,
            [GLsizei, ctypes.POINTER(GLuint)],
        )
        self.glPixelStorei = self._load_function("glPixelStorei", None, [GLenum, GLint])
        self.glGenerateMipmap = self._load_function(
            ("glGenerateMipmap", "glGenerateMipmapEXT"),
            None,
            [GLenum],
            optional=True,
        )
        self.glActiveTexture = self._load_function(
            ("glActiveTexture", "glActiveTextureARB"),
            None,
            [GLenum],
            optional=True,
        )

    # ------------------------------------------------------------------ helpers
    def _load_function(
        self,
        names: Sequence[str] | str,
        restype: type | None,
        argtypes: Sequence[type],
        *,
        optional: bool = False,
    ):
        if isinstance(names, str):
            candidates = (names,)
        else:
            candidates = tuple(names)

        for name in candidates:
            address = self._context.get_proc_address(name)
            if address:
                prototype = ctypes.CFUNCTYPE(restype, *argtypes)  # type: ignore[arg-type]
                func = prototype(address)
                func.__name__ = name
                return func

        if optional:
            self._logger.debug("Optional GL function missing: %s", ", ".join(candidates))
            return None

        raise HeadlessContextError(
            "Required GL function(s) unavailable: " + ", ".join(candidates)
        )

    # --------------------------------------------------------------- introspect
    def get_string(self, name: int) -> str:
        value = self.glGetString(name)
        if not value:
            return ""
        return ctypes.cast(ctypes.c_void_p(value), ctypes.c_char_p).value.decode("utf-8", "ignore")

    # --------------------------------------------------------------- buffer API
    def create_buffer(self, target: int, data: np.ndarray, usage: int = GL_STATIC_DRAW) -> "GPUBuffer":
        array = np.ascontiguousarray(data)
        size = array.nbytes
        buffer_id = GLuint()
        self.glGenBuffers(1, ctypes.byref(buffer_id))
        if buffer_id.value == 0:
            raise HeadlessContextError("glGenBuffers returned 0")

        self.glBindBuffer(target, buffer_id.value)
        ptr = ctypes.c_void_p(array.ctypes.data if size else 0)
        self.glBufferData(target, GLsizeiptr(size), ptr, usage)
        self.glBindBuffer(target, 0)
        return GPUBuffer(self, int(buffer_id.value), target, size)

    def delete_buffer(self, handle: int) -> None:
        if handle == 0:
            return
        buffer_id = GLuint(handle)
        self.glDeleteBuffers(1, ctypes.byref(buffer_id))

    # --------------------------------------------------------------- texture API
    def create_texture2d(
        self,
        image: np.ndarray,
        *,
        srgb: bool = True,
        generate_mipmaps: bool = True,
        min_filter: Optional[int] = None,
        mag_filter: Optional[int] = None,
        wrap_s: Optional[int] = None,
        wrap_t: Optional[int] = None,
    ) -> "Texture2D":
        texels = np.ascontiguousarray(image)
        if texels.ndim != 3 or texels.shape[2] != 4:
            raise ValueError("Texture uploads require RGBA data")

        height, width, _ = texels.shape
        texture_id = GLuint()
        self.glGenTextures(1, ctypes.byref(texture_id))
        if texture_id.value == 0:
            raise HeadlessContextError("glGenTextures returned 0")

        self.glBindTexture(GL_TEXTURE_2D, texture_id.value)
        self.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        internal_format = GL_SRGB8_ALPHA8 if srgb else GL_RGBA8
        ptr = ctypes.c_void_p(texels.ctypes.data if texels.size else 0)
        self.glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GLint(internal_format),
            GLsizei(width),
            GLsizei(height),
            GLint(0),
            GLenum(GL_RGBA),
            GLenum(GL_UNSIGNED_BYTE),
            ptr,
        )

        if srgb:
            err = self.glGetError()
            if err != GL_NO_ERROR:
                self._logger.debug(
                    "glTexImage2D SRGB upload failed (0x%04x); retrying as linear RGBA",
                    err,
                )
                internal_format = GL_RGBA8
                self.glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GLint(internal_format),
                    GLsizei(width),
                    GLsizei(height),
                    GLint(0),
                    GLenum(GL_RGBA),
                    GLenum(GL_UNSIGNED_BYTE),
                    ptr,
                )

        min_value = min_filter if min_filter is not None else GL_LINEAR
        if generate_mipmaps and self.glGenerateMipmap is not None:
            if min_filter is None:
                min_value = GL_LINEAR_MIPMAP_LINEAR
        mag_value = mag_filter if mag_filter is not None else GL_LINEAR
        wrap_s_val = wrap_s if wrap_s is not None else GL_CLAMP_TO_EDGE
        wrap_t_val = wrap_t if wrap_t is not None else GL_CLAMP_TO_EDGE

        self.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GLint(min_value))
        self.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GLint(mag_value))
        self.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GLint(wrap_s_val))
        self.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GLint(wrap_t_val))

        if generate_mipmaps and self.glGenerateMipmap is not None:
            self.glGenerateMipmap(GL_TEXTURE_2D)

        self.glBindTexture(GL_TEXTURE_2D, 0)
        return Texture2D(
            bindings=self,
            handle=int(texture_id.value),
            width=int(width),
            height=int(height),
            internal_format=int(internal_format),
            srgb=srgb,
        )

    def delete_texture(self, handle: int) -> None:
        if handle == 0:
            return
        texture_id = GLuint(handle)
        self.glDeleteTextures(1, ctypes.byref(texture_id))


@dataclass
class GPUBuffer:
    bindings: GLBindings
    handle: int
    target: int
    size: int

    def release(self) -> None:
        if self.handle:
            self.bindings.delete_buffer(self.handle)
            self.handle = 0


@dataclass
class Texture2D:
    bindings: GLBindings
    handle: int
    width: int
    height: int
    internal_format: int
    srgb: bool = True

    def release(self) -> None:
        if self.handle:
            self.bindings.delete_texture(self.handle)
            self.handle = 0
