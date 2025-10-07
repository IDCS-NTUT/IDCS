"""OpenGL bindings and GPU resource helpers for the headless renderer."""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .context import HeadlessContextError, HeadlessGLContext

__all__ = [
    "GLBindings",
    "GPUBuffer",
    "Texture2D",
    "ShaderProgram",
    "VertexArray",
    "GL_ARRAY_BUFFER",
    "GL_ELEMENT_ARRAY_BUFFER",
    "GL_STATIC_DRAW",
    "GL_TEXTURE_2D",
    "GL_FLOAT",
    "GL_UNSIGNED_INT",
    "GL_UNSIGNED_BYTE",
    "GL_RGBA",
    "GL_COLOR_BUFFER_BIT",
    "GL_DEPTH_BUFFER_BIT",
    "GL_TRIANGLES",
    "GL_DEPTH_TEST",
    "GL_CULL_FACE",
    "GL_BACK",
    "GL_FRONT",
    "GL_CCW",
    "GL_LEQUAL",
    "GL_PACK_ALIGNMENT",
    "GL_TEXTURE0",
]


_LOGGER = logging.getLogger(__name__)


# Common GL enum values -------------------------------------------------------
GLenum = ctypes.c_uint
GLuint = ctypes.c_uint
GLint = ctypes.c_int
GLsizei = ctypes.c_int
GLsizeiptr = ctypes.c_size_t
GLboolean = ctypes.c_ubyte
GLfloat = ctypes.c_float

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
GL_UNSIGNED_INT = 0x1405
GL_FLOAT = 0x1406

GL_UNPACK_ALIGNMENT = 0x0CF5
GL_PACK_ALIGNMENT = 0x0D05

GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02

GL_TEXTURE0 = 0x84C0

GL_NO_ERROR = 0
GL_FALSE = 0
GL_TRUE = 1

GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_TRIANGLES = 0x0004

GL_DEPTH_TEST = 0x0B71
GL_CULL_FACE = 0x0B44
GL_BACK = 0x0405
GL_FRONT = 0x0404
GL_CCW = 0x0901
GL_LEQUAL = 0x0203

GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_INFO_LOG_LENGTH = 0x8B84


class GLBindings:
    """Resolve OpenGL entry points for the active headless context."""

    def __init__(self, context: HeadlessGLContext, *, logger: Optional[logging.Logger] = None) -> None:
        self._context = context
        self._logger = logger or _LOGGER

        self.glGetError = self._load_function("glGetError", ctypes.c_uint, [])
        self.glGetString = self._load_function("glGetString", ctypes.c_void_p, [GLenum])

        self.glViewport = self._load_function("glViewport", None, [GLint, GLint, GLsizei, GLsizei])
        self.glClearColor = self._load_function("glClearColor", None, [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float])
        self.glClear = self._load_function("glClear", None, [ctypes.c_uint])
        self.glEnable = self._load_function("glEnable", None, [GLenum])
        self.glDisable = self._load_function("glDisable", None, [GLenum])
        self.glDepthMask = self._load_function("glDepthMask", None, [GLboolean])
        self.glDepthFunc = self._load_function("glDepthFunc", None, [GLenum])
        self.glCullFace = self._load_function("glCullFace", None, [GLenum])
        self.glFrontFace = self._load_function("glFrontFace", None, [GLenum])

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

        self.glCreateShader = self._load_function("glCreateShader", GLuint, [GLenum])
        self.glShaderSource = self._load_function(
            "glShaderSource",
            None,
            [GLuint, GLsizei, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(GLint)],
        )
        self.glCompileShader = self._load_function("glCompileShader", None, [GLuint])
        self.glGetShaderiv = self._load_function("glGetShaderiv", None, [GLuint, GLenum, ctypes.POINTER(GLint)])
        self.glGetShaderInfoLog = self._load_function(
            "glGetShaderInfoLog",
            None,
            [GLuint, GLsizei, ctypes.POINTER(GLsizei), ctypes.c_char_p],
        )
        self.glDeleteShader = self._load_function("glDeleteShader", None, [GLuint])

        self.glCreateProgram = self._load_function("glCreateProgram", GLuint, [])
        self.glAttachShader = self._load_function("glAttachShader", None, [GLuint, GLuint])
        self.glLinkProgram = self._load_function("glLinkProgram", None, [GLuint])
        self.glGetProgramiv = self._load_function(
            "glGetProgramiv",
            None,
            [GLuint, GLenum, ctypes.POINTER(GLint)],
        )
        self.glGetProgramInfoLog = self._load_function(
            "glGetProgramInfoLog",
            None,
            [GLuint, GLsizei, ctypes.POINTER(GLsizei), ctypes.c_char_p],
        )
        self.glUseProgram = self._load_function("glUseProgram", None, [GLuint])
        self.glDeleteProgram = self._load_function("glDeleteProgram", None, [GLuint])

        self.glGetUniformLocation = self._load_function("glGetUniformLocation", GLint, [GLuint, ctypes.c_char_p])
        self.glUniformMatrix4fv = self._load_function(
            "glUniformMatrix4fv",
            None,
            [GLint, GLsizei, GLboolean, ctypes.POINTER(GLfloat)],
        )
        self.glUniformMatrix3fv = self._load_function(
            "glUniformMatrix3fv",
            None,
            [GLint, GLsizei, GLboolean, ctypes.POINTER(GLfloat)],
        )
        self.glUniform1f = self._load_function("glUniform1f", None, [GLint, ctypes.c_float])
        self.glUniform1i = self._load_function("glUniform1i", None, [GLint, ctypes.c_int])
        self.glUniform3fv = self._load_function(
            "glUniform3fv",
            None,
            [GLint, GLsizei, ctypes.POINTER(GLfloat)],
        )
        self.glUniform4fv = self._load_function(
            "glUniform4fv",
            None,
            [GLint, GLsizei, ctypes.POINTER(GLfloat)],
        )

        self.glGenVertexArrays = self._load_function("glGenVertexArrays", None, [GLsizei, ctypes.POINTER(GLuint)])
        self.glBindVertexArray = self._load_function("glBindVertexArray", None, [GLuint])
        self.glDeleteVertexArrays = self._load_function(
            "glDeleteVertexArrays",
            None,
            [GLsizei, ctypes.POINTER(GLuint)],
        )
        self.glEnableVertexAttribArray = self._load_function("glEnableVertexAttribArray", None, [GLuint])
        self.glDisableVertexAttribArray = self._load_function("glDisableVertexAttribArray", None, [GLuint])
        self.glVertexAttribPointer = self._load_function(
            "glVertexAttribPointer",
            None,
            [GLuint, GLint, GLenum, GLboolean, GLsizei, ctypes.c_void_p],
        )

        self.glDrawArrays = self._load_function(
            "glDrawArrays",
            None,
            [GLenum, GLint, GLsizei],
        )
        self.glDrawElements = self._load_function(
            "glDrawElements",
            None,
            [GLenum, GLsizei, GLenum, ctypes.c_void_p],
        )

        self.glReadPixels = self._load_function(
            "glReadPixels",
            None,
            [GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, ctypes.c_void_p],
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

    # --------------------------------------------------------------- program API
    def create_program(self, vertex_src: str, fragment_src: str) -> "ShaderProgram":
        vertex = self._compile_shader(GL_VERTEX_SHADER, vertex_src)
        fragment = self._compile_shader(GL_FRAGMENT_SHADER, fragment_src)
        try:
            program_handle = self.glCreateProgram()
            if program_handle == 0:
                raise HeadlessContextError("glCreateProgram returned 0")
            self.glAttachShader(program_handle, vertex)
            self.glAttachShader(program_handle, fragment)
            self.glLinkProgram(program_handle)

            status = GLint()
            self.glGetProgramiv(program_handle, GL_LINK_STATUS, ctypes.byref(status))
            if status.value != GL_TRUE:
                message = self._get_program_log(program_handle)
                raise HeadlessContextError(f"Program link failed: {message}")
        except Exception:
            self.glDeleteProgram(program_handle)
            raise
        finally:
            self.glDeleteShader(vertex)
            self.glDeleteShader(fragment)

        return ShaderProgram(self, int(program_handle))

    def _compile_shader(self, shader_type: int, source: str) -> int:
        handle = self.glCreateShader(shader_type)
        if handle == 0:
            raise HeadlessContextError("glCreateShader returned 0")
        encoded = source.encode("utf-8")
        src_buffer = ctypes.c_char_p(encoded)
        length = GLint(len(encoded))
        self.glShaderSource(handle, 1, ctypes.byref(src_buffer), ctypes.byref(length))
        self.glCompileShader(handle)

        status = GLint()
        self.glGetShaderiv(handle, GL_COMPILE_STATUS, ctypes.byref(status))
        if status.value != GL_TRUE:
            message = self._get_shader_log(handle)
            self.glDeleteShader(handle)
            raise HeadlessContextError(f"Shader compilation failed: {message}")

        return handle

    def _get_shader_log(self, handle: int) -> str:
        length = GLint()
        self.glGetShaderiv(handle, GL_INFO_LOG_LENGTH, ctypes.byref(length))
        if length.value <= 1:
            return "<empty>"
        buffer = ctypes.create_string_buffer(length.value)
        written = GLsizei()
        self.glGetShaderInfoLog(handle, length, ctypes.byref(written), buffer)
        return buffer.value.decode("utf-8", "ignore")

    def _get_program_log(self, handle: int) -> str:
        length = GLint()
        self.glGetProgramiv(handle, GL_INFO_LOG_LENGTH, ctypes.byref(length))
        if length.value <= 1:
            return "<empty>"
        buffer = ctypes.create_string_buffer(length.value)
        written = GLsizei()
        self.glGetProgramInfoLog(handle, length, ctypes.byref(written), buffer)
        return buffer.value.decode("utf-8", "ignore")

    # --------------------------------------------------------------- vertex arrays
    def create_vertex_array(
        self,
        vertex_buffer: "GPUBuffer",
        *,
        index_buffer: Optional["GPUBuffer"] = None,
        layout: Optional[Dict[str, object]] = None,
    ) -> "VertexArray":
        array_id = GLuint()
        self.glGenVertexArrays(1, ctypes.byref(array_id))
        if array_id.value == 0:
            raise HeadlessContextError("glGenVertexArrays returned 0")

        self.glBindVertexArray(array_id.value)
        self.glBindBuffer(vertex_buffer.target, vertex_buffer.handle)
        if index_buffer is not None:
            self.glBindBuffer(index_buffer.target, index_buffer.handle)

        stride = 0
        attribs = []
        if layout is not None:
            stride = int(layout.get("stride", 0))
            raw_attribs = layout.get("attributes", [])
            if isinstance(raw_attribs, Sequence):
                for entry in raw_attribs:
                    if not isinstance(entry, Sequence) or len(entry) != 5:
                        continue
                    location, size, atype, normalised, offset = entry
                    attribs.append(
                        (
                            int(location),
                            int(size),
                            int(atype),
                            1 if normalised else 0,
                            int(offset),
                        )
                    )

        for entry in attribs:
            location, size, atype, normalised, offset = entry
            self.glEnableVertexAttribArray(GLuint(location))
            self.glVertexAttribPointer(
                GLuint(location),
                GLint(size),
                GLenum(atype),
                GLboolean(normalised),
                GLsizei(stride),
                ctypes.c_void_p(offset),
            )

        self.glBindVertexArray(0)
        self.glBindBuffer(GL_ARRAY_BUFFER, 0)
        if index_buffer is None:
            self.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        return VertexArray(self, int(array_id.value), attribs)

    def delete_vertex_array(self, handle: int) -> None:
        if handle == 0:
            return
        array_id = GLuint(handle)
        self.glDeleteVertexArrays(1, ctypes.byref(array_id))


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


@dataclass
class ShaderProgram:
    bindings: GLBindings
    handle: int

    def use(self) -> None:
        self.bindings.glUseProgram(self.handle)

    def get_uniform_location(self, name: str) -> int:
        location = self.bindings.glGetUniformLocation(self.handle, name.encode("utf-8"))
        return int(location)

    def release(self) -> None:
        if self.handle:
            self.bindings.glDeleteProgram(self.handle)
            self.handle = 0


@dataclass
class VertexArray:
    bindings: GLBindings
    handle: int
    attributes: Sequence[tuple[int, int, int, int, int]]

    def bind(self) -> None:
        self.bindings.glBindVertexArray(self.handle)

    def release(self) -> None:
        if self.handle:
            self.bindings.delete_vertex_array(self.handle)
            self.handle = 0
