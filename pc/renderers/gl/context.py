"""Utilities for creating headless OpenGL contexts.

The goal for the upcoming GL renderer is to run completely offscreen, both on
standard developer workstations and on Jetson devices.  The helpers below focus
on creating EGL backed pixel buffers and expose a small abstraction that keeps
resource lifetime management contained in one place.

The implementation deliberately avoids external Python dependencies so that the
code can run in minimal environments (e.g. the default Jetson rootfs).  Only the
subset of EGL calls required for an offscreen pixel buffer are wrapped.  The
helper raises :class:`HeadlessContextError` with descriptive messages whenever an
initialisation step fails so the caller can fall back to the CPU renderer and
log the specific reason.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

__all__ = [
    "HeadlessContextError",
    "HeadlessGLContext",
    "create_headless_context",
]


_LOGGER = logging.getLogger(__name__)


class HeadlessContextError(RuntimeError):
    """Raised when the headless GL context fails to initialise."""


def _compat_dataclass(*args, **kwargs):
    """Return ``dataclass`` with ``slots`` where supported."""

    if sys.version_info >= (3, 10):
        kwargs.setdefault("slots", True)
    return dataclass(*args, **kwargs)


@_compat_dataclass()
class HeadlessGLContext:
    """Simple wrapper around an underlying EGL context."""

    width: int
    height: int
    backend: str
    api: str
    samples: int
    _impl: Optional["_EGLContext"] = field(init=False, default=None, repr=False)

    def attach_impl(self, impl: "_EGLContext") -> None:
        self._impl = impl

    # The low level context is reference counted implicitly by the owning
    # renderer.  Exposing a ``close`` method keeps destruction explicit and easy
    # to invoke from ``SimCamera`` shutdown hooks when those are added.
    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
            self._impl = None

    def make_current(self) -> None:
        if self._impl is None:
            raise HeadlessContextError("GL context is not initialised")
        self._impl.make_current()

    def get_proc_address(self, name: str) -> int:
        if self._impl is None:
            raise HeadlessContextError("GL context is not initialised")
        return self._impl.get_proc_address(name)

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


def create_headless_context(
    width: int,
    height: int,
    *,
    backend: str = "auto",
    msaa_samples: int = 0,
    logger: Optional[logging.Logger] = None,
) -> HeadlessGLContext:
    """Create a headless GL context for ``width``×``height`` rendering."""

    backend_name = (backend or "auto").strip().lower()
    samples = max(0, int(msaa_samples))

    context = HeadlessGLContext(
        width=int(width),
        height=int(height),
        backend=backend_name,
        api="",
        samples=samples,
    )

    errors: List[str] = []

    attempts: List[tuple[str, str]] = []
    if backend_name in {"auto", "egl", "desktop"}:
        attempts.append(("egl", "opengl"))
    if backend_name in {"auto", "egl", "jetson", "gles", "opengles"}:
        attempts.append(("egl", "opengles"))

    if not attempts:
        raise HeadlessContextError(f"Unsupported renderer backend '{backend_name}'")

    try:
        egl_loader = _EGLLoader()
    except HeadlessContextError as exc:
        if os.name == "nt":
            raise HeadlessContextError(
                f"{exc}. Install an EGL runtime (ANGLE/mesa-dist-win) or set"
                " EGL_LIBRARY; WGL is not supported for headless rendering"
            ) from exc
        raise

    for driver, api in attempts:
        try:
            impl = _EGLContext(
                egl_loader,
                width=context.width,
                height=context.height,
                samples=context.samples,
                api=api,
            )
        except HeadlessContextError as exc:
            errors.append(f"{driver}/{api}: {exc}")
            continue

        context.backend = driver
        context.api = api
        context.attach_impl(impl)
        active_logger = logger or _LOGGER
        active_logger.info(
            "Created headless GL context via %s/%s (samples=%d)",
            driver,
            api,
            context.samples,
        )
        return context

    raise HeadlessContextError("; ".join(errors))


# --------------------------------------------------------------------------- EGL


def _load_egl_library() -> ctypes.CDLL:
    env_path = os.environ.get("EGL_LIBRARY")
    candidates: List[str] = []
    if env_path:
        candidates.append(env_path)

    found = ctypes.util.find_library("EGL")
    if found:
        candidates.append(found)

    candidates.extend(
        [
            "libEGL.so.1",
            "libEGL.so",
        ]
    )

    if os.name == "nt":
        # Windows builds typically ship ANGLE or vendor supplied ``libEGL.dll``
        # next to the executable.  Probe the common dll names explicitly so a
        # simple copy of those runtime binaries is enough to satisfy the loader.
        dll_names = ["libEGL.dll", "EGL.dll"]
        system_root = os.environ.get("SystemRoot")
        for dll in dll_names:
            candidates.append(dll)
            if system_root:
                candidates.append(os.path.join(system_root, "System32", dll))
                candidates.append(os.path.join(system_root, "SysWOW64", dll))

    errors: List[str] = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    raise HeadlessContextError(
        "Unable to load libEGL. Tried: " + ", ".join(errors or ["<none>"])
    )


class _EGLLoader:
    """Lazy loader that exposes the handful of EGL entry points we use."""

    def __init__(self) -> None:
        self.lib = _load_egl_library()
        self._setup_prototypes()

    def _setup_prototypes(self) -> None:
        lib = self.lib

        # Basic type aliases to keep signatures tidy.
        EGLNativeDisplayType = ctypes.c_void_p
        EGLDisplay = ctypes.c_void_p
        EGLConfig = ctypes.c_void_p
        EGLContext = ctypes.c_void_p
        EGLSurface = ctypes.c_void_p
        EGLBoolean = ctypes.c_uint
        EGLint = ctypes.c_int

        lib.eglGetDisplay.argtypes = [EGLNativeDisplayType]
        lib.eglGetDisplay.restype = EGLDisplay

        lib.eglInitialize.argtypes = [EGLDisplay, ctypes.POINTER(EGLint), ctypes.POINTER(EGLint)]
        lib.eglInitialize.restype = EGLBoolean

        lib.eglTerminate.argtypes = [EGLDisplay]
        lib.eglTerminate.restype = EGLBoolean

        lib.eglBindAPI.argtypes = [EGLint]
        lib.eglBindAPI.restype = EGLBoolean

        lib.eglGetError.argtypes = []
        lib.eglGetError.restype = EGLint

        lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        lib.eglGetProcAddress.restype = ctypes.c_void_p

        lib.eglChooseConfig.argtypes = [
            EGLDisplay,
            ctypes.POINTER(EGLint),
            ctypes.POINTER(EGLConfig),
            EGLint,
            ctypes.POINTER(EGLint),
        ]
        lib.eglChooseConfig.restype = EGLBoolean

        lib.eglCreatePbufferSurface.argtypes = [EGLDisplay, EGLConfig, ctypes.POINTER(EGLint)]
        lib.eglCreatePbufferSurface.restype = EGLSurface

        lib.eglDestroySurface.argtypes = [EGLDisplay, EGLSurface]
        lib.eglDestroySurface.restype = EGLBoolean

        lib.eglCreateContext.argtypes = [
            EGLDisplay,
            EGLConfig,
            EGLContext,
            ctypes.POINTER(EGLint),
        ]
        lib.eglCreateContext.restype = EGLContext

        lib.eglDestroyContext.argtypes = [EGLDisplay, EGLContext]
        lib.eglDestroyContext.restype = EGLBoolean

        lib.eglMakeCurrent.argtypes = [EGLDisplay, EGLSurface, EGLSurface, EGLContext]
        lib.eglMakeCurrent.restype = EGLBoolean

        lib.eglReleaseThread.argtypes = []
        lib.eglReleaseThread.restype = EGLBoolean

    # Convenience wrappers -------------------------------------------------
    def __getattr__(self, item: str) -> ctypes.CDLL:
        return getattr(self.lib, item)


# EGL constant values -----------------------------------------------------------
EGL_DEFAULT_DISPLAY = ctypes.c_void_p(0)
EGL_NO_DISPLAY = ctypes.c_void_p(0)
EGL_NO_CONTEXT = ctypes.c_void_p(0)
EGL_NO_SURFACE = ctypes.c_void_p(0)

EGL_NONE = 0x3038
EGL_PBUFFER_BIT = 0x0001
EGL_SURFACE_TYPE = 0x3033
EGL_RENDERABLE_TYPE = 0x3040
EGL_RED_SIZE = 0x3024
EGL_GREEN_SIZE = 0x3023
EGL_BLUE_SIZE = 0x3022
EGL_ALPHA_SIZE = 0x3021
EGL_DEPTH_SIZE = 0x3025
EGL_STENCIL_SIZE = 0x3026
EGL_SAMPLE_BUFFERS = 0x3032
EGL_SAMPLES = 0x3031
EGL_WIDTH = 0x3057
EGL_HEIGHT = 0x3056

EGL_OPENGL_API = 0x30A2
EGL_OPENGL_ES_API = 0x30A0
EGL_OPENGL_BIT = 0x0008
EGL_OPENGL_ES2_BIT = 0x0004
EGL_OPENGL_ES3_BIT = 0x00000040  # KHR extension value used on Jetson
EGL_CONTEXT_CLIENT_VERSION = 0x3098


class _EGLContext:
    """Represents a single EGL pixel buffer context."""

    def __init__(
        self,
        loader: _EGLLoader,
        *,
        width: int,
        height: int,
        samples: int,
        api: str,
    ) -> None:
        self._egl = loader
        self.width = int(width)
        self.height = int(height)
        self.samples = max(0, int(samples))
        self.api = api

        self._display = EGL_NO_DISPLAY
        self._context = EGL_NO_CONTEXT
        self._surface = EGL_NO_SURFACE
        self._gl_lib: Optional[ctypes.CDLL] = None
        self._gl_lib_failed = False

        self._initialise()

    # ------------------------------------------------------------------ setup
    def _initialise(self) -> None:
        egl = self._egl

        display = egl.eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if display == EGL_NO_DISPLAY:
            raise HeadlessContextError("eglGetDisplay returned EGL_NO_DISPLAY")

        major = ctypes.c_int()
        minor = ctypes.c_int()
        if not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
            raise HeadlessContextError(
                f"eglInitialize failed (error=0x{egl.eglGetError():04x})"
            )

        api_constant, renderable_bits, context_attribs = self._select_api(self.api)

        if not egl.eglBindAPI(api_constant):
            raise HeadlessContextError(
                f"eglBindAPI({api_constant:#x}) failed (error=0x{egl.eglGetError():04x})"
            )

        config = self._choose_config(display, renderable_bits)
        surface = self._create_surface(display, config)
        context = self._create_context(display, config, context_attribs)

        if not egl.eglMakeCurrent(display, surface, surface, context):
            raise HeadlessContextError(
                f"eglMakeCurrent failed (error=0x{egl.eglGetError():04x})"
            )

        self._display = display
        self._context = context
        self._surface = surface

    def get_proc_address(self, name: str) -> int:
        if not name:
            return 0

        egl = self._egl
        encoded = name.encode("ascii")
        proc = egl.eglGetProcAddress(encoded)
        if proc:
            value = ctypes.cast(proc, ctypes.c_void_p).value
            if value:
                return int(value)

        if self._gl_lib is None and not self._gl_lib_failed:
            self._gl_lib = self._load_gl_library()
            if self._gl_lib is None:
                self._gl_lib_failed = True

        if self._gl_lib is not None:
            try:
                symbol = getattr(self._gl_lib, name)
            except AttributeError:
                return 0
            value = ctypes.cast(symbol, ctypes.c_void_p).value
            return int(value) if value else 0

        return 0

    def _load_gl_library(self) -> Optional[ctypes.CDLL]:
        candidates: List[str] = []
        env_value = os.environ.get("GL_LIBRARY")
        if env_value:
            candidates.append(env_value)

        if self.api == "opengles":
            candidates.extend(
                [
                    "libGLESv3.so",
                    "libGLESv2.so.2",
                    "libGLESv2.so",
                ]
            )
        else:
            found = ctypes.util.find_library("OpenGL")
            if found:
                candidates.append(found)
            candidates.extend(
                [
                    "libOpenGL.so.0",
                    "libGL.so.1",
                    "libGL.so",
                ]
            )

        errors: List[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return ctypes.CDLL(candidate)
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")

        if errors:
            _LOGGER.debug("Unable to load GL library (%s)", ", ".join(errors))
        return None

    # ---------------------------------------------------------------- helpers
    def _select_api(self, api: str) -> tuple[int, int, Sequence[int]]:
        requested = (api or "opengl").strip().lower()
        if requested == "opengles":
            context_attribs = [EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE]
            return EGL_OPENGL_ES_API, EGL_OPENGL_ES3_BIT | EGL_OPENGL_ES2_BIT, context_attribs

        # Default to desktop OpenGL.  We do not request a specific profile here
        # to keep the setup portable; drivers will usually expose at least 3.3.
        context_attribs = [EGL_NONE]
        return EGL_OPENGL_API, EGL_OPENGL_BIT, context_attribs

    def _choose_config(self, display: ctypes.c_void_p, renderable_bits: int) -> ctypes.c_void_p:
        egl = self._egl

        attribs: List[int] = [
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE,
            renderable_bits,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_ALPHA_SIZE,
            8,
            EGL_DEPTH_SIZE,
            24,
            EGL_STENCIL_SIZE,
            8,
        ]

        if self.samples > 0:
            attribs.extend([EGL_SAMPLE_BUFFERS, 1, EGL_SAMPLES, self.samples])

        attribs.append(EGL_NONE)

        attrib_array = (ctypes.c_int * len(attribs))(*attribs)
        configs = (ctypes.c_void_p * 1)()
        num_configs = ctypes.c_int()

        if not egl.eglChooseConfig(
            display,
            attrib_array,
            configs,
            1,
            ctypes.byref(num_configs),
        ):
            raise HeadlessContextError(
                f"eglChooseConfig failed (error=0x{egl.eglGetError():04x})"
            )

        if num_configs.value < 1:
            raise HeadlessContextError("No suitable EGL configs were found")

        return configs[0]

    def _create_surface(self, display: ctypes.c_void_p, config: ctypes.c_void_p) -> ctypes.c_void_p:
        egl = self._egl
        attribs = (ctypes.c_int * 5)(EGL_WIDTH, self.width, EGL_HEIGHT, self.height, EGL_NONE)
        surface = egl.eglCreatePbufferSurface(display, config, attribs)
        if surface == EGL_NO_SURFACE:
            raise HeadlessContextError(
                f"eglCreatePbufferSurface failed (error=0x{egl.eglGetError():04x})"
            )
        return surface

    def _create_context(
        self,
        display: ctypes.c_void_p,
        config: ctypes.c_void_p,
        context_attribs: Sequence[int],
    ) -> ctypes.c_void_p:
        egl = self._egl
        attrib_array = (ctypes.c_int * len(context_attribs))(*context_attribs)
        context = egl.eglCreateContext(display, config, EGL_NO_CONTEXT, attrib_array)
        if context == EGL_NO_CONTEXT:
            raise HeadlessContextError(
                f"eglCreateContext failed (error=0x{egl.eglGetError():04x})"
            )
        return context

    # ---------------------------------------------------------------- control
    def make_current(self) -> None:
        egl = self._egl
        if not egl.eglMakeCurrent(self._display, self._surface, self._surface, self._context):
            raise HeadlessContextError(
                f"eglMakeCurrent failed (error=0x{egl.eglGetError():04x})"
            )

    def close(self) -> None:
        egl = self._egl
        try:
            if self._display != EGL_NO_DISPLAY:
                egl.eglMakeCurrent(self._display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
                if self._context != EGL_NO_CONTEXT:
                    egl.eglDestroyContext(self._display, self._context)
                if self._surface != EGL_NO_SURFACE:
                    egl.eglDestroySurface(self._display, self._surface)
                egl.eglTerminate(self._display)
        finally:
            egl.eglReleaseThread()
            self._display = EGL_NO_DISPLAY
            self._context = EGL_NO_CONTEXT
            self._surface = EGL_NO_SURFACE
