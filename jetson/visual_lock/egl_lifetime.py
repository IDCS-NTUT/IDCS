"""Explicit EGL ownership for a fresh, standalone GPU verification process.

Initialize BEFORE loading/using CUDA or VPI, and terminate AFTER every wrapper,
surface and allocation is released. Never insert this owner into an existing
DeepStream process: its pipeline must own its shared EGL display/lifetime.
This performs real initialization/cleanup; it does not skip process finalizers.
"""
from __future__ import annotations

import ctypes as ct
import threading


class StandaloneEGL:
    _mutex = threading.Lock()
    _owner = None

    def __init__(self, *, loader=ct.CDLL):
        self._loader = loader
        self._egl = None
        self._display = None
        self._thread = None
        self.version = None
        self.vendor = None

    @property
    def active(self):
        return self._display is not None and self._thread == threading.get_ident()

    def __enter__(self):
        with self._mutex:
            if StandaloneEGL._owner is not None:
                raise RuntimeError("standalone EGL owner already active")
            StandaloneEGL._owner = self
        try:
            lib = self._loader("libEGL.so.1")
            self._egl = lib
            lib.eglGetDisplay.argtypes = [ct.c_void_p]
            lib.eglGetDisplay.restype = ct.c_void_p
            lib.eglInitialize.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int),ct.POINTER(ct.c_int)]
            lib.eglInitialize.restype = ct.c_uint
            lib.eglTerminate.argtypes = [ct.c_void_p]
            lib.eglTerminate.restype = ct.c_uint
            lib.eglGetError.restype = ct.c_uint
            lib.eglQueryString.argtypes = [ct.c_void_p,ct.c_int]
            lib.eglQueryString.restype = ct.c_char_p
            display = lib.eglGetDisplay(None)
            major,minor = ct.c_int(),ct.c_int()
            if not display or not lib.eglInitialize(display,ct.byref(major),ct.byref(minor)):
                raise RuntimeError(f"EGL initialization failed: 0x{lib.eglGetError():04x}")
            self._display = display
            self._thread = threading.get_ident()
            self.version = (major.value,minor.value)
            self.vendor = (lib.eglQueryString(display,0x3053) or b"unknown").decode()
            if "NVIDIA" not in self.vendor:
                raise RuntimeError(f"NVIDIA EGL required, got {self.vendor}")
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._display is not None and self._thread != threading.get_ident():
            raise RuntimeError("EGL owner must close on its owning thread")
        display,self._display = self._display,None
        try:
            if display is not None and not self._egl.eglTerminate(display):
                raise RuntimeError(f"EGL termination failed: 0x{self._egl.eglGetError():04x}")
        finally:
            self._thread = None
            with self._mutex:
                if StandaloneEGL._owner is self:
                    StandaloneEGL._owner = None

    def __exit__(self,*_):
        self.close()
