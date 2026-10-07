"""Small ctypes boundary to VPI; OpenCV is NOT the tracking backend.

Host RGBA supports recorded replay. CUDA/EGL input passes standalone tests with
early EGL initialization, but live pipeline frame ownership remains experimental.
Callers must retain their allocation through commit/discard and supply proper
process-level EGL initialization/lifetime; standalone replay is not live qualification.
Each instance is owned by exactly one thread. No background queue is created.
"""
from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np

from .fusion import Proposal


class _Result(ct.Structure):
    _fields_ = [("box", ct.c_float * 4), ("psr", ct.c_float), ("peak", ct.c_float),
                ("count", ct.c_int32), ("before", ct.c_float * 128),
                ("after", ct.c_float * 128), ("fb", ct.c_float * 64),
                ("elapsed_ms", ct.c_double)]


class VPIBackend:
    def __init__(self, library: str | Path, width: int, height: int, *,
                 allow_experimental_gpu_interop: bool = False):
        self.width, self.height = width, height
        self._allow_gpu_interop = allow_experimental_gpu_interop
        self._image = None
        self._handle = None
        self._lib = ct.CDLL(str(Path(library).resolve()))
        lib = self._lib
        lib.vl_abi_version.restype = ct.c_int
        if lib.vl_abi_version() != 1:
            raise RuntimeError("unsupported visual-lock native ABI")
        lib.vl_error.restype = ct.c_char_p
        lib.vl_create.argtypes = [ct.c_int, ct.c_int]
        lib.vl_create.restype = ct.c_void_p
        lib.vl_destroy.argtypes = [ct.c_void_p]
        lib.vl_destroy.restype = None
        lib.vl_process.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int,
                                  ct.POINTER(ct.c_float), ct.POINTER(_Result)]
        lib.vl_process.restype = ct.c_int
        lib.vl_commit.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int]
        lib.vl_commit.restype = ct.c_int
        lib.vl_discard.argtypes = [ct.c_void_p]
        lib.vl_discard.restype = ct.c_int
        self._handle = lib.vl_create(width, height)
        if not self._handle:
            self._raise()

    def _raise(self):
        raise RuntimeError(self._lib.vl_error().decode("utf-8", errors="replace"))

    def process_pointer(self, pointer: int, pitch: int, *, kind: int, seed=None) -> Proposal:
        if kind != 0 and not self._allow_gpu_interop:
            raise RuntimeError("live GPU pointer input is unqualified; explicit experimental opt-in required")
        if not self._handle:
            raise RuntimeError("tracker closed")
        r = _Result()
        b = (ct.c_float * 4)(*seed) if seed is not None else None
        if self._lib.vl_process(self._handle, pointer, pitch, kind, b, ct.byref(r)):
            self._raise()
        n = r.count
        if not 0 <= n <= 64:
            raise RuntimeError("corrupt native point count")
        return Proposal(tuple(r.box), r.psr, r.peak,
                        np.array(r.before[:2*n]).reshape(-1, 2),
                        np.array(r.after[:2*n]).reshape(-1, 2),
                        np.array(r.fb[:n]), r.elapsed_ms, n)

    def process(self, rgba: np.ndarray, *, seed=None) -> Proposal:
        if rgba.dtype != np.uint8 or rgba.shape != (self.height, self.width, 4):
            raise ValueError("expected uint8 HxWx4 RGBA frame")
        # Replay only: explicit host buffer. Production callers use GPU pointers.
        image = np.ascontiguousarray(rgba)
        result = self.process_pointer(image.ctypes.data, image.strides[0], kind=0, seed=seed)
        # A rejected second proposal must not free the pending input allocation.
        self._image = image
        return result

    def commit(self, box, *, update_template: bool):
        if not self._handle:
            raise RuntimeError("tracker closed")
        if self._lib.vl_commit(self._handle, (ct.c_float * 4)(*box), int(update_template)):
            self._raise()
        self._image = None

    def discard(self):
        if self._handle and self._lib.vl_discard(self._handle):
            self._raise()
        self._image = None

    def close(self):
        if self._handle:
            self._lib.vl_destroy(self._handle)
            self._handle = None
        self._image = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
