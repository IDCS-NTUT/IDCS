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
from .seed import refine_seed


class _Result(ct.Structure):
    _fields_ = [("box", ct.c_float * 4), ("psr", ct.c_float), ("peak", ct.c_float),
                ("count", ct.c_int32), ("before", ct.c_float * 128),
                ("after", ct.c_float * 128), ("fb", ct.c_float * 64),
                ("elapsed_ms", ct.c_double), ("point_ids", ct.c_uint64 * 64),
                ("input_count",ct.c_int32), ("forward_count",ct.c_int32), ("backward_count",ct.c_int32),
                ("reference_count",ct.c_int32),("reference_before",ct.c_float*128),
                ("reference_after",ct.c_float*128),("reference_fb",ct.c_float*64)]


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
        if lib.vl_abi_version() != 4:
            raise RuntimeError("unsupported visual-lock native ABI")
        lib.vl_error.restype = ct.c_char_p
        lib.vl_create.argtypes = [ct.c_int, ct.c_int]
        lib.vl_create.restype = ct.c_void_p
        lib.vl_destroy.argtypes = [ct.c_void_p]
        lib.vl_destroy.restype = None
        lib.vl_process.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int,
                                  ct.POINTER(ct.c_float), ct.POINTER(_Result)]
        lib.vl_process.restype = ct.c_int
        lib.vl_commit.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int,ct.c_uint64,ct.c_int,
                                 ct.POINTER(ct.c_int32)]
        lib.vl_commit.restype = ct.c_int
        lib.vl_discard.argtypes = [ct.c_void_p]
        lib.vl_discard.restype = ct.c_int
        lib.vl_release_input.argtypes = [ct.c_void_p]
        lib.vl_release_input.restype = ct.c_int
        self._handle = lib.vl_create(width, height)
        if not self._handle:
            self._raise()
        self.commit_stats = None
        self.seed_min_points = 8

    def _probe_seed(self, pointer, pitch, kind, seed):
        # Resolve only for seeds; ABI 4 process/commit layouts are unchanged.
        try:
            probe = self._lib.vl_seed_probe
        except AttributeError as exc:
            raise RuntimeError("rebuild native library for additive seed probe API") from exc
        probe.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int,
            ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p, ct.c_int,
            ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int,
            ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
        probe.restype = ct.c_int
        xy = np.empty((20000, 2), dtype=np.float32)
        scores = np.empty(20000, dtype=np.uint32)
        crop = np.empty(self.width*self.height, dtype=np.uint8)
        n, w, h = ct.c_int(), ct.c_int(), ct.c_int()
        if probe(self._handle, pointer, pitch, kind, (ct.c_float*4)(*seed),
                 xy.ctypes.data, scores.ctypes.data, len(xy), ct.byref(n),
                 crop.ctypes.data, len(crop), ct.byref(w), ct.byref(h)):
            self._raise()
        if not 0 <= n.value <= 20000 or not 0 < w.value*h.value <= len(crop):
            raise RuntimeError("corrupt seed evidence")
        report = refine_seed(seed, xy[:n.value], scores[:n.value], min_points=self.seed_min_points)
        return report, crop[:w.value*h.value].reshape(h.value, w.value).copy()

    def _raise(self):
        raise RuntimeError(self._lib.vl_error().decode("utf-8", errors="replace"))

    def process_pointer(self, pointer: int, pitch: int, *, kind: int, seed=None) -> Proposal:
        if kind != 0 and not self._allow_gpu_interop:
            raise RuntimeError("live GPU pointer input is unqualified; explicit experimental opt-in required")
        if not self._handle:
            raise RuntimeError("tracker closed")
        report = gray = None
        if seed is not None:
            report, gray = self._probe_seed(pointer, pitch, kind, seed)
            if report.refined_box is None:
                return Proposal(tuple(seed), 0., 0., np.empty((0, 2)), np.empty((0, 2)),
                                np.empty(0), seed_report=report, seed_gray=gray)
            seed = report.refined_box
        r = _Result()
        b = (ct.c_float * 4)(*seed) if seed is not None else None
        if self._lib.vl_process(self._handle, pointer, pitch, kind, b, ct.byref(r)):
            self._raise()
        n = r.count
        rn = r.reference_count
        if not 0 <= n <= 64 or not 0 <= rn <= 64:
            raise RuntimeError("corrupt native point count")
        return Proposal(tuple(r.box), r.psr, r.peak,
                        np.array(r.before[:2*n]).reshape(-1, 2),
                        np.array(r.after[:2*n]).reshape(-1, 2),
                        np.array(r.fb[:n]), r.elapsed_ms, n,np.array(r.point_ids[:n],dtype=np.uint64),
                        r.input_count,r.forward_count,r.backward_count,
                        np.array(r.reference_before[:2*rn]).reshape(-1,2),
                        np.array(r.reference_after[:2*rn]).reshape(-1,2),np.array(r.reference_fb[:rn]),
                        report, gray)

    def process(self, rgba: np.ndarray, *, seed=None) -> Proposal:
        if rgba.dtype != np.uint8 or rgba.shape != (self.height, self.width, 4):
            raise ValueError("expected uint8 HxWx4 RGBA frame")
        # Replay only: explicit host buffer. Production callers use GPU pointers.
        image = np.ascontiguousarray(rgba)
        result = self.process_pointer(image.ctypes.data, image.strides[0], kind=0, seed=seed)
        # A rejected second proposal must not free the pending input allocation.
        self._image = image
        return result

    def commit(self, box, *, update_template: bool, retain_mask=None, replenish=True):
        if not self._handle:
            raise RuntimeError("tracker closed")
        stats = (ct.c_int32 * 3)()
        mask = (1 << 64)-1 if retain_mask is None else retain_mask
        if self._lib.vl_commit(self._handle, (ct.c_float * 4)(*box), int(update_template),mask,int(replenish),stats):
            self._raise()
        self.commit_stats = dict(retained=stats[0],added=stats[1],total=stats[2])
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

    def release_input(self):
        if not self._handle:
            raise RuntimeError("tracker closed")
        if self._lib.vl_release_input(self._handle):
            self._raise()
        self._image = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
