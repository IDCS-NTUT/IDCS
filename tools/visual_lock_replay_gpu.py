"""Test-only host-fixture upload to real CUDA pitch or NVMM/EGL input.

One allocation, no queue. Never use this to read back production GPU frames.
The standalone process's explicit EGL owner must outlive all adapters.
"""
import ctypes as ct
from pathlib import Path

import numpy as np

from jetson.visual_lock.backend import VPIBackend


class ReplayGPUBackend:
    def __init__(self,library,width,height,*,kind,egl_owner,surface_library=None):
        if kind not in ("cuda","egl") or not getattr(egl_owner,"active",False):
            raise RuntimeError("GPU replay needs an active standalone EGL owner and valid kind")
        if not 64 <= width <= 4096 or not 64 <= height <= 2160:
            raise ValueError("unsupported replay frame dimensions")
        self.kind,self.width,self.height = kind,width,height
        self._egl_owner = egl_owner
        self._backend = None
        self._pointer = ct.c_void_p()
        self._surface = None
        self._image = None
        self._bytes = width*height*4
        self._pending = False
        try:
            if kind == "cuda":
                self._memory = ct.CDLL("libcudart.so")
                self._memory.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p),ct.c_size_t]
                self._memory.cudaMemcpy.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int]
                self._memory.cudaFree.argtypes = [ct.c_void_p]
                self._check_cuda(self._memory.cudaMalloc(ct.byref(self._pointer),self._bytes))
            else:
                if not surface_library: raise ValueError("test surface library required")
                lib = self._memory = ct.CDLL(str(Path(surface_library).resolve()))
                lib.vlt_error.restype = ct.c_char_p
                lib.vlt_create.argtypes = [ct.c_int,ct.c_int]
                lib.vlt_create.restype = ct.c_void_p
                lib.vlt_egl_image.argtypes = [ct.c_void_p]
                lib.vlt_egl_image.restype = ct.c_void_p
                lib.vlt_upload.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_uint64]
                lib.vlt_destroy.argtypes = [ct.c_void_p]
                self._surface = lib.vlt_create(width,height)
                if not self._surface: self._check_surface(-1)
                self._pointer = ct.c_void_p(lib.vlt_egl_image(self._surface))
            self._backend = VPIBackend(library,width,height,allow_experimental_gpu_interop=True)
        except BaseException:
            self.close()
            raise

    def _check_cuda(self,status):
        if status: raise RuntimeError(f"CUDA replay memory API failed: {status}")

    @property
    def seed_min_points(self): return self._backend.seed_min_points

    @seed_min_points.setter
    def seed_min_points(self, value): self._backend.seed_min_points = value

    def _check_surface(self,status):
        if status: raise RuntimeError(self._memory.vlt_error().decode())

    def process(self,rgba,*,seed=None):
        if not self._backend or not self._egl_owner.active:
            raise RuntimeError("GPU replay owner/backend closed")
        if self._pending:
            raise RuntimeError("previous proposal not committed/discarded; input retained")
        if rgba.dtype != np.uint8 or rgba.shape != (self.height,self.width,4):
            raise ValueError("expected uint8 HxWx4 RGBA")
        image = np.ascontiguousarray(rgba)
        if self.kind == "cuda":
            self._check_cuda(self._memory.cudaMemcpy(self._pointer,image.ctypes.data,self._bytes,1))
        else:
            self._check_surface(self._memory.vlt_upload(self._surface,image.ctypes.data,self._bytes))
        result = self._backend.process_pointer(self._pointer.value,self.width*4,
                                              kind=1 if self.kind == "cuda" else 2,seed=seed)
        self._pending = seed is None
        return result

    def commit(self,*args,**kwargs):
        result = self._backend.commit(*args,**kwargs)
        self._pending = False
        return result

    def discard(self):
        result = self._backend.discard()
        self._pending = False
        return result

    def close(self):
        # Wrapper/stream destruction precedes CUDA free or EGL unmap/NVMM release.
        if self._backend:
            self._backend.close()
            self._backend = None
        if self.kind == "cuda" and self._pointer.value:
            pointer,self._pointer = self._pointer,ct.c_void_p()
            self._check_cuda(self._memory.cudaFree(pointer))
        elif self._surface:
            surface,self._surface = self._surface,None
            self._pointer = ct.c_void_p()
            self._check_surface(self._memory.vlt_destroy(surface))

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
