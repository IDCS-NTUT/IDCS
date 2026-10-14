"""Single-frame synchronous GPU borrowing; no host pixel readback or queue."""
import ctypes as ct
from pathlib import Path

from .backend import VPIBackend


class NVMMBorrower:
    def __init__(self,library):
        self.lib = ct.CDLL(str(Path(library).resolve()))
        self.lib.vll_error.restype = ct.c_char_p
        self.lib.vll_borrow.argtypes = [ct.c_void_p]
        self.lib.vll_borrow.restype = ct.c_void_p
        self.lib.vll_image.argtypes = [ct.c_void_p]
        self.lib.vll_image.restype = ct.c_void_p
        self.lib.vll_release.argtypes = [ct.c_void_p]

    def borrow(self,buffer):
        handle = self.lib.vll_borrow(hash(buffer))
        if not handle: raise RuntimeError(self.lib.vll_error().decode())
        return BorrowedImage(self.lib,handle)

    def clone(self,buffer):
        # Resolve the new ABI only when an opt-in panel trial needs it. Older
        # fixed-seed trials remain compatible with their existing adapter.
        self.lib.vll_clone.argtypes = [ct.c_void_p]
        self.lib.vll_clone.restype = ct.c_void_p
        self.lib.vll_clone_image.argtypes = [ct.c_void_p]
        self.lib.vll_clone_image.restype = ct.c_void_p
        self.lib.vll_clone_release.argtypes = [ct.c_void_p]
        self.lib.vll_clone_release.restype = ct.c_int
        handle = self.lib.vll_clone(hash(buffer))
        if not handle: raise RuntimeError(self.lib.vll_error().decode())
        return ClonedImage(self.lib,handle)


class BorrowedImage:
    def __init__(self,lib,handle):
        self.lib,self.handle = lib,handle
        self.pointer = lib.vll_image(handle)

    def __enter__(self): return self

    def __exit__(self,*_):
        handle,self.handle = self.handle,None
        if self.lib.vll_release(handle): raise RuntimeError(self.lib.vll_error().decode())


class ClonedImage:
    def __init__(self,lib,handle):
        self.lib,self.handle = lib,handle
        self.pointer = lib.vll_clone_image(handle)

    def close(self):
        if self.handle is not None:
            handle,self.handle = self.handle,None
            if self.lib.vll_clone_release(handle): raise RuntimeError(self.lib.vll_error().decode())

    def __enter__(self): return self

    def __exit__(self,*_): self.close()


class BorrowedBackend(VPIBackend):
    def __init__(self,library,width,height):
        super().__init__(library,width,height,allow_experimental_gpu_interop=True)

    def process(self,image,*,seed=None):
        result = self.process_pointer(image.pointer,0,kind=2,seed=seed)
        if seed is not None: self.release_input()
        return result

    def commit(self,*args,**kwargs):
        super().commit(*args,**kwargs)
        self.release_input()

    def discard(self):
        super().discard()
        self.release_input()
