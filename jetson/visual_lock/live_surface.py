"""Synchronous GPU borrowing; base backend reads gray only for seed evidence."""
import ctypes as ct
import os
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
        # Trial switch (default off): hand the tracker a CUDA pointer to each
        # pool frame, registered once per pool buffer, instead of an EGLImage
        # wrapped per frame. Frames that cannot be registered use the EGLImage.
        self.cuda_input = os.environ.get("IDCS_VLL_CUDA_INPUT") == "1"
        self.cuda_fallbacks = 0
        if self.cuda_input:
            self.lib.vll_cuda.argtypes = [ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_int)]
            self.lib.vll_cuda.restype = ct.c_int

    def borrow(self,buffer):
        handle = self.lib.vll_borrow(hash(buffer))
        if not handle: raise RuntimeError(self.lib.vll_error().decode())
        image = BorrowedImage(self.lib,handle)
        if self.cuda_input:
            pointer,pitch = ct.c_void_p(),ct.c_int()
            if self.lib.vll_cuda(handle,ct.byref(pointer),ct.byref(pitch)):
                self.cuda_fallbacks += 1
                self.cuda_error = self.lib.vll_error().decode()
            else:
                image.cuda = (pointer.value,pitch.value)
        return image

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
    cuda = None

    def __init__(self,lib,handle):
        self.lib,self.handle = lib,handle
        self.pointer = lib.vll_image(handle)

    def __enter__(self): return self

    def __exit__(self,*_):
        handle,self.handle = self.handle,None
        if self.lib.vll_release(handle): raise RuntimeError(self.lib.vll_error().decode())


class ClonedImage:
    cuda = None

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
        if image.cuda is not None:
            result = self.process_pointer(*image.cuda,kind=1,seed=seed)
        else:
            result = self.process_pointer(image.pointer,0,kind=2,seed=seed)
        if seed is not None: self.release_input()
        return result

    def commit(self,*args,**kwargs):
        super().commit(*args,**kwargs)
        self.release_input()

    def discard(self):
        super().discard()
        self.release_input()


class CudaInputVerifier:
    """Trial check: the registered CUDA view of a live frame equals a fresh EGLImage wrapper.

    Reads the full-frame gray image both ways through a separate, never-seeded
    backend. Only for opt-in trials; each check costs a few milliseconds.
    """
    def __init__(self,library,width=1280,height=720):
        self.library,self.size = library,(width,height)
        self.backend = None  # created on the frame owner's thread at first check
        self.box = (0.,0.,float(width),float(height))
        self.checked = self.mismatched = 0
        self.max_abs_difference = 0

    def check(self,image):
        if image.cuda is None: return None
        if self.backend is None: self.backend = BorrowedBackend(self.library,*self.size)
        _,cuda_gray = self.backend._probe_seed(*image.cuda,1,self.box)
        _,egl_gray = self.backend._probe_seed(image.pointer,0,2,self.box)
        self.backend.release_input()
        difference = int(abs(cuda_gray.astype(int)-egl_gray.astype(int)).max())
        self.checked += 1
        self.mismatched += difference != 0
        self.max_abs_difference = max(self.max_abs_difference,difference)
        return difference == 0

    def report(self):
        return dict(checked=self.checked,mismatched=self.mismatched,max_abs_difference=self.max_abs_difference)

    def close(self):
        if self.backend is not None: self.backend.close()
