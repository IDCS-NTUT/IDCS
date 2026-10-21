"""Synchronous GPU borrowing; base backend reads gray only for seed evidence."""
import ctypes as ct
import os
import threading
from collections import deque
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
        # Hand the tracker a CUDA pointer to each pool frame, registered once
        # per pool buffer, instead of an EGLImage wrapped per frame (~2.5 ms;
        # verified byte-identical live 2026-10-19). Frames that cannot be
        # registered use the EGLImage. IDCS_VLL_CUDA_INPUT=0 restores EGLImage.
        self.cuda_input = os.environ.get("IDCS_VLL_CUDA_INPUT") != "0"
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


class PooledImage:
    """One exclusive use of a persistent GPU allocation/registration."""
    def __init__(self,pool,image):
        self.pool,self.image=pool,image
        self.pointer,self.cuda=image.pointer,image.cuda
    def close(self):
        image,self.image=self.image,None
        if image is not None:self.pool.release(image)
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


class NVMMClonePool:
    def __init__(self,borrower,capacity=34):
        self.borrower,self.capacity=borrower,capacity
        self.images,self.idle=[],deque()
        self.lock=threading.Lock()
        self.io_lock=threading.Lock()
        self.closed=False
        lib=borrower.lib
        lib.vll_clone_copy.argtypes=[ct.c_void_p,ct.c_void_p]
        lib.vll_clone_copy.restype=ct.c_int
        lib.vll_clone_cuda.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_int)]
        lib.vll_clone_cuda.restype=ct.c_int

    def acquire(self,buffer):
        with self.io_lock:
            with self.lock:
                if self.closed:return None
                image=self.idle.popleft() if self.idle else None
                if image is None and len(self.images)>=self.capacity:return None
            lib=self.borrower.lib
            if image is None:
                image=self.borrower.clone(buffer)
                pointer,pitch=ct.c_void_p(),ct.c_int()
                if lib.vll_clone_cuda(image.handle,ct.byref(pointer),ct.byref(pitch)):
                    message=lib.vll_error().decode();image.close()
                    raise RuntimeError('resident clone CUDA view: '+message)
                image.cuda=(pointer.value,pitch.value)
                self.images.append(image)
            elif lib.vll_clone_copy(image.handle,hash(buffer)):
                raise RuntimeError(lib.vll_error().decode())
            return PooledImage(self,image)

    def release(self,image):
        with self.lock:
            if not self.closed:self.idle.append(image)

    def close(self):
        with self.io_lock:
            with self.lock:
                self.closed=True
                self.idle.clear()
            for image in self.images:image.close()
            self.images.clear()


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
