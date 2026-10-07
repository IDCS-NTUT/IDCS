"""Minimal normal-exit CUDA/VPI reproducer, independent of our tracker.

No cameras, services, sockets, kernels, device reset or motor access. Initialize
EGL before CUDA/VPI to avoid the reproduced late driver-finalizer fault.
--without-egl-bootstrap retains the failing baseline. NEVER use os._exit to turn
a crash into success. Run alone; do not import into an existing live GPU owner.
"""
import ctypes as ct
import argparse
from contextlib import nullcontext

from jetson.visual_lock.egl_lifetime import StandaloneEGL


def probe():
    cuda = ct.CDLL("libcudart.so")
    vpi = ct.CDLL("libnvvpi.so.4")
    cuda.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p),ct.c_size_t]
    cuda.cudaFree.argtypes = [ct.c_void_p]
    vpi.vpiContextCreate.argtypes = [ct.c_uint64,ct.POINTER(ct.c_void_p)]
    vpi.vpiContextDestroy.argtypes = [ct.c_void_p]
    vpi.vpiContextDestroy.restype = None
    ptr,context = ct.c_void_p(),ct.c_void_p()
    status = vpi.vpiContextCreate(0,ct.byref(context))
    print(f"vpiContextCreate={status}",flush=True)
    if status: return 1
    try:
        status = cuda.cudaMalloc(ct.byref(ptr),4096)
        print(f"cudaMalloc={status}",flush=True)
        if status: return 1
        status = cuda.cudaFree(ptr)
        print(f"cudaFree={status}",flush=True)
        if status: return 1
    finally:
        vpi.vpiContextDestroy(context)
    print("VPI context destroyed; allowing NORMAL process exit (no tracker loaded)",flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-egl-bootstrap",action="store_true")
    args = parser.parse_args()
    with nullcontext() if args.without_egl_bootstrap else StandaloneEGL():
        return probe()


if __name__ == "__main__":
    raise SystemExit(main())
