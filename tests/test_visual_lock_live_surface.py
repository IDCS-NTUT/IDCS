"""Input selection of the live NVMM surface (no GPU; native calls are faked)."""
from jetson.visual_lock import live_surface
from jetson.visual_lock.live_surface import BorrowedBackend, BorrowedImage, NVMMBorrower


class _Backend(BorrowedBackend):
    def __init__(self):  # skip native construction
        self.calls, self.released = [], 0

    def process_pointer(self, pointer, pitch, *, kind, seed=None):
        self.calls.append((pointer, pitch, kind, seed))
        return "proposal"

    def release_input(self):
        self.released += 1


class _Image:
    pointer, cuda = 0xE6, None


def test_registered_frame_uses_cuda_pointer_and_unregistered_uses_eglimage():
    backend = _Backend()
    image = _Image()
    assert backend.process(image) == "proposal"
    image.cuda = (0xC0DA, 5120)
    backend.process(image, seed=(1, 2, 30, 40))
    assert backend.calls == [(0xE6, 0, 2, None), (0xC0DA, 5120, 1, (1, 2, 30, 40))]
    assert backend.released == 1  # seeds release their input as before


class _Fn:
    """Stands in for a ctypes function (accepts argtypes/restype)."""
    def __init__(self, fn): self.fn = fn
    def __call__(self, *a): return self.fn(*a)


class _Lib:
    def __init__(self, fail):
        def cuda(handle, pointer, pitch):
            if fail: return -1
            pointer._obj.value, pitch._obj.value = 0xC0DA, 5120
            return 0
        self.vll_borrow = _Fn(lambda buffer: 7)
        self.vll_image = _Fn(lambda handle: 0xE6)
        self.vll_release = _Fn(lambda handle: 0)
        self.vll_cuda = _Fn(cuda)
        self.vll_error = _Fn(lambda: b"EGL mapping not owned by this borrow")


def _borrower(monkeypatch, *, enabled, fail=False):
    lib = _Lib(fail)
    monkeypatch.setattr(live_surface.ct, "CDLL", lambda path: lib)
    if enabled is None: monkeypatch.delenv("IDCS_VLL_CUDA_INPUT", raising=False)
    else: monkeypatch.setenv("IDCS_VLL_CUDA_INPUT", "1" if enabled else "0")
    return NVMMBorrower("unused.so")


def test_cuda_input_is_on_by_default_and_can_be_disabled(monkeypatch):
    assert _borrower(monkeypatch, enabled=None).borrow(object()).cuda == (0xC0DA, 5120)
    borrower = _borrower(monkeypatch, enabled=False)
    image = borrower.borrow(object())
    assert isinstance(image, BorrowedImage) and image.cuda is None and image.pointer == 0xE6


def test_cuda_input_switch_registers_and_counts_fallbacks(monkeypatch):
    assert _borrower(monkeypatch, enabled=True).borrow(object()).cuda == (0xC0DA, 5120)
    borrower = _borrower(monkeypatch, enabled=True, fail=True)
    image = borrower.borrow(object())
    assert image.cuda is None and image.pointer == 0xE6
    assert borrower.cuda_fallbacks == 1
    assert borrower.cuda_error == "EGL mapping not owned by this borrow"
