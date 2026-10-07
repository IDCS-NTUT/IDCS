import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import pytest

from jetson.visual_lock.egl_lifetime import StandaloneEGL


class Function:
    def __init__(self,fn): self.fn = fn
    def __call__(self,*args): return self.fn(*args)


class EGL:
    def __init__(self,*,initialize=1,terminate=1,vendor=b"NVIDIA",display=123):
        self.terminated = []
        self.eglGetDisplay = Function(lambda _:display)
        self.eglGetError = Function(lambda:0x3001)
        self.eglQueryString = Function(lambda *_:vendor)
        def init(_,major,minor):
            ct.cast(major,ct.POINTER(ct.c_int))[0] = 1
            ct.cast(minor,ct.POINTER(ct.c_int))[0] = 5
            return initialize
        self.eglInitialize = Function(init)
        def end(d):
            self.terminated.append(d)
            return terminate
        self.eglTerminate = Function(end)


def owner(lib): return StandaloneEGL(loader=lambda _:lib)


def test_normal_lifetime_nested_owner_rejected_and_idempotent_cleanup():
    lib = EGL()
    instance = owner(lib)
    with instance as current:
        assert current.active and current.version == (1,5) and current.vendor == "NVIDIA"
        with pytest.raises(RuntimeError,match="already active"):
            with owner(EGL()): pass
    assert not instance.active and lib.terminated == [123]
    instance.close()
    assert lib.terminated == [123]
    with owner(EGL()): pass


@pytest.mark.parametrize("kwargs",[{"initialize":0},{"display":None},{"vendor":b"Mesa"}])
def test_initialization_failure_releases_owner_without_fake_success(kwargs):
    lib = EGL(**kwargs)
    with pytest.raises(RuntimeError):
        with owner(lib): pytest.fail("must fail closed")
    assert lib.terminated == ([123] if kwargs.get("vendor") else [])
    with owner(EGL()): pass


def test_body_exception_cleans_display():
    lib = EGL()
    with pytest.raises(ValueError,match="injected"):
        with owner(lib): raise ValueError("injected")
    assert lib.terminated == [123]


def test_close_failure_is_failure_and_releases_owner():
    lib = EGL(terminate=0)
    with pytest.raises(RuntimeError,match="termination"):
        with owner(lib): pass
    with owner(EGL()): pass


def test_cross_thread_access_not_active_and_close_rejected():
    lib = EGL()
    with owner(lib) as current:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert not pool.submit(lambda:current.active).result()
            with pytest.raises(RuntimeError,match="owning thread"):
                pool.submit(current.close).result()
        assert current.active and not lib.terminated
    assert lib.terminated == [123]
