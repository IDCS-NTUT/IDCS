import numpy as np
import pytest

from tools.verify_visual_lock import preload_frames
from tools.visual_lock_replay_gpu import ReplayGPUBackend


def test_preload_bounded_and_preserves_pixels_and_labels():
    image = np.arange(12,dtype=np.uint8).reshape(2,2,3)
    truth = (1,2,3,4)
    prepared = preload_frames(iter([(image,truth)]),max_bytes=12)
    assert prepared[0][0] is image and prepared[0][1] == truth
    with pytest.raises(ValueError,match="budget"):
        preload_frames(iter([(image,truth)]),max_bytes=11)


def test_preload_limits_frame_count():
    image = np.zeros((1,1,3),dtype=np.uint8)
    assert len(preload_frames(((image,None) for _ in range(700)))) == 600


@pytest.mark.parametrize("kind",["cuda","egl","bad"])
def test_gpu_replay_rejects_missing_owner_before_loading_any_gpu_library(kind):
    class Owner:
        active = False
    with pytest.raises(RuntimeError,match="active standalone"):
        ReplayGPUBackend("does-not-exist",320,240,kind=kind,egl_owner=Owner())


def test_gpu_replay_checks_dimensions_before_allocation():
    class Owner:
        active = True
    with pytest.raises(ValueError,match="dimensions"):
        ReplayGPUBackend("does-not-exist",32,240,kind="cuda",egl_owner=Owner())
