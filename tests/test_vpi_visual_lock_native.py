"""Optional real VPI CUDA gate; never opens sockets or commands a motor.

IDCS_VPI_LIBRARY names the separately built library. Without it these tests
skip, rather than claiming Windows/fake tests validate NVIDIA kernels.
"""
import ctypes as ct
import os

import numpy as np
import pytest

from jetson.visual_lock.backend import VPIBackend

LIBRARY = os.environ.get("IDCS_VPI_LIBRARY")
pytestmark = pytest.mark.skipif(not LIBRARY,reason="built VPI CUDA library required")


@pytest.fixture(scope="session",autouse=True)
def standalone_egl_owner():
    # Only a standalone test process may own/terminate the default display.
    # This is explicitly selected, not a hidden fixture masking teardown errors.
    if os.environ.get("IDCS_VPI_STANDALONE_EGL") == "1":
        from jetson.visual_lock.egl_lifetime import StandaloneEGL
        with StandaloneEGL() as owner:
            yield owner
    else:
        yield None


def pixels():
    rng = np.random.default_rng(736)
    # Larger four-pixel cells support pyramidal flow without OpenCV tracking.
    texture = np.repeat(np.repeat(rng.integers(30,240,(18,24,3),dtype=np.uint8),4,axis=0),4,axis=1)
    image = np.full((240,320,4),32,np.uint8); image[:,:,3] = 255
    image[60:132,80:176,:3] = texture
    moved = np.full_like(image,32); moved[:,:,3] = 255
    moved[61:133,82:178,:3] = texture
    return image,moved


def test_real_native_seed_flow_lifetime_and_commit():
    image,moved = pixels()
    with VPIBackend(LIBRARY,320,240) as backend:
        seed = backend.process(image,seed=(80,60,96,72))
        assert seed.seed_features >= 8
        p = backend.process(moved)
        assert p.psr > 6 and len(p.before) >= 8
        valid = p.fb < 1
        assert valid.sum() >= 8
        assert np.median(p.after[valid]-p.before[valid],axis=0) == pytest.approx((2,1),abs=.15)
        ref_valid = p.reference_fb < 1
        assert ref_valid.sum() >= 8
        assert np.median(p.reference_after[ref_valid]-p.reference_before[ref_valid],axis=0) == pytest.approx((2,1),abs=.15)
        retained = backend._image
        with pytest.raises(RuntimeError,match="previous proposal"):
            backend.process(image)
        with pytest.raises(RuntimeError,match="pending proposal"):
            backend.release_input()
        assert backend._image is retained
        backend.commit((82,61,96,72),update_template=True)
        assert backend.commit_stats["retained"] >= 8
        backend.release_input()
        still = backend.process(moved)
        assert len(set(p.point_ids.tolist()) & set(still.point_ids.tolist())) >= 8
        valid = still.fb < 1
        assert valid.sum() >= 8
        assert np.median(still.after[valid]-still.before[valid],axis=0) == pytest.approx((0,0),abs=.15)
        ref_valid = still.reference_fb < 1
        # Adjacent flow is zero, but original reference must still measure (2,1).
        assert np.median(still.reference_after[ref_valid]-still.reference_before[ref_valid],axis=0) == pytest.approx((2,1),abs=.15)
        backend.discard()
        backend.release_input()
        with pytest.raises(RuntimeError,match="no pending"):
            backend.commit((82,61,96,72),update_template=True)
        with pytest.raises(RuntimeError,match="new seed"):
            backend.process(moved,seed=(82,61,96,72))


def test_native_retains_only_approved_ids_and_selectively_replenishes():
    image,moved = pixels()
    with VPIBackend(LIBRARY,320,240) as backend:
        backend.process(image,seed=(80,60,96,72))
        p = backend.process(moved)
        valid = np.flatnonzero(p.fb < 1)[:12]
        assert len(valid) == 12
        mask = sum(1 << int(i) for i in valid)
        approved = set(p.point_ids[valid].tolist())
        rejected = set(p.point_ids.tolist())-approved
        backend.commit((82,61,96,72),update_template=False,retain_mask=mask,replenish=True)
        assert backend.commit_stats["retained"] == 12
        assert backend.commit_stats["added"] > 0
        p = backend.process(moved)
        assert approved <= set(p.point_ids.tolist())
        assert not rejected & set(p.point_ids.tolist())
        backend.commit((82,61,96,72),update_template=False,retain_mask=1,replenish=False)
        assert backend.commit_stats == dict(retained=1,added=0,total=1)


def test_native_bad_dimensions_and_box_rejected():
    with pytest.raises(RuntimeError,match="dimensions"):
        VPIBackend(LIBRARY,32,32)
    image,_ = pixels()
    with VPIBackend(LIBRARY,320,240) as backend:
        with pytest.raises(ValueError,match="uint8"):
            backend.process(image.astype(np.float32),seed=(80,60,96,72))
        with pytest.raises(RuntimeError,match="box outside"):
            backend.process(image,seed=(319,60,96,72))


def test_seed_probe_refines_before_native_initialization_and_copies_own_gray():
    image, moved = pixels()
    with VPIBackend(LIBRARY,320,240) as backend:
        seed = backend.process(image,seed=(40,25,200,150))
        assert seed.seed_features >= 8
        assert seed.seed_report.operator_box == (40,25,200,150)
        assert seed.seed_report.refined_box == pytest.approx(seed.box)
        assert seed.box[2] < 200 and seed.box[3] < 150
        assert seed.seed_gray.shape == (150,200) and seed.seed_gray.dtype == np.uint8
        # A gray source remains exactly gray through VPI conversion.
        assert seed.seed_gray[0,0] == 32
        p = backend.process(moved)
        assert p.psr >= 6 and np.count_nonzero(p.fb < 1) >= 8
        backend.discard()


@pytest.mark.parametrize("hand,tight", [
    ("thumb", (486,298,62,89)), ("open", (438,299,124,91))])
def test_camera_hand_loose_seed_retains_at_least_as_long_as_tight(hand,tight,tmp_path):
    import cv2
    from pathlib import Path
    from tools.verify_visual_lock import evaluate
    path = Path(__file__).parent/"fixtures/camera_roi"/f"camera_hand_{hand}_1280x720.png"
    image = cv2.imread(str(path))
    assert image is not None
    loose = evaluate("loose",((image,None) for _ in range(120)),LIBRARY,tmp_path,(426,281,176,145))
    control = evaluate("tight",((image,None) for _ in range(120)),LIBRARY,tmp_path,tight)
    assert loose["seed_accepted"] and control["seed_accepted"]
    assert (loose["first_loss"] or {"frame":120})["frame"] >= (control["first_loss"] or {"frame":120})["frame"]
    import json
    rows = lambda name: [json.loads(line) for line in (tmp_path/f"{name}.jsonl").read_text().splitlines()]
    assert sum(r["selected"] for r in rows("loose")) >= sum(r["selected"] for r in rows("tight")) > 0


def test_camera_plain_wall_seed_refuses_without_learning(tmp_path):
    import cv2
    from pathlib import Path
    from tools.verify_visual_lock import evaluate
    image = cv2.imread(str(Path(__file__).parent/"fixtures/camera_roi/camera_hand_thumb_1280x720.png"))
    report = evaluate("wall",((image,None) for _ in range(10)),LIBRARY,tmp_path,(200,180,120,100))
    assert not report["seed_accepted"]
    assert report["first_loss"]["frame"] == 0
    assert report["false_tracking_absent"] == 0


@pytest.mark.skipif(os.environ.get("IDCS_VPI_CUDA_INTEROP") != "1",
                   reason="explicit experimental GPU opt-in required")
def test_cuda_pitch_pointer_same_tracking_without_host_frame_wrapper():
    # Use the CUDA runtime primary context, shared with VPI in this process.
    ptr = ct.c_void_p()
    image,moved = pixels()
    with VPIBackend(LIBRARY,320,240,allow_experimental_gpu_interop=True) as backend:
        cuda = ct.CDLL("libcudart.so")
        cuda.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p),ct.c_size_t]
        cuda.cudaMemcpy.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int]
        cuda.cudaFree.argtypes = [ct.c_void_p]
        assert cuda.cudaMalloc(ct.byref(ptr),image.nbytes) == 0
        try:
            assert cuda.cudaMemcpy(ptr,image.ctypes.data,image.nbytes,1) == 0
            seed = backend.process_pointer(ptr.value,image.strides[0],kind=1,seed=(80,60,96,72))
            assert seed.seed_features >= 8
            assert cuda.cudaMemcpy(ptr,moved.ctypes.data,moved.nbytes,1) == 0
            p = backend.process_pointer(ptr.value,moved.strides[0],kind=1)
            valid = p.fb < 1
            assert p.psr > 6 and valid.sum() >= 8
            assert np.median(p.after[valid]-p.before[valid],axis=0) == pytest.approx((2,1),abs=.15)
            assert backend._image is None
            backend.discard()
        finally:
            assert cuda.cudaFree(ptr) == 0


@pytest.mark.skipif(not os.environ.get("IDCS_VPI_TEST_SURFACE"),reason="built Jetson NVMM test library required")
@pytest.mark.parametrize("kind",["cuda","egl"])
def test_replay_gpu_lifetime_commit_and_discard(kind,standalone_egl_owner):
    from tools.visual_lock_replay_gpu import ReplayGPUBackend
    assert standalone_egl_owner is not None, "select IDCS_VPI_STANDALONE_EGL=1"
    image,moved = pixels()
    for _ in range(10):
        with ReplayGPUBackend(LIBRARY,320,240,kind=kind,egl_owner=standalone_egl_owner,
                              surface_library=os.environ["IDCS_VPI_TEST_SURFACE"]) as backend:
            seed = backend.process(image,seed=(80,60,96,72))
            assert seed.seed_features >= 8
            p = backend.process(moved)
            valid = p.fb < 1
            assert p.psr > 6 and valid.sum() >= 8
            assert np.median(p.after[valid]-p.before[valid],axis=0) == pytest.approx((2,1),abs=.15)
            with pytest.raises(RuntimeError,match="input retained"):
                backend.process(image)
            backend.commit((82,61,96,72),update_template=True)
            p = backend.process(moved)
            valid = p.fb < 1
            assert p.psr > 6 and valid.sum() >= 8
            assert np.median(p.after[valid]-p.before[valid],axis=0) == pytest.approx((0,0),abs=.15)
            backend.discard()
        backend.close()
