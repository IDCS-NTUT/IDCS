import json

import numpy as np
import pytest

from jetson.visual_lock.fusion import Proposal
from jetson.visual_lock.seed import SeedDiagnostics, inside_seed, refine_seed, seed_feature_count
from jetson.visual_lock.session import VisualLockSession
from common.perception import NormalizedBoxV2
from tests.test_visual_lock import frame


def grid(x, y, w=24, h=24):
    return np.array([(a, b) for a in np.linspace(x, x+w, 4) for b in np.linspace(y, y+h, 4)])


def refine(box, points):
    return refine_seed(box, points, np.ones(len(points)))


def test_tight_target_in_loose_box():
    points = grid(80, 70)
    result = refine((20, 20, 180, 140), points)
    assert result.refined_box == (76, 66, 32, 32)
    assert result.operator_box == (20, 20, 180, 140)
    assert refine(result.refined_box, points).refined_box == result.refined_box


@pytest.mark.parametrize("x,y", [(4, 4), (70, 4), (4, 70), (70, 70)])
def test_target_in_any_corner(x, y):
    result = refine((0, 0, 100, 100), grid(x, y))
    assert result.refined_box == (x-4, y-4, 32, 32)


def test_two_separate_textured_blobs_keep_operator_box():
    box = (0, 0, 200, 200)
    result = refine(box, np.vstack((grid(10, 10), grid(100, 100))))
    assert result.refined_box == box
    assert result.reason.startswith("refinement skipped:")
    assert "support" in result.reason or "competing" in result.reason


@pytest.mark.parametrize("points", [[], grid(50, 50)[:7]])
def test_only_insufficient_operator_texture_refuses(points):
    assert refine((0, 0, 100, 100), points).refined_box is None


@pytest.mark.parametrize("points", [[[50, 50]]*16, grid(50, 50, 2, 2)])
def test_compact_support_keeps_operator_box(points):
    result = refine((0, 0, 100, 100), points)
    assert result.refined_box == result.operator_box
    assert result.reason == "refinement skipped: support is too compact"


@pytest.mark.parametrize("points", [
    # Small drone: body plus four separated propellers (seg1159/2520/5155).
    np.vstack([grid(90, 80, 12, 12), grid(40, 35, 12, 12),
               grid(140, 35, 12, 12), grid(40, 125, 12, 12), grid(140, 125, 12, 12)]),
    # Open palm with separate finger supports, in a loose or tight operator ROI.
    np.vstack([grid(65, 110, 60, 20), *[grid(x, 35, 8, 45) for x in (40, 75, 110, 145)]]),
])
@pytest.mark.parametrize("tight", [False, True])
def test_drone_and_open_hand_layouts_never_refuse_or_drop_separated_support(points, tight):
    low = points.min(axis=0)-4
    size = np.ptp(points, axis=0)+8
    box = (*low, *size) if tight else (0, 0, 200, 180)
    result = refine(box, points)
    assert result.refined_box is not None
    assert inside_seed(result.refined_box, points).all()
    assert seed_feature_count(points) >= 8


def test_shrinking_cannot_discard_points_needed_for_native_minimum():
    # Eight spread-out useful points in the operator box, but the dominant
    # component's dense corners collapse under native 4px suppression.
    dense = grid(80, 70, 6, 6)
    outliers = np.array([(20, 20), (180, 20), (20, 150), (180, 150)])
    points = np.vstack((dense, outliers))
    assert seed_feature_count(points) >= 8
    result = refine((0, 0, 200, 180), points)
    assert result.refined_box == result.operator_box
    assert result.reason == "refinement skipped: too few points after shrinking"


def test_dominant_body_cannot_trim_small_separated_propeller_clusters():
    body = grid(80, 70)
    props = np.array([(20, 20), (24, 20), (160, 20), (164, 20)])
    result = refine((0, 0, 200, 180), np.vstack((body, props)))
    assert result.refined_box == result.operator_box
    assert result.reason == "refinement skipped: competing textured regions"


def test_native_border_filter_is_applied_before_counting_refined_points():
    points = np.array([(3, 3), (16.999, 3), (17, 3), (3, 17), (2.999, 4)])
    assert inside_seed((0, 0, 20, 20), points).tolist() == [True, True, False, False, False]


def test_thumb_like_loose_box_tightens_with_usable_support():
    points = grid(490, 300, 23, 65)
    result = refine((426, 281, 176, 145), points)
    assert result.reason == "refined"
    assert result.refined_box[2:] == pytest.approx((31, 84.5))
    assert inside_seed(result.refined_box, points).all()
    assert seed_feature_count(points) >= 8


@pytest.mark.parametrize("box", [(0, 0, 100, 100), (1180, 620, 100, 100), (0, 0, 12, 12)])
def test_image_edges_and_floor(box):
    x, y, w, h = box
    points = grid(x+3, y+3, min(24, w-6.1), min(24, h-6.1))
    result = refine(box, points)
    if result.refined_box is None:
        assert len(result.corners) < 8
    else:
        rx, ry, rw, rh = result.refined_box
        assert x <= rx and y <= ry and rw >= 12 and rh >= 12
        assert rx+rw <= x+w and ry+rh <= y+h


def test_scores_nonfinite_outside_and_permutation_are_deterministic():
    p = np.vstack((grid(40, 40), [[-1, -1], [np.nan, 40], [50, 50]]))
    scores = np.r_[np.ones(16), 1, 1, 0]
    a = refine_seed((0, 0, 100, 100), p, scores)
    b = refine_seed((0, 0, 100, 100), p[::-1], scores[::-1])
    assert a == b
    assert len(a.corners) == 16


def test_tight_seed_does_not_change():
    assert refine((20, 20, 40, 40), grid(24, 24, 30, 30)).refined_box == (20, 20, 40, 40)


def test_session_uses_refined_geometry_and_writes_only_at_seed(tmp_path):
    report = refine((80, 60, 80, 60), grid(100, 80))
    gray = np.arange(80*60, dtype=np.uint8).reshape(60, 80)
    class Backend:
        def process(self, image, seed=None):
            return Proposal(report.refined_box, 0, 0, np.empty((0, 2)), np.empty((0, 2)),
                            np.empty(0), seed_features=16, seed_report=report, seed_gray=gray)
        def close(self): pass
    session = VisualLockSession(lambda w, h: Backend(), seed_output=tmp_path)
    result = session.seed(frame(), None, NormalizedBoxV2(x=.25, y=.25, w=.25, h=.25))
    assert result.decision.box == report.refined_box
    assert session.lock.seed_size.tolist() == list(report.refined_box[2:])
    assert result.seed_report["operator_box"] == report.operator_box
    assert (tmp_path/"seeds/seed-00.pgm").read_bytes().endswith(gray.tobytes())
    saved = (tmp_path/"seeds/seed-00.json").read_bytes()
    session.close()
    session.update(frame(2), None)
    assert (tmp_path/"seeds/seed-00.json").read_bytes() == saved


def test_diagnostics_bound_only_own_slots_and_preserve_other_files(tmp_path):
    writer = SeedDiagnostics(tmp_path)
    writer.output.mkdir()
    (writer.output/"recording.txt").write_text("keep")
    report = refine((0, 0, 100, 100), grid(30, 30))
    from jetson.visual_lock.fusion import Decision
    for _ in range(70):
        writer.write(report, np.zeros((100, 100), np.uint8), frame(), Decision("acquiring", "seed"))
    assert len(list(writer.output.glob("*.json"))) == 64
    assert len(list(writer.output.glob("*.pgm"))) == 64
    assert json.loads((writer.output/"seed-00.json").read_text())["sequence"] == 65
    assert (writer.output/"recording.txt").read_text() == "keep"


def test_session_refuses_weak_seed_and_preserves_report(tmp_path):
    report = refine((80, 60, 80, 60), [])
    class Backend:
        closed = False
        def process(self, image, seed=None):
            return Proposal((80,60,80,60),0,0,np.empty((0,2)),np.empty((0,2)),np.empty(0),
                            seed_report=report,seed_gray=np.zeros((60,80),np.uint8))
        def close(self): self.closed = True
    backend = Backend()
    session = VisualLockSession(lambda w,h:backend,seed_output=tmp_path)
    out = session.seed(frame(),None,NormalizedBoxV2(x=.25,y=.25,w=.25,h=.25))
    assert out.decision.state == "lost" and backend.closed
    assert out.seed_report["refined_box"] is None
    assert not session.update(frame(2),None).snapshot.selection


@pytest.mark.parametrize("strong", [True, False])
def test_native_initialization_only_receives_accepted_refined_geometry(strong):
    import ctypes as ct
    from jetson.visual_lock.backend import VPIBackend, _Result
    backend = VPIBackend.__new__(VPIBackend)
    backend._handle = 1
    backend._allow_gpu_interop = False
    events = []
    report = refine((20,20,180,140),grid(80,70) if strong else [])
    def probe(*args):
        events.append("probe")
        return report,np.zeros((140,180),np.uint8)
    backend._probe_seed = probe
    class Library:
        def vl_process(self,handle,pointer,pitch,kind,seed,out):
            assert events == ["probe"]
            assert tuple(seed) == report.refined_box
            events.append("initialize")
            result = ct.cast(out,ct.POINTER(_Result)).contents
            result.count = 8
            result.box[:] = seed[:]
            return 0
    backend._lib = Library()
    proposal = backend.process_pointer(1,1280,kind=0,seed=(20,20,180,140))
    assert events == (["probe","initialize"] if strong else ["probe"])
    assert proposal.seed_report == report
    assert proposal.seed_features == (8 if strong else 0)


@pytest.mark.parametrize("native_points", [0, 7, 16])
def test_changed_harris_window_is_verified_before_single_initialization(native_points):
    import ctypes as ct
    from jetson.visual_lock.backend import VPIBackend, _Result
    operator = (20, 20, 180, 140)
    points = grid(80, 70).astype(np.float32)
    expected = refine(operator, points)
    events = []
    class Probe:
        def __call__(self, handle, pointer, pitch, kind, box, xy, scores, capacity,
                     count, crop, crop_capacity, width, height):
            events.append(("probe", tuple(box)))
            original = tuple(box) == operator
            n = 16 if original else native_points
            np.ctypeslib.as_array(ct.cast(xy, ct.POINTER(ct.c_float)), shape=(2*n,))[:] = points[:n].ravel()
            np.ctypeslib.as_array(ct.cast(scores, ct.POINTER(ct.c_uint32)), shape=(n,))[:] = 1
            ct.cast(count, ct.POINTER(ct.c_int)).contents.value = n
            w, h = map(int, box[2:])
            ct.cast(width, ct.POINTER(ct.c_int)).contents.value = w
            ct.cast(height, ct.POINTER(ct.c_int)).contents.value = h
            np.ctypeslib.as_array(ct.cast(crop, ct.POINTER(ct.c_uint8)), shape=(w*h,))[:] = 31 if original else 99
            return 0
    class Library:
        vl_seed_probe = Probe()
        def vl_process(self, handle, pointer, pitch, kind, box, out):
            events.append(("initialize", tuple(box)))
            result = ct.cast(out, ct.POINTER(_Result)).contents
            result.count = 16
            result.box[:] = box[:]
            return 0
    backend = VPIBackend.__new__(VPIBackend)
    backend._handle, backend._allow_gpu_interop = 1, False
    backend.width, backend.height, backend.seed_min_points = 256, 200, 8
    backend._lib = Library()
    proposal = backend.process_pointer(1, 1024, kind=0, seed=operator)
    chosen = expected.refined_box if native_points >= 8 else operator
    assert events == [("probe", operator), ("probe", expected.refined_box), ("initialize", chosen)]
    assert proposal.box == chosen and proposal.seed_features >= 8
    assert proposal.seed_report.operator_box == operator
    assert proposal.seed_report.corners == expected.corners
    assert proposal.seed_gray.shape == (140, 180) and (proposal.seed_gray == 31).all()
    if native_points < 8:
        assert proposal.seed_report.reason == "refinement skipped: too few native points in refined window"


def test_session_accepts_divided_seed_with_operator_geometry():
    operator = (20, 20, 180, 140)
    report = refine(operator, np.vstack((grid(30, 30), grid(140, 100))))
    class Backend:
        def process(self, image, seed=None):
            assert tuple(seed) == operator
            return Proposal(operator, 0, 0, np.empty((0, 2)), np.empty((0, 2)),
                            np.empty(0), seed_features=32, seed_report=report)
        def close(self): pass
    session = VisualLockSession(lambda w, h: Backend())
    f = frame()
    out = session.seed(f, None, NormalizedBoxV2(x=20/f.width, y=20/f.height,
        w=180/f.width, h=140/f.height))
    assert out.decision.state == "acquiring" and out.decision.box == operator
    assert out.seed_report["reason"].startswith("refinement skipped:")
    session.close()
