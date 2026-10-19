import json

import numpy as np
import pytest

from jetson.visual_lock.fusion import Proposal
from jetson.visual_lock.seed import SeedDiagnostics, refine_seed
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


def test_two_separate_textured_blobs_refuse_even_if_one_nearer_centre():
    result = refine((0, 0, 200, 200), np.vstack((grid(10, 10), grid(100, 100))))
    assert result.refined_box is None
    assert "support" in result.reason or "competing" in result.reason


@pytest.mark.parametrize("points", [[], [[50, 50]]*16, grid(50, 50, 2, 2), grid(50, 50)[:7]])
def test_uniform_or_noise_refuses(points):
    assert refine((0, 0, 100, 100), points).refined_box is None


@pytest.mark.parametrize("box", [(0, 0, 100, 100), (1180, 620, 100, 100), (0, 0, 12, 12)])
def test_image_edges_and_floor(box):
    x, y, w, h = box
    points = grid(x+3, y+3, min(24, w-6.1), min(24, h-6.1))
    result = refine(box, points)
    if result.refined_box is None:
        assert min(w, h) == 12  # not enough spread: refusal beats shrinking noise
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
