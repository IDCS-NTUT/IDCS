from dataclasses import replace

import numpy as np
import pytest

from common.perception import NormalizedBoxV2, PerceptionFrameV2, perception_snapshot_from_json
from jetson.visual_lock.fusion import Gates, Proposal, VisualLock


def frame(i=0, **changes):
    args = dict(tracker_session="pixels-A", frame_id=i, source_time_ns=1_000_000_000+i*16_666_667,
                observed_time_ns=5_000_000_000+i*16_666_667, source_clock_domain="pc_monotonic",
                observation_clock_domain="jetson_monotonic", source_identity_verified=True,
                width=320, height=240)
    args.update(changes)
    return PerceptionFrameV2(**args)


def seeded():
    lock = VisualLock()
    d = lock.seed(frame(), NormalizedBoxV2(x=.25, y=.25, w=.25, h=.25), 16)
    assert d.state == "acquiring"
    return lock


def proposal(lock, dx=2., dy=1., scale=1., **changes):
    x, y, w, h = lock.box
    a = np.array([(x+w*u, y+h*v) for u in (.2,.4,.6,.8) for v in (.2,.4,.6,.8)])
    c = np.array([x+w/2, y+h/2])
    b = (a-c)*scale + c + [dx, dy]
    args = dict(box=(x+dx,y+dy,w,h), psr=15., peak=.8, before=a, after=b,
                fb=np.zeros(len(a)))
    args.update(changes)
    return Proposal(**args)


def test_acquisition_v2_and_no_invented_detection_or_depth():
    lock = seeded()
    for i in range(1,4):
        d = lock.evaluate(frame(i), proposal(lock))
        s = lock.snapshot(i, d)
        assert not s.detections and not s.assessments
        assert (s.selection is not None) == (i == 3)
        assert d.update_template == (i == 3)
    assert s.selection.policy == "operator_visual_lock"
    assert s.tracks[0].class_id == "visual"
    assert perception_snapshot_from_json(s.model_dump_json()) == s


def test_scale_from_flow_not_dcf_size():
    lock = seeded()
    d = lock.evaluate(frame(1), proposal(lock, scale=1.02))
    assert d.accepted
    assert d.scale == pytest.approx(1.02)
    assert d.box[2:] == pytest.approx((81.6,61.2))


def test_outliers_removed_deterministically():
    lock = seeded()
    p = proposal(lock)
    b = p.after.copy(); b[:4] += [30,-20]
    d = lock.evaluate(frame(1), replace(p, after=b))
    assert d.accepted and d.inliers == 12


@pytest.mark.parametrize("change,reason", [
    ({"psr":2.},"weak DCF"),
    ({"fb":np.ones(16)*2},"insufficient forward"),
    ({"box":(130.,60.,80.,60.)},"disagreement"),
])
def test_bad_proposal_never_updates_template_and_loss_latches(change,reason):
    lock = seeded(); saved = lock.box
    for i in range(1,4):
        d = lock.evaluate(frame(i), proposal(lock, **change))
        assert reason in d.reason
        assert not d.accepted and not d.update_template
        assert not lock.snapshot(i,d).selection
        assert lock.box == saved
    assert d.state == "lost"
    d = lock.evaluate(frame(4), proposal(lock))
    assert d.state == "lost" and not d.accepted


@pytest.mark.parametrize("change", [
    {"tracker_session":"pixels-B"}, {"source_clock_domain":"other"},
    {"source_identity_verified":False}, {"width":640},
    {"frame_id":0}, {"source_time_ns":1_000_000_000},
    {"source_time_ns":1_300_000_000},
])
def test_namespace_order_and_gap_fail_closed(change):
    lock = seeded()
    assert lock.evaluate(frame(1,**change), proposal(lock)).state == "lost"


@pytest.mark.parametrize("pchange", [
    {"psr":float("nan")}, {"box":(1.,2.,3.,float("inf"))},
    {"after":np.zeros((4,2))}, {"fb":np.ones(16)*float("nan")},
    {"before":np.zeros((65,2)),"after":np.zeros((65,2)),"fb":np.zeros(65)},
])
def test_invalid_backend_result_latches_loss(pchange):
    lock = seeded()
    assert lock.evaluate(frame(1), proposal(lock,**pchange)).state == "lost"


def test_textureless_seed_is_not_accepted():
    lock = VisualLock()
    assert lock.seed(frame(),NormalizedBoxV2(x=.2,y=.2,w=.2,h=.2),0).state == "lost"


def test_requires_verified_source_and_session():
    lock = VisualLock()
    with pytest.raises(ValueError):
        lock.seed(frame(tracker_session=None),NormalizedBoxV2(x=.2,y=.2,w=.2,h=.2),64)


def test_uncertain_recovery_requires_full_acquisition_again():
    lock = seeded()
    for i in range(1,4):
        assert lock.evaluate(frame(i),proposal(lock)).accepted
    assert lock.state == "tracking"
    assert lock.evaluate(frame(4),proposal(lock,psr=0.)).state == "uncertain"
    for i in range(5,8):
        d = lock.evaluate(frame(i),proposal(lock))
        assert (d.state == "tracking") == (i == 7)


def test_spatially_concentrated_features_rejected():
    lock = seeded()
    a = np.array([(100+u,80+v) for u in range(4) for v in range(4)],dtype=float)
    d = lock.evaluate(frame(1),proposal(lock,before=a,after=a+[2,1]))
    assert not d.accepted


def test_remote_observation_clock_not_subtracted_from_source():
    lock = seeded()
    assert lock.evaluate(frame(1,observed_time_ns=0),proposal(lock)).accepted


@pytest.mark.parametrize("kwargs",[{"min_psr":float("nan")},{"acquire_frames":0},{"min_inlier_fraction":1.1}])
def test_bad_gate_values(kwargs):
    with pytest.raises(ValueError):
        Gates(**kwargs)


def test_session_checks_frame_before_gpu_and_closes_on_fault():
    from jetson.visual_lock.session import VisualLockSession
    class Backend:
        closed = 0
        calls = 0
        fault = False
        def process(self,rgba,seed=None):
            self.calls += 1
            if self.fault: raise RuntimeError("injected CUDA error")
            if seed is not None:
                return Proposal(tuple(seed),0.,0.,np.zeros((0,2)),np.zeros((0,2)),np.zeros(0),seed_features=16)
            return proposal(session.lock)
        def close(self): self.closed += 1
        def commit(self,*_,**kwargs): pass
        def discard(self): pass
    back = Backend()
    session = VisualLockSession(lambda w,h: back)
    session.seed(frame(),None,NormalizedBoxV2(x=.25,y=.25,w=.25,h=.25))
    out = session.update(frame(1,tracker_session="new"),None)
    assert out.decision.state == "lost" and out.snapshot.frame.tracker_session == "new"
    assert not out.snapshot.selection and back.calls == 1 and back.closed == 1
    session.seed(frame(2),None,NormalizedBoxV2(x=.25,y=.25,w=.25,h=.25))
    back.fault = True
    out = session.update(frame(3),None)
    assert "injected CUDA" in out.decision.reason
    assert not out.snapshot.selection and back.closed == 2


def test_session_commit_fault_never_publishes_accepted_track():
    from jetson.visual_lock.session import VisualLockSession
    class Backend:
        def process(self,rgba,seed=None):
            if seed is not None:
                return Proposal(tuple(seed),0.,0.,np.zeros((0,2)),np.zeros((0,2)),np.zeros(0),seed_features=16)
            return proposal(session.lock)
        def commit(self,*_,**kwargs): raise RuntimeError("commit failed")
        def close(self): pass
    session = VisualLockSession(lambda w,h: Backend(),Gates(acquire_frames=1))
    session.seed(frame(),None,NormalizedBoxV2(x=.25,y=.25,w=.25,h=.25))
    out = session.update(frame(1),None)
    assert out.decision.state == "lost" and not out.snapshot.tracks and not out.snapshot.selection


def test_session_update_before_seed_is_empty_and_never_creates_backend():
    from jetson.visual_lock.session import VisualLockSession
    session = VisualLockSession(lambda w,h: pytest.fail("must not create backend"))
    out = session.update(frame(),None)
    assert out.decision.state == "idle" and not out.snapshot.selection
    assert out.snapshot.frame == frame()


def test_selection_names_current_observation_not_initial_seed():
    lock = seeded()
    for i in range(1,4): d = lock.evaluate(frame(i),proposal(lock))
    snapshot = lock.snapshot(3,d)
    assert snapshot.selection.source_frame_id == snapshot.frame.frame_id == 3
