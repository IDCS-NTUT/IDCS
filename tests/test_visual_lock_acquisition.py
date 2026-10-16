import json
from types import SimpleNamespace

import numpy as np
import pytest

from common.perception import NormalizedBoxV2
from jetson.visual_lock.acquisition import acquire_cached
from jetson.visual_lock.frame_cache import ExactFrameCache
from jetson.visual_lock.fusion import Gates, Proposal
from jetson.visual_lock.session import VisualLockSession
from tests.test_visual_lock import frame, proposal
from tools.roi_relock_fixture import box_at, seed_for


BOX = NormalizedBoxV2(x=.25, y=.25, w=.25, h=.25)


def coordinator(*, texture=16, fault_at=None):
    backends = []
    class Backend:
        calls = 0
        closed = False
        def process(self, image, seed=None):
            self.calls += 1
            if self.calls == fault_at:
                raise RuntimeError("injected GPU fault")
            if seed is not None:
                return Proposal(tuple(seed), 0., 0., np.zeros((0, 2)),
                    np.zeros((0, 2)), np.zeros(0), seed_features=0 if image == "blank" else texture)
            return proposal(session.lock)
        def close(self): self.closed = True
        def commit(self, *args, **kwargs): pass
        def discard(self): pass
    def factory(w, h):
        backend = Backend()
        backends.append(backend)
        return backend
    session = VisualLockSession(factory, Gates(acquire_frames=1))
    return session, backends


def replay(*indices):
    return [(frame(i), object()) for i in indices]


@pytest.mark.parametrize("texture,fault,reason", [
    (0, None, "seed: seed lacks texture"),
    (16, 1, "seed: backend fault: injected GPU fault"),
    (16, 2, "catch-up: backend fault: injected GPU fault")])
def test_first_failure_is_not_overwritten_or_updated(texture, fault, reason):
    session, backends = coordinator(texture=texture, fault_at=fault)
    acquisition = acquire_cached(session, replay(0, 1, 2, 3), BOX)
    assert not acquisition.accepted and acquisition.reason == reason
    assert backends[0].calls == (fault or 1) and backends[0].closed
    assert not acquisition.result.snapshot.selection
    assert acquisition.result.snapshot.frame.frame_id == len(acquisition.steps)-1
    if texture == 0:
        assert acquisition.result.proposal.seed_features == 0
    rows = acquisition.audit(lambda f: f.frame_id * 100)
    assert rows[-1]["decoded_pts_ns"] == (len(rows)-1)*100
    assert rows[-1]["decision"]["reason"] in reason
    json.dumps(rows)


def test_source_gap_remains_distinguishable_from_quality_failure():
    session, backends = coordinator()
    acquisition = acquire_cached(session, replay(0, 20, 21), BOX)
    assert not acquisition.accepted
    assert "gap" in acquisition.reason
    assert len(acquisition.steps) == 2 and backends[0].calls == 1
    assert acquisition.result.snapshot.frame.frame_id == 20


def test_repeated_release_explicit_seed_and_recovery_after_blank_seed():
    session, backends = coordinator()
    for index in (0, 10, 20):
        acquisition = acquire_cached(session, replay(index, index+1, index+2), BOX)
        assert acquisition.accepted and acquisition.result.snapshot.selection
        assert acquisition.result.snapshot.frame.frame_id == index+2
        session.close()
        assert backends[-1].closed
        assert not session.update(frame(index+3), None).snapshot.selection
    # Refusal requires another explicit seed, not an update/retry.
    refused = acquire_cached(session, [(frame(40), "blank"), (frame(41), object())], BOX)
    assert not refused.accepted and backends[-1].closed
    assert not session.update(frame(42), None).snapshot.selection
    recovered = acquire_cached(session, replay(50, 51, 52), BOX)
    assert recovered.accepted and recovered.result.snapshot.selection
    session.close()
    assert all(b.closed for b in backends)


def test_empty_replay_cannot_invent_seed():
    with pytest.raises(ValueError, match="exact seed"):
        acquire_cached(SimpleNamespace(), [], BOX)


def test_cache_pts_provenance_has_same_bounded_lifetime_as_pixels():
    cache = ExactFrameCache(capacity=1)
    a, b = frame(1), frame(2)
    cache.add(a, object(), decoded_pts_ns=123)
    assert cache.decoded_pts(a) == 123
    cache.add(b, object(), decoded_pts_ns=456)
    assert cache.decoded_pts(a) is None and cache.decoded_pts(b) == 456
    cache.clear()
    assert cache.decoded_pts(b) is None


def test_acquisition_audit_joins_real_cache_frames_to_decode_pts():
    session, _ = coordinator()
    cache = ExactFrameCache()
    frames = replay(0, 1, 2)
    for f, pixels in frames:
        cache.add(f, pixels, decoded_pts_ns=500+f.frame_id)
    acquisition = acquire_cached(session, frames, BOX)
    assert [r['decoded_pts_ns'] for r in acquisition.audit(cache.decoded_pts)] == [500,501,502]


def test_fixture_operator_seed_uses_exact_requested_pixels_not_old_track():
    request = dict(type="visual_roi_seed_v1", tracker_session="source-A", frame_id=5,
        source_time_ns=2_000_000_000, source_clock_domain="pc_monotonic",
        width=1280, height=720, x=1, y=2, w=30, h=40)
    selected = seed_for(request)
    assert tuple(selected[k] for k in ("x", "y", "w", "h")) == box_at(request["source_time_ns"])
    later = seed_for({**request, "frame_id":125, "source_time_ns":4_000_000_000})
    assert later["x"] != selected["x"] and later["y"] != selected["y"]
    assert selected["frame_id"] == 5 and selected["tracker_session"] == "source-A"
    blank = seed_for(request, blank=True)
    assert (blank["x"], blank["y"]) == (80, 80)
    for changes in ({"source_clock_domain":"guess"}, {"tracker_session":""},
                    {"source_time_ns":None}, {"frame_id":True}):
        with pytest.raises((ValueError, TypeError)):
            seed_for({**request, **changes})
