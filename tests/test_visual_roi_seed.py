"""Exact operator seed validation and source-cache replay selection."""

from dataclasses import replace
import uuid

import pytest

from common.visual_roi_seed import VisualRoiSeed, VisualRoiRelease
from jetson.visual_lock.frame_cache import ExactFrameCache
from common.perception import PerceptionFrameV2


def _request(frame_id=5):
    return VisualRoiSeed.from_mapping(dict(type="visual_roi_seed_v1",
        request_id=str(uuid.uuid4()), tracker_session="run-a", frame_id=frame_id,
        source_time_ns=frame_id * 16_000_000, source_clock_domain="pc_monotonic",
        x=100, y=200, w=80, h=60, width=1280, height=720))


def _frame(frame_id):
    return PerceptionFrameV2(tracker_session="run-a", frame_id=frame_id,
        source_time_ns=frame_id * 16_000_000, source_clock_domain="pc_monotonic",
        source_identity_verified=True, observed_time_ns=frame_id * 16_000_000,
        observation_clock_domain="jetson_monotonic", width=1280, height=720)


def test_exact_frame_replay_and_no_nearest_fallback():
    cache = ExactFrameCache(capacity=8)
    for i in range(3, 8):
        cache.add(_frame(i), f"pixels-{i}")
    assert [f.frame_id for f, _ in cache.replay_from(_request(5))] == [5, 6, 7]
    assert cache.replay_from(_request(8)) is None
    assert cache.replay_from(replace(_request(5), source_time_ns=999)) is None
    assert cache.replay_from(replace(_request(5), tracker_session="other")) is None


def test_cache_bound_restart_and_unverified_rejection():
    cache = ExactFrameCache(capacity=2)
    for i in range(4, 7): cache.add(_frame(i), i)
    assert cache.replay_from(_request(4)) is None
    assert cache.replay_from(_request(5)) is not None
    cache.add(_frame(1), "new source")
    assert cache.replay_from(_request(5)) is None


def test_cache_releases_evicted_and_cleared_gpu_images():
    class Image:
        def __init__(self): self.closed = False
        def close(self): self.closed = True

    cache = ExactFrameCache(capacity=1)
    first, second = Image(), Image()
    cache.add(_frame(1), first)
    cache.add(_frame(2), second)
    assert first.closed and not second.closed
    cache.clear()
    assert second.closed
    cache.add(_frame(5).model_copy(update={"source_identity_verified": False}), "bad")
    assert cache.replay_from(_request(5)) is None


def test_request_rejects_bad_geometry_and_identity():
    original = _request().as_message()
    for change in ({"w": 0}, {"x": 1270}, {"frame_id": True},
                   {"source_clock_domain": "unverified_clock"}, {"request_id": "bad"}):
        with pytest.raises(ValueError):
            VisualRoiSeed.from_mapping({**original, **change})


def test_release_is_scoped_to_one_seed_and_source_session():
    original = VisualRoiRelease.from_mapping(dict(type="visual_roi_release_v1",
        request_id=str(uuid.uuid4()), tracker_session="run-a"))
    assert VisualRoiRelease.from_mapping(original.as_message()) == original
    for change in ({"request_id": "bad"}, {"tracker_session": ""},
                   {"tracker_session": None}, {"type": "visual_roi_seed_v1"}):
        with pytest.raises(ValueError):
            VisualRoiRelease.from_mapping({**original.as_message(), **change})
