from __future__ import annotations

import pytest

from common.perception import PerceptionFrameV2, PerceptionSnapshotV2
from jetson.control.video_input import stamp_verified_snapshot


def _snapshot(verified: bool = True) -> PerceptionSnapshotV2:
    return PerceptionSnapshotV2(
        sequence=1,
        frame=PerceptionFrameV2(
            frame_id=11, source_time_ns=1_000_000_000,
            observed_time_ns=1_000_000_000,
            source_clock_domain="pc_monotonic",
            observation_clock_domain="pc_monotonic",
            source_identity_verified=verified,
            width=1920, height=1080,
        ),
        tracks=(), assessments=(), selection=None,
    )


def test_stamping_preserves_verified_source_and_adds_measured_local_times() -> None:
    original = _snapshot()
    stamped = stamp_verified_snapshot(
        original, received_ns=1_060_000_000, observed_ns=1_061_000_000,
    )
    assert original.frame.received_time_ns is None
    assert stamped.frame.source_time_ns == original.frame.source_time_ns
    assert stamped.frame.source_identity_verified is True
    assert stamped.frame.received_time_ns == 1_060_000_000
    assert stamped.frame.receive_clock_domain == "jetson_monotonic"
    assert stamped.frame.observation_clock_domain == "jetson_monotonic"


def test_stamping_rejects_unverified_identity_and_invalid_order() -> None:
    with pytest.raises(ValueError, match="identity"):
        stamp_verified_snapshot(_snapshot(False), received_ns=2, observed_ns=3)
    with pytest.raises(ValueError, match="order"):
        stamp_verified_snapshot(_snapshot(), received_ns=3, observed_ns=2)


def test_deepstream_receipt_is_kept_on_the_jetson_and_replaced_elsewhere():
    from common.perception import PerceptionFrameV2, PerceptionSnapshotV2
    from jetson.control.video_input import stamp_verified_snapshot

    frame = PerceptionFrameV2(frame_id=1, source_time_ns=100, source_clock_domain="pc_monotonic",
                              source_identity_verified=True, received_time_ns=500,
                              receive_clock_domain="jetson_monotonic", observed_time_ns=600,
                              observation_clock_domain="jetson_monotonic", width=1280, height=720)
    snapshot = PerceptionSnapshotV2(sequence=1, frame=frame)
    kept = stamp_verified_snapshot(snapshot, received_ns=900, observed_ns=950, keep_upstream_receipt=True)
    assert kept.frame.received_time_ns == 500 and kept.frame.observed_time_ns == 600
    local = stamp_verified_snapshot(snapshot, received_ns=900, observed_ns=950)
    assert local.frame.received_time_ns == 900 and local.frame.observed_time_ns == 950
