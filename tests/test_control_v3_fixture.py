from __future__ import annotations

import pytest

from jetson.control_v3.fixture import stamp_sim_truth_receive
from pc.sim_camera import SimCamera


def _snapshot():
    camera = SimCamera(
        width=1280, height=720, renderer_name="cpu",
        scene={
            "mode": "static_targets",
            "targets": [{"sprite": "drone", "width": 0.35, "ground": [0.0, -1.0], "ground_y": 0.9}],
            "buildings": [], "cubes": [],
        },
    )
    camera.next_frame()
    return camera.build_ground_truth_snapshot(12, 1_000_000_000)


def test_exact_truth_receives_a_named_jetson_timestamp() -> None:
    stamped = stamp_sim_truth_receive(_snapshot(), received_monotonic_ns=1_050_000_000)
    assert stamped.frame.frame_id == 12
    assert stamped.frame.source_time_ns == 1_000_000_000
    assert stamped.frame.source_identity_verified is True
    assert stamped.frame.received_time_ns == 1_050_000_000
    assert stamped.frame.observed_time_ns == 1_050_000_000
    assert stamped.frame.receive_clock_domain == "jetson_monotonic"


def test_unverified_or_non_fixture_snapshot_cannot_be_stamped() -> None:
    snapshot = _snapshot()
    bad = snapshot.model_copy(update={"frame": snapshot.frame.model_copy(update={"source_identity_verified": None})})
    with pytest.raises(ValueError, match="unverified"):
        stamp_sim_truth_receive(bad, received_monotonic_ns=1_050_000_000)
    with pytest.raises(ValueError, match="already"):
        stamp_sim_truth_receive(
            stamp_sim_truth_receive(snapshot, received_monotonic_ns=1_050_000_000),
            received_monotonic_ns=1_060_000_000,
        )
