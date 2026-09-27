"""The live canary must keep video provenance and synthetic target separate."""

import math
import sys

import pytest

from common.perception import PerceptionFrameV2, PerceptionSnapshotV2
from tools.shadow_verified_video import _p95, _p99, run, synthetic_observation


def _snapshot(verified: bool) -> PerceptionSnapshotV2:
    return PerceptionSnapshotV2(
        sequence=9,
        frame=PerceptionFrameV2(
            frame_id=17,
            source_time_ns=100,
            received_time_ns=200,
            observed_time_ns=300,
            source_clock_domain="pc_monotonic",
            source_identity_verified=verified,
            receive_clock_domain="jetson_monotonic",
            observation_clock_domain="jetson_monotonic",
            width=1280,
            height=720,
        ),
    )


def test_synthetic_target_preserves_video_provenance_without_detection():
    obs = synthetic_observation(_snapshot(True), sequence=1, decision_ns=400)
    assert (obs.source_frame_id, obs.source_time_ns) == (17, 100)
    assert (obs.frame_received_time_ns, obs.frame_observed_time_ns) == (200, 300)
    assert obs.source_identity_verified is True
    assert obs.target.valid and obs.target.class_id == "synthetic_guaranteed_target"
    assert obs.target.aim_reference_px == (640, 360)
    assert obs.target.bearing_error_rad == (0.0, 0.04)
    assert obs.gimbal.valid and obs.safety.auto_allowed


def test_synthetic_target_moves_but_does_not_upgrade_unverified_video():
    first = synthetic_observation(_snapshot(False), sequence=1, decision_ns=400)
    later = synthetic_observation(_snapshot(False), sequence=16, decision_ns=500)
    assert first.source_identity_verified is False
    assert later.source_identity_verified is False
    assert math.isclose(later.target.bearing_error_rad[0], 0.08)
    assert later.target.bearing_error_rad != first.target.bearing_error_rad


def test_canary_upper_age_quantiles_are_conservative():
    assert _p95([]) is None and _p99([]) is None
    values = list(range(1, 101))
    assert _p95(values) == 95
    assert _p99(values) == 99


def test_check_rejects_shadow_policy_that_exceeds_uncertainty_budget(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "shadow_verified_video",
        "--snapshot-endpoint", "tcp://127.0.0.1:56198",
        "--clock-endpoint", "tcp://127.0.0.1:56199",
        "--duration-s", "1",
        "--empirical-drift-ppm", "20000",
        "--ack-empirical-bound-shadow-only",
        "--check",
    ])
    with pytest.raises(SystemExit) as exc:
        run()
    assert exc.value.code == 2
