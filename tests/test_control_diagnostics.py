from __future__ import annotations

from types import SimpleNamespace

import pytest

from common.schemas import ControlIntent
from jetson.control.diagnostics import build_diagnostics
from jetson.control.pid import AxisPIDDecision, PIDDecision


def test_diagnostics_decompose_the_decision_and_normalize_geometry() -> None:
    intent = ControlIntent(sequence=9, observation_sequence=4, issued_monotonic_ns=1,
                           valid_until_monotonic_ns=2, mode="live", yaw_rate_rad_s=0.25,
                           pitch_rate_rad_s=-0.1, reason="tracking")
    decision = SimpleNamespace(
        intent=intent,
        pid=SimpleNamespace(
            pid=PIDDecision("tracking", yaw=AxisPIDDecision(proportional_rad_s=0.2, pre_limit_rad_s=0.26,
                                                            rate_limited_rad_s=0.26),
                            pitch=AxisPIDDecision(proportional_rad_s=-0.08)),
            timing=SimpleNamespace(capture_age_ns=SimpleNamespace(latest_ns=120_000_000)),
        ),
        feedforward=SimpleNamespace(valid=True, yaw_rate_rad_s=0.1, pitch_rate_rad_s=-0.04),
        applied_feedforward_rad_s=(0.05, -0.02),
    )
    observation = SimpleNamespace(
        sequence=4,
        target=SimpleNamespace(bearing_error_rad=(0.04, -0.016), track_id=3,
                               target_center_px=(960.0, 270.0), aim_reference_px=(640.0, 360.0)),
        gimbal=SimpleNamespace(sample_age_ms=12.0),
    )
    diag = build_diagnostics(observation, decision, feedforward_scale=0.5, created_monotonic_ns=5,
                             frame_size_px=(1280, 720))
    assert diag.reason == "tracking" and diag.intent_sequence == 9 and diag.track_id == 3
    assert diag.yaw.feedback_term_rad_s == pytest.approx(0.2)
    assert diag.yaw.feedforward_term_rad_s == pytest.approx(0.05)
    assert diag.yaw.final_rate_rad_s == pytest.approx(0.25)
    assert diag.pitch.estimated_target_rate_rad_s == pytest.approx(-0.04)
    assert diag.timing.source_frame_age_ms == pytest.approx(120.0)
    assert diag.target_center_norm == pytest.approx((0.75, 0.375))
    assert diag.aim_reference_norm == pytest.approx((0.5, 0.5))
    assert build_diagnostics(observation, decision, feedforward_scale=0.0,
                             created_monotonic_ns=5).aim_reference_norm is None
