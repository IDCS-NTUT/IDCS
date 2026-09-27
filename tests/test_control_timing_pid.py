from __future__ import annotations

import pytest

from jetson.control.pid import AxisPIDConfig, BasicPID, PIDInput
from jetson.control.timing import (
    ClockBounds,
    FrameTimingEvidence,
    TimingVerdict,
    verify_frame_timing,
)
from tools.survey_clock_bounds import summarize


def _clock(*, drift: float | None = 0.0) -> ClockBounds:
    # Exact synthetic offset is 100 ms; 2 ms round trip bounds it to +/-1 ms.
    return ClockBounds.from_exchange(
        jetson_send_ns=1_000_000_000,
        pc_receive_ns=1_101_000_000,
        pc_send_ns=1_101_000_000,
        jetson_receive_ns=1_002_000_000,
        max_drift_ppm=drift,
    )


def _frame(**changes: object) -> FrameTimingEvidence:
    values = dict(
        frame_id=7, identity_verified=True, pc_source_ns=1_150_000_000,
        jetson_received_ns=1_065_000_000,
        jetson_observed_ns=1_080_000_000,
        jetson_decision_ns=1_100_000_000,
    )
    values.update(changes)
    return FrameTimingEvidence(**values)


def _verdict(frame: FrameTimingEvidence, clock: ClockBounds | None = None) -> TimingVerdict:
    return verify_frame_timing(
        frame, _clock() if clock is None else clock,
        max_clock_sample_age_ns=200_000_000,
        max_capture_age_ns=100_000_000,
    )


def test_clock_exchange_produces_interval_not_exact_offset() -> None:
    clock = _clock()
    assert (clock.offset_min_ns, clock.offset_max_ns) == (99_000_000, 101_000_000)
    result = _verdict(_frame())
    assert result.valid and result.reason == "verified"
    assert result.capture_age_ns is not None
    assert (result.capture_age_ns.earliest_ns, result.capture_age_ns.latest_ns) == (
        49_000_000, 51_000_000,
    )
    assert result.receive_to_observe_ns == 15_000_000
    assert result.observe_to_decision_ns == 20_000_000


@pytest.mark.parametrize("changes,reason", [
    ({"identity_verified": False}, "frame_identity_unverified"),
    ({"jetson_observed_ns": 1_060_000_000}, "local_timestamps_out_of_order"),
    ({"jetson_decision_ns": 1_300_000_000}, "clock_sample_stale"),
    ({"pc_source_ns": 1_165_000_000}, "capture_after_receive_possible"),
    ({"pc_source_ns": 1_050_000_000}, "capture_age_out_of_bounds"),
])
def test_timing_refuses_guesses(changes: dict, reason: str) -> None:
    assert _verdict(_frame(**changes)).reason == reason


def test_clock_drift_must_be_bounded_and_expands_interval() -> None:
    with pytest.raises(ValueError, match="drift bound"):
        _clock(drift=None).map_pc_event(1_150_000_000, jetson_now_ns=1_100_000_000)
    interval = _clock(drift=100.0).map_pc_event(
        1_150_000_000, jetson_now_ns=1_100_000_000
    )
    assert interval.earliest_ns < 1_049_000_000
    assert interval.latest_ns > 1_051_000_000


def test_clock_survey_does_not_claim_drift_bound() -> None:
    report = summarize([_clock()])
    assert report["best_interval_width_ms"] == 2.0
    assert report["observed_span_s"] == 0.0
    assert report["all_intervals_intersect"] is True
    assert report["drift_bound_established"] is False


def _pid() -> BasicPID:
    axis = AxisPIDConfig(
        kp=1.0, ki=1.0, kd=0.5,
        integral_limit_rad_s=0.1, rate_limit_rad_s=0.2,
        acceleration_limit_rad_s2=1.0,
    )
    return BasicPID(axis, axis)


def _input(index: int, **changes: object) -> PIDInput:
    values = dict(
        decision_ns=1_000_000_000 + index * 100_000_000,
        track_id=7, error_rad=(0.1, 0.0),
        gimbal_rate_rad_s=(0.0, 0.0),
        timing=TimingVerdict(True, "verified"),
        safety_allowed=True, gimbal_valid=True,
    )
    values.update(changes)
    return PIDInput(**values)


def test_basic_pid_is_raw_feedback_with_explicit_terms() -> None:
    pid = _pid()
    first = pid.decide(_input(0))
    second = pid.decide(_input(1, gimbal_rate_rad_s=(0.04, 0.0)))
    assert first.yaw.proportional_rad_s == pytest.approx(0.1)
    assert first.yaw.final_rad_s == 0.0  # no invented first elapsed period
    assert second.yaw.integral_rad_s == pytest.approx(0.01)
    assert second.yaw.derivative_rad_s == pytest.approx(-0.02)
    assert second.yaw.final_rad_s == pytest.approx(0.09)
    assert not hasattr(second.yaw, "feedforward_term_rad_s")


def test_pid_antiwindup_and_acceleration_limit() -> None:
    pid = _pid()
    pid.decide(_input(0, error_rad=(1.0, 0.0)))
    second = pid.decide(_input(1, error_rad=(1.0, 0.0)))
    assert second.yaw.integral_rad_s == 0.0
    assert second.yaw.rate_limited
    assert second.yaw.acceleration_limited
    assert second.yaw.final_rad_s == pytest.approx(0.1)


@pytest.mark.parametrize("changes,reason", [
    ({"safety_allowed": False}, "safety_hold"),
    ({"gimbal_valid": False}, "gimbal_invalid"),
    ({"timing": TimingVerdict(False, "frame_identity_unverified")}, "frame_identity_unverified"),
])
def test_pid_holds_and_resets_on_invalid_input(changes: dict, reason: str) -> None:
    pid = _pid()
    pid.decide(_input(0))
    hold = pid.decide(_input(1, **changes))
    assert hold.reason == reason and hold.yaw.final_rad_s == 0.0
    assert pid.decide(_input(2)).yaw.final_rad_s == 0.0


def test_pid_holds_on_track_switch_and_nonmonotonic_time() -> None:
    pid = _pid()
    pid.decide(_input(0))
    assert pid.decide(_input(1, track_id=8)).reason == "target_switch_hold"
    assert pid.decide(_input(2, track_id=8)).reason == "tracking"
    assert pid.decide(_input(2, track_id=8)).reason == "nonmonotonic_decision_time"


def test_raw_pid_closes_error_in_deterministic_integrator_fixture() -> None:
    # This verifies controller sign and state progression, not hardware gains.
    axis = AxisPIDConfig(
        kp=2.0, ki=0.0, kd=0.0, integral_limit_rad_s=0.0,
        rate_limit_rad_s=0.5, acceleration_limit_rad_s2=10.0,
    )
    pid = BasicPID(axis, axis)
    camera_angle = 0.0
    target_angle = 0.2
    dt_s = 0.02
    for tick in range(150):
        command = pid.decide(_input(
            tick, decision_ns=1_000_000_000 + tick * 20_000_000,
            error_rad=(target_angle - camera_angle, 0.0),
        ))
        camera_angle += command.yaw.final_rad_s * dt_s
    assert abs(target_angle - camera_angle) < 0.001
