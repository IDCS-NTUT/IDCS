from __future__ import annotations

from common.schemas import CamState, ManualControlState
from jetson.control_v3.local_pid_trial import reference_offset_rad, safe_state


def _gimbal() -> CamState:
    return CamState(frame_id=1, src_ts_ms=0, state_monotonic_ns=1_000_000_000,
                    pan=0.0, tilt=0.0, pan_rate=0.0, tilt_rate=0.0)


def _manual() -> ManualControlState:
    return ManualControlState(src_ts_ms=0, source="test", active=False,
                              emergency=False, control_cmd_enabled=True,
                              joystick_raw=(0, 0), joystick_rate_cmd=(0.0, 0.0))


def _state(**overrides):
    fields = dict(now_ns=1_020_000_000, gimbal=_gimbal(),
                  gimbal_receipt_ns=1_010_000_000, manual=_manual(),
                  manual_receipt_ns=1_010_000_000, home_yaw_rad=0.0,
                  home_pitch_rad=0.0)
    fields.update(overrides)
    return safe_state(**fields)


def test_reference_is_bounded_symmetric_and_returns_home() -> None:
    assert [reference_offset_rad(t) for t in (0, 3, 8, 13)] == [0, 0.06, -0.06, 0]


def test_ready_requires_fresh_safe_encoder_and_manual_state() -> None:
    assert _state() == "ready"
    assert _state(gimbal_receipt_ns=800_000_000) == "gimbal_stale"
    assert _state(manual_receipt_ns=100_000_000) == "safety_stale"
    assert _state(gimbal=_gimbal().model_copy(update={"pan": 0.16})) == "yaw_travel_limit"
    assert _state(gimbal=_gimbal().model_copy(update={"tilt": 0.04})) == "pitch_travel_limit"
    assert _state(manual=_manual().model_copy(update={"emergency": True})) == "safety_hold"
