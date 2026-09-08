from __future__ import annotations

import pytest

from common.control import AxisPair, ControlConfig, LaserAimingControlConfig
from common.schemas import (
    Box, CamState, ControlIntent, ControlIntentLimits, DetectionMsg, ManualControlState,
)
from jetson.control_observation import ControlObservationAssembler


def _config() -> ControlConfig:
    return ControlConfig(mode="rate", loop_hz=50.0, fx_px=1000.0, fy_px=1000.0,
                         cx_px=640.0, cy_px=360.0, aim_mode="camera_center",
                         kp=AxisPair(1, 1), kd=AxisPair(0, 0), ki=AxisPair(0, 0),
                         rate_limits=AxisPair(1, 1), accel_limits=AxisPair(1, 1),
                         deadband_px=0, smooth_px_alpha=0, lost_target_timeout_ms=100,
                         reinit_on_lost=True, target_selector="preselected", yaw_sign=1,
                         pitch_sign=-1, frame_size=(1280, 720), fov_deg=None,
                         laser=LaserAimingControlConfig(1, "infinite", 10))


def _detection() -> DetectionMsg:
    return DetectionMsg(frame_id=3, src_ts_ms=0, rx_ts_ms=0, infer_ts_ms=0, img_w=1280, img_h=720,
                        target_idx=0, target_velocity_px_s=(100.0, -50.0),
                        boxes=[Box(x=0.5, y=0.5, w=0.1, h=0.1, cls="drone", conf=0.9, track_id=8)])


def _manual(**updates) -> ManualControlState:
    values = dict(src_ts_ms=0, source="test", active=False, emergency=False,
                  control_cmd_enabled=True, joystick_raw=(0, 0), joystick_rate_cmd=(0.0, 0.0))
    values.update(updates)
    return ManualControlState(**values)


def test_assembler_marks_complete_fresh_snapshot_valid() -> None:
    assembler = ControlObservationAssembler(_config())
    assembler.update_detection(_detection(), received_at=10.0)
    assembler.update_cam_state(CamState(frame_id=3, src_ts_ms=0, pan=0.2, tilt=-0.1), received_at=10.01)
    assembler.update_manual_state(_manual(), received_at=10.02)
    observation = assembler.build(now=10.04, serial_acceptance_ms=2.0)
    assert observation.sequence == 1
    assert observation.target.valid and observation.target.track_id == 8
    assert observation.target.bearing_error_rad == pytest.approx((0.0639, -0.0360), abs=1e-3)
    assert observation.target.bearing_rate_rad_s == pytest.approx((0.1, 0.05))
    assert observation.gimbal.valid and observation.safety.auto_allowed


def test_assembler_never_invents_stale_or_missing_inputs() -> None:
    assembler = ControlObservationAssembler(_config())
    assembler.update_detection(_detection(), received_at=1.0)
    observation = assembler.build(now=1.2)
    assert not observation.target.valid
    assert observation.target.source_age_ms == pytest.approx(200.0)
    assert not observation.gimbal.valid
    assert not observation.safety.valid and not observation.safety.auto_allowed


def test_intent_is_strict_and_cannot_expire_before_issue() -> None:
    with pytest.raises(ValueError):
        ControlIntent(sequence=1, observation_sequence=1, issued_monotonic_ns=20,
                      valid_until_monotonic_ns=19, yaw_rate_rad_s=0.0, pitch_rate_rad_s=0.0,
                      limits=ControlIntentLimits(), reason="hold")
    with pytest.raises(ValueError):
        ControlIntent(sequence=1, observation_sequence=1, issued_monotonic_ns=20,
                      valid_until_monotonic_ns=21, yaw_rate_rad_s=float("nan"), pitch_rate_rad_s=0.0,
                      limits=ControlIntentLimits(), reason="hold")
