from __future__ import annotations

import pytest

from common.control import AxisPair, ControlConfig, LaserAimingControlConfig
from common.perception import (
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
    TargetSelectionV2,
)
from common.schemas import (
    Box, CamState, ControlIntent, ControlIntentLimits, DetectionMsg, ManualControlState,
)
from jetson.controller import ControlLoop
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


def _snapshot() -> PerceptionSnapshotV2:
    return PerceptionSnapshotV2(
        sequence=1,
        frame=PerceptionFrameV2(
            frame_id=3,
            source_time_ns=0,
            observed_time_ns=0,
            source_clock_domain="test",
            observation_clock_domain="test",
            width=1280,
            height=720,
        ),
        tracks=(PerceptionTrackV2(
            track_id=8,
            box={"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
            class_id="drone",
            confidence=0.9,
            missed_frames=0,
        ),),
        selection=TargetSelectionV2(
            track_id=8,
            source_frame_id=3,
            applied_frame_id=3,
            selected_time_ns=0,
            selection_clock_domain="test",
            policy="preselected",
        ),
    )


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


def test_v2_snapshot_target_matches_legacy_geometry_without_mutation() -> None:
    legacy = ControlObservationAssembler(_config())
    legacy.update_detection(_detection(), received_at=10.0)
    legacy_target = legacy.build(now=10.04).target

    v2 = ControlObservationAssembler(_config())
    snapshot = _snapshot()
    v2.update_perception_snapshot(snapshot, received_at=10.0)
    v2_target = v2.build(now=10.04).target

    assert snapshot.selection.track_id == 8
    assert v2_target.valid and v2_target.track_id == legacy_target.track_id
    assert v2_target.class_id == legacy_target.class_id
    assert v2_target.confidence == pytest.approx(legacy_target.confidence)
    assert v2_target.bearing_error_rad == pytest.approx(legacy_target.bearing_error_rad)
    assert v2_target.bearing_rate_rad_s is None


def test_control_loop_accepts_immutable_observation_without_mutation() -> None:
    assembler = ControlObservationAssembler(_config())
    assembler.update_perception_snapshot(_snapshot(), received_at=10.0)
    observation = assembler.build(now=10.04)
    loop = ControlLoop(_config(), object())

    loop.update_control_observation(observation, received_at=10.04)

    assert observation.target.valid
    assert loop._latest_detection is not None
    assert loop._latest_detection.target_uv == pytest.approx((704.0, 396.0))
    assert loop._latest_target_track_id == 8


def test_intent_is_strict_and_cannot_expire_before_issue() -> None:
    with pytest.raises(ValueError):
        ControlIntent(sequence=1, observation_sequence=1, issued_monotonic_ns=20,
                      valid_until_monotonic_ns=19, yaw_rate_rad_s=0.0, pitch_rate_rad_s=0.0,
                      limits=ControlIntentLimits(), reason="hold")
    with pytest.raises(ValueError):
        ControlIntent(sequence=1, observation_sequence=1, issued_monotonic_ns=20,
                      valid_until_monotonic_ns=21, yaw_rate_rad_s=float("nan"), pitch_rate_rad_s=0.0,
                      limits=ControlIntentLimits(), reason="hold")
