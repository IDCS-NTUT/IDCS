from __future__ import annotations

from unittest.mock import patch

import pytest

from common.control import AxisPair, ControlConfig, LaserAimingControlConfig
from common.perception import (
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
    TargetSelectionV2,
)
from common.schemas import (
    CamState, ControlIntent, ControlIntentLimits, ControlObservation,
    ManualControlState,
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
    assembler.update_perception_snapshot(_snapshot(), received_at=10.0)
    assembler.update_cam_state(CamState(frame_id=3, src_ts_ms=0, pan=0.2, tilt=-0.1), received_at=10.01)
    assembler.update_manual_state(_manual(), received_at=10.02)
    observation = assembler.build(now=10.04, serial_acceptance_ms=2.0)
    assert observation.sequence == 1
    assert observation.target.valid and observation.target.track_id == 8
    assert observation.target.bearing_error_rad == pytest.approx((0.0639, -0.0360), abs=1e-3)
    assert observation.target.target_center_px == pytest.approx((704.0, 396.0))
    assert observation.target.aim_reference_px == pytest.approx((640.0, 360.0))
    assert observation.target.pixel_error == pytest.approx((64.0, 36.0))
    assert observation.target.bearing_rate_rad_s is None
    assert observation.gimbal.valid and observation.safety.auto_allowed


def test_assembler_never_invents_stale_or_missing_inputs() -> None:
    assembler = ControlObservationAssembler(_config())
    assembler.update_perception_snapshot(_snapshot(), received_at=1.0)
    observation = assembler.build(now=1.2)
    assert not observation.target.valid
    assert observation.target.source_age_ms == pytest.approx(200.0)
    assert not observation.gimbal.valid
    assert not observation.safety.valid and not observation.safety.auto_allowed


def test_v2_snapshot_target_geometry_is_complete_without_mutation() -> None:
    v2 = ControlObservationAssembler(_config())
    snapshot = _snapshot()
    v2.update_perception_snapshot(snapshot, received_at=10.0)
    v2_observation = v2.build(now=10.04)
    v2_target = v2_observation.target

    assert snapshot.selection.track_id == 8
    assert v2_target.valid and v2_target.track_id == 8
    assert v2_target.class_id == "drone"
    assert v2_target.confidence == pytest.approx(0.9)
    assert v2_target.target_center_px == pytest.approx((704.0, 396.0))
    assert v2_target.aim_reference_px == pytest.approx((640.0, 360.0))
    assert v2_target.bearing_error_rad == pytest.approx((0.0639, -0.0360), abs=1e-3)
    assert v2_target.bearing_rate_rad_s is None
    assert v2_observation.source_frame_id == 3
    assert v2_observation.source_clock_domain == "test"


def test_control_loop_accepts_immutable_observation_without_mutation() -> None:
    assembler = ControlObservationAssembler(_config())
    assembler.update_perception_snapshot(_snapshot(), received_at=10.0)
    observation = assembler.build(now=10.04)
    loop = ControlLoop(_config(), object())

    loop.update_control_observation(observation, received_at=10.04)

    assert observation.target.valid
    assert loop._latest_detection is not None
    assert loop._latest_detection.target_uv == pytest.approx((704.0, 396.0))
    assert loop._latest_detection.frame_id == 3
    assert loop._latest_detection.src_ts_ms == 0
    assert loop._latest_target_track_id == 8


def test_v2_observation_drives_deterministic_simulation_command() -> None:
    assembler = ControlObservationAssembler(_config())
    assembler.update_perception_snapshot(_snapshot(), received_at=10.0)
    assembler.update_manual_state(_manual(), received_at=10.0)
    observation = assembler.build(now=10.0)
    loop = ControlLoop(_config(), object())
    loop.update_control_observation(observation, received_at=10.0)

    with patch.object(loop, "_send_cmd") as send:
        loop.tick(now=10.02)

    send.assert_called_once()
    command = send.call_args.args[0]
    assert command.frame_id == 3
    assert command.target_ok
    assert command.target_uv == pytest.approx((704.0, 396.0))
    assert command.err_rad == pytest.approx(observation.target.bearing_error_rad)


def test_control_loop_emits_zero_when_v2_authority_is_missing() -> None:
    assembler = ControlObservationAssembler(_config())
    assembler.update_perception_snapshot(_snapshot(), received_at=10.0)
    observation = assembler.build(now=10.0)
    loop = ControlLoop(_config(), object())
    loop.update_control_observation(observation, received_at=10.0)

    with patch.object(loop, "_send_cmd") as send:
        loop.tick(now=10.02)

    command = send.call_args.args[0]
    assert not command.target_ok
    assert command.pan_rate_cmd == 0.0
    assert command.tilt_rate_cmd == 0.0
    assert command.pan_abs_cmd is None
    assert command.tilt_abs_cmd is None


def test_control_observation_rejects_partial_source_provenance() -> None:
    assembler = ControlObservationAssembler(_config())
    assembler.update_perception_snapshot(_snapshot(), received_at=10.0)
    payload = assembler.build(now=10.0).model_dump(mode="json")
    payload["source_clock_domain"] = None

    with pytest.raises(ValueError, match="provenance fields must be set together"):
        ControlObservation.model_validate(payload)


def test_intent_is_strict_and_cannot_expire_before_issue() -> None:
    with pytest.raises(ValueError):
        ControlIntent(sequence=1, observation_sequence=1, issued_monotonic_ns=20,
                      valid_until_monotonic_ns=19, yaw_rate_rad_s=0.0, pitch_rate_rad_s=0.0,
                      limits=ControlIntentLimits(), reason="hold")
    with pytest.raises(ValueError):
        ControlIntent(sequence=1, observation_sequence=1, issued_monotonic_ns=20,
                      valid_until_monotonic_ns=21, yaw_rate_rad_s=float("nan"), pitch_rate_rad_s=0.0,
                      limits=ControlIntentLimits(), reason="hold")
