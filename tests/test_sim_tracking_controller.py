from __future__ import annotations

import math

import pytest

from common.control import ControlConfig, LaserMountConfig
from common.perception import (
    NormalizedBoxV2,
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
    TargetSelectionV2,
    TrackAssessmentV2,
)
from common.schemas import (
    ControlGimbalObservation,
    ControlIntent,
    ControlObservation,
    ControlSafetyObservation,
    ControlTargetObservation,
    ControlTransportObservation,
)
from jetson.sim_control_runtime import (
    _acquisition_time_s,
    apply_sim_camera_intrinsics,
    control_cmd_from_intent,
    load_sim_baseline_policy_config,
    load_sim_evaluation_contract,
    require_loopback_endpoint,
)


def _observation(*, target_valid: bool = True) -> ControlObservation:
    target_px = (768.0, 360.0)
    aim_px = (640.0, 422.3538291)
    error_px = (target_px[0] - aim_px[0], target_px[1] - aim_px[1])
    return ControlObservation(
        sequence=3,
        created_monotonic_ns=2_000_000_000,
        source_frame_id=17,
        source_time_ns=1_900_000_000,
        source_clock_domain="test",
        target=ControlTargetObservation(
            valid=target_valid,
            track_id=4 if target_valid else None,
            class_id="person" if target_valid else None,
            confidence=0.9 if target_valid else None,
            target_center_px=target_px if target_valid else None,
            aim_reference_px=aim_px if target_valid else None,
            pixel_error=error_px if target_valid else None,
            bearing_error_rad=(
                math.atan(128.0 / 640.0),
                math.atan(62.3538291 / 623.5382907),
            ) if target_valid else None,
            distance_m=4.0 if target_valid else None,
            distance_source="known_size:width" if target_valid else None,
            parallax_active=target_valid,
            on_target=False if target_valid else None,
            source_age_ms=5.0,
        ),
        gimbal=ControlGimbalObservation(
            valid=True,
            yaw_rad=0.2,
            pitch_rad=-0.1,
            yaw_rate_rad_s=0.0,
            pitch_rate_rad_s=0.0,
            sample_age_ms=3.0,
        ),
        transport=ControlTransportObservation(),
        safety=ControlSafetyObservation(
            valid=True,
            auto_allowed=True,
            manual_active=False,
            emergency_active=False,
            sample_age_ms=0.0,
        ),
    )


def _snapshot() -> PerceptionSnapshotV2:
    return PerceptionSnapshotV2(
        sequence=17,
        frame=PerceptionFrameV2(
            frame_id=17,
            source_time_ns=1_900_000_000,
            observed_time_ns=1_950_000_000,
            source_clock_domain="test",
            observation_clock_domain="test",
            width=1280,
            height=720,
        ),
        tracks=(
            PerceptionTrackV2(
                track_id=4,
                box=NormalizedBoxV2(x=0.55, y=0.40, w=0.10, h=0.20),
                class_id="person",
                confidence=0.9,
                missed_frames=0,
            ),
        ),
        assessments=(
            TrackAssessmentV2(track_id=4, distance_m=4.0, distance_src="width"),
        ),
        selection=TargetSelectionV2(
            track_id=4,
            source_frame_id=16,
            applied_frame_id=17,
            selected_time_ns=1_960_000_000,
            selection_clock_domain="test",
            policy="test",
        ),
    )


def _intent(*, reason: str = "tracking") -> ControlIntent:
    return ControlIntent(
        sequence=2,
        observation_sequence=3,
        issued_monotonic_ns=2_000_000_000,
        valid_until_monotonic_ns=2_050_000_000,
        yaw_rate_rad_s=0.3,
        pitch_rate_rad_s=-0.2,
        reason=reason,
    )


def _projection_config() -> tuple[ControlConfig, LaserMountConfig]:
    config = {
        "control": {
            "mode": "rate",
            "aim_mode": "laser_point",
            "fx_fy_from_fov": True,
            "fov_deg": {"h": 90.0, "v": 60.0},
            "pid": {
                "kp": {"yaw": 1.0, "pitch": 1.0},
                "kd": {"yaw": 0.0, "pitch": 0.0},
                "ki": {"yaw": 0.0, "pitch": 0.0},
                "rate_limits": {"yaw": 0.5, "pitch": 0.5},
                "accel_limits": {"yaw": 1.0, "pitch": 1.0},
            },
            "laser": {
                "tolerance_px": 30.0,
                "use_range": "known_size",
                "default_distance_m": 10.0,
            },
            "sign_convention": {
                "yaw_positive": "right",
                "pitch_positive": "up",
            },
        },
        "laser": {
            "offset_m": {"x": 0.0, "y": -0.4, "z": 0.0},
            "dir_cam": {"x": 0.0, "y": 0.0, "z": 1.0},
        },
    }
    return (
        ControlConfig.from_raw_config(config, (1280, 720)),
        LaserMountConfig.from_raw_config(config),
    )


def test_loopback_endpoint_gate_rejects_lan_and_missing_port() -> None:
    assert require_loopback_endpoint("tcp://127.0.0.1:5571", "test") == "tcp://127.0.0.1:5571"
    with pytest.raises(ValueError, match="loopback"):
        require_loopback_endpoint("tcp://192.168.0.5:5557", "test")
    with pytest.raises(ValueError, match="valid port"):
        require_loopback_endpoint("tcp://127.0.0.1", "test")


def test_sim_camera_intrinsics_replace_hardware_fov() -> None:
    config = {
        "control": {
            "fx_fy_from_fov": True,
            "fov_deg": {"h": 135.0, "v": 73.0},
        },
        "sim": {"camera": {"fov_y_deg": 60.0}},
    }

    model = apply_sim_camera_intrinsics(config, (1280, 720))

    assert model["fov_x_deg"] == pytest.approx(91.4928445)
    assert model["fx_px"] == pytest.approx(623.5382907)
    assert model["fy_px"] == pytest.approx(623.5382907)
    assert config["control"]["fov_deg"]["h"] == pytest.approx(91.4928445)
    assert config["control"]["fov_deg"]["v"] == pytest.approx(60.0)


def test_sim_baseline_policy_is_estimator_free_and_hardware_independent() -> None:
    config = {
        "sim": {
            "baseline_controller": {
                "type": "bounded_p",
                "loop_hz": 50.0,
                "valid_for_ms": 50.0,
                "kp": {"yaw": 3.0, "pitch": 2.5},
                "rate_limits_rad_s": {"yaw": 0.3, "pitch": 0.2},
                "accel_limits_rad_s2": {"yaw": 1.0, "pitch": 0.8},
                "acceptance": {
                    "min_duration_s": 20.0,
                    "warmup_s": 5.0,
                    "max_acquisition_time_s": 5.0,
                    "acquisition_error_px": 22.0,
                    "acquisition_hold_s": 1.0,
                    "min_tracking_fraction": 0.98,
                    "max_rms_error_px": 14.0,
                    "max_p95_error_px": 22.0,
                    "max_rate_limited_fraction": 0.05,
                    "max_command_drops": 0,
                    "min_yaw_pose_span_rad": 0.05,
                },
            }
        }
    }

    policy, acceptance = load_sim_baseline_policy_config(config)

    assert policy.yaw_kp == pytest.approx(3.0)
    assert policy.pitch_kp == pytest.approx(2.5)
    assert policy.yaw_los_kalman is None
    assert policy.pitch_los_kalman is None
    assert policy.yaw_feedforward_gain == 0.0
    assert policy.pitch_feedforward_gain == 0.0
    assert policy.yaw_position_limits_rad is None
    assert acceptance["max_rms_error_px"] == pytest.approx(14.0)


def test_sim_evaluation_contract_forbids_hardware_tuning() -> None:
    config = {
        "sim": {
            "evaluation_contract": {
                "motion_model_role": "stable_substitute",
                "detection_fidelity_goal": "real_camera_equivalent",
                "hardware_controller_simulation_role": "interface_observation_only",
                "hardware_tuning_from_sim_allowed": False,
            }
        }
    }

    contract = load_sim_evaluation_contract(config)

    assert contract["hardware_tuning_from_sim_allowed"] is False
    with pytest.raises(ValueError, match="hardware tuning"):
        config["sim"]["evaluation_contract"][
            "hardware_tuning_from_sim_allowed"
        ] = True
        load_sim_evaluation_contract(config)


def test_acquisition_time_requires_a_sustained_p95_window() -> None:
    times = [index * 0.1 for index in range(31)]
    errors = [40.0] * 5 + [10.0] * 26

    acquired = _acquisition_time_s(
        times,
        errors,
        threshold_px=22.0,
        hold_s=1.0,
    )

    assert acquired == pytest.approx(0.5)


def test_tracking_intent_maps_to_simulator_control_cmd() -> None:
    control_config, laser_mount = _projection_config()
    command = control_cmd_from_intent(
        _observation(),
        _intent(),
        snapshot=_snapshot(),
        control_config=control_config,
        laser_mount=laser_mount,
        now_s=2.01,
    )

    assert command.target_ok is True
    assert command.frame_id == 17
    assert command.target_uv == pytest.approx((768.0, 360.0))
    assert command.err_uv == pytest.approx((128.0, -62.3538291))
    assert command.err_rad == pytest.approx(
        (
            math.atan(128.0 / control_config.fx_px),
            math.atan(62.3538291 / control_config.fy_px),
        )
    )
    assert command.pan_rate_cmd == pytest.approx(0.3)
    assert command.tilt_rate_cmd == pytest.approx(-0.2)
    assert command.parallax_compensation_active is True
    assert command.laser_origin_px == pytest.approx((640.0, 360.0))
    assert command.laser_dot_px == pytest.approx((640.0, 422.3538291))
    assert command.laser_on_target is False
    assert command.laser_range_m == pytest.approx(4.0)
    assert command.laser_range_source == "known_size:width"


def test_nontracking_intent_forces_zero_rates() -> None:
    control_config, laser_mount = _projection_config()
    command = control_cmd_from_intent(
        _observation(target_valid=False),
        _intent(reason="target_invalid"),
        snapshot=None,
        control_config=control_config,
        laser_mount=laser_mount,
        now_s=2.01,
    )

    assert command.target_ok is False
    assert command.pan_rate_cmd == 0.0
    assert command.tilt_rate_cmd == 0.0
    assert command.parallax_compensation_active is False
    assert command.laser_on_target is None
    assert command.laser_range_m is None
    assert command.laser_range_source is None
