from __future__ import annotations

from common.control import ControlConfig, LaserMountConfig
from common.schemas import (
    ControlGimbalObservation,
    ControlObservation,
    ControlSafetyObservation,
    ControlTargetObservation,
    ControlTransportObservation,
)
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig
from tools.compare_legacy_v2_controller_trace import compare_policies


def _config() -> ControlConfig:
    return ControlConfig.from_raw_config(
        {
            "control": {
                "mode": "rate",
                "controller": "pid",
                "loop_hz": 50.0,
                "fx_fy_from_fov": True,
                "fov_deg": {"h": 65.2, "v": 39.6},
                "pid": {
                    "kp": {"yaw": 1.0, "pitch": 1.0},
                    "kd": {"yaw": 0.0, "pitch": 0.0},
                    "ki": {"yaw": 0.0, "pitch": 0.0},
                    "rate_limits": {"yaw": 0.5, "pitch": 0.5},
                    "accel_limits": {"yaw": 10.0, "pitch": 10.0},
                },
                "laser": {
                    "tolerance_px": 20.0,
                    "use_range": "infinite",
                    "default_distance_m": 10.0,
                },
            }
        },
        (1280, 720),
    )


def _observation(sequence: int, *, auto_allowed: bool) -> ControlObservation:
    return ControlObservation(
        sequence=sequence,
        created_monotonic_ns=sequence * 20_000_000,
        target=ControlTargetObservation(
            valid=True,
            track_id=1,
            confidence=0.9,
            target_center_px=(700.0, 360.0),
            aim_reference_px=(640.0, 360.0),
            pixel_error=(60.0, 0.0),
            bearing_error_rad=(0.06, 0.0),
        ),
        gimbal=ControlGimbalObservation(
            valid=True, yaw_rad=0.0, pitch_rad=0.0,
            yaw_rate_rad_s=0.0, pitch_rate_rad_s=0.0,
        ),
        transport=ControlTransportObservation(),
        safety=ControlSafetyObservation(
            valid=True,
            auto_allowed=auto_allowed,
            manual_active=not auto_allowed,
            emergency_active=False,
        ),
    )


def test_same_snapshot_parity_reports_safety_and_rate_deltas() -> None:
    observations = [_observation(1, auto_allowed=True), _observation(2, auto_allowed=False)]
    policy = ShadowRatePolicy(
        ShadowRatePolicyConfig(
            yaw_kp=1.0,
            pitch_kp=1.0,
            yaw_accel_limit_rad_s2=10.0,
            pitch_accel_limit_rad_s2=10.0,
        )
    )

    result = compare_policies(
        observations,
        legacy_config=_config(),
        laser_mount=LaserMountConfig.from_raw_config({"laser": {}}),
        redesigned_policy=policy,
        max_safety_mismatches=0,
        max_rate_delta_rad_s=0.5,
    )

    assert result["physical_control_disabled"] is True
    assert result["observations"] == 2
    assert result["safety_decision_mismatches"] == 0
    assert result["qualified"] is True
