"""Load a qualified offline controller result into the shadow policy contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Tuple

from jetson.los_kalman import LOSKalmanConfig
from jetson.shadow_rate_policy import ShadowRatePolicyConfig


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"qualified controller report has no {key} mapping")
    return value


def load_qualified_shadow_policy_config(
    report_path: Path,
    *,
    yaw_position_limits_rad: Optional[Tuple[float, float]] = None,
    pitch_position_limits_rad: Optional[Tuple[float, float]] = None,
    valid_for_ns: int = 50_000_000,
    intent_mode: Literal["shadow", "live"] = "shadow",
    sequence_base: int = 0,
) -> ShadowRatePolicyConfig:
    """Construct an opt-in policy solely from a passing estimator report."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != "idcs.offline_los_estimator_validation":
        raise ValueError("unsupported qualified controller report format")
    qualification = _mapping(report, "qualification")
    if qualification.get("qualified") is not True:
        raise ValueError("controller estimator report is not qualified")
    axes = _mapping(report, "axes")
    simulation = _mapping(report, "measurement_scenario")
    yaw = _mapping(axes, "yaw")
    pitch = _mapping(axes, "pitch")
    for axis_name, axis in (("yaw", yaw), ("pitch", pitch)):
        axis_qualification = _mapping(axis, "qualification")
        if axis_qualification.get("qualified") is not True:
            raise ValueError(f"{axis_name} controller estimator result is not qualified")

    yaw_pid = _mapping(yaw, "pid_gains")
    pitch_pid = _mapping(pitch, "pid_gains")
    for axis_name, pid in (("yaw", yaw_pid), ("pitch", pitch_pid)):
        if abs(float(pid.get("ki", 0.0))) > 1e-12:
            raise ValueError(
                f"{axis_name} controller has nonzero integral gain, which the shadow policy does not implement"
            )
    yaw_kalman = LOSKalmanConfig(**dict(_mapping(yaw, "kalman_config")))
    pitch_kalman = LOSKalmanConfig(**dict(_mapping(pitch, "kalman_config")))
    controller_hz = float(simulation.get("controller_hz", 0.0))
    if controller_hz <= 0.0 or float(simulation.get("vision_hz", 0.0)) <= 0.0:
        raise ValueError("qualified controller report has invalid measurement cadence")
    nominal_period_s = 1.0 / controller_hz
    return ShadowRatePolicyConfig(
        yaw_kp=float(yaw_pid["kp"]),
        pitch_kp=float(pitch_pid["kp"]),
        yaw_kd=float(yaw_pid["kd"]),
        pitch_kd=float(pitch_pid["kd"]),
        yaw_rate_limit_rad_s=0.5,
        pitch_rate_limit_rad_s=0.5,
        yaw_accel_limit_rad_s2=3.5,
        pitch_accel_limit_rad_s2=3.5,
        nominal_period_s=nominal_period_s,
        valid_for_ns=valid_for_ns,
        yaw_position_limits_rad=yaw_position_limits_rad,
        pitch_position_limits_rad=pitch_position_limits_rad,
        yaw_los_kalman=yaw_kalman,
        pitch_los_kalman=pitch_kalman,
        yaw_feedforward_gain=float(yaw["feedforward_gain"]),
        pitch_feedforward_gain=float(pitch["feedforward_gain"]),
        intent_mode=intent_mode,
        sequence_base=sequence_base,
    )
