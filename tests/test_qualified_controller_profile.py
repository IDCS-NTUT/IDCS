from __future__ import annotations

import json
from pathlib import Path

import pytest

from jetson.qualified_controller_profile import load_qualified_shadow_policy_config


def _report(*, qualified: bool = True) -> dict:
    axis = lambda kp, kd, q: {
        "pid_gains": {"kp": kp, "ki": 0.0, "kd": kd},
        "kalman_config": {
            "acceleration_spectral_density": q,
            "measurement_variance_rad2": 9e-6,
            "initial_position_variance_rad2": 1e-4,
            "initial_rate_variance_rad2_s2": 0.25,
            "innovation_gate_nis": 16.0,
            "max_gap_s": 0.25,
            "max_consecutive_rejections": 2,
        },
        "feedforward_gain": 0.5,
        "qualification": {"qualified": qualified},
    }
    return {
        "format": "idcs.offline_los_estimator_validation",
        "qualification": {"qualified": qualified},
        "measurement_scenario": {"controller_hz": 50.0, "vision_hz": 30.0},
        "axes": {"yaw": axis(8.0, 0.1, 0.001), "pitch": axis(6.0, 0.0, 0.005)},
    }


def test_loader_preserves_frozen_axis_parameters(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")

    config = load_qualified_shadow_policy_config(path, yaw_position_limits_rad=(-1.0, 1.0))

    assert config.yaw_kp == 8.0 and config.yaw_kd == 0.1
    assert config.pitch_kp == 6.0 and config.pitch_kd == 0.0
    assert config.yaw_feedforward_gain == config.pitch_feedforward_gain == 0.5
    assert config.yaw_los_kalman is not None
    assert config.yaw_los_kalman.acceleration_spectral_density == 0.001
    assert config.pitch_los_kalman is not None
    assert config.pitch_los_kalman.acceleration_spectral_density == 0.005
    assert config.yaw_position_limits_rad == (-1.0, 1.0)


def test_loader_rejects_unqualified_report(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(qualified=False)), encoding="utf-8")

    with pytest.raises(ValueError, match="not qualified"):
        load_qualified_shadow_policy_config(path)


def test_loader_rejects_nonzero_integral_gain(tmp_path: Path) -> None:
    report = _report()
    report["axes"]["yaw"]["pid_gains"]["ki"] = 0.25
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="nonzero integral gain"):
        load_qualified_shadow_policy_config(path)
