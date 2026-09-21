from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.schemas import ControlIntent
from jetson.control_runtime import _build_shutdown_intents, load_runtime_settings


def _qualified_report() -> dict:
    def axis(kp: float, kd: float, q: float) -> dict:
        return {
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
            "qualification": {"qualified": True},
        }

    return {
        "format": "idcs.offline_los_estimator_validation",
        "qualification": {"qualified": True},
        "measurement_scenario": {"controller_hz": 50.0, "vision_hz": 30.0},
        "axes": {
            "yaw": axis(8.0, 0.1, 0.001),
            "pitch": axis(6.0, 0.0, 0.005),
        },
    }


def _config(report: str) -> dict:
    return {
        "net": {
            "zmq_perception_v2": "tcp://jetson:5564",
            "zmq_gimbal_state": "tcp://jetson:5558",
            "zmq_manual_state": "tcp://jetson:5559",
            "zmq_control": "tcp://jetson:5557",
        },
        "gimbal": {
            "yaw_min_rad": None,
            "yaw_max_rad": None,
            "pitch_min_rad": -0.2,
            "pitch_max_rad": 1.0,
        },
        "controller_v2": {"qualified_report": report, "valid_for_ms": 80.0},
    }


def test_runtime_loads_only_qualified_live_policy(tmp_path: Path) -> None:
    report = tmp_path / "qualified.json"
    report.write_text(json.dumps(_qualified_report()), encoding="utf-8")

    settings, policy = load_runtime_settings(
        _config(report.name), base_dir=tmp_path, sequence_base=123_000
    )

    assert policy.intent_mode == "live"
    assert policy.yaw_kp == pytest.approx(8.0)
    assert policy.pitch_kp == pytest.approx(6.0)
    assert policy.pitch_position_limits_rad == (-0.2, 1.0)
    assert policy.valid_for_ns == 80_000_000
    assert policy.sequence_base == 123_000
    assert len(settings["qualified_report_sha256"]) == 64
    assert settings["intent_bind"] == "tcp://0.0.0.0:5557"


def test_runtime_rejects_missing_report_and_long_intent_lifetime(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_runtime_settings(_config("missing.json"), base_dir=tmp_path)

    report = tmp_path / "qualified.json"
    report.write_text(json.dumps(_qualified_report()), encoding="utf-8")
    config = _config(report.name)
    config["controller_v2"]["valid_for_ms"] = 500.0
    with pytest.raises(ValueError, match="valid_for_ms"):
        load_runtime_settings(config, base_dir=tmp_path)


def test_shutdown_redundancy_uses_unique_monotonic_sequences() -> None:
    last = ControlIntent(
        sequence=70,
        observation_sequence=41,
        issued_monotonic_ns=1_000_000,
        valid_until_monotonic_ns=2_000_000,
        mode="live",
        yaw_rate_rad_s=0.2,
        pitch_rate_rad_s=-0.1,
        reason="tracking",
    )

    stops = _build_shutdown_intents(
        last,
        issued_monotonic_ns=3_000_000,
        valid_for_ns=50_000_000,
    )

    assert [intent.sequence for intent in stops] == [71, 72, 73]
    assert all(intent.observation_sequence == 41 for intent in stops)
    assert all(intent.yaw_rate_rad_s == 0.0 for intent in stops)
    assert all(intent.pitch_rate_rad_s == 0.0 for intent in stops)
    assert all(intent.reason == "controller_shutdown" for intent in stops)


def test_shutdown_redundancy_starts_at_one_without_prior_intent() -> None:
    stops = _build_shutdown_intents(
        None,
        issued_monotonic_ns=3_000_000,
        valid_for_ns=50_000_000,
    )

    assert [intent.sequence for intent in stops] == [1, 2, 3]
    assert all(intent.observation_sequence == 0 for intent in stops)
