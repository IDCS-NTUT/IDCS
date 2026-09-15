from __future__ import annotations

import numpy as np

from jetson.los_kalman import LOSKalmanConfig
from tools.offline_los_estimator_sim import TrackingScenario, _qualification, _write_trace, simulate_comparison
from tools.offline_pid_sim import AxisPlant, PIDGains


def test_comparison_is_deterministic() -> None:
    plant = AxisPlant("yaw", 25.0, 25.0, 25.0, 0.0, 0.0, 0.02, "test")
    gains = PIDGains(6.0, 0.0, 0.0)
    scenario = TrackingScenario("sine", 3.0, lambda t: 0.1 * __import__("math").sin(t))
    config = LOSKalmanConfig(acceleration_spectral_density=0.05)

    first = simulate_comparison(plant, gains, scenario, config, feedforward_gain=1.0, seed=4)
    second = simulate_comparison(plant, gains, scenario, config, feedforward_gain=1.0, seed=4)

    assert first["raw_metrics"] == second["raw_metrics"]
    assert first["estimated_metrics"] == second["estimated_metrics"]


def test_qualification_requires_improvement_and_latency() -> None:
    result = {
        "raw_metrics": {"rms_error_rad": 0.02, "command_total_variation_rad_s": 1.0},
        "estimated_metrics": {"rms_error_rad": 0.01, "command_total_variation_rad_s": 1.2},
        "estimator": {"updates": 10, "rejected_updates": 0, "update_p95_us": 10.0, "predict_p95_us": 10.0},
    }
    qualification = _qualification({"scenarios": {"case": result}})

    assert qualification["qualified"] is True
    assert qualification["relative_rms_change"] == -0.5


def test_trace_writer_uses_repository_lf_line_endings(tmp_path) -> None:
    path = tmp_path / "trace.csv"
    _write_trace(path, "yaw", "test", {
        "time_s": np.asarray([0.0, 0.02]),
        "target_rad": np.asarray([0.0, 0.1]),
    })

    assert b"\r\n" not in path.read_bytes()
