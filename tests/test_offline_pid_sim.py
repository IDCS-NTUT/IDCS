from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.offline_pid_sim import (
    AxisPlant,
    PIDGains,
    _continuous_decay_rate,
    _step,
    holdout_failures,
    load_qualified_plants,
    simulate_pid,
)


def test_negative_but_stable_pole_uses_fitter_euler_conversion() -> None:
    assert _continuous_decay_rate(-0.01, 0.02) == pytest.approx(50.5)


def test_unqualified_report_is_rejected(tmp_path: Path) -> None:
    fit_path = tmp_path / "fit.json"
    validation_path = tmp_path / "validation.json"
    fit_path.write_text("{}", encoding="utf-8")
    validation_path.write_text(json.dumps({"qualification": {"qualified": False}}), encoding="utf-8")

    with pytest.raises(ValueError, match="not qualified"):
        load_qualified_plants(fit_path, validation_path)


def test_pid_simulation_is_bounded_and_converges() -> None:
    plant = AxisPlant("yaw", a_f=20.0, b_pos=20.0, b_neg=20.0, disturbance=0.0, delay_s=0.0, fit_dt_s=0.02, source_model="test")
    result = simulate_pid(plant, PIDGains(kp=4.0, ki=0.2, kd=0.0), _step("step", 0.1), dt_s=0.02)

    assert max(abs(value) for value in result["command_rad_s"]) <= 0.5
    assert result["metrics"]["final_abs_error_rad"] < 0.01


def test_pid_simulation_is_deterministic() -> None:
    plant = AxisPlant("pitch", a_f=10.0, b_pos=9.0, b_neg=11.0, disturbance=0.01, delay_s=0.0, fit_dt_s=0.03, source_model="test")
    gains = PIDGains(kp=3.0, ki=0.5, kd=0.02)
    scenario = _step("step", -0.2)

    first = simulate_pid(plant, gains, scenario)
    second = simulate_pid(plant, gains, scenario)

    assert first["metrics"] == second["metrics"]
    assert (first["theta_rad"] == second["theta_rad"]).all()


def test_holdout_gate_reports_specific_failures() -> None:
    suite = {
        "scenarios": {
            "holdout_step": {"metrics": {"final_abs_error_rad": 0.006, "overshoot_rad": 0.0, "settling_time_s": 0.5}},
            "holdout_sine": {"metrics": {"rms_error_rad": 0.02, "max_abs_error_rad": 0.061}},
        }
    }

    assert holdout_failures(suite) == ["holdout_step:final_error", "holdout_sine:max_error"]
