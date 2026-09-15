from __future__ import annotations

from pathlib import Path
import json

from tools.validate_gimbal_fit import (
    _axis_failures,
    _selected_discrete_candidate,
    build_validation_report,
)


def test_selected_discrete_candidate_returns_exact_coefficients() -> None:
    coefficients = {"c_omega": 0.1, "c_u": 0.9, "bias": 0.0, "dt_s": 0.02}
    entry = {
        "selected_model": "discrete-first-order",
        "model_comparison": [
            {"model": "continuous-derivative", "parameters": {}},
            {"model": "discrete-first-order", "coefficients": coefficients},
        ],
    }

    selected = _selected_discrete_candidate(entry)

    assert selected is not None
    assert selected["coefficients"] == coefficients


def test_selected_discrete_candidate_rejects_missing_coefficients() -> None:
    entry = {
        "selected_model": "discrete-first-order",
        "model_comparison": [{"model": "discrete-first-order"}],
    }

    try:
        _selected_discrete_candidate(entry)
    except ValueError as exc:
        assert "coefficients" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("discrete model without coefficients was accepted")


def test_validation_requires_both_axis_parameter_sets(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("axis\n", encoding="utf-8")
    bad_report = {"axes": {"yaw": {"parameters": {"a_u": 1, "a_f": 1, "bias": 0}}}}
    try:
        build_validation_report(bad_report, fit_report_path=tmp_path / "fit.json", validation_csv=csv_path)
    except ValueError as exc:
        assert "pitch" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("missing pitch parameters were accepted")


def test_accuracy_gate_rejects_insufficient_range_and_error() -> None:
    failures = _axis_failures(
        axis="yaw",
        fit_metrics={"command_max_abs": 0.31},
        validation_metrics={
            "sample_count": 199,
            "command_max_abs": 0.21,
            "omega_rmse": 0.051,
            "theta_rmse": 0.011,
            "direction_omega_bias": {"-1": -0.021, "1": 0.0},
        },
        min_samples_per_axis=200,
        min_command_abs_rad_s=0.45,
        max_omega_rmse_rad_s=0.05,
        max_theta_rmse_rad=0.01,
        max_direction_bias_rad_s=0.02,
    )
    assert set(failures) == {
        "yaw:insufficient_samples",
        "yaw:fit_command_coverage",
        "yaw:validation_command_coverage",
        "yaw:omega_rmse",
        "yaw:theta_rmse",
        "yaw:direction_bias",
    }


def test_accuracy_gate_accepts_metrics_at_thresholds() -> None:
    failures = _axis_failures(
        axis="pitch",
        fit_metrics={"command_max_abs": 0.45},
        validation_metrics={
            "sample_count": 200,
            "command_max_abs": 0.45,
            "omega_rmse": 0.05,
            "theta_rmse": 0.01,
            "direction_omega_bias": {"-1": -0.02, "1": 0.02},
        },
        min_samples_per_axis=200,
        min_command_abs_rad_s=0.45,
        max_omega_rmse_rad_s=0.05,
        max_theta_rmse_rad=0.01,
        max_direction_bias_rad_s=0.02,
    )
    assert failures == []


def test_existing_low_range_fit_is_not_offline_control_qualified() -> None:
    root = Path(__file__).resolve().parents[1]
    fit_path = root / "artifacts/gimbal_fit/controller_sysid_step_timed_unloaded_20260908/fit_report_external_selected.json"
    validation_path = root / "logs/controller_sysid_step_timed_unloaded_20260908_validation.csv"
    fit_report = json.loads(fit_path.read_text(encoding="utf-8"))

    report = build_validation_report(
        fit_report,
        fit_report_path=fit_path,
        validation_csv=validation_path,
    )

    assert report["qualification"]["qualified"] is False
    assert set(report["qualification"]["failures"]) == {
        "yaw:fit_command_coverage",
        "yaw:validation_command_coverage",
        "pitch:fit_command_coverage",
        "pitch:validation_command_coverage",
    }
