#!/usr/bin/env python3
"""Score a frozen gimbal fit report against an independent sweep CSV.

This never fits, tunes, publishes, or opens hardware.  Rows affected by motor
limits or transport-quality faults are excluded by the same quality filter as
the fitter; their counts remain in the report so a seemingly good score cannot
hide an invalid experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from tools.fit_gimbal_response import (
    _axis_samples,
    _discrete_metrics,
    _metrics,
    load_sweep_samples,
)


REPORT_FORMAT = "idcs.gimbal_frozen_fit_validation"
REPORT_VERSION = 3

DEFAULT_MIN_SAMPLES_PER_AXIS = 200
DEFAULT_MIN_COMMAND_ABS_RAD_S = 0.45
DEFAULT_MAX_OMEGA_RMSE_RAD_S = 0.05
DEFAULT_MAX_THETA_RMSE_RAD = 0.01
DEFAULT_MAX_DIRECTION_BIAS_RAD_S = 0.02


def _selected_discrete_candidate(entry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the coefficients for the selected discrete model, if recorded."""

    selected_model = str(entry.get("selected_model", ""))
    if not selected_model.startswith("discrete-"):
        return None
    candidates = entry.get("model_comparison")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError(f"selected model {selected_model!r} has no model_comparison list")
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or candidate.get("model") != selected_model:
            continue
        coefficients = candidate.get("coefficients")
        if not isinstance(coefficients, Mapping):
            raise ValueError(f"selected model {selected_model!r} has no discrete coefficients")
        return candidate
    raise ValueError(f"selected model {selected_model!r} is absent from model_comparison")


def _validation_metrics(entry: Mapping[str, Any], axis_samples: Sequence[Any]) -> dict[str, Any]:
    """Score samples with the exact transition type selected by the fitter."""

    candidate = _selected_discrete_candidate(entry)
    parameters = entry["parameters"]
    delay_s = float(parameters.get("delay_s", 0.0))
    if candidate is not None:
        return _discrete_metrics(
            axis_samples,
            coeffs=candidate["coefficients"],
            delay_s=float(candidate.get("delay_s", delay_s)),
            model_name=str(entry["selected_model"]),
        )
    params = tuple(float(parameters[name]) for name in ("a_u", "a_f", "bias"))
    return _metrics(axis_samples, params=params, delay_s=delay_s)


def _resolved(path: Path) -> Path:
    """Resolve a provenance path without requiring it to still exist."""

    return path.expanduser().resolve(strict=False)


def _metric_float(value: Any, *, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) else fallback


def _axis_failures(
    *,
    axis: str,
    fit_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    min_samples_per_axis: int,
    min_command_abs_rad_s: float,
    max_omega_rmse_rad_s: float,
    max_theta_rmse_rad: float,
    max_direction_bias_rad_s: float,
) -> list[str]:
    failures: list[str] = []
    if int(validation_metrics.get("sample_count", 0)) < min_samples_per_axis:
        failures.append(f"{axis}:insufficient_samples")
    if _metric_float(fit_metrics.get("command_max_abs"), fallback=0.0) < min_command_abs_rad_s:
        failures.append(f"{axis}:fit_command_coverage")
    if _metric_float(validation_metrics.get("command_max_abs"), fallback=0.0) < min_command_abs_rad_s:
        failures.append(f"{axis}:validation_command_coverage")
    if _metric_float(validation_metrics.get("omega_rmse"), fallback=float("inf")) > max_omega_rmse_rad_s:
        failures.append(f"{axis}:omega_rmse")
    if _metric_float(validation_metrics.get("theta_rmse"), fallback=float("inf")) > max_theta_rmse_rad:
        failures.append(f"{axis}:theta_rmse")
    direction_bias = validation_metrics.get("direction_omega_bias", {})
    if not isinstance(direction_bias, Mapping) or not direction_bias:
        failures.append(f"{axis}:missing_direction_bias")
    elif max(abs(_metric_float(value, fallback=float("inf"))) for value in direction_bias.values()) > max_direction_bias_rad_s:
        failures.append(f"{axis}:direction_bias")
    return failures


def build_validation_report(
    fit_report: Mapping[str, Any],
    *,
    fit_report_path: Path,
    validation_csv: Path,
    min_samples_per_axis: int = DEFAULT_MIN_SAMPLES_PER_AXIS,
    min_command_abs_rad_s: float = DEFAULT_MIN_COMMAND_ABS_RAD_S,
    max_omega_rmse_rad_s: float = DEFAULT_MAX_OMEGA_RMSE_RAD_S,
    max_theta_rmse_rad: float = DEFAULT_MAX_THETA_RMSE_RAD,
    max_direction_bias_rad_s: float = DEFAULT_MAX_DIRECTION_BIAS_RAD_S,
) -> dict[str, Any]:
    samples, quality_filter = load_sweep_samples(validation_csv)
    limit_blocks = int(quality_filter.get("rejected_limit_blocked", 0))
    axes_raw = fit_report.get("axes")
    if not isinstance(axes_raw, Mapping):
        raise ValueError("fit report has no axes mapping")
    source = fit_report.get("source")
    fit_csv = source.get("csv") if isinstance(source, Mapping) else None
    independent_validation = bool(fit_csv) and _resolved(Path(str(fit_csv))) != _resolved(validation_csv)
    axes: dict[str, Any] = {}
    failures: list[str] = []
    for axis in ("yaw", "pitch"):
        entry = axes_raw.get(axis)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("parameters"), Mapping):
            raise ValueError(f"fit report has no usable {axis} parameters")
        parameters = entry["parameters"]
        params = tuple(float(parameters[name]) for name in ("a_u", "a_f", "bias"))
        delay_s = float(parameters.get("delay_s", 0.0))
        axis_samples = _axis_samples(samples, axis)
        fit_metrics = entry.get("train_metrics")
        if not isinstance(fit_metrics, Mapping):
            fit_metrics = {}
        metrics = _validation_metrics(entry, axis_samples)
        axis_failures = _axis_failures(
            axis=axis,
            fit_metrics=fit_metrics,
            validation_metrics=metrics,
            min_samples_per_axis=min_samples_per_axis,
            min_command_abs_rad_s=min_command_abs_rad_s,
            max_omega_rmse_rad_s=max_omega_rmse_rad_s,
            max_theta_rmse_rad=max_theta_rmse_rad,
            max_direction_bias_rad_s=max_direction_bias_rad_s,
        )
        failures.extend(axis_failures)
        axes[axis] = {
            "selected_model": entry.get("selected_model"),
            "parameters": {"a_u": params[0], "a_f": params[1], "bias": params[2]},
            "delay_s": delay_s,
            "accepted_sample_count": len(axis_samples),
            "fit_command_max_abs_rad_s": _metric_float(fit_metrics.get("command_max_abs"), fallback=0.0),
            "metrics": metrics,
            "qualified": not axis_failures,
            "failures": axis_failures,
        }
    if not samples:
        failures.append("no_accepted_samples")
    if limit_blocks:
        failures.append("limit_blocked")
    if not independent_validation:
        failures.append("validation_not_independent")
    qualified = not failures
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "fit_report": str(fit_report_path),
        "validation_csv": str(validation_csv),
        "quality_filter": quality_filter,
        "acceptance": {
            "min_samples_per_axis": min_samples_per_axis,
            "min_fit_command_abs_rad_s": min_command_abs_rad_s,
            "min_validation_command_abs_rad_s": min_command_abs_rad_s,
            "max_omega_rmse_rad_s": max_omega_rmse_rad_s,
            "max_theta_rmse_rad": max_theta_rmse_rad,
            "max_abs_direction_bias_rad_s": max_direction_bias_rad_s,
            "independent_validation_required": True,
        },
        "qualification": {
            "qualified": qualified,
            "reason": "qualified_model_accuracy" if qualified else "not_qualified_model_accuracy",
            "failures": failures,
            "independent_validation": independent_validation,
        },
        "axes": axes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-samples-per-axis", type=int, default=DEFAULT_MIN_SAMPLES_PER_AXIS)
    parser.add_argument("--min-command-abs-rad-s", type=float, default=DEFAULT_MIN_COMMAND_ABS_RAD_S)
    parser.add_argument("--max-omega-rmse-rad-s", type=float, default=DEFAULT_MAX_OMEGA_RMSE_RAD_S)
    parser.add_argument("--max-theta-rmse-rad", type=float, default=DEFAULT_MAX_THETA_RMSE_RAD)
    parser.add_argument("--max-direction-bias-rad-s", type=float, default=DEFAULT_MAX_DIRECTION_BIAS_RAD_S)
    args = parser.parse_args(argv)
    fit_report = json.loads(args.fit_report.read_text(encoding="utf-8"))
    if not isinstance(fit_report, Mapping):
        raise SystemExit("fit report must be a JSON object")
    report = build_validation_report(
        fit_report,
        fit_report_path=args.fit_report,
        validation_csv=args.validation_csv,
        min_samples_per_axis=args.min_samples_per_axis,
        min_command_abs_rad_s=args.min_command_abs_rad_s,
        max_omega_rmse_rad_s=args.max_omega_rmse_rad_s,
        max_theta_rmse_rad=args.max_theta_rmse_rad,
        max_direction_bias_rad_s=args.max_direction_bias_rad_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "qualification": report["qualification"], "quality_filter": report["quality_filter"],
                      "axes": {axis: values["metrics"] for axis, values in report["axes"].items()}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
