#!/usr/bin/env python3
"""Qualified-fit-backed plant simulation and reproducible PID gain search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from common.gimbal.gray_box import (
    AxisPlant,
    _continuous_decay_rate,
    load_qualified_plants,
)


REPORT_FORMAT = "idcs.offline_pid_search"
REPORT_VERSION = 2


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float
    reference: Callable[[float], float]
    step_time_s: float | None = None
    final_reference: float | None = None


@dataclass
class PIDState:
    integral: float = 0.0
    previous_command: float = 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _step(name: str, amplitude: float, *, at_s: float = 0.25, duration_s: float = 3.0) -> Scenario:
    return Scenario(
        name=name,
        duration_s=duration_s,
        reference=lambda t, amplitude=amplitude, at_s=at_s: amplitude if t >= at_s else 0.0,
        step_time_s=at_s,
        final_reference=amplitude,
    )


def search_scenarios() -> tuple[Scenario, ...]:
    return (
        _step("step_pos_0p10", 0.10),
        _step("step_neg_0p10", -0.10),
        _step("step_pos_0p25", 0.25, duration_s=4.0),
        _step("step_neg_0p25", -0.25, duration_s=4.0),
        Scenario(
            "reversal_0p20",
            5.0,
            lambda t: 0.0 if t < 0.25 else (0.20 if t < 2.0 else (-0.20 if t < 3.75 else 0.0)),
        ),
    )


def holdout_scenarios() -> tuple[Scenario, ...]:
    return (
        _step("holdout_step_pos_0p15", 0.15, at_s=0.4, duration_s=3.5),
        _step("holdout_step_neg_0p20", -0.20, at_s=0.4, duration_s=3.5),
        Scenario("holdout_sine", 8.0, lambda t: 0.12 * math.sin(2.0 * math.pi * 0.20 * t)),
        Scenario(
            "holdout_piecewise",
            7.0,
            lambda t: float(np.interp(t, [0, 1, 2.5, 4, 5.5, 7], [0, 0.12, -0.18, 0.22, -0.08, 0])),
        ),
    )


def simulate_pid(
    plant: AxisPlant,
    gains: PIDGains,
    scenario: Scenario,
    *,
    dt_s: float = 0.02,
    rate_limit_rad_s: float = 0.5,
    accel_limit_rad_s2: float = 3.5,
) -> dict[str, Any]:
    count = int(math.ceil(scenario.duration_s / dt_s)) + 1
    time_s = np.arange(count, dtype=float) * dt_s
    reference = np.asarray([scenario.reference(float(t)) for t in time_s], dtype=float)
    theta = np.zeros(count, dtype=float)
    omega = np.zeros(count, dtype=float)
    command = np.zeros(count, dtype=float)
    rate_limited = np.zeros(count, dtype=bool)
    accel_limited = np.zeros(count, dtype=bool)
    state = PIDState()
    history_t = [0.0]
    history_u = [0.0]

    for idx in range(count - 1):
        error = reference[idx] - theta[idx]
        p_and_d = gains.kp * error - gains.kd * omega[idx]
        candidate_integral = state.integral + error * dt_s
        raw_candidate = p_and_d + gains.ki * candidate_integral
        bounded_candidate = float(np.clip(raw_candidate, -rate_limit_rad_s, rate_limit_rad_s))
        saturated_outward = raw_candidate != bounded_candidate and math.copysign(1.0, raw_candidate) == math.copysign(1.0, error)
        if not saturated_outward:
            state.integral = candidate_integral
        raw = p_and_d + gains.ki * state.integral
        bounded = float(np.clip(raw, -rate_limit_rad_s, rate_limit_rad_s))
        rate_limited[idx] = not math.isclose(raw, bounded, rel_tol=0.0, abs_tol=1e-12)
        max_delta = accel_limit_rad_s2 * dt_s
        applied = float(np.clip(bounded, state.previous_command - max_delta, state.previous_command + max_delta))
        accel_limited[idx] = not math.isclose(bounded, applied, rel_tol=0.0, abs_tol=1e-12)
        command[idx] = applied
        state.previous_command = applied
        history_t.append(float(time_s[idx]))
        history_u.append(applied)
        delayed_at = float(time_s[idx] - plant.delay_s)
        delayed_index = int(np.searchsorted(history_t, delayed_at, side="right") - 1)
        delayed_command = history_u[max(0, delayed_index)]
        theta[idx + 1], omega[idx + 1] = plant.advance(theta[idx], omega[idx], delayed_command, dt_s)

    command[-1] = command[-2]
    error = reference - theta
    metrics = _metrics(time_s, reference, theta, command, rate_limited, accel_limited, scenario)
    return {
        "time_s": time_s,
        "reference_rad": reference,
        "theta_rad": theta,
        "omega_rad_s": omega,
        "command_rad_s": command,
        "error_rad": error,
        "rate_limited": rate_limited,
        "accel_limited": accel_limited,
        "metrics": metrics,
    }


def _metrics(
    time_s: np.ndarray,
    reference: np.ndarray,
    theta: np.ndarray,
    command: np.ndarray,
    rate_limited: np.ndarray,
    accel_limited: np.ndarray,
    scenario: Scenario,
) -> dict[str, Any]:
    error = reference - theta
    abs_error = np.abs(error)
    result: dict[str, Any] = {
        "rms_error_rad": float(np.sqrt(np.mean(error * error))),
        "p95_abs_error_rad": float(np.percentile(abs_error, 95.0)),
        "max_abs_error_rad": float(np.max(abs_error)),
        "final_abs_error_rad": float(abs_error[-1]),
        "command_rms_rad_s": float(np.sqrt(np.mean(command * command))),
        "command_total_variation_rad_s": float(np.sum(np.abs(np.diff(command)))),
        "rate_limited_fraction": float(np.mean(rate_limited)),
        "accel_limited_fraction": float(np.mean(accel_limited)),
    }
    if scenario.step_time_s is not None and scenario.final_reference is not None:
        active = time_s >= scenario.step_time_s
        signed_response = np.sign(scenario.final_reference) * theta[active]
        target = abs(scenario.final_reference)
        result["overshoot_rad"] = float(max(0.0, np.max(signed_response) - target))
        tolerance = max(0.005, 0.02 * target)
        active_indices = np.flatnonzero(active)
        settling_time = None
        for index in active_indices:
            if np.all(abs_error[index:] <= tolerance):
                settling_time = float(time_s[index] - scenario.step_time_s)
                break
        result["settling_time_s"] = settling_time
        result["settling_tolerance_rad"] = tolerance
    return result


def _suite_score(results: Mapping[str, Mapping[str, Any]]) -> float:
    metrics = [result["metrics"] for result in results.values()]
    rms = float(np.mean([entry["rms_error_rad"] for entry in metrics]))
    p95 = float(np.mean([entry["p95_abs_error_rad"] for entry in metrics]))
    final = float(np.mean([entry["final_abs_error_rad"] for entry in metrics]))
    effort = float(np.mean([entry["command_rms_rad_s"] for entry in metrics]))
    accel_sat = float(np.mean([entry["accel_limited_fraction"] for entry in metrics]))
    overshoot = float(np.mean([entry.get("overshoot_rad", 0.0) for entry in metrics]))
    unsettled = sum(entry.get("settling_time_s") is None for entry in metrics if "settling_time_s" in entry)
    return rms + 0.35 * p95 + 0.8 * final + 0.5 * overshoot + 0.01 * effort + 0.02 * accel_sat + 0.1 * unsettled


def evaluate_suite(plant: AxisPlant, gains: PIDGains, scenarios: Iterable[Scenario], *, dt_s: float) -> dict[str, Any]:
    results = {scenario.name: simulate_pid(plant, gains, scenario, dt_s=dt_s) for scenario in scenarios}
    return {"score": _suite_score(results), "scenarios": results}


def search_axis(
    plant: AxisPlant,
    *,
    kp_values: Sequence[float],
    ki_values: Sequence[float],
    kd_values: Sequence[float],
    dt_s: float,
) -> tuple[PIDGains, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for kp, ki, kd in itertools.product(kp_values, ki_values, kd_values):
        gains = PIDGains(float(kp), float(ki), float(kd))
        result = evaluate_suite(plant, gains, search_scenarios(), dt_s=dt_s)
        rows.append({"kp": gains.kp, "ki": gains.ki, "kd": gains.kd, "score": float(result["score"])})
    rows.sort(key=lambda row: (row["score"], row["kp"], row["ki"], row["kd"]))
    best = rows[0]
    return PIDGains(best["kp"], best["ki"], best["kd"]), rows


def _json_suite(suite: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "score": suite["score"],
        "scenarios": {name: result["metrics"] for name, result in suite["scenarios"].items()},
    }


def holdout_failures(suite: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, result in suite["scenarios"].items():
        metrics = result["metrics"]
        if name.startswith("holdout_step"):
            if metrics["final_abs_error_rad"] > 0.005:
                failures.append(f"{name}:final_error")
            if metrics.get("overshoot_rad", math.inf) > 0.01:
                failures.append(f"{name}:overshoot")
            settling = metrics.get("settling_time_s")
            if settling is None or settling > 1.0:
                failures.append(f"{name}:settling_time")
        else:
            if metrics["rms_error_rad"] > 0.03:
                failures.append(f"{name}:rms_error")
            if metrics["max_abs_error_rad"] > 0.06:
                failures.append(f"{name}:max_error")
    return failures


def _write_holdout_plot(path: Path, axis: str, suite: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(suite["scenarios"].items())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 2.6 * len(scenarios)), constrained_layout=True)
    for plot_axis, (name, result) in zip(np.atleast_1d(axes), scenarios):
        plot_axis.plot(result["time_s"], result["reference_rad"], "k--", label="reference")
        plot_axis.plot(result["time_s"], result["theta_rad"], label="plant")
        plot_axis.plot(result["time_s"], result["command_rad_s"], alpha=0.65, label="rate command")
        plot_axis.set_title(name)
        plot_axis.set_ylabel("rad / rad s$^{-1}$")
        plot_axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="best", ncol=3)
    fig.suptitle(f"{axis} selected PID on held-out scenarios")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_trace(path: Path, axis: str, scenario: str, result: Mapping[str, Any]) -> None:
    fields = ["axis", "scenario", "time_s", "reference_rad", "theta_rad", "omega_rad_s", "command_rad_s", "error_rad", "rate_limited", "accel_limited"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(result["time_s"])):
            writer.writerow({
                "axis": axis,
                "scenario": scenario,
                "time_s": result["time_s"][index],
                "reference_rad": result["reference_rad"][index],
                "theta_rad": result["theta_rad"][index],
                "omega_rad_s": result["omega_rad_s"][index],
                "command_rad_s": result["command_rad_s"][index],
                "error_rad": result["error_rad"][index],
                "rate_limited": int(result["rate_limited"][index]),
                "accel_limited": int(result["accel_limited"][index]),
            })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.dt_s) or args.dt_s <= 0.0:
        raise SystemExit("--dt-s must be finite and positive")

    plants = load_qualified_plants(args.fit_report, args.validation_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "sources": {
            "fit_report": str(args.fit_report.resolve()),
            "fit_report_sha256": _sha256(args.fit_report),
            "validation_report": str(args.validation_report.resolve()),
            "validation_report_sha256": _sha256(args.validation_report),
        },
        "simulation": {"dt_s": args.dt_s, "rate_limit_rad_s": 0.5, "accel_limit_rad_s2": 3.5},
        "axes": {},
    }
    grid = {
        "kp": (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0),
        "ki": (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
        "kd": (0.0, 0.01, 0.02, 0.05, 0.1, 0.2),
    }
    for axis, plant in plants.items():
        gains, candidates = search_axis(plant, kp_values=grid["kp"], ki_values=grid["ki"], kd_values=grid["kd"], dt_s=args.dt_s)
        search_result = evaluate_suite(plant, gains, search_scenarios(), dt_s=args.dt_s)
        holdout_result = evaluate_suite(plant, gains, holdout_scenarios(), dt_s=args.dt_s)
        failures = holdout_failures(holdout_result)
        axis_dir = args.output_dir / axis
        axis_dir.mkdir(parents=True, exist_ok=True)
        with (axis_dir / "gain_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("kp", "ki", "kd", "score"))
            writer.writeheader()
            writer.writerows(candidates)
        for name, result in {**search_result["scenarios"], **holdout_result["scenarios"]}.items():
            _write_trace(axis_dir / f"{name}.csv", axis, name, result)
        if args.plot:
            _write_holdout_plot(axis_dir / "holdout_response.png", axis, holdout_result)
        report["axes"][axis] = {
            "plant": plant.__dict__,
            "selected_gains": gains.__dict__,
            "search": _json_suite(search_result),
            "holdout": _json_suite(holdout_result),
            "qualification": {"qualified": not failures, "failures": failures},
        }
        print(f"{axis}: kp={gains.kp:g} ki={gains.ki:g} kd={gains.kd:g} search={search_result['score']:.6g} holdout={holdout_result['score']:.6g}")
    all_failures = [f"{axis}:{failure}" for axis, values in report["axes"].items() for failure in values["qualification"]["failures"]]
    report["acceptance"] = {
        "step_final_abs_error_rad_max": 0.005,
        "step_overshoot_rad_max": 0.01,
        "step_settling_time_s_max": 1.0,
        "moving_rms_error_rad_max": 0.03,
        "moving_max_abs_error_rad_max": 0.06,
    }
    report["qualification"] = {"qualified": not all_failures, "failures": all_failures}
    report_path = args.output_dir / "pid_search_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
