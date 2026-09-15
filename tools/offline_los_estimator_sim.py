#!/usr/bin/env python3
"""Tune and validate LOS Kalman feedforward against the qualified plant/PID."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jetson.los_kalman import AxisLOSKalman, LOSKalmanConfig
from tools.offline_pid_sim import AxisPlant, PIDGains, load_qualified_plants


REPORT_FORMAT = "idcs.offline_los_estimator_validation"
REPORT_VERSION = 1


@dataclass(frozen=True)
class TrackingScenario:
    name: str
    duration_s: float
    reference: Callable[[float], float]


@dataclass
class _PIDState:
    integral: float = 0.0
    previous_command: float = 0.0


def tuning_scenarios() -> tuple[TrackingScenario, ...]:
    return (
        TrackingScenario("tune_sine_slow", 8.0, lambda t: 0.12 * math.sin(2 * math.pi * 0.15 * t)),
        TrackingScenario("tune_sine_fast", 7.0, lambda t: 0.09 * math.sin(2 * math.pi * 0.35 * t + 0.4)),
        TrackingScenario(
            "tune_piecewise",
            7.0,
            lambda t: float(np.interp(t, [0, 1, 2.2, 3.6, 5.0, 7], [0, 0.15, -0.12, 0.20, -0.18, 0.05])),
        ),
        TrackingScenario(
            "tune_ramp_hold",
            6.0,
            lambda t: float(np.interp(t, [0, 0.5, 1.5, 3.5, 4.5, 6.0], [0, 0, 0.15, 0.15, -0.10, -0.10])),
        ),
    )


def holdout_scenarios() -> tuple[TrackingScenario, ...]:
    return (
        TrackingScenario("holdout_sine", 9.0, lambda t: 0.15 * math.sin(2 * math.pi * 0.22 * t + 0.7)),
        TrackingScenario(
            "holdout_piecewise",
            8.0,
            lambda t: float(np.interp(t, [0, 0.8, 2.0, 3.1, 4.5, 6.2, 8], [0, -0.11, 0.19, -0.20, 0.14, -0.06, 0.1])),
        ),
        TrackingScenario("holdout_step", 5.0, lambda t: 0.0 if t < 0.5 else 0.15),
    )


def _controller_step(
    state: _PIDState,
    gains: PIDGains,
    *,
    error: float,
    omega: float,
    feedforward: float,
    dt_s: float,
    rate_limit: float = 0.5,
    accel_limit: float = 3.5,
) -> tuple[float, bool, bool]:
    p_d_ff = gains.kp * error - gains.kd * omega + feedforward
    candidate_integral = state.integral + error * dt_s
    raw_candidate = p_d_ff + gains.ki * candidate_integral
    bounded_candidate = float(np.clip(raw_candidate, -rate_limit, rate_limit))
    outward = raw_candidate != bounded_candidate and math.copysign(1.0, raw_candidate) == math.copysign(1.0, error)
    if not outward:
        state.integral = candidate_integral
    raw = p_d_ff + gains.ki * state.integral
    bounded = float(np.clip(raw, -rate_limit, rate_limit))
    max_delta = accel_limit * dt_s
    applied = float(np.clip(bounded, state.previous_command - max_delta, state.previous_command + max_delta))
    rate_limited = not math.isclose(raw, bounded, rel_tol=0.0, abs_tol=1e-12)
    accel_limited = not math.isclose(bounded, applied, rel_tol=0.0, abs_tol=1e-12)
    state.previous_command = applied
    return applied, rate_limited, accel_limited


def _measurement_events(
    scenario: TrackingScenario,
    *,
    seed: int,
    vision_hz: float,
    latency_s: float,
    noise_std_rad: float,
    dropout_fraction: float,
) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    events: list[tuple[float, float, float]] = []
    for sample_time in np.arange(0.0, scenario.duration_s + 1e-12, 1.0 / vision_hz):
        if float(rng.random()) < dropout_fraction:
            continue
        measurement = scenario.reference(float(sample_time)) + float(rng.normal(0.0, noise_std_rad))
        events.append((float(sample_time + latency_s), float(sample_time), measurement))
    return events


def simulate_comparison(
    plant: AxisPlant,
    gains: PIDGains,
    scenario: TrackingScenario,
    estimator_config: LOSKalmanConfig,
    *,
    feedforward_gain: float,
    seed: int,
    dt_s: float = 0.02,
    vision_hz: float = 30.0,
    latency_s: float = 0.05,
    noise_std_rad: float = 0.003,
    dropout_fraction: float = 0.10,
) -> dict[str, Any]:
    count = int(math.ceil(scenario.duration_s / dt_s)) + 1
    times = np.arange(count, dtype=float) * dt_s
    target = np.asarray([scenario.reference(float(value)) for value in times])
    raw_theta = np.zeros(count)
    raw_omega = np.zeros(count)
    raw_command = np.zeros(count)
    est_theta = np.zeros(count)
    est_omega = np.zeros(count)
    est_command = np.zeros(count)
    estimated_angle = np.full(count, np.nan)
    estimated_rate = np.full(count, np.nan)
    raw_state = _PIDState()
    est_state = _PIDState()
    estimator = AxisLOSKalman(estimator_config)
    events = _measurement_events(
        scenario,
        seed=seed,
        vision_hz=vision_hz,
        latency_s=latency_s,
        noise_std_rad=noise_std_rad,
        dropout_fraction=dropout_fraction,
    )
    event_index = 0
    latest_measurement: float | None = None
    update_ns: list[int] = []
    predict_ns: list[int] = []
    raw_rate_limited = np.zeros(count, dtype=bool)
    raw_accel_limited = np.zeros(count, dtype=bool)
    est_rate_limited = np.zeros(count, dtype=bool)
    est_accel_limited = np.zeros(count, dtype=bool)

    for index in range(count - 1):
        now = float(times[index])
        while event_index < len(events) and events[event_index][0] <= now + 1e-12:
            _delivery, sample_time, measurement = events[event_index]
            latest_measurement = measurement
            before = time.perf_counter_ns()
            estimator.update(measurement, sample_time_s=sample_time)
            update_ns.append(time.perf_counter_ns() - before)
            event_index += 1

        if latest_measurement is not None:
            raw_command[index], raw_rate_limited[index], raw_accel_limited[index] = _controller_step(
                raw_state,
                gains,
                error=latest_measurement - raw_theta[index],
                omega=raw_omega[index],
                feedforward=0.0,
                dt_s=dt_s,
            )
        before = time.perf_counter_ns()
        estimate = estimator.estimate(query_time_s=now)
        predict_ns.append(time.perf_counter_ns() - before)
        if estimate is not None:
            estimated_angle[index] = estimate.angle_rad
            estimated_rate[index] = estimate.rate_rad_s
            est_command[index], est_rate_limited[index], est_accel_limited[index] = _controller_step(
                est_state,
                gains,
                error=estimate.angle_rad - est_theta[index],
                omega=est_omega[index],
                feedforward=feedforward_gain * estimate.rate_rad_s,
                dt_s=dt_s,
            )

        raw_theta[index + 1], raw_omega[index + 1] = plant.advance(
            raw_theta[index], raw_omega[index], raw_command[index], dt_s
        )
        est_theta[index + 1], est_omega[index + 1] = plant.advance(
            est_theta[index], est_omega[index], est_command[index], dt_s
        )

    raw_command[-1] = raw_command[-2]
    est_command[-1] = est_command[-2]
    valid_estimates = np.flatnonzero(np.isfinite(estimated_angle))
    if valid_estimates.size:
        estimated_angle[-1] = estimated_angle[valid_estimates[-1]]
        estimated_rate[-1] = estimated_rate[valid_estimates[-1]]
    evaluation = times >= max(0.5, latency_s + 2.0 / vision_hz)
    return {
        "time_s": times,
        "target_rad": target,
        "raw_theta_rad": raw_theta,
        "raw_command_rad_s": raw_command,
        "estimated_theta_rad": est_theta,
        "estimated_command_rad_s": est_command,
        "target_estimate_rad": estimated_angle,
        "target_rate_estimate_rad_s": estimated_rate,
        "raw_metrics": _tracking_metrics(target[evaluation], raw_theta[evaluation], raw_command[evaluation], raw_rate_limited[evaluation], raw_accel_limited[evaluation]),
        "estimated_metrics": _tracking_metrics(target[evaluation], est_theta[evaluation], est_command[evaluation], est_rate_limited[evaluation], est_accel_limited[evaluation]),
        "estimator": {
            "updates": len(update_ns),
            "accepted_updates": 0 if estimate is None else estimate.accepted_updates,
            "rejected_updates": 0 if estimate is None else estimate.rejected_updates,
            "reinitialized_updates": 0 if estimate is None else estimate.reinitialized_updates,
            "update_p95_us": float(np.percentile(update_ns, 95) / 1000.0) if update_ns else None,
            "predict_p95_us": float(np.percentile(predict_ns, 95) / 1000.0) if predict_ns else None,
        },
    }


def _tracking_metrics(target: np.ndarray, theta: np.ndarray, command: np.ndarray, rate_limited: np.ndarray, accel_limited: np.ndarray) -> dict[str, float]:
    error = target - theta
    return {
        "rms_error_rad": float(np.sqrt(np.mean(error * error))),
        "p95_abs_error_rad": float(np.percentile(np.abs(error), 95)),
        "max_abs_error_rad": float(np.max(np.abs(error))),
        "command_rms_rad_s": float(np.sqrt(np.mean(command * command))),
        "command_total_variation_rad_s": float(np.sum(np.abs(np.diff(command)))),
        "rate_limited_fraction": float(np.mean(rate_limited)),
        "accel_limited_fraction": float(np.mean(accel_limited)),
    }


def _suite(
    plant: AxisPlant,
    gains: PIDGains,
    scenarios: Sequence[TrackingScenario],
    config: LOSKalmanConfig,
    feedforward_gain: float,
    *,
    seed_base: int,
) -> dict[str, Any]:
    results = {
        scenario.name: simulate_comparison(
            plant,
            gains,
            scenario,
            config,
            feedforward_gain=feedforward_gain,
            seed=seed_base + index * 1009,
        )
        for index, scenario in enumerate(scenarios)
    }
    estimated = [result["estimated_metrics"] for result in results.values()]
    score = float(np.mean([item["rms_error_rad"] for item in estimated])) + 0.2 * float(
        np.mean([item["p95_abs_error_rad"] for item in estimated])
    ) + 0.003 * float(np.mean([item["command_total_variation_rad_s"] for item in estimated]))
    return {"score": score, "scenarios": results}


def _qualification(suite: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    raw_rms = float(np.mean([result["raw_metrics"]["rms_error_rad"] for result in suite["scenarios"].values()]))
    estimated_rms = float(np.mean([result["estimated_metrics"]["rms_error_rad"] for result in suite["scenarios"].values()]))
    if estimated_rms > 0.90 * raw_rms:
        failures.append("aggregate_rms_improvement")
    for name, result in suite["scenarios"].items():
        raw = result["raw_metrics"]
        estimated = result["estimated_metrics"]
        if estimated["rms_error_rad"] > 1.10 * raw["rms_error_rad"]:
            failures.append(f"{name}:rms_regression")
        if estimated["command_total_variation_rad_s"] > 1.50 * raw["command_total_variation_rad_s"]:
            failures.append(f"{name}:command_variation")
        timing = result["estimator"]
        if timing["update_p95_us"] is None or timing["update_p95_us"] > 1000.0:
            failures.append(f"{name}:update_latency")
        if timing["predict_p95_us"] is None or timing["predict_p95_us"] > 1000.0:
            failures.append(f"{name}:predict_latency")
        if timing["updates"] and timing["rejected_updates"] / timing["updates"] > 0.25:
            failures.append(f"{name}:innovation_rejections")
    return {
        "qualified": not failures,
        "failures": failures,
        "raw_mean_rms_error_rad": raw_rms,
        "estimated_mean_rms_error_rad": estimated_rms,
        "relative_rms_change": estimated_rms / raw_rms - 1.0,
    }


def _load_pid_gains(path: Path) -> dict[str, PIDGains]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("qualification", {}).get("qualified") is not True:
        raise ValueError("PID baseline is not qualified")
    return {axis: PIDGains(**values["selected_gains"]) for axis, values in report["axes"].items()}


def _write_trace(path: Path, axis: str, scenario: str, result: Mapping[str, Any]) -> None:
    vectors = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["axis", "scenario", *vectors.keys()]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index in range(len(result["time_s"])):
            writer.writerow({"axis": axis, "scenario": scenario, **{key: values[index] for key, values in vectors.items()}})


def _write_plot(path: Path, axis: str, suite: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(suite["scenarios"].items())
    figure, axes = plt.subplots(len(rows), 1, figsize=(10, 3 * len(rows)), constrained_layout=True)
    for plot_axis, (name, result) in zip(np.atleast_1d(axes), rows):
        plot_axis.plot(result["time_s"], result["target_rad"], "k--", label="target")
        plot_axis.plot(result["time_s"], result["raw_theta_rad"], label="raw PID")
        plot_axis.plot(result["time_s"], result["estimated_theta_rad"], label="KF + feedforward")
        plot_axis.set_title(name)
        plot_axis.set_ylabel("angle (rad)")
        plot_axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="best", ncol=3)
    figure.suptitle(f"{axis} LOS estimator/feedforward holdout")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _metrics_only(suite: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "score": suite["score"],
        "scenarios": {
            name: {
                "raw_metrics": result["raw_metrics"],
                "estimated_metrics": result["estimated_metrics"],
                "estimator": result["estimator"],
            }
            for name, result in suite["scenarios"].items()
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", required=True, type=Path)
    parser.add_argument("--plant-validation-report", required=True, type=Path)
    parser.add_argument("--pid-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    plants = load_qualified_plants(args.fit_report, args.plant_validation_report)
    gains = _load_pid_gains(args.pid_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "sources": {
            "fit_report": str(args.fit_report.resolve()),
            "fit_report_sha256": _sha256(args.fit_report),
            "plant_validation_report": str(args.plant_validation_report.resolve()),
            "plant_validation_report_sha256": _sha256(args.plant_validation_report),
            "pid_report": str(args.pid_report.resolve()),
            "pid_report_sha256": _sha256(args.pid_report),
        },
        "measurement_scenario": {"controller_hz": 50.0, "vision_hz": 30.0, "latency_s": 0.05, "noise_std_rad": 0.003, "dropout_fraction": 0.10},
        "acceptance": {"aggregate_rms_improvement_min": 0.10, "per_scenario_rms_regression_max": 0.10, "command_variation_ratio_max": 1.5, "estimator_p95_us_max": 1000.0, "innovation_rejection_fraction_max": 0.25},
        "axes": {},
    }
    all_failures: list[str] = []
    for axis_index, axis in enumerate(("yaw", "pitch")):
        best: tuple[float, LOSKalmanConfig, float, dict[str, Any]] | None = None
        for spectral_density in (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0):
            for feedforward_gain in (0.5, 0.75, 1.0, 1.25):
                config = LOSKalmanConfig(acceleration_spectral_density=spectral_density)
                suite = _suite(plants[axis], gains[axis], tuning_scenarios(), config, feedforward_gain, seed_base=10000 + axis_index * 100000)
                candidate = (float(suite["score"]), config, feedforward_gain, suite)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        assert best is not None
        _score, config, feedforward_gain, tuning = best
        holdout = _suite(plants[axis], gains[axis], holdout_scenarios(), config, feedforward_gain, seed_base=50000 + axis_index * 100000)
        qualification = _qualification(holdout)
        all_failures.extend(f"{axis}:{failure}" for failure in qualification["failures"])
        axis_dir = args.output_dir / axis
        axis_dir.mkdir(parents=True, exist_ok=True)
        for name, result in holdout["scenarios"].items():
            _write_trace(axis_dir / f"{name}.csv", axis, name, result)
        if args.plot:
            _write_plot(axis_dir / "holdout_comparison.png", axis, holdout)
        report["axes"][axis] = {
            "kalman_config": config.__dict__,
            "feedforward_gain": feedforward_gain,
            "pid_gains": gains[axis].__dict__,
            "tuning": _metrics_only(tuning),
            "holdout": _metrics_only(holdout),
            "qualification": qualification,
        }
        print(f"{axis}: q={config.acceleration_spectral_density:g} ff={feedforward_gain:g} rms_change={qualification['relative_rms_change']:+.1%} qualified={qualification['qualified']}")
    report["qualification"] = {"qualified": not all_failures, "failures": all_failures}
    output = args.output_dir / "los_estimator_validation_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0 if not all_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
