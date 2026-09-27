#!/usr/bin/env python3
"""Latency-aware PID gain sweep on the qualified gimbal plant.

One axis at a time. A camera captures frames at ``fps``; each frame's bearing
error (target minus gimbal angle at capture) reaches the controller only
after a sampled latency. The real V3 ``BasicPID`` runs at ``tick_hz`` on the
newest available frame, with derivative on the fresh encoder-derived gimbal
rate, as in V3. Its output passes the trial rate and slew limits, then the
F6 integer-RPM encoding (``MksServo42Axis.quantized_speed_rad_s``), then the
qualified fitted plant, which is solved exactly between events.

The score is true pointing error, target(t) - gimbal(t), sampled on a fixed
grid: the quantity a synthetic-target hardware trial compares against the
motor's step position. Gains are searched per latency level: a dense
log-spaced P-only sweep, then a bounded Nelder-Mead refinement of (Kp, Ki,
Kd). Flatness is reported as the Kp band whose cost is within 5% of the
optimum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from common.gimbal.gray_box import AxisPlant, load_qualified_plants
from common.gimbal.mks_servo42_rs485 import MksServo42Axis
from jetson.control_v3.pid import AxisPIDConfig, BasicPID, PIDInput
from jetson.control_v3.timing import TimingVerdict

REPORT_FORMAT = "idcs.latency_gain_sweep"
REPORT_VERSION = 1
COUNTS_PER_REV = 16384
_VALID = TimingVerdict(True, "ok")


@dataclass(frozen=True)
class LoopConfig:
    tick_hz: float = 50.0
    fps: float = 60.0
    rate_limit_rad_s: float = 0.2
    accel_limit_rad_s2: float = 3.5
    integral_limit_rad_s: float = 0.2
    gear_ratio: float = 1.0
    quantize_f6: bool = True
    encoder_quantize: bool = True
    score_hz: float = 100.0


@dataclass(frozen=True)
class LatencySpec:
    base_s: float
    jitter_s: float = 0.002
    seed: int = 7

    def sampler(self) -> Callable[[], float]:
        rng = random.Random(self.seed)
        return lambda: self.base_s + rng.uniform(0.0, self.jitter_s)


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float
    target: Callable[[float], float]
    step_at_s: float | None = None
    step_to: float | None = None


def _step(name: str, amplitude: float, at_s: float = 0.5, duration_s: float = 4.0) -> Scenario:
    return Scenario(name, duration_s, lambda t: amplitude if t >= at_s else 0.0, at_s, amplitude)


def _ramp(name: str, rate: float, start_s: float, stop_s: float, duration_s: float) -> Scenario:
    def target(t: float) -> float:
        return rate * (min(max(t, start_s), stop_s) - start_s)
    return Scenario(name, duration_s, target)


def _sine(name: str, amplitude: float, period_s: float, duration_s: float) -> Scenario:
    return Scenario(name, duration_s, lambda t: amplitude * math.sin(2.0 * math.pi * t / period_s))


def search_scenarios() -> tuple[Scenario, ...]:
    return (
        _step("step_pos_0p06", 0.06),
        _step("step_neg_0p06", -0.06),
        _ramp("ramp_0p05", 0.05, 0.5, 2.5, 4.0),
        _sine("sine_0p06_3s", 0.06, 3.0, 6.0),
        _sine("sine_0p03_1p5s", 0.03, 1.5, 4.5),
    )


def holdout_scenarios() -> tuple[Scenario, ...]:
    def piecewise(t: float) -> float:
        return float(np.interp(t, [0, 1, 2.5, 4, 5.5, 7], [0, 0.03, -0.09, 0.02, 0.06, 0]))

    def mixed(t: float) -> float:
        return 0.03 * math.sin(2 * math.pi * t / 2.3 + 0.4) + 0.02 * math.sin(2 * math.pi * t / 0.9 + 1.7)

    return (
        _step("holdout_step_pos_0p04", 0.04, at_s=0.4),
        _step("holdout_step_neg_0p08", -0.08, at_s=0.4),
        Scenario("holdout_piecewise_velocity", 7.0, piecewise),
        _sine("holdout_sine_0p05_4s", 0.05, 4.0, 8.0),
        Scenario("holdout_mixed_sines", 7.0, mixed),
    )


@dataclass(frozen=True)
class Gains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0


def _quantize_counts(theta: float) -> float:
    step = 2.0 * math.pi / COUNTS_PER_REV
    return round(theta / step) * step


def simulate(
    plant: AxisPlant,
    gains: Gains,
    scenario: Scenario,
    latency: LatencySpec,
    loop: LoopConfig = LoopConfig(),
) -> dict:
    """Event-driven closed loop; returns time series and metrics."""

    axis_cfg = AxisPIDConfig(
        kp=gains.kp, ki=gains.ki, kd=gains.kd,
        integral_limit_rad_s=loop.integral_limit_rad_s,
        rate_limit_rad_s=loop.rate_limit_rad_s,
        acceleration_limit_rad_s2=loop.accel_limit_rad_s2,
    )
    idle = AxisPIDConfig(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    pid = BasicPID(axis_cfg, idle)
    sample_latency = latency.sampler()
    tick = 1.0 / loop.tick_hz
    frame = 1.0 / loop.fps
    score = 1.0 / loop.score_hz

    events: list[tuple[float, int]] = []  # (time, kind): 0 capture, 1 tick, 2 score
    n = 0
    while n * frame <= scenario.duration_s:
        events.append((n * frame, 0)); n += 1
    n = 0
    while n * tick <= scenario.duration_s:
        events.append((n * tick, 1)); n += 1
    n = 0
    while n * score <= scenario.duration_s:
        events.append((n * score, 2)); n += 1
    events.sort()

    # A stepper commanded to 0 RPM holds position; the fitted bias term only
    # describes moving commands, so it is switched off while holding.
    holding_plant = replace(plant, disturbance=0.0)
    theta = omega = 0.0
    now = 0.0
    applied = 0.0
    pending: list[tuple[float, float]] = []  # (available_at, bearing_error)
    latest_error: float | None = None
    prev_meas: tuple[float, float] | None = None
    t_score, err_score, cmd_ticks = [], [], []

    for when, kind in events:
        if when > now:
            active = holding_plant if applied == 0.0 else plant
            theta, omega = active.advance(theta, omega, applied, when - now)
            now = when
        if kind == 0:
            pending.append((when + sample_latency(), scenario.target(when) - theta))
        elif kind == 1:
            while pending and pending[0][0] <= now:
                latest_error = pending.pop(0)[1]
            meas = _quantize_counts(theta) if loop.encoder_quantize else theta
            rate = 0.0 if prev_meas is None else (meas - prev_meas[1]) / (now - prev_meas[0])
            prev_meas = (now, meas)
            if latest_error is None:
                command = 0.0
            else:
                decision = pid.decide(PIDInput(
                    decision_ns=int(round(now * 1e9)) + 1,
                    track_id=1,
                    error_rad=(latest_error, 0.0),
                    gimbal_rate_rad_s=(rate, 0.0),
                    timing=_VALID,
                    safety_allowed=True,
                    gimbal_valid=True,
                ))
                command = decision.yaw.final_rad_s
            applied = (MksServo42Axis.quantized_speed_rad_s(command, loop.gear_ratio)
                       if loop.quantize_f6 else command)
            cmd_ticks.append(applied)
        else:
            t_score.append(when)
            err_score.append(scenario.target(when) - theta)

    return {"time_s": np.asarray(t_score), "error_rad": np.asarray(err_score),
            "commands": np.asarray(cmd_ticks), "metrics": _metrics(scenario, t_score, err_score, cmd_ticks)}


def _metrics(scenario: Scenario, t: Sequence[float], err: Sequence[float], cmds: Sequence[float]) -> dict:
    t = np.asarray(t); err = np.asarray(err); cmds = np.asarray(cmds)
    abs_err = np.abs(err)
    out = {
        "rms_error_rad": float(np.sqrt(np.mean(err ** 2))),
        "p95_abs_error_rad": float(np.percentile(abs_err, 95)),
        "max_abs_error_rad": float(abs_err.max()),
        "final_window_mean_abs_error_rad": float(abs_err[t >= scenario.duration_s - 1.0].mean()),
        "command_total_variation_rad_s": float(np.abs(np.diff(cmds)).sum()) if len(cmds) > 1 else 0.0,
    }
    if scenario.step_at_s is not None and scenario.step_to is not None:
        after = t >= scenario.step_at_s
        position = scenario.step_to - err[after]
        out["overshoot_rad"] = float(max(0.0, np.max(np.sign(scenario.step_to) * position) - abs(scenario.step_to)))
    return out


def suite_cost(plant: AxisPlant, gains: Gains, scenarios: Sequence[Scenario],
               latency: LatencySpec, loop: LoopConfig) -> tuple[float, dict]:
    """Objective: mean true-pointing RMS error across scenarios."""
    per = {s.name: simulate(plant, gains, s, latency, loop)["metrics"] for s in scenarios}
    return float(np.mean([m["rms_error_rad"] for m in per.values()])), per


def optimize_axis(plant: AxisPlant, latency: LatencySpec, loop: LoopConfig,
                  kp_grid: Sequence[float]) -> dict:
    from scipy.optimize import minimize

    scenarios = search_scenarios()
    p_sweep = []
    for kp in kp_grid:
        cost, _ = suite_cost(plant, Gains(kp), scenarios, latency, loop)
        p_sweep.append({"kp": float(kp), "cost": cost})
    best_p = min(p_sweep, key=lambda row: row["cost"])
    band = [row["kp"] for row in p_sweep if row["cost"] <= 1.05 * best_p["cost"]]

    def objective(x: np.ndarray) -> float:
        gains = Gains(abs(float(x[0])), abs(float(x[1])), abs(float(x[2])))
        return suite_cost(plant, gains, scenarios, latency, loop)[0]

    start = np.array([best_p["kp"], 0.05 * best_p["kp"], 0.01])
    result = minimize(objective, start, method="Nelder-Mead",
                      options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 400, "initial_simplex": np.array([
                          start, start + [0.3 * best_p["kp"], 0, 0],
                          start + [0, 0.5, 0], start + [0, 0, 0.05]])})
    pid = Gains(*(abs(float(v)) for v in result.x))
    pid_cost, _ = suite_cost(plant, pid, scenarios, latency, loop)
    if pid_cost > best_p["cost"]:
        pid, pid_cost = Gains(best_p["kp"]), best_p["cost"]
    holdout_cost, holdout = suite_cost(plant, pid, holdout_scenarios(), latency, loop)
    holdout_p_cost, _ = suite_cost(plant, Gains(best_p["kp"]), holdout_scenarios(), latency, loop)
    return {
        "p_only_sweep": p_sweep,
        "p_only_optimum": best_p,
        "p_only_within_5pct_kp_band": [min(band), max(band)],
        "pid_optimum": {**asdict(pid), "cost": pid_cost, "nelder_mead_iterations": int(result.nit)},
        "holdout_cost_pid": holdout_cost,
        "holdout_cost_p_only": holdout_p_cost,
        "holdout_metrics_pid": holdout,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    fit_dir = root / "artifacts/gimbal_fit/controller_sysid_pid_range_wire_20260914"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit-report", type=Path, default=fit_dir / "fit_report.json")
    parser.add_argument("--validation-report", type=Path, default=fit_dir / "independent_validation_report.json")
    parser.add_argument("--latencies-ms", default="0,30,60,120,200")
    parser.add_argument("--jitter-ms", type=float, default=2.0)
    parser.add_argument("--axes", default="yaw,pitch")
    parser.add_argument("--compare", default="yaw:8,pitch:4", help="current gains to score, axis:kp")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    plants = load_qualified_plants(args.fit_report, args.validation_report)
    loop = LoopConfig(quantize_f6=not args.no_quantize)
    kp_grid = np.geomspace(0.25, 40.0, 48)
    compare = dict(item.split(":") for item in args.compare.split(",") if item)
    report = {
        "format": REPORT_FORMAT, "version": REPORT_VERSION,
        "objective": "mean true-pointing RMS error over search scenarios",
        "loop": asdict(loop),
        "sources": {
            "fit_report": str(args.fit_report), "fit_report_sha256": _sha256(args.fit_report),
            "validation_report": str(args.validation_report),
            "validation_report_sha256": _sha256(args.validation_report),
            "sweep_tool_sha256": _sha256(Path(__file__)),
        },
        "search_scenarios": [s.name for s in search_scenarios()],
        "holdout_scenarios": [s.name for s in holdout_scenarios()],
        "axes": {},
    }
    for axis in args.axes.split(","):
        plant = plants[axis]
        report["axes"][axis] = {}
        for ms in (float(v) for v in args.latencies_ms.split(",")):
            latency = LatencySpec(base_s=ms / 1000.0, jitter_s=args.jitter_ms / 1000.0)
            result = optimize_axis(plant, latency, loop, kp_grid)
            if axis in compare:
                result["current_kp"] = float(compare[axis])
                result["current_cost"] = suite_cost(
                    plant, Gains(float(compare[axis])), search_scenarios(), latency, loop)[0]
            result["latency"] = asdict(latency)
            report["axes"][axis][f"{ms:g}ms"] = result
            p, g = result["p_only_optimum"], result["pid_optimum"]
            print(f"{axis} L={ms:g}ms  P-only kp*={p['kp']:.3g} cost={p['cost']*1e3:.3f}mrad "
                  f"band={result['p_only_within_5pct_kp_band'][0]:.3g}-{result['p_only_within_5pct_kp_band'][1]:.3g}  "
                  f"PID kp={g['kp']:.3g} ki={g['ki']:.3g} kd={g['kd']:.3g} cost={g['cost']*1e3:.3f}mrad  "
                  f"holdout={result['holdout_cost_pid']*1e3:.3f}mrad"
                  + (f"  current kp={result['current_kp']:g} cost={result['current_cost']*1e3:.3f}mrad"
                     if "current_cost" in result else ""), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
