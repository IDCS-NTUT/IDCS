#!/usr/bin/env python3
"""Feedforward / latency-prediction sweep on the qualified plant.

For each latency level and actuator model (measured F6 table or ideal
fine-resolution rate), every feedforward configuration (rate scale,
prediction fraction, Kalman acceleration noise) gets its own dense P-only Kp
sweep, so configurations are compared at their own best gain. Scenarios are
fast moving targets (0.3-0.6 rad/s ramps, 1-2 s sines, a constant-velocity
back-and-forth) with a 1 rad/s rate cap; the earlier slow scenarios are
repeated at one latency for contrast.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from common.gimbal.gray_box import load_qualified_plants
from tools.latency_gain_sweep import (
    FeedforwardConfig,
    Gains,
    LatencySpec,
    LoopConfig,
    Scenario,
    search_scenarios,
    suite_cost,
)


BLEND_S = 0.2  # velocity changes are cosine-blended over this long (finite acceleration)


def _velocity_profile(name: str, knots: Sequence[tuple[float, float]], duration_s: float) -> Scenario:
    """Position from a piecewise-constant velocity with smooth transitions.

    ``knots`` are (time, velocity) switch points; each change ramps over
    ``BLEND_S`` with a raised-cosine profile, so acceleration stays finite
    (peak pi/2 * dv / BLEND_S).
    """

    dt = 0.001
    t_grid = np.arange(0.0, duration_s + dt, dt)
    v = np.zeros_like(t_grid)
    previous = 0.0
    for when, velocity in knots:
        blend = np.clip((t_grid - when) / BLEND_S, 0.0, 1.0)
        v += (velocity - previous) * (0.5 - 0.5 * np.cos(np.pi * blend))
        previous = velocity
    position = np.concatenate(([0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * dt)))
    return Scenario(name, duration_s, lambda t: float(np.interp(t, t_grid, position)))


def _ramp_hold(name: str, rate: float, start_s: float, stop_s: float, duration_s: float) -> Scenario:
    return _velocity_profile(name, [(start_s, rate), (stop_s, 0.0)], duration_s)


def _triangle(name: str, speed: float, half_period_s: float, duration_s: float) -> Scenario:
    knots = [(0.3, speed)]
    t = 0.3 + half_period_s
    sign = -1.0
    while t < duration_s:
        knots.append((t, sign * speed))
        sign = -sign
        t += 2 * half_period_s if len(knots) > 1 else half_period_s
    return _velocity_profile(name, knots, duration_s)


def fast_scenarios() -> tuple[Scenario, ...]:
    return (
        _ramp_hold("ramp_0p3", 0.3, 0.5, 1.5, 3.5),
        _ramp_hold("ramp_0p6", 0.6, 0.5, 1.0, 3.0),
        Scenario("sine_0p2_2s", 6.0, lambda t: 0.2 * math.sin(2 * math.pi * t / 2.0)),
        Scenario("sine_0p1_1s", 5.0, lambda t: 0.1 * math.sin(2 * math.pi * t / 1.0)),
        _triangle("triangle_0p4", 0.4, 1.0, 6.0),
    )


def configs() -> list[FeedforwardConfig]:
    out = [FeedforwardConfig()]
    for scale, predict, sigma in itertools.product((0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.4, 2.0, 8.0)):
        if scale == 0.0 and predict == 0.0:
            continue
        out.append(FeedforwardConfig(rate_scale=scale, predict=predict, accel_sigma_rad_s2=sigma))
    return out


def best_over_kp(plant, scenarios, latency, loop, ff, kp_grid) -> dict:
    rows = [(float(kp), suite_cost(plant, Gains(float(kp)), scenarios, latency, loop, ff)[0]) for kp in kp_grid]
    kp, cost = min(rows, key=lambda row: row[1])
    return {"kp": kp, "cost": cost, "at_grid_edge": kp in (kp_grid[0], kp_grid[-1]),
            "curve": rows}


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    fit_dir = root / "artifacts/gimbal_fit/controller_sysid_pid_range_wire_20260914"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axis", default="yaw")
    parser.add_argument("--latencies-ms", default="30,60,120")
    parser.add_argument("--rate-limit", type=float, default=1.0)
    parser.add_argument("--accel-limit", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plant = load_qualified_plants(fit_dir / "fit_report.json", fit_dir / "independent_validation_report.json")[args.axis]
    kp_grid = [float(v) for v in np.geomspace(1.0, 40.0, 16)]
    actuators = {
        "f6_measured": dict(quantize_f6=True),
        "ideal": dict(quantize_f6=False),
    }
    report = {"format": "idcs.feedforward_sweep", "version": 1, "axis": args.axis,
              "rate_limit_rad_s": args.rate_limit, "accel_limit_rad_s2": args.accel_limit,
              "blend_s": BLEND_S, "kp_grid": kp_grid,
              "fast_scenarios": [s.name for s in fast_scenarios()], "results": []}
    for ms in (float(v) for v in args.latencies_ms.split(",")):
        latency = LatencySpec(ms / 1000.0)
        for actuator, kwargs in actuators.items():
            loop = LoopConfig(fps=60.0, rate_limit_rad_s=args.rate_limit, accel_limit_rad_s2=args.accel_limit,
                              step_count_angle=True, **kwargs)
            baseline = None
            for ff in configs():
                best = best_over_kp(plant, fast_scenarios(), latency, loop, ff, kp_grid)
                baseline = baseline or best["cost"]
                report["results"].append({"latency_ms": ms, "actuator": actuator, "feedforward": asdict(ff),
                                          **{k: v for k, v in best.items() if k != "curve"},
                                          "improvement_vs_pid_only": 1.0 - best["cost"] / baseline,
                                          "curve": best["curve"]})
            rows = [r for r in report["results"] if r["latency_ms"] == ms and r["actuator"] == actuator]
            top = min(rows, key=lambda r: r["cost"])
            f = top["feedforward"]
            print(f"L={ms:g}ms {actuator:11s} PID-only {rows[0]['cost']*1e3:6.2f} mrad (kp {rows[0]['kp']:.3g})"
                  f" | best FF scale {f['rate_scale']:g} predict {f['predict']:g} sigma {f['accel_sigma_rad_s2']:g}:"
                  f" {top['cost']*1e3:6.2f} mrad (kp {top['kp']:.3g}{', edge' if top['at_grid_edge'] else ''})"
                  f"  -> {100*top['improvement_vs_pid_only']:.0f}% better", flush=True)
    # Contrast: the earlier slow scenarios at 60 ms. The trials' 0.2 rad/s cap is
    # below the slowest real F6 speed, so use the lowest reachable cap instead.
    slow_loop = LoopConfig(fps=60.0, rate_limit_rad_s=0.25, step_count_angle=True)
    slow = {}
    for ff in (FeedforwardConfig(), FeedforwardConfig(rate_scale=1.0, predict=1.0, accel_sigma_rad_s2=2.0)):
        slow[f"scale{ff.rate_scale:g}_predict{ff.predict:g}"] = best_over_kp(
            plant, search_scenarios(), LatencySpec(0.06), slow_loop, ff, kp_grid)
    report["slow_contrast_60ms_f6"] = {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in slow.items()}
    a, b = (slow[k]["cost"] for k in slow)
    print(f"slow targets, 60 ms, F6, 0.2 cap: PID-only {a*1e3:.2f} mrad vs full FF+prediction {b*1e3:.2f} mrad"
          f" -> {100*(1-b/a):.0f}% better")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
