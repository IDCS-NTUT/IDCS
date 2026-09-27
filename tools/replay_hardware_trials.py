#!/usr/bin/env python3
"""Check the qualified plant against recorded V3 local-target hardware trials.

Two replays per trial (``local_pid_trial`` ``trial.jsonl``):

* open loop: the recorded per-tick intent rates, F6-quantised as on the
  wire, drive the fitted plant from the first measured angle. Isolates the
  plant and command path from the controller.
* closed loop: the real ``BasicPID`` with the recorded gains tracks the
  recorded reference in simulation (``tools.latency_gain_sweep.simulate``),
  sweeping the feedback latency; the best-matching latency estimates the
  trial's effective loop delay.

All angles are relative to the first measured angle ("home").
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from common.gimbal.gray_box import AxisPlant, load_qualified_plants
from common.gimbal.mks_servo42_rs485 import MksServo42Axis
from tools.latency_gain_sweep import Gains, LatencySpec, LoopConfig, Scenario, simulate


def load_trial(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    meta = next(row for row in rows if row.get("type") == "meta")
    axis = meta.get("axis", "yaw")
    key = "yaw" if axis == "yaw" else "pitch"
    gains = meta.get("active_axis_gains") or meta.get(f"{key}_gains")
    ticks = [row for row in rows if row.get("type") == "tick"
             and row.get(f"measured_{key}_rad") is not None
             and row.get(f"reference_{key}_rad") is not None]
    t = np.array([row["elapsed_s"] for row in ticks])
    measured = np.array([row[f"measured_{key}_rad"] for row in ticks])
    reference = np.array([row[f"reference_{key}_rad"] for row in ticks])
    intent_key = f"{key}_rate_rad_s"
    command = np.array([
        (row.get("intent") or {}).get(intent_key, 0.0)
        if (row.get("intent") or {}).get("reason") == "tracking" else 0.0
        for row in ticks
    ])
    home = measured[0]
    return {
        "name": path.parent.name, "axis": key, "gains": gains,
        "observation_delay_s": meta.get("observation_delay_ms", 0) / 1000.0,
        "rate_limit_rad_s": meta.get("yaw_rate_limit_rad_s", 0.2),
        "accel_limit_rad_s2": meta.get("yaw_acceleration_limit_rad_s2", 3.5),
        "t": t - t[0], "measured": measured - home, "reference": reference - home,
        "command": command,
    }


def open_loop(plant: AxisPlant, trial: dict) -> dict:
    holding = replace(plant, disturbance=0.0)
    t, cmd = trial["t"], trial["command"]
    theta, omega = 0.0, 0.0
    sim = [0.0]
    for i in range(len(t) - 1):
        applied = MksServo42Axis.quantized_speed_rad_s(float(cmd[i]), 1.0)
        active = holding if applied == 0.0 else plant
        theta, omega = active.advance(theta, omega, applied, float(t[i + 1] - t[i]))
        sim.append(theta)
    sim = np.array(sim)
    diff = sim - trial["measured"]
    moving = np.array([MksServo42Axis.quantized_speed_rad_s(float(c), 1.0) != 0.0 for c in cmd])
    travel_hw = float(np.sum(np.abs(np.diff(trial["measured"]))))
    travel_sim = float(np.sum(np.abs(np.diff(sim))))
    return {
        "rms_diff_rad": float(np.sqrt(np.mean(diff ** 2))),
        "final_diff_rad": float(diff[-1]),
        "travel_ratio_sim_over_hw": travel_sim / travel_hw if travel_hw else math.nan,
        "nonzero_wire_command_fraction": float(moving.mean()),
        "series": {"t": t, "sim": sim},
    }


def closed_loop(plant: AxisPlant, trial: dict, extra_delays_s: Sequence[float]) -> dict:
    t, ref, meas = trial["t"], trial["reference"], trial["measured"]
    scenario = Scenario(trial["name"], float(t[-1]), lambda x: float(np.interp(x, t, ref)))
    gains = Gains(trial["gains"]["kp"], trial["gains"].get("ki", 0.0), trial["gains"].get("kd", 0.0))
    loop = LoopConfig(fps=50.0, rate_limit_rad_s=trial["rate_limit_rad_s"],
                      accel_limit_rad_s2=trial["accel_limit_rad_s2"])
    hw_error = ref - meas
    hw_rms = float(np.sqrt(np.mean(hw_error ** 2)))
    rows = []
    for extra in extra_delays_s:
        latency = LatencySpec(trial["observation_delay_s"] + extra, 0.0)
        result = simulate(plant, gains, scenario, latency, loop)
        sim_theta = np.interp(t, result["time_s"], np.interp(result["time_s"], t, ref) - result["error_rad"])
        rows.append({
            "extra_delay_s": extra,
            "trajectory_rms_diff_rad": float(np.sqrt(np.mean((sim_theta - meas) ** 2))),
            "sim_tracking_rms_rad": float(np.sqrt(np.mean((ref - sim_theta) ** 2))),
            "sim_theta": sim_theta,
        })
    best = min(rows, key=lambda row: row["trajectory_rms_diff_rad"])
    return {"hw_tracking_rms_rad": hw_rms, "best": best,
            "by_delay": [{k: v for k, v in row.items() if k != "sim_theta"} for row in rows]}


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    fit_dir = root / "artifacts/gimbal_fit/controller_sysid_pid_range_wire_20260914"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trials", nargs="+", type=Path, help="trial.jsonl files")
    parser.add_argument("--fit-report", type=Path, default=fit_dir / "fit_report.json")
    parser.add_argument("--validation-report", type=Path, default=fit_dir / "independent_validation_report.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path)
    args = parser.parse_args(argv)
    plants = load_qualified_plants(args.fit_report, args.validation_report)
    delays = [i * 0.01 for i in range(0, 13)]
    report = {"format": "idcs.hardware_trial_replay", "version": 1, "trials": {}}
    for path in args.trials:
        trial = load_trial(path)
        plant = plants[trial["axis"]]
        ol = open_loop(plant, trial)
        cl = closed_loop(plant, trial, delays)
        report["trials"][trial["name"]] = {
            "axis": trial["axis"], "gains": trial["gains"],
            "observation_delay_s": trial["observation_delay_s"],
            "open_loop": {k: v for k, v in ol.items() if k != "series"},
            "closed_loop": {"hw_tracking_rms_rad": cl["hw_tracking_rms_rad"],
                            "best_extra_delay_s": cl["best"]["extra_delay_s"],
                            "best_trajectory_rms_diff_rad": cl["best"]["trajectory_rms_diff_rad"],
                            "sim_tracking_rms_at_best_rad": cl["best"]["sim_tracking_rms_rad"],
                            "by_delay": cl["by_delay"]},
        }
        print(f"{trial['name']:28s} {trial['axis']:5s} kp={trial['gains']['kp']:g} "
              f"| open-loop rms diff {ol['rms_diff_rad']*1e3:6.2f} mrad, travel sim/hw {ol['travel_ratio_sim_over_hw']:.2f}, "
              f"wire nonzero {ol['nonzero_wire_command_fraction']:.2f} "
              f"| closed-loop hw rms {cl['hw_tracking_rms_rad']*1e3:6.2f} sim rms {cl['best']['sim_tracking_rms_rad']*1e3:6.2f} mrad "
              f"(traj diff {cl['best']['trajectory_rms_diff_rad']*1e3:5.2f}, extra delay {cl['best']['extra_delay_s']*1e3:.0f} ms)")
        if args.plot_dir:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            args.plot_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(trial["t"], trial["reference"] * 1e3, "k--", lw=1, label="reference")
            ax.plot(trial["t"], trial["measured"] * 1e3, lw=1.5, label="hardware")
            ax.plot(trial["t"], ol["series"]["sim"] * 1e3, lw=1, label="open-loop replay")
            ax.plot(trial["t"], cl["best"]["sim_theta"] * 1e3, lw=1, label=f"closed-loop sim (+{cl['best']['extra_delay_s']*1e3:.0f} ms)")
            ax.set_xlabel("s"); ax.set_ylabel("mrad from home"); ax.set_title(trial["name"])
            ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
            fig.savefig(args.plot_dir / f"{trial['name']}.png", dpi=110); plt.close(fig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
