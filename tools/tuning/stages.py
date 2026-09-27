"""Stages of the tuning procedure (docs/tuning_procedure.md).

Each stage reads the plan and earlier stages' reports from one run directory,
runs its tool, writes its own report, and evaluates a gate. A stage runs only
when the stages it depends on passed; re-running a stage clears everything
downstream of it, so a run directory never mixes results from different
inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

REPO = Path(__file__).resolve().parents[2]

ORDER = ("latency", "sysid", "fit", "limits", "sim", "hardware", "agreement", "emit", "live_ab")
REQUIRES: dict[str, tuple[str, ...]] = {
    "latency": (), "sysid": (), "fit": ("sysid",), "limits": (),
    "sim": ("latency", "fit", "limits"), "hardware": ("sim",), "agreement": ("hardware",),
    "emit": ("agreement",), "live_ab": ("emit",),
}
HARDWARE_STAGES = ("sysid", "limits", "hardware", "live_ab")


class StageError(RuntimeError):
    """The stage could not run (not a failed gate)."""


@dataclass
class Gate:
    failures: list[str] = field(default_factory=list)

    def check(self, ok: bool, message: str) -> None:
        if not ok:
            self.failures.append(message)

    @property
    def passed(self) -> bool:
        return not self.failures


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def jsonl_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.startswith("{")]


class Run:
    """One tuning run directory: plan copy, manifest, per-stage outputs."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.exists():
            raise StageError(f"{self.root} is not a tuning run (run `init` first)")
        self.manifest = read_json(self.manifest_path)
        self.plan = yaml.safe_load((self.root / "plan.yaml").read_text(encoding="utf-8"))["tuning"]

    @staticmethod
    def init(root: Path, plan_path: Path) -> "Run":
        root = Path(root)
        if (root / "manifest.json").exists():
            raise StageError(f"{root} already holds a tuning run")
        plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
        if not isinstance(plan, dict) or "tuning" not in plan:
            raise StageError(f"{plan_path} has no `tuning` section")
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(plan_path, root / "plan.yaml")
        write_json(root / "manifest.json", {
            "format": "idcs.tuning_run", "version": 1, "created_unix_s": time.time(),
            "plan_source": str(plan_path), "plan_sha256": sha256(root / "plan.yaml"),
            "git_head": _git_head(), "stages": {},
        })
        return Run(root)

    def dir(self, stage: str) -> Path:
        path = self.root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def passed(self, stage: str) -> bool:
        return self.manifest["stages"].get(stage, {}).get("passed") is True

    def report(self, stage: str) -> dict:
        return read_json(self.root / stage / "report.json")

    def require(self, stage: str) -> None:
        missing = [dep for dep in REQUIRES[stage] if not self.passed(dep)]
        if missing:
            raise StageError(f"{stage} needs passed stage(s): {', '.join(missing)}")

    def record(self, stage: str, report: dict, gate: Gate) -> dict:
        """Write the stage report, clear downstream stages, update the manifest."""
        report = {**report, "gate": {"passed": gate.passed, "failures": gate.failures}}
        path = write_json(self.dir(stage) / "report.json", report)
        downstream = {name for name in ORDER if _depends_on(name, stage)}
        for name in downstream:
            if name in self.manifest["stages"]:
                del self.manifest["stages"][name]
        self.manifest["stages"][stage] = {
            "passed": gate.passed, "failures": gate.failures, "completed_unix_s": time.time(),
            "report_sha256": sha256(path), "git_head": _git_head(),
        }
        write_json(self.manifest_path, self.manifest)
        return report


def _depends_on(stage: str, upstream: str) -> bool:
    deps = REQUIRES[stage]
    return upstream in deps or any(_depends_on(dep, upstream) for dep in deps)


def _git_head() -> str | None:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_tool(args: list[str], log: Path, *, check: bool = True) -> int:
    """Run a repo tool, teeing its output to ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(args) + "\n")
        handle.flush()
        proc = subprocess.run(args, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if check and proc.returncode != 0:
        raise StageError(f"{args[1:3]} exited {proc.returncode}; see {log}")
    return proc.returncode


def require_exclusive_bus(plan: dict) -> None:
    """Hardware stages own the bus: the port must exist and no runtime may hold it."""
    from tools.runtime_process_guard import find_module_owners

    if not Path(plan["port"]).exists():
        raise StageError(f"serial port {plan['port']} not present (hardware stages run on the Jetson)")
    owners = find_module_owners(["tools.serial_io_service", "jetson.gimbal_bridge",
                                 "jetson.control.video_runtime"])
    if owners:
        raise StageError("stop the gimbal stack first (sudo systemctl stop idcs-hil.target): "
                         + ", ".join(str(owner) for owner in owners))


def home(run: "Run", stage: str) -> dict:
    """Drive the axes to their envelope centres before a hardware stage.

    Relative probes and sweeps accumulate drift on an uncoupled axis until the
    envelope blocks one direction (the first long-step sysid lost every
    positive step that way).
    """
    plan = run.plan
    cfg = plan.get("home") or {}
    log = run.dir(stage) / "home.jsonl"
    run_tool([sys.executable, "jetson/tools/home_axes.py", "--config", plan["config"],
              "--config-extra", plan["config_extra"], "--port", plan["port"],
              "--speed-rpm", str(cfg.get("speed_rpm", 20)), "--acc", str(cfg.get("acc", 2)),
              "--tolerance-counts", str(cfg.get("tolerance_counts", 30)), "--execute"], log, check=False)
    summary = next((e for e in jsonl_events(log) if e.get("event") in ("summary", "abort")), None)
    if summary is None or summary.get("event") == "abort" or not summary.get("homed"):
        raise StageError(f"homing failed before {stage}: {summary}")
    return summary["results"]


# --------------------------------------------------------------- 1 latency
def stage_latency(run: Run, traces: list[Path]) -> dict:
    """Loop latency = capture age at decision (upper bound) on tracking ticks."""
    ages_ms: list[float] = []
    verified = tracking = 0
    for trace in traces:
        for row in jsonl_events(trace):
            if row.get("type") != "tick" or row.get("pid_reason") != "tracking":
                continue
            tracking += 1
            verified += row.get("clock_reason") == "verified_under_configured_policy"
            age = row.get("capture_age_ns")
            if isinstance(age, list) and len(age) == 2:
                ages_ms.append(age[1] / 1e6)
    cfg = run.plan["latency"]
    gate = Gate()
    gate.check(len(ages_ms) >= cfg["min_tracking_ticks"],
               f"{len(ages_ms)} tracking ticks with capture age < {cfg['min_tracking_ticks']}")
    fraction = verified / tracking if tracking else 0.0
    gate.check(fraction >= cfg["min_clock_verified_fraction"],
               f"clock-verified fraction {fraction:.3f} < {cfg['min_clock_verified_fraction']}")
    ages_ms.sort()
    pct = (lambda q: ages_ms[min(len(ages_ms) - 1, int(q * len(ages_ms)))]) if ages_ms else (lambda q: None)
    return run.record("latency", {
        "p50_ms": pct(0.50), "p95_ms": pct(0.95), "max_ms": ages_ms[-1] if ages_ms else None,
        "tracking_ticks": tracking, "clock_verified_fraction": fraction,
        "sources": {str(t): sha256(t) for t in traces},
    }, gate)


# ----------------------------------------------------------------- 2 sysid
def enable_axes(plan: dict) -> None:
    """Energize each plan axis and require its F3 ACK.

    Stopping the gimbal stack de-energizes the motors, and the response sweep
    does not enable them; an unpowered axis would record no motion.
    """
    from common.gimbal.mks_servo42_rs485 import RS485Bus

    with RS485Bus(plan["port"], baudrate=38400, timeout=0.12, max_retries=1) as bus:
        for axis, spec in plan["axes"].items():
            if bus.send_command(int(spec["addr"]), 0xF3, [1], expected_response_len=1) != b"\x01":
                raise StageError(f"{axis} (addr {spec['addr']}) did not acknowledge enable")


def motion_fraction(csv_path: Path, axis: str) -> float:
    """Share of commanded samples where the axis actually moved (>30% of command)."""
    import csv

    moving = commanded = 0
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("axis") != axis:
                continue
            try:
                cmd = abs(float(row["cmd_rate_encoded_rad_s"]))
                omega = abs(float(row["omega_rad_s"]))
            except (KeyError, ValueError):
                continue
            if cmd > 0:
                commanded += 1
                moving += omega > 0.3 * cmd
    return moving / commanded if commanded else 0.0


def stage_sysid(run: Run) -> dict:
    plan = run.plan
    require_exclusive_bus(plan)
    homed = home(run, "sysid")
    enable_axes(plan)
    out = run.dir("sysid")
    common = plan["sysid"]["common"]
    report: dict[str, Any] = {"home": homed}
    gate = Gate()
    for split, seed in (("train", 1), ("validation", 2)):
        cfg = {**common, **plan["sysid"][split]}
        csv, manifest = out / f"{split}.csv", out / f"{split}.json"
        run_tool([sys.executable, "-m", "jetson.tools.gimbal_response_sweep",
                  "--config", plan["config"], "--config-extra", plan["config_extra"], "--axis", "both",
                  "--profile", str(cfg["profile"]), "--rates", ",".join(str(r) for r in cfg["rates"]),
                  "--directions", str(cfg["directions"]), "--repeat", str(cfg["repeat"]),
                  "--sample-hz", str(cfg["sample_hz"]), "--command-refresh-s", str(cfg["command_refresh_s"]),
                  "--command-runtime-ms", str(cfg["command_runtime_ms"]), "--pre-roll-s", str(cfg["pre_roll_s"]),
                  "--step-s", str(cfg["step_s"]), "--post-roll-s", str(cfg["post_roll_s"]),
                  "--rest-s", str(cfg["rest_s"]), "--seed", str(seed),
                  "--operator-note", f"tuning {run.root.name} {split}",
                  "--output", str(csv), "--manifest", str(manifest), "--start-serial-io"],
                 out / f"{split}.log", check=False)
        status = read_json(manifest).get("status") if manifest.exists() else "missing"
        gate.check(status == "complete", f"{split} sweep status {status}")
        report[split] = {"csv": str(csv), "manifest": str(manifest), "status": status,
                         "csv_sha256": sha256(csv) if csv.exists() else None}
        if csv.exists():
            # A model fits a motionless axis perfectly; require real motion.
            report[split]["motion_fraction"] = {}
            for axis in plan["axes"]:
                fraction = motion_fraction(csv, axis)
                report[split]["motion_fraction"][axis] = fraction
                gate.check(fraction >= plan["sysid"].get("min_motion_fraction", 0.5),
                           f"{split} {axis}: moved in only {fraction:.0%} of commanded samples")
    return run.record("sysid", report, gate)


# ------------------------------------------------------------------- 3 fit
def stage_fit(run: Run) -> dict:
    run.require("fit")
    sysid = run.report("sysid")
    out = run.dir("fit")
    fit_report, validation = out / "fit_report.json", out / "validation_report.json"
    run_tool([sys.executable, "tools/fit_gimbal_response.py", sysid["train"]["csv"],
              "--manifest", sysid["train"]["manifest"], "--out-dir", str(out),
              "--report-json", str(fit_report)], out / "fit.log")
    extra: list[str] = []
    for key, value in (run.plan.get("fit") or {}).items():
        extra += [f"--{key.replace('_', '-')}", str(value)]
    run_tool([sys.executable, "tools/validate_gimbal_fit.py", "--fit-report", str(fit_report),
              "--validation-csv", sysid["validation"]["csv"], "--output", str(validation), *extra],
             out / "validate.log", check=False)
    gate = Gate()
    qual = read_json(validation).get("qualification", {}) if validation.exists() else {}
    gate.check(qual.get("qualified") is True, f"fit not qualified: {qual.get('failures') or 'no report'}")
    return run.record("fit", {"fit_report": str(fit_report), "validation_report": str(validation),
                              "fit_report_sha256": sha256(fit_report),
                              "qualification": qual}, gate)


# ---------------------------------------------------------------- 4 limits
def choose_rate_limit(summary: dict, cfg: dict, gear_ratio: float) -> tuple[int | None, float | None, str]:
    """Rate limit ``margin_levels`` below the fastest passing level at the chosen accel byte."""
    from common.gimbal.mks_servo42_rs485 import f6_level_speed_rad_s

    entry = summary.get(str(cfg["accel_byte"]))
    if entry is None:
        return None, None, f"accel byte {cfg['accel_byte']} not probed"
    best = entry.get("max_passing_level")
    if best is None:
        return None, None, "no level passed at the chosen accel byte"
    levels = sorted(cfg["levels"])
    index = levels.index(best) - int(cfg["margin_levels"])
    if index < 0:
        return None, None, f"fastest passing level {best} leaves no margin"
    level = levels[index]
    return level, min(float(cfg["max_rate_rad_s"]), f6_level_speed_rad_s(level, gear_ratio)), "ok"


def stage_limits(run: Run) -> dict:
    plan = run.plan
    require_exclusive_bus(plan)
    homed = home(run, "limits")
    cfg = plan["limits"]
    out = run.dir("limits")
    report: dict[str, Any] = {"axes": {}, "home": homed}
    gate = Gate()
    for axis, spec in plan["axes"].items():
        log = out / f"{axis}.jsonl"
        run_tool([sys.executable, "jetson/tools/limit_probe.py", "--port", plan["port"],
                  "--addr", str(spec["addr"]), "--levels", ",".join(map(str, cfg["levels"])),
                  "--accel-bytes", ",".join(map(str, cfg["accel_bytes"])), "--execute"], log, check=False)
        events = jsonl_events(log)
        summary = next((e for e in events if e.get("event") == "summary"), None)
        if summary is None:
            reason = next((e.get("reason") for e in events if e.get("event") == "abort"), "no summary")
            gate.check(False, f"{axis}: probe failed ({reason})")
            continue
        level, rate, why = choose_rate_limit(summary["by_accel_byte"], cfg, float(plan["gear_ratio"]))
        gate.check(rate is not None, f"{axis}: {why}")
        if rate is not None:
            gate.check(rate >= cfg["min_rate_rad_s"],
                       f"{axis}: rate limit {rate:.3f} < required {cfg['min_rate_rad_s']} rad/s")
        report["axes"][axis] = {"by_accel_byte": summary["by_accel_byte"], "level": level,
                                "rate_limit_rad_s": rate, "accel_byte": cfg["accel_byte"]}
    rates = [a["rate_limit_rad_s"] for a in report["axes"].values() if a["rate_limit_rad_s"] is not None]
    report["rate_limit_rad_s"] = min(rates) if len(rates) == len(plan["axes"]) else None
    return run.record("limits", report, gate)


# ------------------------------------------------------------------- 5 sim
def _sim_setup(run: Run):
    from common.gimbal.gray_box import load_qualified_plants
    from tools.feedforward_sweep import fast_scenarios
    from tools.latency_gain_sweep import LatencySpec, LoopConfig, search_scenarios

    plan = run.plan
    latency = run.report("latency")
    fit = run.report("fit")
    plants = load_qualified_plants(Path(fit["fit_report"]), Path(fit["validation_report"]))
    spec = LatencySpec(base_s=latency["p50_ms"] / 1e3,
                       jitter_s=max(0.002, (latency["p95_ms"] - latency["p50_ms"]) / 1e3))
    loop = LoopConfig(tick_hz=plan["loop"]["tick_hz"], fps=plan["loop"]["fps"],
                      rate_limit_rad_s=run.report("limits")["rate_limit_rad_s"],
                      accel_limit_rad_s2=plan["loop"]["accel_limit_rad_s2"],
                      gear_ratio=float(plan["gear_ratio"]), step_count_angle=True)
    by_name = {s.name: s for s in (*search_scenarios(), *fast_scenarios())}
    unknown = [n for n in plan["scenarios"] if n not in by_name]
    if unknown:
        raise StageError(f"unknown scenarios: {unknown}")
    return plants, spec, loop, [by_name[n] for n in plan["scenarios"]]


def ff_config(values: list[float]):
    from tools.latency_gain_sweep import FeedforwardConfig

    scale, predict, sigma = (float(v) for v in values)
    return None if scale == 0 and predict == 0 else FeedforwardConfig(scale, predict, sigma)


def ff_tag(values: list[float]) -> str:
    return ":".join(f"{float(v):g}" for v in values)


def stage_sim(run: Run) -> dict:
    import numpy as np

    from tools.feedforward_sweep import best_over_kp

    run.require("sim")
    plan = run.plan
    plants, spec, loop, scenarios = _sim_setup(run)
    grid = np.geomspace(plan["sim"]["kp_min"], plan["sim"]["kp_max"], int(plan["sim"]["kp_points"]))
    configs = [list(map(float, c)) for c in plan["feedforward_configs"]]
    if ff_config(configs[0]) is not None:
        raise StageError("the first feedforward config must be PID only (0, 0, sigma)")
    results: dict[str, dict] = {}
    gate = Gate()
    for axis in plan["axes"]:
        results[axis] = {}
        for values in configs:
            best = best_over_kp(plants[axis], scenarios, spec, loop, ff_config(values), grid)
            results[axis][ff_tag(values)] = {"kp": best["kp"], "cost_rad": best["cost"],
                                             "at_grid_edge": best["at_grid_edge"]}
    pid_tag = ff_tag(configs[0])

    def normalized(tag: str) -> float:
        return sum(results[a][tag]["cost_rad"] / results[a][pid_tag]["cost_rad"] for a in plan["axes"])

    chosen = min((ff_tag(c) for c in configs), key=normalized)
    for tag in {pid_tag, chosen}:
        for axis in plan["axes"]:
            gate.check(not results[axis][tag]["at_grid_edge"],
                       f"{axis} {tag}: optimum Kp {results[axis][tag]['kp']:.2f} at the grid edge")
    return run.record("sim", {
        "latency": {"base_s": spec.base_s, "jitter_s": spec.jitter_s},
        "loop": {k: getattr(loop, k) for k in ("tick_hz", "fps", "rate_limit_rad_s", "accel_limit_rad_s2",
                                               "gear_ratio")},
        "scenarios": [s.name for s in scenarios], "results": results,
        "pid_only": pid_tag, "chosen": chosen,
        "chosen_improvement": {a: 1 - results[a][chosen]["cost_rad"] / results[a][pid_tag]["cost_rad"]
                               for a in plan["axes"]},
    }, gate)


# -------------------------------------------------------------- 6 hardware
def stage_hardware(run: Run) -> dict:
    run.require("hardware")
    plan = run.plan
    require_exclusive_bus(plan)
    homed = home(run, "hardware")
    sim = run.report("sim")
    limits = run.report("limits")
    latency_ms = round(run.report("latency")["p50_ms"])
    hw = plan["hardware_sweep"]
    out = run.dir("hardware")
    report: dict[str, Any] = {"latency_ms": latency_ms, "runs": {}, "home": homed}
    gate = Gate()
    for axis, spec in plan["axes"].items():
        for tag in dict.fromkeys((sim["pid_only"], sim["chosen"])):
            kp0 = sim["results"][axis][tag]["kp"]
            kps = [round(kp0 * f, 3) for f in hw["kp_factors"]]
            log = out / f"{axis}_{tag.replace(':', '_')}.jsonl"
            run_tool([sys.executable, "jetson/tools/step_count_gain_sweep.py", "--port", plan["port"],
                      "--addr", str(spec["addr"]), "--motor-sign", str(spec["motor_sign"]),
                      "--step-sign", str(spec["step_sign"]), "--kps", ",".join(map(str, kps)),
                      "--latencies-ms", str(latency_ms), "--repeats", str(hw["repeats"]),
                      "--scenarios", ",".join(plan["scenarios"]), "--ff-configs", tag,
                      "--tick-hz", str(plan["loop"]["tick_hz"]), "--fps", str(plan["loop"]["fps"]),
                      "--rate-limit", str(limits["rate_limit_rad_s"]),
                      "--accel-limit", str(plan["loop"]["accel_limit_rad_s2"]),
                      "--acc", str(plan["limits"]["accel_byte"]), "--gear-ratio", str(plan["gear_ratio"]),
                      "--guard-rad", str(hw["guard_rad"]), "--execute"], log, check=False)
            events = jsonl_events(log)
            runs = [e for e in events if e.get("event") == "run"]
            aborted = [e.get("reason") for e in events if e.get("event") == "abort"]
            expected = len(kps) * int(hw["repeats"])
            gate.check(not aborted and len(runs) == expected,
                       f"{axis} {tag}: {len(runs)}/{expected} runs" + (f", abort: {aborted[0]}" if aborted else ""))
            report["runs"].setdefault(axis, {})[tag] = {"log": str(log), "log_sha256": sha256(log),
                                                        "kps": kps, "completed_runs": len(runs)}
    return run.record("hardware", report, gate)


# ------------------------------------------------------------- 7 agreement
def hardware_costs(log: Path) -> dict[float, float]:
    """Mean cost per Kp over repeats, from a step_count_gain_sweep log."""
    by_kp: dict[float, list[float]] = {}
    for event in jsonl_events(log):
        if event.get("event") == "run":
            by_kp.setdefault(float(event["kp"]), []).append(float(event["cost_rad"]))
    return {kp: statistics.fmean(costs) for kp, costs in sorted(by_kp.items())}


def compare(hw: dict[float, float], sim: dict[float, float]) -> dict:
    """Agreement metrics between matched hardware and simulated cost curves."""
    kps = sorted(set(hw) & set(sim))
    errors = [abs(sim[kp] / hw[kp] - 1.0) for kp in kps]
    hw_best = min(kps, key=lambda kp: hw[kp])
    sim_choice = min(kps, key=lambda kp: sim[kp])
    return {"kps": kps, "hardware_cost_rad": [hw[k] for k in kps], "sim_cost_rad": [sim[k] for k in kps],
            "median_cost_error": statistics.median(errors), "hardware_best_kp": hw_best,
            "sim_choice_kp": sim_choice, "sim_choice_penalty": hw[sim_choice] / hw[hw_best] - 1.0}


def stage_agreement(run: Run, *, simulate_cost: Callable | None = None) -> dict:
    run.require("agreement")
    plan = run.plan
    sim = run.report("sim")
    hardware = run.report("hardware")
    if simulate_cost is None:
        from tools.latency_gain_sweep import Gains, LatencySpec, suite_cost

        plants, _spec, loop, scenarios = _sim_setup(run)
        spec = LatencySpec(base_s=hardware["latency_ms"] / 1e3)  # the sweep injects this latency

        def simulate_cost(axis: str, tag: str, kp: float) -> float:
            return suite_cost(plants[axis], Gains(kp), scenarios, spec, loop,
                              ff_config(tag.split(":")))[0]

    cfg = plan["agreement"]
    gate = Gate()
    comparisons: dict[str, dict] = {}
    for axis, by_tag in hardware["runs"].items():
        comparisons[axis] = {}
        for tag, info in by_tag.items():
            hw = hardware_costs(Path(info["log"]))
            result = compare(hw, {kp: simulate_cost(axis, tag, kp) for kp in hw})
            comparisons[axis][tag] = result
            gate.check(result["median_cost_error"] <= cfg["max_median_cost_error"],
                       f"{axis} {tag}: median sim/hardware cost error {result['median_cost_error']:.2f}")
            gate.check(result["sim_choice_penalty"] <= cfg["max_sim_choice_penalty"],
                       f"{axis} {tag}: sim-chosen Kp costs {result['sim_choice_penalty']:.0%} more on hardware")
    # Hardware is the arbiter: keep the sim-chosen feedforward only if it also
    # beats PID only on hardware on every axis, at each config's hardware best.
    chosen, pid = sim["chosen"], sim["pid_only"]

    def best_cost(axis: str, tag: str) -> float:
        c = comparisons[axis][tag]
        return c["hardware_cost_rad"][c["kps"].index(c["hardware_best_kp"])]

    ff_wins = chosen != pid and all(best_cost(a, chosen) < best_cost(a, pid) for a in comparisons)
    selected = chosen if ff_wins else pid
    return run.record("agreement", {
        "comparisons": comparisons, "selected": selected,
        "kp": {a: comparisons[a][selected]["hardware_best_kp"] for a in comparisons},
        "pid_only_kp": {a: comparisons[a][pid]["hardware_best_kp"] for a in comparisons},
        "feedforward_confirmed_on_hardware": ff_wins,
    }, gate)


# ------------------------------------------------------------------ 8 emit
def tuned_overlay(run: Run) -> dict:
    agreement = run.report("agreement")
    limits = run.report("limits")
    plan = run.plan
    scale, predict, sigma = (float(v) for v in agreement["selected"].split(":"))
    rate = limits["rate_limit_rad_s"]
    accel_byte = int(plan["limits"]["accel_byte"])
    return {
        "controller": {
            "yaw_kp": round(agreement["kp"]["yaw"], 3), "pitch_kp": round(agreement["kp"]["pitch"], 3),
            "feedforward_scale": scale, "predict": predict, "feedforward_accel_sigma_rad_s2": sigma,
            "rate_limit_rad_s": round(rate, 4), "accel_limit_rad_s2": float(plan["loop"]["accel_limit_rad_s2"]),
        },
        "gimbal": {
            "yaw_rate_limit_rad_s": round(rate, 4), "pitch_rate_limit_rad_s": round(rate, 4),
            "yaw_accel_byte": accel_byte, "pitch_accel_byte": accel_byte,
            "yaw_gear_ratio": float(plan["gear_ratio"]), "pitch_gear_ratio": float(plan["gear_ratio"]),
        },
    }


def stage_emit(run: Run) -> dict:
    run.require("emit")
    overlay = tuned_overlay(run)
    out = run.dir("emit")
    stamp = {stage: run.manifest["stages"][stage]["report_sha256"] for stage in ORDER[:7]}
    header = (f"# Generated by tools.tuning from run {run.root.resolve()}; do not edit by hand.\n"
              f"# plan sha256 {run.manifest['plan_sha256']}\n"
              + "".join(f"# {stage} report sha256 {digest}\n" for stage, digest in stamp.items()))
    tuned = out / "tuned_config.yaml"
    tuned.write_text(header + yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    pid_variant = out / "pid_only_overlay.yaml"
    pid_kp = run.report("agreement")["pid_only_kp"]
    pid_variant.write_text("# Live A/B baseline: PID only at its own hardware optimum.\n" + yaml.safe_dump({
        "controller": {"yaw_kp": round(pid_kp["yaw"], 3), "pitch_kp": round(pid_kp["pitch"], 3),
                       "feedforward_scale": 0.0, "predict": 0.0}}, sort_keys=False), encoding="utf-8")
    gate = Gate()
    try:
        from common.config import load_config_bundle, resolve_config_paths
        from jetson.control.runtime_config import ControlRuntimeConfig

        paths = resolve_config_paths(run.plan["config"], ",".join([run.plan["config_extra"], str(tuned)]))
        ControlRuntimeConfig.from_config(load_config_bundle(paths).mutable_copy())
    except (ValueError, KeyError, TypeError) as exc:
        gate.check(False, f"emitted config does not validate: {exc}")
    return run.record("emit", {"tuned_config": str(tuned), "tuned_config_sha256": sha256(tuned),
                               "pid_only_overlay": str(pid_variant), "overlay": overlay}, gate)


# --------------------------------------------------------------- 9 live A/B
def score_live_ab(results: list[dict], overlay: dict, cfg: dict, *, feedforward: bool) -> tuple[dict, Gate]:
    """Gate analyzed live trials: the tuned config must meet the RMS bounds and,
    when it uses feedforward, beat PID only on both axes."""
    gate = Gate()
    tuned = [r for r in results if abs(r["yaw_kp"] - overlay["controller"]["yaw_kp"]) < 1e-6
             and r["feedforward_scale"] == overlay["controller"]["feedforward_scale"]]
    baseline = [r for r in results if r["feedforward_scale"] == 0 and r not in tuned]
    pairs = min(len(tuned), len(baseline)) if feedforward else len(tuned)
    gate.check(pairs >= cfg["min_pairs"], f"{pairs} usable trial pairs < {cfg['min_pairs']}")
    summary: dict[str, Any] = {"tuned_trials": len(tuned), "baseline_trials": len(baseline)}
    for axis in ("yaw", "pitch"):
        key = f"{axis}_capture_rms_rad"
        tuned_rms = [r[key] * 1e3 for r in tuned]
        summary[f"{axis}_tuned_rms_mrad"] = tuned_rms
        if tuned_rms:
            gate.check(max(tuned_rms) <= cfg["max_capture_rms_mrad"][axis],
                       f"{axis}: tuned RMS {max(tuned_rms):.1f} mrad > {cfg['max_capture_rms_mrad'][axis]}")
        if feedforward:
            base_rms = [r[key] * 1e3 for r in baseline]
            summary[f"{axis}_baseline_rms_mrad"] = base_rms
            if tuned_rms and base_rms:
                gate.check(statistics.fmean(tuned_rms) < statistics.fmean(base_rms),
                           f"{axis}: feedforward does not beat PID only live")
    return summary, gate


def stage_live_ab(run: Run, streamer_check: Path, *, duration_s: int = 30,
                  controller_unit: str = "idcs-controller") -> dict:
    """ABBA live trials against the running HIL streamer, scored by the HIL analyzer."""
    from tools.analyze_video_hil import analyze_trial

    run.require("live_ab")
    plan = run.plan
    emit = run.report("emit")
    out = run.dir("live_ab")
    feedforward = emit["overlay"]["controller"]["feedforward_scale"] > 0
    variants = {"tuned": [emit["tuned_config"]],
                "baseline": [emit["tuned_config"], emit["pid_only_overlay"]]}
    order = ("baseline", "tuned", "tuned", "baseline") if feedforward else ("tuned", "tuned")
    subprocess.run(["sudo", "systemctl", "stop", controller_unit], check=True)
    results = []
    try:
        for index, name in enumerate(order):
            host, jetson = out / f"{index}_{name}" / "host", out / f"{index}_{name}" / "jetson"
            host.mkdir(parents=True, exist_ok=True)
            jetson.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(streamer_check, host / "streamer-check.json")
            extra = ",".join([plan["config_extra"], "configs/controller_sim_hil.yaml", *variants[name]])
            run_tool([sys.executable, "-m", "jetson.control.video_runtime", "--config", plan["config"],
                      "--config-extra", extra, "--duration-s", str(duration_s),
                      "--trace", str(jetson / "trace.jsonl"), "--report", str(jetson / "report.json")],
                     jetson / "controller.log")
            try:
                results.append({"trial": f"{index}_{name}", **analyze_trial(host, jetson)})
            except ValueError as exc:
                results.append({"trial": f"{index}_{name}", "rejected": str(exc)})
    finally:
        subprocess.run(["sudo", "systemctl", "start", controller_unit], check=False)
    scored = [r for r in results if "rejected" not in r]
    summary, gate = score_live_ab(scored, emit["overlay"], plan["live_ab"], feedforward=feedforward)
    for r in results:
        gate.check("rejected" not in r, f"trial {r['trial']} rejected: {r.get('rejected')}")
    report = run.record("live_ab", {"streamer_check_sha256": sha256(streamer_check), "trials": results,
                                    "summary": summary}, gate)
    if gate.passed:
        shutil.copyfile(emit["tuned_config"], run.root / "qualified_config.yaml")
    return report
