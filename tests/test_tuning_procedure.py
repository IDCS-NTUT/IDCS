from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.tuning import stages
from tools.tuning.stages import Gate, Run, StageError

REPO = Path(__file__).resolve().parents[1]
FIT_DIR = REPO / "artifacts/gimbal_fit/controller_sysid_pid_range_wire_20260914"


def _run(tmp_path: Path, **plan_overrides) -> Run:
    plan = yaml.safe_load((REPO / "configs/tuning/plan.yaml").read_text())
    plan["tuning"].update(plan_overrides)
    plan["tuning"]["config"] = str(REPO / plan["tuning"]["config"])
    plan["tuning"]["config_extra"] = ",".join(str(REPO / p) for p in plan["tuning"]["config_extra"].split(","))
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    return Run.init(tmp_path / "run", path)


def _trace(path: Path, ticks: int, age_ms: float, verified: bool = True) -> Path:
    rows = [{"type": "meta"}] + [
        {"type": "tick", "pid_reason": "tracking", "capture_age_ns": [0, int((age_ms + i % 10) * 1e6)],
         "clock_reason": "verified_under_configured_policy" if verified else "clock_unavailable"}
        for i in range(ticks)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def _pass_upstream(run: Run, *, rate: float = 0.8) -> None:
    """Record sysid/fit/limits as passed with the committed qualified fit."""
    run.record("sysid", {}, Gate())
    run.record("fit", {"fit_report": str(FIT_DIR / "fit_report.json"),
                       "validation_report": str(FIT_DIR / "independent_validation_report.json")}, Gate())
    run.record("limits", {"rate_limit_rad_s": rate}, Gate())


def test_latency_stage_gates_on_ticks_and_clock(tmp_path: Path) -> None:
    run = _run(tmp_path)
    report = stages.stage_latency(run, [_trace(tmp_path / "t.jsonl", 600, 120.0)])
    assert report["gate"]["passed"] and 120 <= report["p50_ms"] <= report["p95_ms"] <= 130
    report = stages.stage_latency(run, [_trace(tmp_path / "u.jsonl", 600, 120.0, verified=False)])
    assert not report["gate"]["passed"] and "clock-verified" in report["gate"]["failures"][0]
    assert not run.passed("latency")


def test_stage_order_is_enforced_and_rerun_clears_downstream(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with pytest.raises(StageError, match="needs passed stage"):
        stages.stage_sim(run)
    _pass_upstream(run)
    run.record("latency", {"p50_ms": 100.0, "p95_ms": 110.0}, Gate())
    run.record("sim", {}, Gate())
    run.record("hardware", {}, Gate())
    run.record("fit", {"fit_report": "x", "validation_report": "y"}, Gate())  # re-fit
    assert run.passed("fit") and not run.passed("sim") and not run.passed("hardware")
    assert run.passed("limits") and run.passed("latency")  # not downstream of fit


def test_rate_limit_keeps_a_margin_below_the_fastest_passing_level() -> None:
    cfg = {"levels": [1, 2, 4, 6, 8, 10], "accel_byte": 10, "margin_levels": 1,
           "max_rate_rad_s": 0.8, "min_rate_rad_s": 0.5}
    summary = {"10": {"max_passing_level": 6}, "50": {"max_passing_level": 2}}
    from common.gimbal.mks_servo42_rs485 import f6_level_speed_rad_s

    level, rate, _ = stages.choose_rate_limit(summary, cfg, 1.0)
    assert level == 4 and rate == pytest.approx(f6_level_speed_rad_s(4))
    assert stages.choose_rate_limit({"10": {"max_passing_level": 10}}, cfg, 1.0)[1] == 0.8  # capped
    assert stages.choose_rate_limit({"10": {"max_passing_level": 1}}, cfg, 1.0)[1] is None
    # A gear reduction makes every level slower at the axis.
    assert stages.choose_rate_limit(summary, cfg, 5.0)[1] == pytest.approx(f6_level_speed_rad_s(4) / 5)


def test_sim_stage_picks_feedforward_on_fast_targets(tmp_path: Path) -> None:
    run = _run(tmp_path, scenarios=["ramp_0p3"], sim={"kp_min": 1.0, "kp_max": 20.0, "kp_points": 7},
               feedforward_configs=[[0.0, 0.0, 0.4], [0.5, 0.5, 2.0]])
    _pass_upstream(run)
    run.record("latency", {"p50_ms": 120.0, "p95_ms": 125.0}, Gate())
    report = stages.stage_sim(run)
    assert set(report["results"]) == {"yaw", "pitch"}
    assert report["chosen"] == "0.5:0.5:2"
    assert all(v > 0 for v in report["chosen_improvement"].values())


def _hardware_log(path: Path, costs: dict[float, float]) -> Path:
    path.write_text("\n".join(json.dumps({"event": "run", "kp": kp, "cost_rad": c + d})
                              for kp, c in costs.items() for d in (-0.0002, 0.0002)))
    return path


def test_agreement_gates_and_hardware_decides_feedforward(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _pass_upstream(run)
    run.record("latency", {"p50_ms": 120.0, "p95_ms": 125.0}, Gate())
    run.record("sim", {"pid_only": "0:0:0.4", "chosen": "0.5:0.5:2"}, Gate())
    pid = {3.0: 0.030, 4.0: 0.026, 5.0: 0.025, 6.0: 0.027}
    ff = {4.0: 0.016, 5.0: 0.014, 6.0: 0.013, 7.0: 0.015}
    runs = {axis: {"0:0:0.4": {"log": str(_hardware_log(tmp_path / f"{axis}_pid.jsonl", pid))},
                   "0.5:0.5:2": {"log": str(_hardware_log(tmp_path / f"{axis}_ff.jsonl", ff))}}
            for axis in ("yaw", "pitch")}
    run.record("hardware", {"latency_ms": 120, "runs": runs}, Gate())
    truth = {"0:0:0.4": pid, "0.5:0.5:2": ff}
    report = stages.stage_agreement(run, simulate_cost=lambda axis, tag, kp: truth[tag][kp] * 1.1)
    assert report["gate"]["passed"]
    assert report["selected"] == "0.5:0.5:2" and report["kp"] == {"yaw": 6.0, "pitch": 6.0}
    assert report["pid_only_kp"]["yaw"] == 5.0 and report["feedforward_confirmed_on_hardware"]
    # A simulator that prefers the wrong gain fails the gate.
    bad = stages.stage_agreement(run, simulate_cost=lambda axis, tag, kp: truth[tag][kp] * (2 if kp > 3 else 1))
    assert not bad["gate"]["passed"]


def test_emit_writes_a_validated_overlay_with_provenance(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _pass_upstream(run, rate=0.7)
    run.record("latency", {"p50_ms": 120.0, "p95_ms": 125.0}, Gate())
    run.record("sim", {}, Gate())
    run.record("hardware", {}, Gate())
    run.record("agreement", {"selected": "0.5:0.5:2", "kp": {"yaw": 6.0, "pitch": 5.5},
                             "pid_only_kp": {"yaw": 5.0, "pitch": 4.5}}, Gate())
    report = stages.stage_emit(run)
    assert report["gate"]["passed"], report["gate"]
    text = Path(report["tuned_config"]).read_text()
    assert "report sha256" in text and "plan sha256" in text
    overlay = yaml.safe_load(text)
    assert overlay["controller"]["yaw_kp"] == 6.0 and overlay["controller"]["predict"] == 0.5
    assert overlay["gimbal"]["yaw_rate_limit_rad_s"] == 0.7 == overlay["controller"]["rate_limit_rad_s"]
    assert yaml.safe_load(Path(report["pid_only_overlay"]).read_text())["controller"]["feedforward_scale"] == 0.0


def test_live_ab_requires_bounds_and_a_feedforward_win() -> None:
    overlay = {"controller": {"yaw_kp": 6.0, "feedforward_scale": 0.5}}
    cfg = {"min_pairs": 2, "max_capture_rms_mrad": {"yaw": 20.0, "pitch": 20.0}}

    def trial(kp, scale, yaw, pitch):
        return {"yaw_kp": kp, "feedforward_scale": scale,
                "yaw_capture_rms_rad": yaw / 1e3, "pitch_capture_rms_rad": pitch / 1e3}

    good = [trial(5.0, 0.0, 25, 16), trial(6.0, 0.5, 11, 10), trial(6.0, 0.5, 12, 11), trial(5.0, 0.0, 22, 15)]
    assert stages.score_live_ab(good, overlay, cfg, feedforward=True)[1].passed
    worse = [trial(5.0, 0.0, 10, 9), trial(6.0, 0.5, 11, 10), trial(6.0, 0.5, 12, 11), trial(5.0, 0.0, 10, 9)]
    assert not stages.score_live_ab(worse, overlay, cfg, feedforward=True)[1].passed
    too_big = [trial(6.0, 0.5, 25, 10), trial(6.0, 0.5, 26, 10)]
    assert not stages.score_live_ab(too_big, overlay, cfg, feedforward=False)[1].passed


def test_cli_status_and_missing_run(tmp_path: Path, capsys) -> None:
    from tools.tuning.__main__ import main

    assert main(["status", str(tmp_path / "nope")]) == 2
    run = _run(tmp_path)
    assert main(["status", str(run.root)]) == 0
    assert "not qualified yet" in capsys.readouterr().out


def test_motion_check_catches_an_axis_that_never_moved(tmp_path: Path) -> None:
    rows = ["axis,cmd_rate_encoded_rad_s,omega_rad_s"]
    rows += [f"yaw,0.3,{0.28 + 0.01 * (i % 3)}" for i in range(20)] + ["yaw,0.0,0.0"]
    rows += ["pitch,0.3,0.0" for _ in range(20)] + ["pitch,-0.5,"]
    path = tmp_path / "sweep.csv"
    path.write_text("\n".join(rows))
    assert stages.motion_fraction(path, "yaw") == 1.0
    assert stages.motion_fraction(path, "pitch") == 0.0
