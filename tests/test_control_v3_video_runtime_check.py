from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _command(tmp_path: Path) -> list[str]:
    return [
        sys.executable, "-m", "jetson.control_v3.video_runtime",
        "--config", "configs/network.yaml",
        "--config-extra", "configs/perception.yaml,configs/control.yaml,configs/system.yaml,configs/deepstream_runtime.yaml",
        "--snapshot-sub", "tcp://127.0.0.1:5574",
        "--gimbal-sub", "tcp://127.0.0.1:5558",
        "--manual-bind", "tcp://127.0.0.1:5559",
        "--clock-endpoint", "tcp://127.0.0.1:5575",
        "--intent-bind", "tcp://127.0.0.1:5557",
        "--duration-s", "10",
        "--feedforward-scale", "0.5",
        "--clock-drift-ppm", "1000",
        "--trace", str(tmp_path / "trace.jsonl"),
        "--report", str(tmp_path / "report.json"),
        "--check",
    ]


def test_runtime_check_defaults_to_non_actuating_shadow(tmp_path: Path) -> None:
    result = subprocess.run(
        [*_command(tmp_path), "--ack-shadow-only"], cwd=REPO,
        capture_output=True, text=True, check=True,
    )
    startup = json.loads(result.stdout)
    assert startup["mode"] == "v3_video_check"
    assert startup["motor_authority"] is False
    assert startup["requested_live_publication"] is False
    assert not (tmp_path / "trace.jsonl").exists()


def test_runtime_rejects_live_without_both_explicit_acknowledgements(tmp_path: Path) -> None:
    result = subprocess.run(
        [*_command(tmp_path), "--enable-live-intent-publish"], cwd=REPO,
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "acknowledgements" in result.stderr
    assert not (tmp_path / "trace.jsonl").exists()


def test_runtime_live_check_is_non_operational_test_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [*_command(tmp_path), "--enable-live-intent-publish",
         "--ack-empirical-test-clock", "--acknowledge-unloaded-hardware",
         "--max-capture-age-ms", "250",
         "--camera-fov-y-deg", "60", "--pitch-kp", "8"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    startup = json.loads(result.stdout)
    assert startup["motor_authority"] is False
    assert startup["requested_live_publication"] is True
    assert startup["check_only"] is True
    assert startup["clock_policy_basis"] == "empirical_test_only"
    assert startup["max_capture_age_ms"] == 250
    assert "pose_source" not in startup
    assert startup["camera_fov_x_deg"] == pytest.approx(91.4928445)
    assert startup["aim_fx_px"] == pytest.approx(935.3074, rel=1e-4)
    assert startup["pitch_kp"] == 8.0
    assert not (tmp_path / "trace.jsonl").exists()


def test_runtime_has_no_pose_source_modes(tmp_path: Path) -> None:
    result = subprocess.run(
        [*_command(tmp_path), "--pose-source", "frame", "--ack-shadow-only"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --pose-source" in result.stderr


def test_runtime_rejects_out_of_range_camera_fov(tmp_path: Path) -> None:
    result = subprocess.run(
        [*_command(tmp_path), "--camera-fov-y-deg", "0.5", "--ack-shadow-only"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--camera-fov-y-deg" in result.stderr


def test_runtime_check_accepts_only_bounded_explicit_crossover(tmp_path: Path) -> None:
    base = _command(tmp_path)
    index = base.index("--feedforward-scale")
    base[index:index + 2] = ["--feedforward-schedule", "off-on-off"]
    duration = base.index("--duration-s")
    base[duration + 1] = "30"
    result = subprocess.run(
        [*base, "--ack-shadow-only", "--camera-fov-y-deg", "60"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    startup = json.loads(result.stdout)
    assert startup["feedforward_schedule"] == "off-on-off"
    assert startup["motor_authority"] is False
    base[duration + 1] = "20"
    rejected = subprocess.run([*base, "--ack-shadow-only"], cwd=REPO,
                              capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "exactly 30 seconds" in rejected.stderr
