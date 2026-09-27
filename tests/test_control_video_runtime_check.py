from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from jetson.control.runtime_config import ControlRuntimeConfig


REPO = Path(__file__).resolve().parents[1]
EXTRA = "configs/perception.yaml,configs/control.yaml,configs/system.yaml,configs/deepstream_runtime.yaml"


def _run(tmp_path: Path, *overlays: dict, extra_args: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    paths = [EXTRA]
    for index, overlay in enumerate(overlays):
        path = tmp_path / f"overlay{index}.yaml"
        path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
        paths.append(str(path))
    return subprocess.run(
        [sys.executable, "-m", "jetson.control.video_runtime",
         "--config", "configs/network.yaml", "--config-extra", ",".join(paths),
         "--trace", str(tmp_path / "trace.jsonl"), "--report", str(tmp_path / "report.json"),
         *extra_args, "--check"],
        cwd=REPO, capture_output=True, text=True,
    )


def test_repo_config_defaults_to_non_actuating_shadow(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    startup = json.loads(result.stdout)
    assert startup["mode"] == "video_check"
    assert startup["motor_authority"] is False
    assert startup["controller"]["mode"] == "shadow"
    assert startup["controller"]["clock_basis"] == "assumed"
    assert not (tmp_path / "trace.jsonl").exists()


def test_sim_hil_overlay_is_live_with_simulated_camera_intrinsics(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jetson.control.video_runtime",
         "--config", "configs/network.yaml",
         "--config-extra", f"{EXTRA},configs/controller_sim_hil.yaml", "--check"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    startup = json.loads(result.stdout)
    cfg = startup["controller"]
    assert cfg["mode"] == "live" and startup["check_only"] is True
    assert startup["motor_authority"] is False  # --check never actuates
    assert cfg["snapshot_endpoint"] == "tcp://192.168.0.1:5574"
    assert startup["camera_fov_x_deg"] == pytest.approx(91.4928445)
    assert startup["aim_fx_px"] == pytest.approx(935.3074, rel=1e-4)
    assert (cfg["yaw_kp"], cfg["feedforward_scale"], cfg["predict"]) == (5.9, 0.5, 0.5)


def test_runtime_rejects_invalid_config_values(tmp_path: Path) -> None:
    for overlay, message in (
        ({"controller": {"rate_limit_rad_s": 0.2}}, "slowest F6 speed"),
        ({"controller": {"yaw_kp": 25}}, "yaw_kp"),
        ({"controller": {"mode": "test_live"}}, "mode"),
        ({"controller": {"camera_fov_y_deg": 0.5}}, "camera_fov_y_deg"),
        ({"controller": {"clock": {"basis": "trust_me", "drift_ppm": 1000}}}, "clock.basis"),
    ):
        result = _run(tmp_path, overlay)
        assert result.returncode != 0, overlay
        assert message in result.stderr, (overlay, result.stderr)


def test_trial_era_flags_are_gone(tmp_path: Path) -> None:
    for flag in ("--enable-live-intent-publish", "--feedforward-schedule", "--pose-source",
                 "--trial-rate-limit", "--ack-shadow-only"):
        result = _run(tmp_path, extra_args=(flag,))
        assert result.returncode != 0
        assert "unrecognized arguments" in result.stderr


def test_runtime_config_requires_section_and_clock_basis() -> None:
    with pytest.raises(ValueError, match="missing controller"):
        ControlRuntimeConfig.from_config({"net": {}})
    with pytest.raises(KeyError):
        ControlRuntimeConfig.from_config({"controller": {
            "yaw_kp": 5, "pitch_kp": 5, "rate_limit_rad_s": 0.8, "clock": {"drift_ppm": 500}}})


def test_runtime_config_endpoints_fall_back_to_network_config() -> None:
    cfg = ControlRuntimeConfig.from_config({
        "net": {"zmq_perception_v2": "tcp://a:1", "zmq_gimbal_state": "tcp://a:2",
                "zmq_manual_state": "tcp://a:3", "zmq_source_clock_sync": "tcp://a:4",
                "zmq_control": "tcp://a:5"},
        "controller": {"yaw_kp": 5, "pitch_kp": 5, "rate_limit_rad_s": 0.8,
                          "endpoints": {"snapshot_sub": "tcp://b:9"},
                          "clock": {"basis": "same_host", "drift_ppm": 0}},
    })
    assert cfg.snapshot_endpoint == "tcp://b:9" and cfg.intent_bind == "tcp://a:5"
    assert cfg.mode == "shadow"
