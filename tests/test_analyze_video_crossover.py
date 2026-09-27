from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.analyze_video_crossover import analyze_crossover, compare_crossovers


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _trial(root: Path, schedule: str) -> tuple[Path, Path]:
    host, jetson = root / "host", root / "jetson"
    host.mkdir(parents=True)
    jetson.mkdir()
    (host / "streamer-check.json").write_text(
        json.dumps({"config_sources": [{
            "path": "/external/v3_video_hil_fixture.yaml", "sha256": "fixture",
        }]}) + "\n" + json.dumps({
            "source": "sim", "sim_motion_mode": "hardware_in_loop",
            "moves_physical_mount": True, "sim_camera_fov_y_deg": 60.0,
        }) + "\n", encoding="utf-8",
    )
    scales = (0.0, 0.5, 0.0) if schedule == "off-on-off" else (0.5, 0.0, 0.5)
    rows = [{
        "type": "meta", "mode": "v3_video_test_live", "motor_authority": True,
        "feedforward_scale": None,
        "feedforward_schedule": schedule, "duration_s": 30.0,
        "config_digest": "control", "clock_policy_basis": "empirical_test_only",
        "camera_fov_y_deg": 60.0, "aim_fx_px": 935.3074360871939,
        "max_capture_age_ms": 250, "yaw_kp": 8.0, "pitch_kp": 8.0,
    }]
    for index in range(1500):
        block = index // 500
        scale = scales[block]
        issued = 1_000_000_000 + index * 20_000_000
        rows.append({
            "type": "tick", "pid_reason": "tracking", "source_frame_id": index,
            "raw_bearing_error_rad": [0.1 if scale == 0 else 0.08,
                                      0.02 if scale == 0 else 0.015],
            "schedule_block": block, "feedforward_scale": scale,
            "clock_reason": "verified_under_configured_policy",
            "capture_age_ns": [150_000_000, 160_000_000],
            "estimated_capture_midpoint_ns": issued - 150_000_000,
            "capture_camera_pose_rad": [0.0, 0.0],
            "measured_target_world_rad": [0.1, 0.02],
            "feedforward_rad_s": [scale * 0.02, 0.0], "ff_reason": "ready",
            "intent": {"issued_monotonic_ns": issued,
                       "valid_until_monotonic_ns": issued + 50_000_000,
                       "mode": "live"},
        })
    _jsonl(jetson / "trace.jsonl", rows)
    (jetson / "report.json").write_text(
        json.dumps({"ticks": 1500, "missed_periods": 0}), encoding="utf-8",
    )
    _jsonl(jetson / "pitch-a-guard.jsonl", [{"summary": {
        "windows": 12, "failure": None, "pitch_a_span": 100,
        "pitch_b_origin": 1855, "pitch_b_final": 1855,
    }}])
    _jsonl(jetson / "serial-events.jsonl", [
        {"type": "SerialCommandEventV1", "addr": addr, "func": "F6",
         "payload": [0, 1, 10], "event": "wire_sent"} for addr in (1, 2)
    ])
    return host, jetson


def test_balanced_crossover_scores_all_six_blocks(tmp_path: Path) -> None:
    first_host, first_jetson = _trial(tmp_path / "first", "off-on-off")
    second_host, second_jetson = _trial(tmp_path / "second", "on-off-on")
    result = compare_crossovers(
        analyze_crossover(first_host, first_jetson),
        analyze_crossover(second_host, second_jetson),
    )
    assert result["pooled"]["off"]["blocks"] == 3
    assert result["pooled"]["on"]["blocks"] == 3
    assert result["yaw_rms_change_percent"] == pytest.approx(-20)
    assert result["pitch_rms_change_percent"] == pytest.approx(-25)


def test_crossover_rejects_unbalanced_schedule(tmp_path: Path) -> None:
    first_host, first_jetson = _trial(tmp_path / "first", "off-on-off")
    second_host, second_jetson = _trial(tmp_path / "second", "off-on-off")
    with pytest.raises(ValueError, match="both reverse"):
        compare_crossovers(
            analyze_crossover(first_host, first_jetson),
            analyze_crossover(second_host, second_jetson),
        )
