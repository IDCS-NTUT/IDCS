from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.analyze_video_hil import analyze_trial, compare


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _trial(root: Path, *, scale: float, mode: str = "v3_video_test_live") -> tuple[Path, Path]:
    host, jetson = root / "host", root / "jetson"
    host.mkdir(parents=True)
    jetson.mkdir(parents=True)
    (host / "streamer-check.json").write_text(
        json.dumps({"config_sources": [
            {"path": "/external/v3_video_hil_fixture.yaml", "sha256": "fixture"},
        ]}) + "\n[streamer] profile 1080p\n" + json.dumps({
            "source": "sim", "sim_motion_mode": "hardware_in_loop",
            "moves_physical_mount": True, "sim_camera_fov_y_deg": 60.0,
        }) + "\n", encoding="utf-8",
    )
    rows = [{"type": "meta", "mode": mode, "motor_authority": mode == "v3_video_test_live",
             "feedforward_scale": scale, "config_digest": "control",
             "clock_policy_basis": "empirical_test_only", "camera_fov_y_deg": 60.0,
             "aim_fx_px": 935.3074360871939, "max_capture_age_ms": 250,
             "yaw_kp": 8.0, "pitch_kp": 8.0}]
    for index in range(201):
        rows.append({
            "type": "tick", "pid_reason": "tracking",
            "source_frame_id": index,
            "raw_bearing_error_rad": [0.1 if scale == 0 else 0.08, 0.02],
            "clock_reason": "verified_under_configured_policy",
            "capture_age_ns": [150_000_000, 160_000_000],
            "estimated_capture_midpoint_ns": 900_000_000 + index * 50_000_000,
            "capture_camera_pose_rad": [0.0, 0.0],
            "measured_target_world_rad": [0.1, 0.02],
            "ff_reason": "ready", "feedforward_rad_s": [scale * 0.02, 0.0],
            "intent": {"issued_monotonic_ns": 1_000_000_000 + index * 50_000_000,
                       "valid_until_monotonic_ns": 1_050_000_000 + index * 50_000_000,
                       "mode": "live"},
        })
    _write_jsonl(jetson / "trace.jsonl", rows)
    (jetson / "report.json").write_text(json.dumps({"ticks": 201, "missed_periods": 0}), encoding="utf-8")
    _write_jsonl(jetson / "pitch-a-guard.jsonl", [{"summary": {
        "windows": 12, "failure": None, "pitch_a_span": 100,
        "pitch_b_origin": 1854, "pitch_b_final": 1854,
    }}])
    _write_jsonl(jetson / "serial-events.jsonl", [
        {"type": "SerialCommandEventV1", "addr": addr, "func": "F6",
         "payload": [0, 1, 10], "event": "wire_sent"}
        for addr in (1, 2)
    ])
    return host, jetson


def test_analyzer_scores_unique_capture_frames_and_compares_matched_trials(tmp_path: Path) -> None:
    off_host, off_jetson = _trial(tmp_path / "off", scale=0.0)
    on_host, on_jetson = _trial(tmp_path / "on", scale=0.5)
    off = analyze_trial(off_host, off_jetson)
    on = analyze_trial(on_host, on_jetson)
    result = compare(off, on)
    assert off["scored_frames"] == 121
    assert off["yaw_capture_rms_rad"] == pytest.approx(0.1)
    assert on["yaw_capture_rms_rad"] == pytest.approx(0.08)
    assert result["yaw_rms_change_percent"] == pytest.approx(-20)
    assert off["serial_f6_rpm_histogram"]["2"] == {"1": 1}


def test_analyzer_rejects_shadow_and_pitch_b_motion(tmp_path: Path) -> None:
    host, jetson = _trial(tmp_path / "shadow", scale=0.5, mode="v3_video_shadow")
    with pytest.raises(ValueError, match="not a live"):
        analyze_trial(host, jetson)
    events = [json.loads(line) for line in (jetson / "serial-events.jsonl").read_text().splitlines()]
    events.append({"type": "SerialCommandEventV1", "addr": 3, "func": "F6",
                   "payload": [0, 1, 10], "event": "wire_sent"})
    _write_jsonl(jetson / "serial-events.jsonl", events)
    with pytest.raises(ValueError, match="pitch-B"):
        analyze_trial(host, jetson, allow_shadow=True)


@pytest.mark.parametrize("field,value,reason", [
    ("clock_reason", "clock_unavailable", "capture-time"),
    ("capture_age_ns", [250_000_000, 251_000_000], "capture-time"),
    ("feedforward_rad_s", [0.0, 0.0], "feedforward"),
])
def test_analyzer_rejects_missing_live_timing_or_feedforward(
    tmp_path: Path, field: str, value: object, reason: str,
) -> None:
    host, jetson = _trial(tmp_path, scale=0.5)
    rows = [json.loads(line) for line in (jetson / "trace.jsonl").read_text().splitlines()]
    for row in rows:
        if row.get("type") == "tick":
            row[field] = value
    _write_jsonl(jetson / "trace.jsonl", rows)
    with pytest.raises(ValueError, match=reason):
        analyze_trial(host, jetson)


def _drop_pose(jetson: Path, count: int) -> None:
    rows = [json.loads(line) for line in (jetson / "trace.jsonl").read_text().splitlines()]
    dropped = 0
    for row in rows:
        if row.get("type") == "tick" and dropped < count:
            row["capture_camera_pose_rad"] = None
            dropped += 1
    _write_jsonl(jetson / "trace.jsonl", rows)


def test_a_few_ticks_without_aligned_pose_are_tolerated_and_reported(tmp_path: Path) -> None:
    host, jetson = _trial(tmp_path, scale=0.0)
    _drop_pose(jetson, 4)  # 2% of 201 tracking ticks is 4.02
    assert analyze_trial(host, jetson)["unaligned_pose_ticks"] == 4


def test_many_ticks_without_aligned_pose_reject_the_trial(tmp_path: Path) -> None:
    host, jetson = _trial(tmp_path, scale=0.0)
    _drop_pose(jetson, 5)
    with pytest.raises(ValueError, match="5/201 tracking ticks lack"):
        analyze_trial(host, jetson)


def test_service_era_meta_is_normalized() -> None:
    from tools.analyze_video_hil import normalize_meta
    meta = normalize_meta({"type": "meta", "mode": "v3_video_live", "motor_authority": True,
                           "controller_v3": {"feedforward_scale": 0.5, "yaw_kp": 5.9, "pitch_kp": 5.9,
                                             "max_capture_age_ms": 250, "camera_fov_y_deg": 60.0,
                                             "predict": 0.5, "clock_basis": "assumed",
                                             "clock_drift_ppm": 1000}})
    assert meta["mode"] == "v3_video_test_live"
    assert meta["clock_policy_basis"] == "assumed" and meta["yaw_kp"] == 5.9


def test_renamed_service_meta_is_normalized() -> None:
    from tools.analyze_video_hil import normalize_meta
    meta = normalize_meta({"type": "meta", "mode": "video_live", "motor_authority": True,
                           "controller": {"feedforward_scale": 0.5, "yaw_kp": 5.9, "pitch_kp": 5.9,
                                          "max_capture_age_ms": 250, "camera_fov_y_deg": 60.0,
                                          "predict": 0.5, "clock_basis": "assumed",
                                          "clock_drift_ppm": 1000}})
    assert meta["mode"] == "v3_video_test_live" and meta["feedforward_scale"] == 0.5


def test_intrinsics_must_match_the_simulated_camera() -> None:
    from tools.analyze_video_hil import _intrinsics_match
    host = {"sim_camera_fov_x_deg": 135.0, "sim_camera_fov_y_deg": 73.0}
    assert _intrinsics_match(host, {"aim_fov_deg": [135.0, 73.0]})
    assert not _intrinsics_match(host, {"aim_fov_deg": [91.49, 60.0]})
    legacy_host = {"sim_camera_fov_y_deg": 60.0}
    assert _intrinsics_match(legacy_host, {"camera_fov_y_deg": 60.0, "aim_fx_px": 935.3074360871939})
    assert not _intrinsics_match(legacy_host, {"camera_fov_y_deg": 60.0, "aim_fx_px": 700.0})


def test_pitch_b_guard_from_captured_serial_replies(tmp_path) -> None:
    import json

    from tools.analyze_video_hil import _pitch_b_guard

    rows = [{"type": "SerialReplyData", "addr": 3, "func": "0x31", "reply": {"parsed": {"counts": 1000 + i % 2}}}
            for i in range(8)] + [{"type": "SerialReplyData", "addr": 2, "func": "0x31", "reply": {"parsed": {"counts": 5}}}]
    (tmp_path / "serial-events.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    guard = _pitch_b_guard(tmp_path)
    assert guard == {"failure": None, "windows": 8, "pitch_b_origin": 1000, "pitch_b_final": 1001}
    (tmp_path / "serial-events.jsonl").write_text("")
    assert _pitch_b_guard(tmp_path)["failure"] is not None
