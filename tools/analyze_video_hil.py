"""Score matched hardware/video HIL traces against exact source-frame truth.

One bearing error per unique rendered source frame is scored, so a 50-Hz
controller cannot gain weight by repeating a slower video observation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


MAX_UNALIGNED_POSE_FRACTION = 0.02


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _p95_abs(values: list[float]) -> float:
    ordered = sorted(abs(value) for value in values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def normalize_meta(meta: dict) -> dict:
    """Flatten service trace metadata (a ``controller`` section; ``controller_v3``
    in traces from before the rename) into the trial-era flat keys, so traces
    from every runtime generation are scored identically. The trial-era live
    marker ``v3_video_test_live`` is the canonical live mode here."""
    cfg = meta.get("controller", meta.get("controller_v3"))
    if cfg is None:
        return meta
    flat = dict(meta)
    if meta["mode"] in {"video_live", "v3_video_live"}:
        flat["mode"] = "v3_video_test_live"
    for key in ("feedforward_scale", "yaw_kp", "pitch_kp", "max_capture_age_ms",
                "camera_fov_y_deg", "predict"):
        flat[key] = cfg[key]
    flat["clock_policy_basis"] = cfg["clock_basis"]
    flat["clock_drift_ppm"] = cfg["clock_drift_ppm"]
    return flat


def _intrinsics_match(host_sim: dict, meta: dict) -> bool:
    """The controller's aim field of view equals the simulated camera's.

    Service traces record ``aim_fov_deg``; trial-era traces predate it and were
    all taken with the 60-degree fixture camera (fx 935.307 px at 1080p).
    """
    aim = meta.get("aim_fov_deg")
    if aim is None:
        return (host_sim.get("sim_camera_fov_y_deg") == 60.0
                and meta.get("camera_fov_y_deg") == 60.0
                and math.isclose(meta.get("aim_fx_px", 0), 935.3074360871939, rel_tol=1e-5))
    sim_x, sim_y = host_sim.get("sim_camera_fov_x_deg"), host_sim.get("sim_camera_fov_y_deg")
    return (sim_x is not None and sim_y is not None
            and math.isclose(aim[0], sim_x, abs_tol=0.01) and math.isclose(aim[1], sim_y, abs_tol=0.01))


LIVE_CLOCK_BASES = {"empirical_test_only", "slew_limited_ntp", "same_host", "assumed"}


def analyze_trial(
    host_dir: Path, jetson_dir: Path, *, allow_shadow: bool = False,
    allow_schedule: bool = False,
) -> dict:
    host_records = [
        json.loads(line) for line in (host_dir / "streamer-check.json").read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]
    host_check = next((row for row in host_records if "config_sources" in row), None)
    if host_check is None:
        raise ValueError("host streamer check lacks configuration provenance")
    host_sim = next((row for row in host_records if row.get("source") == "sim"), None)
    fixture = next(
        (item for item in host_check["config_sources"]
         if item["path"].endswith((
             "v3_video_hil_fixture.yaml", "v3_video_hil_smooth_fixture.yaml",
             "v3_live_ff_fast_fixture.yaml", "sim_hil_target_ellipse.yaml",
             "deepstream_pc_moving_drone_opengl.yaml",
         ))), None,
    )
    if fixture is None:
        raise ValueError("wrong or missing video HIL fixture")
    rows = _jsonl(jetson_dir / "trace.jsonl")
    meta = normalize_meta(next(row for row in rows if row.get("type") == "meta"))
    live = meta["mode"] == "v3_video_test_live" and meta["motor_authority"]
    if not allow_shadow and not live:
        raise ValueError("trace is not a live hardware/video trial")
    if meta.get("feedforward_schedule") is not None and not allow_schedule:
        raise ValueError("crossover schedule requires crossover analysis")
    if live and (host_sim is None or host_sim.get("sim_motion_mode") != "hardware_in_loop"
                 or host_sim.get("moves_physical_mount") is not True):
        raise ValueError("host source is not motor-driven simulator HIL")
    if live and meta.get("clock_policy_basis") not in LIVE_CLOCK_BASES:
        raise ValueError("live trial lacks an explicit clock policy")
    if live and not _intrinsics_match(host_sim, meta):
        raise ValueError("controller aim intrinsics do not match the simulated camera")
    ticks = [row for row in rows if row.get("type") == "tick"]
    if not ticks:
        raise ValueError("trial has no controller ticks")
    max_capture_age_ns = int(meta.get("max_capture_age_ms", 0) * 1_000_000)
    tracking_ticks = [row for row in ticks if row["pid_reason"] == "tracking"]
    unaligned = 0
    if live:
        report = json.loads((jetson_dir / "report.json").read_text(encoding="utf-8"))
        if report.get("ticks") != len(ticks) or report.get("missed_periods", len(ticks)) > len(ticks) // 100:
            raise ValueError("controller cadence report failed")
        if len(tracking_ticks) < len(ticks) * 0.9:
            raise ValueError("too few tracking ticks for a live comparison")
        for row in tracking_ticks:
            age = row.get("capture_age_ns")
            intent = row["intent"]
            if (row.get("clock_reason") != "verified_under_configured_policy"
                    or not isinstance(age, list) or len(age) != 2
                    or age[1] > max_capture_age_ns or age[0] < 0):
                raise ValueError("live tracking tick violated capture-time policy")
            if (intent.get("mode") != "live"
                    or not 0 < intent["valid_until_monotonic_ns"] - intent["issued_monotonic_ns"] <= 50_000_000):
                raise ValueError("live intent lacks short finite lease")
        # A late pose sample can leave a tick without an aligned capture pose;
        # PID then uses the raw bearing and feedforward sits the tick out.
        unaligned = sum(
            row.get("capture_camera_pose_rad") is None
            or row.get("measured_target_world_rad") is None
            or row.get("estimated_capture_midpoint_ns") is None
            for row in tracking_ticks
        )
        if unaligned > MAX_UNALIGNED_POSE_FRACTION * len(tracking_ticks):
            raise ValueError(
                f"{unaligned}/{len(tracking_ticks)} tracking ticks lack a capture-time aligned camera pose")
        if meta.get("feedforward_schedule") is None and meta["feedforward_scale"] == 0.5 and sum(
            row.get("ff_reason") == "ready" and any(
                abs(value) > 1e-5 for value in row.get("feedforward_rad_s", [])
            ) for row in tracking_ticks
        ) < 100:
            raise ValueError("feedforward-on trial did not exercise live feedforward")
        if meta.get("feedforward_schedule") is None and meta["feedforward_scale"] == 0 and any(
            abs(value) > 1e-9 for row in tracking_ticks
            for value in row.get("feedforward_rad_s", [])
        ):
            raise ValueError("feedforward-off trial applied feedforward")
    first_ns = ticks[0]["intent"]["issued_monotonic_ns"]
    scored: dict[int, tuple[float, float]] = {}
    source_ids: set[int] = set()
    reason_counts: Counter[str] = Counter()
    for row in ticks:
        reason_counts[row["pid_reason"]] += 1
        frame_id = row.get("source_frame_id")
        elapsed_s = (row["intent"]["issued_monotonic_ns"] - first_ns) / 1e9
        if frame_id is None or not 4.0 <= elapsed_s < 19.0:
            continue
        source_ids.add(frame_id)
        error = row.get("raw_bearing_error_rad")
        if frame_id not in scored and error is not None and all(math.isfinite(value) for value in error):
            scored[frame_id] = (float(error[0]), float(error[1]))
    if len(scored) < 100:
        raise ValueError("fewer than 100 unique source frames have exact bearing truth")
    guard = _jsonl(jetson_dir / "pitch-a-guard.jsonl")[-1]["summary"]
    if guard["failure"] is not None or guard["windows"] < 5:
        raise ValueError("pitch-B safety guard did not pass")
    if abs(guard["pitch_b_origin"] - guard["pitch_b_final"]) > 8:
        raise ValueError("pitch-B encoder moved beyond trial tolerance")
    events = [row for row in _jsonl(jetson_dir / "serial-events.jsonl")
              if row.get("type") == "SerialCommandEventV1"]
    def f6_rpm(row: dict) -> int | None:
        payload = row.get("payload") or []
        if row.get("func") != "F6" or len(payload) < 2:
            return None
        return ((int(payload[0]) & 0x0F) << 8) | int(payload[1])

    def motion_write(row: dict) -> bool:
        return bool(f6_rpm(row))

    if any(row.get("addr") == 3 and (
        motion_write(row) or row.get("func") == "FD"
        or (row.get("func") == "F3" and row.get("payload") == [1])
    ) for row in events):
        raise ValueError("pitch-B received a motion or enable command")
    failures = [row for row in events if row.get("event") in {"write_failed", "wire_uncertain"}]
    if failures:
        raise ValueError(f"serial writes failed or were uncertain: {len(failures)}")
    motion_yaw = sum(
        row.get("addr") == 1 and row.get("event") == "wire_sent" and motion_write(row)
        for row in events
    )
    motion_pitch = sum(
        row.get("addr") == 2 and row.get("event") == "wire_sent" and motion_write(row)
        for row in events
    )
    if motion_yaw == 0 or motion_pitch == 0 or (guard["pitch_a_span"] or 0) < 35:
        raise ValueError("trial lacks confirmed two-axis hardware movement")
    rpm_hist = {
        str(addr): dict(sorted(Counter(
            str(f6_rpm(row)) for row in events
            if row.get("addr") == addr and row.get("event") == "wire_sent"
            and f6_rpm(row) is not None
        ).items())) for addr in (1, 2)
    }
    yaw = [item[0] for item in scored.values()]
    pitch = [item[1] for item in scored.values()]
    return {
        "host_fixture_sha256": fixture["sha256"],
        "jetson_config_digest": meta["config_digest"],
        "feedforward_scale": meta["feedforward_scale"],
        "feedforward_schedule": meta.get("feedforward_schedule"),
        "yaw_kp": meta.get("yaw_kp"),
        "pitch_kp": meta.get("pitch_kp"),
        "mode": meta["mode"],
        "clock_policy_basis": meta.get("clock_policy_basis"),
        "ticks": len(ticks),
        "tracking_ticks": reason_counts["tracking"],
        "unaligned_pose_ticks": unaligned,
        "clock_verified_tracking_ticks": sum(
            row.get("clock_reason") == "verified_under_configured_policy"
            for row in tracking_ticks
        ),
        "capture_age_upper_p95_ms": _p95_abs([
            row["capture_age_ns"][1] / 1e6 for row in tracking_ticks
            if row.get("capture_age_ns") is not None
        ]),
        "feedforward_applied_ticks": sum(
            any(abs(value) > 1e-5 for value in row.get("feedforward_rad_s", []))
            for row in tracking_ticks
        ),
        "reason_counts": dict(reason_counts),
        "unique_source_frames": len(source_ids),
        "scored_frames": len(scored),
        "yaw_capture_rms_rad": _rms(yaw),
        "yaw_capture_p95_abs_rad": _p95_abs(yaw),
        "pitch_capture_rms_rad": _rms(pitch),
        "pitch_capture_p95_abs_rad": _p95_abs(pitch),
        "pitch_guard_windows": guard["windows"],
        "pitch_a_span_counts": guard["pitch_a_span"],
        "pitch_b_origin_counts": guard["pitch_b_origin"],
        "pitch_b_final_counts": guard["pitch_b_final"],
        "serial_motion_writes_yaw": motion_yaw,
        "serial_motion_writes_pitch_a": motion_pitch,
        "serial_f6_rpm_histogram": rpm_hist,
    }


def compare(off: dict, on: dict) -> dict:
    if off["feedforward_scale"] != 0 or on["feedforward_scale"] != 0.5:
        raise ValueError("expected matched FF-off and FF-0.5 trials")
    if off["host_fixture_sha256"] != on["host_fixture_sha256"]:
        raise ValueError("different host target fixtures")
    if off["jetson_config_digest"] != on["jetson_config_digest"]:
        raise ValueError("different Jetson control configurations")
    if (off["yaw_kp"], off["pitch_kp"]) != (on["yaw_kp"], on["pitch_kp"]):
        raise ValueError("different PID gains")
    return {
        "off": off, "on": on,
        "yaw_rms_change_percent": 100 * (
            on["yaw_capture_rms_rad"] / off["yaw_capture_rms_rad"] - 1
        ),
        "pitch_rms_change_percent": 100 * (
            on["pitch_capture_rms_rad"] / off["pitch_capture_rms_rad"] - 1
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-host", type=Path, required=True)
    parser.add_argument("--off-jetson", type=Path, required=True)
    parser.add_argument("--on-host", type=Path, required=True)
    parser.add_argument("--on-jetson", type=Path, required=True)
    parser.add_argument("--allow-shadow", action="store_true")
    args = parser.parse_args()
    result = compare(
        analyze_trial(args.off_host, args.off_jetson, allow_shadow=args.allow_shadow),
        analyze_trial(args.on_host, args.on_jetson, allow_shadow=args.allow_shadow),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
