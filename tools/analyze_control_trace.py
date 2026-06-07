#!/usr/bin/env python3
"""Analyze a JSONL control trace captured by tools.record_control_trace."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from common.config_sync import expand_config_paths, load_merged_config


@dataclass(frozen=True)
class Event:
    stream: str
    rx_monotonic_ns: int
    payload: Mapping[str, Any]

    @property
    def rx_s(self) -> float:
        return self.rx_monotonic_ns / 1e9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="JSONL trace produced by record_control_trace.py")
    parser.add_argument("--config", default=None, help="Optional YAML config for rate-limit context")
    parser.add_argument(
        "--config-extra",
        default="configs/control.yaml,configs/system.yaml",
        help="Comma-separated extra YAML configs used with --config.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional PNG path for a matplotlib summary plot.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path for machine-readable summary JSON.",
    )
    parser.add_argument(
        "--settle-threshold-rad",
        type=float,
        default=0.02,
        help="Error magnitude threshold for settling-time detection.",
    )
    parser.add_argument(
        "--settle-hold-s",
        type=float,
        default=0.2,
        help="Required time under threshold before a segment is considered settled.",
    )
    parser.add_argument(
        "--segment-gap-s",
        type=float,
        default=0.25,
        help="Split tracking segments after this receive-time gap.",
    )
    return parser.parse_args()


def _iter_records(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"[analyze][WARN] skipping invalid JSON line {line_no}: {exc}")
                continue
            if isinstance(record, Mapping):
                yield record
            else:
                print(f"[analyze][WARN] skipping non-object JSON line {line_no}")


def _load_events(path: Path) -> tuple[list[Event], int, Optional[Mapping[str, Any]]]:
    events: list[Event] = []
    decode_errors = 0
    meta: Optional[Mapping[str, Any]] = None
    for record in _iter_records(path):
        record_type = str(record.get("type", ""))
        if record_type == "meta":
            meta = record
            continue
        if record_type == "decode_error":
            decode_errors += 1
            continue
        if record_type != "event":
            continue
        stream = str(record.get("stream", "")).strip()
        payload = record.get("payload")
        rx_raw = record.get("rx_monotonic_ns")
        if not stream or not isinstance(payload, Mapping):
            continue
        try:
            rx_ns = int(rx_raw)
        except (TypeError, ValueError):
            continue
        events.append(Event(stream=stream, rx_monotonic_ns=rx_ns, payload=payload))
    events.sort(key=lambda event: event.rx_monotonic_ns)
    return events, decode_errors, meta


def _float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _pair(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != 2:
        return None
    first = _float(value[0])
    second = _float(value[1])
    if first is None or second is None:
        return None
    return first, second


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rms(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _summary(values: Sequence[float]) -> dict[str, Optional[float] | int]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "rms": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "rms": _rms(values),
    }


def _abs_summary(values: Sequence[float]) -> dict[str, Optional[float] | int]:
    return _summary([abs(value) for value in values])


def _stream_events(events: Sequence[Event], stream: str) -> list[Event]:
    return [event for event in events if event.stream == stream]


def _stream_gap_ms(events: Sequence[Event]) -> list[float]:
    ordered = sorted(events, key=lambda event: event.rx_monotonic_ns)
    return [
        (ordered[idx].rx_monotonic_ns - ordered[idx - 1].rx_monotonic_ns) / 1e6
        for idx in range(1, len(ordered))
    ]


def _stream_rate_hz(events: Sequence[Event]) -> Optional[float]:
    if len(events) < 2:
        return None
    span_s = (events[-1].rx_monotonic_ns - events[0].rx_monotonic_ns) / 1e9
    if span_s <= 0.0:
        return None
    return (len(events) - 1) / span_s


def _frame_map(events: Sequence[Event]) -> dict[int, Event]:
    result: dict[int, Event] = {}
    for event in events:
        frame_id_raw = event.payload.get("frame_id")
        try:
            frame_id = int(frame_id_raw)
        except (TypeError, ValueError):
            continue
        result[frame_id] = event
    return result


def _same_host_delta_ms(
    events: Sequence[Event],
    *,
    later_key: str,
    earlier_key: str,
) -> list[float]:
    values: list[float] = []
    for event in events:
        later = _float(event.payload.get(later_key))
        earlier = _float(event.payload.get(earlier_key))
        if later is None or earlier is None:
            continue
        values.append(later - earlier)
    return values


def _matched_delta_ms(
    later_events: Sequence[Event],
    earlier_by_frame: Mapping[int, Event],
    *,
    later_key: str,
    earlier_key: str,
) -> list[float]:
    values: list[float] = []
    for later_event in later_events:
        frame_id_raw = later_event.payload.get("frame_id")
        try:
            frame_id = int(frame_id_raw)
        except (TypeError, ValueError):
            continue
        earlier_event = earlier_by_frame.get(frame_id)
        if earlier_event is None:
            continue
        later = _float(later_event.payload.get(later_key))
        earlier = _float(earlier_event.payload.get(earlier_key))
        if later is None or earlier is None:
            continue
        values.append(later - earlier)
    return values


def _matched_receive_delta_ms(
    later_events: Sequence[Event],
    earlier_by_frame: Mapping[int, Event],
) -> list[float]:
    values: list[float] = []
    for later_event in later_events:
        frame_id_raw = later_event.payload.get("frame_id")
        try:
            frame_id = int(frame_id_raw)
        except (TypeError, ValueError):
            continue
        earlier_event = earlier_by_frame.get(frame_id)
        if earlier_event is None:
            continue
        values.append((later_event.rx_monotonic_ns - earlier_event.rx_monotonic_ns) / 1e6)
    return values


def _control_vectors(control_events: Sequence[Event]) -> dict[str, list[float]]:
    result = {
        "err_yaw_rad": [],
        "err_pitch_rad": [],
        "err_mag_rad": [],
        "err_u_px": [],
        "err_v_px": [],
        "cmd_yaw_rad_s": [],
        "cmd_pitch_rad_s": [],
        "cmd_mag_rad_s": [],
    }
    for event in control_events:
        err_rad = _pair(event.payload.get("err_rad"))
        if err_rad is not None:
            result["err_yaw_rad"].append(err_rad[0])
            result["err_pitch_rad"].append(err_rad[1])
            result["err_mag_rad"].append(math.hypot(err_rad[0], err_rad[1]))
        err_uv = _pair(event.payload.get("err_uv"))
        if err_uv is not None:
            result["err_u_px"].append(err_uv[0])
            result["err_v_px"].append(err_uv[1])
        yaw_cmd = _float(event.payload.get("pan_rate_cmd"))
        pitch_cmd = _float(event.payload.get("tilt_rate_cmd"))
        if yaw_cmd is not None:
            result["cmd_yaw_rad_s"].append(yaw_cmd)
        if pitch_cmd is not None:
            result["cmd_pitch_rad_s"].append(pitch_cmd)
        if yaw_cmd is not None and pitch_cmd is not None:
            result["cmd_mag_rad_s"].append(math.hypot(yaw_cmd, pitch_cmd))
    return result


def _status_counts(control_events: Sequence[Event]) -> dict[str, Counter[str]]:
    by_axis: dict[str, Counter[str]] = {}
    for event in control_events:
        mpc = event.payload.get("mpc")
        if not isinstance(mpc, Mapping):
            continue
        for axis, diag in mpc.items():
            if not isinstance(diag, Mapping):
                continue
            status = str(diag.get("status", "unknown"))
            by_axis.setdefault(str(axis), Counter())[status] += 1
    return by_axis


def _load_rate_limits(config_path: Optional[str], config_extra: str) -> Optional[tuple[float, float]]:
    if not config_path:
        return None
    cfg = load_merged_config(expand_config_paths(config_path, config_extra))
    control = cfg.get("control", {})
    if not isinstance(control, Mapping):
        return None
    controller = str(control.get("controller", "pid")).strip().lower()
    if controller == "mpc":
        mpc = control.get("mpc", {})
        constraints = mpc.get("constraints", {}) if isinstance(mpc, Mapping) else {}
        if isinstance(constraints, Mapping):
            u_min = _float(constraints.get("u_min"))
            u_max = _float(constraints.get("u_max"))
            candidates = [abs(v) for v in (u_min, u_max) if v is not None]
            if candidates:
                limit = max(candidates)
                return limit, limit

    pid = control.get("pid", control)
    if isinstance(pid, Mapping):
        limits = pid.get("rate_limits")
        if isinstance(limits, Mapping):
            yaw = _float(limits.get("yaw"))
            pitch = _float(limits.get("pitch"))
            if yaw is not None and pitch is not None:
                return abs(yaw), abs(pitch)
    return None


def _saturation_counts(
    control_events: Sequence[Event],
    rate_limits: Optional[tuple[float, float]],
) -> Optional[dict[str, int]]:
    if rate_limits is None:
        return None
    yaw_limit, pitch_limit = rate_limits
    eps = 1e-6
    yaw_count = 0
    pitch_count = 0
    for event in control_events:
        yaw_cmd = _float(event.payload.get("pan_rate_cmd"))
        pitch_cmd = _float(event.payload.get("tilt_rate_cmd"))
        if yaw_cmd is not None and yaw_limit > 0.0 and abs(yaw_cmd) >= yaw_limit - eps:
            yaw_count += 1
        if pitch_cmd is not None and pitch_limit > 0.0 and abs(pitch_cmd) >= pitch_limit - eps:
            pitch_count += 1
    return {"yaw": yaw_count, "pitch": pitch_count}


def _settling_times(
    control_events: Sequence[Event],
    *,
    threshold_rad: float,
    hold_s: float,
    segment_gap_s: float,
) -> list[float]:
    if threshold_rad <= 0.0 or hold_s < 0.0 or not control_events:
        return []

    samples: list[tuple[float, float]] = []
    for event in control_events:
        if not bool(event.payload.get("target_ok")):
            continue
        err_rad = _pair(event.payload.get("err_rad"))
        if err_rad is None:
            continue
        samples.append((event.rx_s, math.hypot(err_rad[0], err_rad[1])))
    if not samples:
        return []

    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for sample in samples:
        if current and sample[0] - current[-1][0] > segment_gap_s:
            segments.append(current)
            current = []
        current.append(sample)
    if current:
        segments.append(current)

    settling: list[float] = []
    for segment in segments:
        if len(segment) < 2:
            continue
        start_t = segment[0][0]
        for idx, (candidate_t, candidate_err) in enumerate(segment):
            if candidate_err > threshold_rad:
                continue
            hold_until = candidate_t + hold_s
            window = [err for t, err in segment[idx:] if t <= hold_until]
            if not window:
                continue
            has_hold_span = segment[-1][0] >= hold_until or hold_s == 0.0
            if has_hold_span and max(window) <= threshold_rad:
                settling.append((candidate_t - start_t) * 1000.0)
                break
    return settling


def _format_metric(value: Optional[float], *, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{digits}f}{suffix}"


def _print_summary(summary: Mapping[str, Any]) -> None:
    print(f"Trace: {summary['trace']}")
    print(
        "Span: %s | events: %s | decode_errors=%d"
        % (
            _format_metric(summary["span_s"], unit="s", digits=2),
            ", ".join(f"{k}={v}" for k, v in summary["counts"].items()),
            summary["decode_errors"],
        )
    )

    print("\nStream rates")
    for name, stream_summary in summary["streams"].items():
        gaps = stream_summary["gap_ms"]
        print(
            "  %-9s rate=%s gap_p50=%s gap_p95=%s max_gap=%s"
            % (
                name,
                _format_metric(stream_summary["rate_hz"], unit="Hz", digits=2),
                _format_metric(gaps["median"], unit="ms", digits=1),
                _format_metric(gaps["p95"], unit="ms", digits=1),
                _format_metric(gaps["max"], unit="ms", digits=1),
            )
        )

    tracking = summary["tracking"]
    print("\nTracking")
    print(
        "  target_ok=%d/%d (%.1f%%)"
        % (
            tracking["target_ok_count"],
            tracking["control_count"],
            tracking["target_ok_fraction"] * 100.0,
        )
    )
    for name, axis_summary in tracking["error"].items():
        print(
            "  %-13s mean_abs=%s rms=%s p95_abs=%s max_abs=%s"
            % (
                name,
                _format_metric(axis_summary["mean"], digits=5),
                _format_metric(axis_summary["rms"], digits=5),
                _format_metric(axis_summary["p95"], digits=5),
                _format_metric(axis_summary["max"], digits=5),
            )
        )

    print("\nCommands")
    for name, cmd_summary in summary["commands"].items():
        print(
            "  %-15s mean_abs=%s rms=%s p95_abs=%s max_abs=%s"
            % (
                name,
                _format_metric(cmd_summary["mean"], digits=5),
                _format_metric(cmd_summary["rms"], digits=5),
                _format_metric(cmd_summary["p95"], digits=5),
                _format_metric(cmd_summary["max"], digits=5),
            )
        )
    saturation = summary.get("saturation")
    if saturation is not None:
        print(f"  saturation hits yaw={saturation['yaw']} pitch={saturation['pitch']}")

    print("\nTiming")
    for label, values in summary["timing"].items():
        print(
            "  %-35s p50=%s p95=%s max=%s n=%d"
            % (
                label,
                _format_metric(values["median"], unit="ms", digits=2),
                _format_metric(values["p95"], unit="ms", digits=2),
                _format_metric(values["max"], unit="ms", digits=2),
                values["count"],
            )
        )

    settling = summary["settling_ms"]
    print(
        "\nSettling: n=%d median=%s p95=%s threshold=%s hold=%s"
        % (
            settling["count"],
            _format_metric(settling["median"], unit="ms", digits=1),
            _format_metric(settling["p95"], unit="ms", digits=1),
            _format_metric(summary["settling_threshold_rad"], unit="rad", digits=4),
            _format_metric(summary["settling_hold_s"], unit="s", digits=2),
        )
    )

    mpc_status = summary.get("mpc_status", {})
    if mpc_status:
        print("\nMPC status")
        for axis, counts in mpc_status.items():
            print("  %-5s %s" % (axis, ", ".join(f"{k}={v}" for k, v in counts.items())))

    print(
        "\nNote: PC src_ts_ms and Jetson monotonic timestamps are not treated as a shared clock."
    )


def _build_summary(
    trace_path: Path,
    events: Sequence[Event],
    *,
    decode_errors: int,
    rate_limits: Optional[tuple[float, float]],
    settle_threshold_rad: float,
    settle_hold_s: float,
    segment_gap_s: float,
) -> dict[str, Any]:
    detection_events = _stream_events(events, "detection")
    control_events = _stream_events(events, "control")
    camstate_events = _stream_events(events, "camstate")
    by_stream = {
        "detection": detection_events,
        "control": control_events,
        "camstate": camstate_events,
    }
    counts = {name: len(stream_events) for name, stream_events in by_stream.items()}

    span_s = None
    if len(events) >= 2:
        span_s = (events[-1].rx_monotonic_ns - events[0].rx_monotonic_ns) / 1e9

    streams_summary = {}
    for name, stream_events in by_stream.items():
        streams_summary[name] = {
            "rate_hz": _stream_rate_hz(stream_events),
            "gap_ms": _summary(_stream_gap_ms(stream_events)),
        }

    vectors = _control_vectors(control_events)
    target_ok_count = sum(1 for event in control_events if bool(event.payload.get("target_ok")))
    control_count = len(control_events)
    target_ok_fraction = target_ok_count / control_count if control_count else 0.0

    detections_by_frame = _frame_map(detection_events)
    controls_by_frame = _frame_map(control_events)

    timing = {
        "jetson_infer_minus_rx": _summary(
            _same_host_delta_ms(
                detection_events,
                later_key="infer_ts_ms",
                earlier_key="rx_ts_ms",
            )
        ),
        "control_cmd_minus_infer": _summary(
            _matched_delta_ms(
                control_events,
                detections_by_frame,
                later_key="cmd_ts_ms",
                earlier_key="infer_ts_ms",
            )
        ),
        "recorder_control_after_detection": _summary(
            _matched_receive_delta_ms(control_events, detections_by_frame)
        ),
        "recorder_camstate_after_control": _summary(
            _matched_receive_delta_ms(camstate_events, controls_by_frame)
        ),
    }

    settling_values = _settling_times(
        control_events,
        threshold_rad=settle_threshold_rad,
        hold_s=settle_hold_s,
        segment_gap_s=segment_gap_s,
    )

    mpc_status = {
        axis: dict(counter)
        for axis, counter in sorted(_status_counts(control_events).items())
    }

    return {
        "trace": str(trace_path),
        "span_s": span_s,
        "counts": counts,
        "decode_errors": decode_errors,
        "streams": streams_summary,
        "tracking": {
            "control_count": control_count,
            "target_ok_count": target_ok_count,
            "target_ok_fraction": target_ok_fraction,
            "error": {
                "yaw_rad": _abs_summary(vectors["err_yaw_rad"]),
                "pitch_rad": _abs_summary(vectors["err_pitch_rad"]),
                "mag_rad": _summary(vectors["err_mag_rad"]),
                "u_px": _abs_summary(vectors["err_u_px"]),
                "v_px": _abs_summary(vectors["err_v_px"]),
            },
        },
        "commands": {
            "yaw_rad_s": _abs_summary(vectors["cmd_yaw_rad_s"]),
            "pitch_rad_s": _abs_summary(vectors["cmd_pitch_rad_s"]),
            "mag_rad_s": _summary(vectors["cmd_mag_rad_s"]),
        },
        "saturation": _saturation_counts(control_events, rate_limits),
        "timing": timing,
        "settling_ms": _summary(settling_values),
        "settling_threshold_rad": settle_threshold_rad,
        "settling_hold_s": settle_hold_s,
        "mpc_status": mpc_status,
    }


def _relative_times(events: Sequence[Event], start_ns: int) -> list[float]:
    return [(event.rx_monotonic_ns - start_ns) / 1e9 for event in events]


def _nan_if_none(value: Optional[float]) -> float:
    return math.nan if value is None else value


def _plot(events: Sequence[Event], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit("matplotlib is required for --plot") from exc

    control_events = _stream_events(events, "control")
    camstate_events = _stream_events(events, "camstate")
    if not control_events and not camstate_events:
        raise SystemExit("trace has no control or camstate events to plot")

    start_ns = min(event.rx_monotonic_ns for event in events)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    if control_events:
        t_ctrl = _relative_times(control_events, start_ns)
        err_yaw = []
        err_pitch = []
        cmd_yaw = []
        cmd_pitch = []
        target_ok = []
        for event in control_events:
            err_rad = _pair(event.payload.get("err_rad")) or (math.nan, math.nan)
            err_yaw.append(err_rad[0])
            err_pitch.append(err_rad[1])
            cmd_yaw.append(_nan_if_none(_float(event.payload.get("pan_rate_cmd"))))
            cmd_pitch.append(_nan_if_none(_float(event.payload.get("tilt_rate_cmd"))))
            target_ok.append(1.0 if bool(event.payload.get("target_ok")) else 0.0)
        axes[0].plot(t_ctrl, err_yaw, label="yaw err")
        axes[0].plot(t_ctrl, err_pitch, label="pitch err")
        axes[1].plot(t_ctrl, cmd_yaw, label="yaw cmd")
        axes[1].plot(t_ctrl, cmd_pitch, label="pitch cmd")
        axes[2].step(t_ctrl, target_ok, where="post", label="target_ok")

    if camstate_events:
        t_cam = _relative_times(camstate_events, start_ns)
        pan = [_nan_if_none(_float(event.payload.get("pan"))) for event in camstate_events]
        tilt = [_nan_if_none(_float(event.payload.get("tilt"))) for event in camstate_events]
        axes[2].plot(t_cam, pan, label="pan")
        axes[2].plot(t_cam, tilt, label="tilt")

    axes[0].set_ylabel("error rad")
    axes[1].set_ylabel("cmd rad/s")
    axes[2].set_ylabel("state / lock")
    axes[2].set_xlabel("recorder time s")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    if args.settle_threshold_rad <= 0.0:
        raise SystemExit("--settle-threshold-rad must be > 0")
    if args.settle_hold_s < 0.0:
        raise SystemExit("--settle-hold-s must be >= 0")
    if args.segment_gap_s <= 0.0:
        raise SystemExit("--segment-gap-s must be > 0")

    trace_path = Path(args.trace)
    events, decode_errors, _meta = _load_events(trace_path)
    if not events:
        raise SystemExit(f"{trace_path} contains no trace events")

    rate_limits = _load_rate_limits(args.config, args.config_extra)
    summary = _build_summary(
        trace_path,
        events,
        decode_errors=decode_errors,
        rate_limits=rate_limits,
        settle_threshold_rad=args.settle_threshold_rad,
        settle_hold_s=args.settle_hold_s,
        segment_gap_s=args.segment_gap_s,
    )

    _print_summary(summary)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote summary JSON: {summary_path}")

    if args.plot:
        plot_path = Path(args.plot)
        _plot(events, plot_path)
        print(f"Wrote plot: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
