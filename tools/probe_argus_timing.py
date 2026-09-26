"""Bounded, control-free Jetson CSI camera buffer-timing survey.

Measures nvarguscamerasrc output-pad arrival against its GstBuffer PTS in the
same GStreamer clock. PTS is reported as such, not claimed to be exposure time.
No RTP, inference, controller, serial, or motor connection is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    return sorted(values)[math.ceil(percent * len(values)) - 1]


def choose_pts_age_ns(clock_ns: int, base_ns: int, pts_ns: int) -> tuple[str, int] | None:
    """Identify an unambiguous absolute-clock or running-time PTS mapping."""
    if min(clock_ns, base_ns, pts_ns) < 0:
        return None
    candidates = (
        ("absolute_clock", clock_ns - pts_ns),
        ("running_time", clock_ns - base_ns - pts_ns),
    )
    plausible = [(name, age) for name, age in candidates if 0 <= age <= 2_000_000_000]
    return plausible[0] if len(plausible) == 1 else None


def summarize(records: list[dict[str, int | str]], *, requested_duration_s: float) -> dict[str, object]:
    if not records:
        return {"frames": 0, "requested_duration_s": requested_duration_s}
    start_ns = int(records[0]["python_arrival_ns"])
    steady = [row for row in records if int(row["python_arrival_ns"]) - start_ns >= 2_000_000_000]
    ages_ms = [int(row["pts_to_pad_ns"]) / 1e6 for row in steady if "pts_to_pad_ns" in row]
    arrivals = [int(row["python_arrival_ns"]) for row in steady]
    intervals_ms = [(right - left) / 1e6 for left, right in zip(arrivals, arrivals[1:])]
    domains: dict[str, int] = {}
    for row in records:
        domain = str(row.get("pts_domain", "unresolved"))
        domains[domain] = domains.get(domain, 0) + 1
    span_s = (int(records[-1]["python_arrival_ns"]) - start_ns) / 1e9
    return {
        "frames": len(records),
        "requested_duration_s": requested_duration_s,
        "observed_span_s": span_s,
        "observed_fps": (len(records) - 1) / span_s if span_s > 0 else None,
        "steady_frames_after_2s": len(steady),
        "pts_domain_counts": domains,
        "pts_to_source_pad_p50_ms": statistics.median(ages_ms) if ages_ms else None,
        "pts_to_source_pad_p95_ms": _percentile(ages_ms, 0.95),
        "pts_to_source_pad_p99_ms": _percentile(ages_ms, 0.99),
        "pts_to_source_pad_max_ms": max(ages_ms) if ages_ms else None,
        "source_pad_interval_p50_ms": statistics.median(intervals_ms) if intervals_ms else None,
        "source_pad_interval_p99_ms": _percentile(intervals_ms, 0.99),
        "source_pad_interval_max_ms": max(intervals_ms) if intervals_ms else None,
        "physical_exposure_time_measured": False,
    }


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--sensor-mode", type=int, default=4)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--samples-jsonl", type=Path)
    args = parser.parse_args()
    if not 0 < args.duration_s <= 120 or args.sensor_id < 0 or args.sensor_mode < 0:
        parser.error("invalid bounded camera survey duration or sensor selection")
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("camera width, height, and fps must be positive")

    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst

    Gst.init(None)
    description = (
        f"nvarguscamerasrc name=camera sensor-id={args.sensor_id} "
        f"sensor-mode={args.sensor_mode} ! "
        f"video/x-raw(memory:NVMM),width={args.width},height={args.height},"
        f"framerate={args.fps}/1,format=NV12 ! fakesink sync=false async=false"
    )
    pipeline = Gst.parse_launch(description)
    source = pipeline.get_by_name("camera")
    if source is None or source.get_static_pad("src") is None:
        raise RuntimeError("Argus source pad unavailable")
    records: list[dict[str, int | str]] = []

    def on_buffer(_pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        python_ns = time.monotonic_ns()
        clock = pipeline.get_clock()
        if clock is None:
            return Gst.PadProbeReturn.OK
        clock_ns = int(clock.get_time())
        base_ns = int(pipeline.get_base_time())
        pts_ns = int(buffer.pts)
        row: dict[str, int | str] = {
            "frame_index": len(records),
            "python_arrival_ns": python_ns,
            "gst_clock_arrival_ns": clock_ns,
            "gst_base_ns": base_ns,
            "buffer_pts_ns": pts_ns,
            "buffer_duration_ns": int(buffer.duration),
        }
        if pts_ns != Gst.CLOCK_TIME_NONE:
            chosen = choose_pts_age_ns(clock_ns, base_ns, pts_ns)
            if chosen is not None:
                row["pts_domain"], row["pts_to_pad_ns"] = chosen
        records.append(row)
        return Gst.PadProbeReturn.OK

    source.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_buffer)
    loop = GLib.MainLoop()
    error_text = None

    def on_bus(_bus, message):
        nonlocal error_text
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            error_text = f"{error}: {debug}"
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            loop.quit()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus)
    GLib.timeout_add(round(args.duration_s * 1000), lambda: (loop.quit(), False)[1])
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
    if error_text is not None:
        raise RuntimeError(error_text)
    report = summarize(records, requested_duration_s=args.duration_s)
    report["sensor"] = {
        "id": args.sensor_id, "mode": args.sensor_mode,
        "width": args.width, "height": args.height, "fps": args.fps,
    }
    report["pipeline"] = description
    if args.samples_jsonl is not None:
        with args.samples_jsonl.open("w", encoding="utf-8") as output:
            for row in records:
                output.write(json.dumps(row, sort_keys=True) + "\n")
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if records and report["pts_domain_counts"].get("unresolved", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
