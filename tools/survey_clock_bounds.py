"""Read-only software-clock exchange survey; reports intervals, not 'true offset'."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict

import zmq

from jetson.control_v3.timing import ClockBounds


def summarize(samples: list[ClockBounds]) -> dict[str, object]:
    if not samples:
        raise ValueError("no valid clock exchanges")
    widths = [sample.offset_max_ns - sample.offset_min_ns for sample in samples]
    lower = max(sample.offset_min_ns for sample in samples)
    upper = min(sample.offset_max_ns for sample in samples)
    best = samples[min(range(len(samples)), key=widths.__getitem__)]
    return {
        "samples": len(samples),
        "observed_span_s": (
            samples[-1].observed_jetson_ns - samples[0].observed_jetson_ns
        ) / 1e9,
        "median_interval_width_ms": statistics.median(widths) / 1e6,
        "max_interval_width_ms": max(widths) / 1e6,
        "best_interval_width_ms": min(widths) / 1e6,
        "best_offset_interval_ns": [best.offset_min_ns, best.offset_max_ns],
        "all_intervals_intersect": lower <= upper,
        "intersection_ns": [lower, upper] if lower <= upper else None,
        "drift_bound_established": False,
        "hardware_timestamped": False,
    }


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--samples-jsonl", type=str)
    args = parser.parse_args()
    if not 0 < args.duration_s <= 600 or not 0 < args.rate_hz <= 200:
        parser.error("duration must be in (0, 600] and rate in (0, 200]")
    context = zmq.Context()
    samples: list[ClockBounds] = []
    records: list[dict[str, int]] = []
    attempted = 0
    deadline = time.monotonic() + args.duration_s
    next_at = time.monotonic()
    try:
        while time.monotonic() < deadline:
            attempted += 1
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 200)
            try:
                socket.connect(args.endpoint)
                sent = time.monotonic_ns()
                socket.send_json({"version": 1, "jetson_send_ns": sent})
                reply = socket.recv_json()
                received = time.monotonic_ns()
                if reply.get("version") == 1 and reply.get("jetson_send_ns") == sent:
                    sample = ClockBounds.from_exchange(
                        jetson_send_ns=sent,
                        pc_receive_ns=int(reply["pc_receive_ns"]),
                        pc_send_ns=int(reply["pc_send_ns"]),
                        jetson_receive_ns=received,
                    )
                    samples.append(sample)
                    records.append({
                        "jetson_send_ns": sent,
                        "pc_receive_ns": int(reply["pc_receive_ns"]),
                        "pc_send_ns": int(reply["pc_send_ns"]),
                        "jetson_receive_ns": received,
                        **asdict(sample),
                    })
            except (KeyError, TypeError, ValueError, zmq.ZMQError):
                pass
            finally:
                socket.close(0)
            next_at += 1.0 / args.rate_hz
            remaining = next_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        context.term()
    if args.samples_jsonl:
        with open(args.samples_jsonl, "w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")
    if not samples:
        print(json.dumps({"error": "no_valid_clock_exchanges"}))
        return 1
    report = summarize(samples)
    report["requested_duration_s"] = args.duration_s
    report["attempted_exchanges"] = attempted
    report["failed_exchanges"] = attempted - len(samples)
    report["valid_span_fraction"] = report["observed_span_s"] / args.duration_s
    report["survey_complete"] = report["valid_span_fraction"] >= 0.95
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["survey_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(run())
