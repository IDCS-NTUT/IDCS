#!/usr/bin/env python3
"""Publish frame header side-channel messages for Jetson ingestion.

This mirrors the minimal metadata that pc/streamer.py sends so Jetson can keep
DetectionMsg frame IDs and source timestamps coherent when video arrives from
an external RTP sender (e.g. Raspberry Pi libcamera pipeline).
"""

from __future__ import annotations

import argparse
import time

import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="tcp://192.168.55.1:5555",
        help="Jetson header_push endpoint (default: tcp://192.168.55.1:5555)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Nominal source FPS used to pace frame_id/src_ts_ms publication",
    )
    parser.add_argument(
        "--start-frame-id",
        type=int,
        default=0,
        help="Initial frame id before first publish",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fps = float(args.fps)
    if fps <= 0.0:
        raise SystemExit("--fps must be positive")

    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.SNDHWM, 1)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(args.endpoint)

    frame_id = int(args.start_frame_id)
    period_s = 1.0 / fps
    next_tick = time.monotonic()

    try:
        while True:
            now_ns = time.time_ns()
            frame_id += 1
            payload = {
                "origin": "rpi2",
                "frame_id": frame_id,
                "src_ts_ms": int(now_ns / 1_000_000),
            }
            try:
                push.send_json(payload, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            push.close(0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
