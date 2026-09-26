"""Read-only V2 subscriber for an opt-in RTP frame-identity canary."""

from __future__ import annotations

import argparse
import json
import time

import zmq

from common.perception import perception_snapshot_from_json


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--duration-s", type=float, default=10.0)
    args = parser.parse_args()
    if not 0 < args.duration_s <= 60:
        parser.error("duration must be in (0, 60]")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(args.endpoint)
    counts = {
        "snapshots": 0,
        "verified": 0,
        "unverified": 0,
        "nonmonotonic_frame_ids": 0,
        "source_submillisecond_timestamps": 0,
        "invalid_payloads": 0,
    }
    first_id = last_id = None
    deadline = time.monotonic() + args.duration_s
    try:
        while time.monotonic() < deadline:
            if not socket.poll(100):
                continue
            try:
                snapshot = perception_snapshot_from_json(socket.recv())
            except (ValueError, TypeError):
                counts["invalid_payloads"] += 1
                continue
            frame = snapshot.frame
            counts["snapshots"] += 1
            counts["verified" if frame.source_identity_verified is True else "unverified"] += 1
            if frame.source_time_ns % 1_000_000:
                counts["source_submillisecond_timestamps"] += 1
            if last_id is not None and frame.frame_id <= last_id:
                counts["nonmonotonic_frame_ids"] += 1
            first_id = frame.frame_id if first_id is None else first_id
            last_id = frame.frame_id
    finally:
        socket.close(0)
        context.term()
    print(json.dumps({**counts, "first_frame_id": first_id, "last_frame_id": last_id}, sort_keys=True))
    return 0 if (
        counts["snapshots"] > 0
        and counts["verified"] == counts["snapshots"]
        and counts["source_submillisecond_timestamps"] > 0
        and not counts["nonmonotonic_frame_ids"]
        and not counts["invalid_payloads"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(run())
