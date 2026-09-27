"""Bounded synthetic camera-pose PUB for V3 shadow-only estimator studies.

This never opens serial or publishes motor intents. Do not use alongside a
gimbal bridge or for a hardware-motion claim.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import zmq

from common.schemas import CamState


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--amplitude-rad", type=float, default=0.04)
    parser.add_argument("--period-s", type=float, default=3.0)
    parser.add_argument("--ack-synthetic-shadow-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.ack_synthetic_shadow_only:
        parser.error("synthetic CamState requires --ack-synthetic-shadow-only")
    if not 0 < args.duration_s <= 30 or not 0 < args.amplitude_rad <= 0.05 or not 1 <= args.period_s <= 10:
        parser.error("synthetic pose bounds exceeded")
    if not args.bind.startswith("tcp://0.0.0.0:"):
        parser.error("bind endpoint must be explicit TCP wildcard")
    startup = {
        "synthetic_camera_pose": True, "motor_authority": False,
        "amplitude_rad": args.amplitude_rad, "period_s": args.period_s,
        "duration_s": args.duration_s, "bind": args.bind,
    }
    if args.check:
        print(json.dumps(startup, sort_keys=True))
        return 0
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(args.bind)
    start = time.monotonic()
    sequence = 0
    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= args.duration_s:
                break
            phase = 2 * math.pi * elapsed / args.period_s
            pan = args.amplitude_rad * math.sin(phase)
            rate = args.amplitude_rad * 2 * math.pi / args.period_s * math.cos(phase)
            sample_ns = time.monotonic_ns()
            state = CamState(
                frame_id=sequence, src_ts_ms=0,
                state_monotonic_ns=sample_ns,
                pan_sample_monotonic_ns=sample_ns, tilt_sample_monotonic_ns=sample_ns,
                pan=pan, tilt=0.0, pan_rate=rate, tilt_rate=0.0,
                home_pan=0.0, home_tilt=0.0,
            )
            pub.send_string(state.model_dump_json(exclude_none=True))
            sequence += 1
            time.sleep(max(0, start + sequence * 0.02 - time.monotonic()))
    finally:
        pub.close(0)
        context.term()
    print(json.dumps({**startup, "published": sequence}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
