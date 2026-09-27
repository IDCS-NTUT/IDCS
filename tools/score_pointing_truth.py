"""True pointing error of a running controller, from the simulator's truth.

When the controller consumes real detections (YOLO + tracker), its own
bearing error is only as good as those detections. This records the
simulator's exact truth snapshots for the same run and computes the bearing
error the controller *should* have seen, with the same observation code
(``ControlObservationAssembler``) and configuration the controller uses. One
value per truth frame; the first ``--skip-s`` (acquisition) is excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import zmq

from common.config import load_config_bundle, resolve_active_video_profile, resolve_config_paths
from common.control import ControlConfig, LaserMountConfig
from common.perception import perception_snapshot_from_json
from jetson.control_observation import ControlObservationAssembler


def bearing_errors(config: dict, payloads: list[bytes]) -> list[tuple[float, float]]:
    video, _ = resolve_active_video_profile(config)
    assembler = ControlObservationAssembler(
        ControlConfig.from_raw_config(config, (int(video["width"]), int(video["height"]))),
        laser_mount=LaserMountConfig.from_raw_config(config),
    )
    errors = []
    for payload in payloads:
        snapshot = perception_snapshot_from_json(payload)
        now = time.monotonic()
        assembler.update_perception_snapshot(snapshot, received_at=now)
        target = assembler.build(now=now).target
        if target.bearing_error_rad is not None:
            errors.append(tuple(target.bearing_error_rad))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--config-extra", required=True, help="the controller's config stack")
    parser.add_argument("--truth", default="tcp://127.0.0.1:5574")
    parser.add_argument("--duration-s", type=float, default=40.0)
    parser.add_argument("--skip-s", type=float, default=10.0)
    args = parser.parse_args()
    config = load_config_bundle(resolve_config_paths(args.config, args.config_extra)).mutable_copy()
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 10000)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(args.truth)
    start = time.monotonic()
    payloads = []
    while time.monotonic() - start < args.duration_s:
        if sub.poll(100):
            payload = sub.recv()
            if time.monotonic() - start >= args.skip_s:
                payloads.append(payload)
    sub.close(0)
    ctx.term()
    errors = bearing_errors(config, payloads)
    rms = (lambda values: math.sqrt(sum(v * v for v in values) / len(values)) * 1000) if errors else None
    print(json.dumps({"truth_frames": len(payloads), "scored": len(errors),
                      "yaw_rms_mrad": rms([e[0] for e in errors]) if errors else None,
                      "pitch_rms_mrad": rms([e[1] for e in errors]) if errors else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
