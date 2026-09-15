#!/usr/bin/env python3
"""Publish a bounded, synthetic moving-target V2/controller-input fixture."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import zmq

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.perception import (
    NormalizedBoxV2,
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
    TargetSelectionV2,
    perception_snapshot_to_json,
)
from common.schemas import CamState, ManualControlState


def _snapshot(sequence: int, timestamp_ns: int, elapsed_s: float) -> PerceptionSnapshotV2:
    width, height = 1280, 720
    box_w, box_h = 0.06, 0.10
    center_x = 0.5 + 0.18 * math.sin(2.0 * math.pi * 0.35 * elapsed_s)
    center_y = 0.5 + 0.12 * math.sin(2.0 * math.pi * 0.23 * elapsed_s + 0.4)
    return PerceptionSnapshotV2(
        sequence=sequence,
        frame=PerceptionFrameV2(
            frame_id=sequence,
            source_time_ns=timestamp_ns,
            observed_time_ns=timestamp_ns,
            source_clock_domain="fixture.monotonic",
            observation_clock_domain="fixture.monotonic",
            width=width,
            height=height,
        ),
        tracks=(PerceptionTrackV2(
            track_id=7,
            box=NormalizedBoxV2(
                x=center_x - box_w / 2.0,
                y=center_y - box_h / 2.0,
                w=box_w,
                h=box_h,
            ),
            class_id="synthetic_target",
            confidence=1.0,
            age_frames=sequence + 1,
            missed_frames=0,
        ),),
        selection=TargetSelectionV2(
            track_id=7,
            source_frame_id=sequence,
            applied_frame_id=sequence,
            selected_time_ns=timestamp_ns,
            selection_clock_domain="fixture.monotonic",
            policy="synthetic_guaranteed_selection",
        ),
    )


def _manual(timestamp_ms: int) -> ManualControlState:
    return ManualControlState(
        src_ts_ms=timestamp_ms,
        source="synthetic_fixture",
        active=False,
        emergency=False,
        control_cmd_enabled=True,
        joystick_raw=(0, 0),
        joystick_rate_cmd=(0.0, 0.0),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-bind", required=True)
    parser.add_argument("--camstate-bind", required=True)
    parser.add_argument("--manual-bind", required=True)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.duration_s <= 0.0 or args.fps <= 0.0:
        parser.error("--duration-s and --fps must be > 0")

    context = zmq.Context()
    publishers = []
    for endpoint in (args.snapshot_bind, args.camstate_bind, args.manual_bind):
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(endpoint)
        publishers.append(socket)
    # Give subscribers a bounded join interval before the first fixture frame.
    time.sleep(0.25)
    start = time.monotonic()
    period = 1.0 / args.fps
    next_frame = start
    frames = 0
    try:
        while True:
            now = time.monotonic()
            if now - start >= args.duration_s:
                break
            if now < next_frame:
                time.sleep(min(0.001, next_frame - now))
                continue
            timestamp_ns = int(now * 1_000_000_000)
            elapsed_s = now - start
            publishers[0].send_string(perception_snapshot_to_json(_snapshot(frames, timestamp_ns, elapsed_s)))
            yaw = 0.015 * math.sin(2.0 * math.pi * 0.20 * elapsed_s)
            pitch = 0.010 * math.sin(2.0 * math.pi * 0.17 * elapsed_s + 0.2)
            yaw_rate = 0.015 * 2.0 * math.pi * 0.20 * math.cos(2.0 * math.pi * 0.20 * elapsed_s)
            pitch_rate = 0.010 * 2.0 * math.pi * 0.17 * math.cos(2.0 * math.pi * 0.17 * elapsed_s + 0.2)
            publishers[1].send_string(CamState(
                frame_id=frames,
                src_ts_ms=timestamp_ns // 1_000_000,
                pan=yaw,
                tilt=pitch,
                pan_rate=yaw_rate,
                tilt_rate=pitch_rate,
            ).model_dump_json())
            publishers[2].send_string(_manual(timestamp_ns // 1_000_000).model_dump_json())
            frames += 1
            next_frame += period
            if now - next_frame >= period:
                next_frame = now + period
    finally:
        for socket in publishers:
            socket.close(0)
        context.term()
    print(json.dumps({
        "mode": "synthetic_control_shadow_fixture",
        "frames": frames,
        "physical_control_disabled": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
