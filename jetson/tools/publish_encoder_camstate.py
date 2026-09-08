#!/usr/bin/env python3
"""Publish read-only MKS encoder telemetry as CamState; never commands motors."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import zmq

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from common.config_sync import expand_config_paths, load_merged_config
from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus
from common.schemas import CamState
from common.shutdown import install_signal_handlers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--config-extra", default="configs/control.yaml,configs/system.yaml")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--publish-hz", type=float, default=None)
    parser.add_argument("--port", default=None, help="read-only RS485 device override")
    parser.add_argument("--baud", type=int, default=None)
    args = parser.parse_args()
    cfg = load_merged_config(expand_config_paths(args.config, args.config_extra))
    gimbal = cfg.get("gimbal", {}) if isinstance(cfg.get("gimbal", {}), Mapping) else {}
    net = cfg.get("net", {}) if isinstance(cfg.get("net", {}), Mapping) else {}
    endpoint = args.endpoint or net.get("zmq_gimbal_state")
    if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
        raise SystemExit("a TCP CamState endpoint is required")
    hz = float(args.publish_hz or gimbal.get("feedback_hz", 35.0))
    if not math.isfinite(hz) or hz <= 0:
        raise SystemExit("publish rate must be finite and > 0")
    port = int(endpoint.rsplit(":", 1)[1])
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB); pub.setsockopt(zmq.LINGER, 0); pub.bind(f"tcp://0.0.0.0:{port}")
    stop = install_signal_handlers()
    previous: tuple[float, float, float] | None = None
    frame = 0
    try:
        with RS485Bus(args.port or str(gimbal.get("serial_port", "/dev/ttyCH341USB0")),
                      int(args.baud or gimbal.get("baudrate", 38400)),
                      timeout=float(gimbal.get("timeout", 0.01)), max_retries=int(gimbal.get("retries", 1))) as bus:
            yaw = MksServo42Axis(bus, int(gimbal.get("yaw_addr", 1)), counts_per_rev=int(gimbal.get("counts_per_rev", 16384)), gear_ratio=float(gimbal.get("yaw_gear_ratio", 1.0)))
            pitch_a = MksServo42Axis(bus, int(gimbal.get("pitch_motor_a_addr", 2)), counts_per_rev=int(gimbal.get("counts_per_rev", 16384)), gear_ratio=float(gimbal.get("pitch_gear_ratio", 1.0)))
            pitch_b = MksServo42Axis(bus, int(gimbal.get("pitch_motor_b_addr", 3)), counts_per_rev=int(gimbal.get("counts_per_rev", 16384)), gear_ratio=float(gimbal.get("pitch_gear_ratio", 1.0)))
            period = 1.0 / hz
            while not stop.is_set():
                began = time.monotonic()
                pan = yaw.read_angle_rad() * float(gimbal.get("camstate_yaw_sign", 1.0))
                authority = pitch_a if str(gimbal.get("pitch_encoder_authority", "a")) == "a" else pitch_b
                tilt = authority.read_angle_rad() * float(gimbal.get("camstate_pitch_sign", 1.0))
                _secondary = (pitch_b if authority is pitch_a else pitch_a).read_angle_rad()
                pan_rate = tilt_rate = None
                if previous is not None:
                    old_t, old_pan, old_tilt = previous; dt = began - old_t
                    if dt > 0: pan_rate, tilt_rate = (pan - old_pan) / dt, (tilt - old_tilt) / dt
                previous = (began, pan, tilt)
                pub.send_json(CamState(frame_id=frame, src_ts_ms=int(began * 1000), pan=pan, tilt=tilt,
                                       pan_rate=pan_rate, tilt_rate=tilt_rate).model_dump(mode="json"))
                frame += 1
                time.sleep(max(0.0, period - (time.monotonic() - began)))
    finally:
        pub.close(0); ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
