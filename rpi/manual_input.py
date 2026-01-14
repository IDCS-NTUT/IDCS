"""Raspberry Pi joystick → SerialUpdate manual gimbal input client.

This script reuses the shared ``common.gimbal`` MKS driver so the Pi and Jetson
share identical serial framing, rate limits, and enable/stop behavior. It reads
an analog joystick via the PCF8591 ADC (I2C) and translates the two axes into
pan/tilt rate commands (rad/s) for a two-axis gimbal:

- Yaw: single motor (default addr=1)
- Pitch: dual motors on a shared group address (default group=0x50, A addr=2, B addr=3)

Pitch motors are expected to be configured in the driver menu with opposing
``Dir`` settings so a single group speed command moves them in mirrored
directions (matching the Jetson bridge assumption).
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path
from threading import Event

import smbus
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.serial_io import SerialUpdatePublisher, SerialReplySubscriber
from common.gimbal.mks_servo42_rs485 import MksServo42Axis

# ===============================
# PCF8591 joystick ADC
# ===============================
ADC_ADDR = 0x48
_DEFAULT_SERIAL_UPDATE = "tcp://127.0.0.1:5571"
_DEFAULT_SERIAL_REPLY = "tcp://127.0.0.1:5572"
_DEFAULT_SERIAL_TARGET = "gimbal"


def read_adc(bus: smbus.SMBus, ch: int) -> int:
    ctrl = 0x40 | ch
    bus.write_byte(ADC_ADDR, ctrl)
    bus.read_byte(ADC_ADDR)
    return bus.read_byte(ADC_ADDR)


def map_value_to_rate(value: int, *, deadzone: int, max_rad_s: float) -> float:
    """Convert 8-bit ADC value into signed rad/s with a deadzone."""

    center = 128
    diff = value - center
    if abs(diff) < deadzone:
        return 0.0
    scale = min(max(abs(diff) / 128.0, 0.0), 1.0)
    return math.copysign(scale * max_rad_s, diff)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config to honor gimbal.auto_control_enabled (default: manual mode)",
    )
    parser.add_argument(
        "--serial-update-endpoint",
        default=_DEFAULT_SERIAL_UPDATE,
        help="Serial I/O service update PUB endpoint",
    )
    parser.add_argument(
        "--serial-reply-endpoint",
        default=_DEFAULT_SERIAL_REPLY,
        help="Serial I/O service reply SUB endpoint",
    )
    parser.add_argument(
        "--serial-target",
        default=_DEFAULT_SERIAL_TARGET,
        help="Serial I/O target name for command routing",
    )

    parser.add_argument("--yaw-addr", default=1, type=int, help="Yaw motor slave address")
    parser.add_argument(
        "--yaw-group-addr",
        default=None,
        type=int,
        help="Optional yaw group address for broadcast writes",
    )
    parser.add_argument(
        "--pitch-group-addr",
        default=0x50,
        type=int,
        help="Group address for dual pitch motors (mirrored Dir configuration)",
    )
    parser.add_argument("--pitch-motor-a-addr", default=2, type=int, help="Pitch motor A address")
    parser.add_argument("--pitch-motor-b-addr", default=3, type=int, help="Pitch motor B address")
    parser.add_argument(
        "--pitch-authority",
        choices=["a", "b"],
        default="a",
        help="Which pitch motor supplies encoder feedback",
    )
    parser.add_argument("--counts-per-rev", default=0x4000, type=int, help="Encoder counts/rev")
    parser.add_argument("--yaw-gear-ratio", default=1.0, type=float, help="Yaw gear ratio")
    parser.add_argument("--pitch-gear-ratio", default=1.0, type=float, help="Pitch gear ratio")
    parser.add_argument(
        "--yaw-accel-byte", default=10, type=int, help="Acceleration byte for yaw (0-255)"
    )
    parser.add_argument(
        "--pitch-accel-byte", default=10, type=int, help="Acceleration byte for pitch (0-255)"
    )
    parser.add_argument(
        "--max-rate-rad-s",
        default=10.0,
        type=float,
        help="Clamp joystick output to this magnitude (rad/s)",
    )
    parser.add_argument(
        "--deadzone",
        default=8,
        type=int,
        help="Ignore joystick deltas smaller than this ADC count",
    )
    parser.add_argument(
        "--invert-yaw",
        action="store_true",
        help="Invert yaw joystick sign (useful if wiring orientation differs)",
    )
    parser.add_argument(
        "--invert-pitch",
        action="store_true",
        help="Invert pitch joystick sign (useful if wiring orientation differs)",
    )
    parser.add_argument(
        "--no-group-writes",
        action="store_true",
        help="Force individual writes instead of group addressing",
    )
    parser.add_argument(
        "--no-respond-on-writes",
        action="store_true",
        help="Skip waiting for write acknowledgements (set when motors have Respond=0)",
    )
    return parser


def install_stop_event() -> Event:
    stop_event = Event()

    def _handler(signum, _frame):
        stop_event.set()
        return None

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return stop_event


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("rpi.manual_input")

    stop_event = install_stop_event()
    adc_bus = smbus.SMBus(1)
    cfg = {}

    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            gimbal_cfg = cfg.get("gimbal") or {}
            net_cfg = cfg.get("net") or {}
            serial_target = gimbal_cfg.get("serial_target")
            if serial_target and args.serial_target == _DEFAULT_SERIAL_TARGET:
                args.serial_target = str(serial_target)

            update_endpoint = gimbal_cfg.get("serial_update_endpoint") or net_cfg.get(
                "zmq_serial_update"
            )
            if update_endpoint and args.serial_update_endpoint == _DEFAULT_SERIAL_UPDATE:
                args.serial_update_endpoint = str(update_endpoint)

            reply_endpoint = gimbal_cfg.get("serial_reply_endpoint") or net_cfg.get(
                "zmq_serial_reply"
            )
            if reply_endpoint and args.serial_reply_endpoint == _DEFAULT_SERIAL_REPLY:
                args.serial_reply_endpoint = str(reply_endpoint)

            if gimbal_cfg.get("auto_control_enabled", False):
                log.warning(
                    "auto_control_enabled=true in %s; Jetson/auto control expected. "
                    "Skipping manual joystick control.",
                    args.config,
                )
                return 0
        except FileNotFoundError:
            log.warning("config file %s not found; continuing with manual defaults", args.config)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to read config %s (%s); continuing with manual defaults", args.config, exc)

    reply_sub = SerialReplySubscriber(
        args.serial_reply_endpoint,
        topics=[f"serial.reply.{args.serial_target}"],
    )
    update_pub = SerialUpdatePublisher(args.serial_update_endpoint)

    def _send_update(commands: list[dict[str, object]]) -> None:
        update_pub.send_update(
            {
                "type": "SerialUpdate",
                "source": "rpi.manual_input",
                "target": args.serial_target,
                "fields": {},
                "commands": commands,
                "update_ts_ms": int(time.time() * 1000),
            }
        )

    def _status_addr_list() -> list[int]:
        return [args.yaw_addr, args.pitch_motor_a_addr, args.pitch_motor_b_addr]

    try:
        _send_update(
            [
                {
                    "cmd_id": "enable:yaw",
                    "func": "F3",
                    "addr": args.yaw_addr,
                    "payload": [0x01],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "enable:pitch",
                    "func": "F3",
                    "addr": args.pitch_group_addr,
                    "payload": [0x01],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
            ]
        )
        _send_update(
            [
                {
                    "cmd_id": "zero:yaw",
                    "func": "0x92",
                    "addr": args.yaw_addr,
                    "payload": [],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "high",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "zero:pitch",
                    "func": "0x92",
                    "addr": args.pitch_group_addr,
                    "payload": [],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "high",
                    "target": args.serial_target,
                },
            ]
        )
        _send_update(
            [
                {
                    "cmd_id": "status:yaw",
                    "func": "F1",
                    "addr": args.yaw_addr,
                    "payload": [],
                    "expect_reply": True,
                    "expected_len": 1,
                    "priority": "high",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "status:pitch_a",
                    "func": "F1",
                    "addr": args.pitch_motor_a_addr,
                    "payload": [],
                    "expect_reply": True,
                    "expected_len": 1,
                    "priority": "high",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "status:pitch_b",
                    "func": "F1",
                    "addr": args.pitch_motor_b_addr,
                    "payload": [],
                    "expect_reply": True,
                    "expected_len": 1,
                    "priority": "high",
                    "target": args.serial_target,
                },
            ]
        )
        deadline = time.monotonic() + 2.0
        expected = set(_status_addr_list())
        while expected and time.monotonic() < deadline:
            for reply in reply_sub.recv_nowait():
                if reply.get("func") != "F1":
                    continue
                addr = reply.get("addr")
                if addr in expected:
                    status = reply.get("reply", {}).get("parsed", {}).get("status")
                    if status in (None, 0):
                        raise SystemExit(f"status query failed for addr={addr}")
                    log.info("axis addr=%s status=%s", addr, status)
                    expected.remove(addr)
            time.sleep(0.01)
        if expected:
            raise SystemExit(f"status query timed out for addr(s): {sorted(expected)}")

        log.info(
            "Joystick control active (yaw addr=%d, pitch group=%s a=%d b=%d). Press Ctrl+C to stop.",
            args.yaw_addr,
            args.pitch_group_addr,
            args.pitch_motor_a_addr,
            args.pitch_motor_b_addr,
        )

        last_log = 0.0
        while not stop_event.is_set():
            try:
                joy_x = read_adc(adc_bus, 0)
                joy_y = read_adc(adc_bus, 1)
            except OSError as exc:
                log.warning(
                    "ADC read failed (%s). Stopping motors and waiting for recovery...",
                    exc,
                )
                _send_update(
                    [
                        {
                            "cmd_id": "stop:yaw",
                            "func": "F6",
                            "addr": args.yaw_addr,
                            "payload": MksServo42Axis._encode_speed_payload(
                                0.0, args.yaw_accel_byte, args.yaw_gear_ratio
                            ),
                            "expect_reply": False,
                            "expected_len": None,
                            "priority": "critical",
                            "target": args.serial_target,
                        },
                        {
                            "cmd_id": "stop:pitch",
                            "func": "F6",
                            "addr": args.pitch_group_addr,
                            "payload": MksServo42Axis._encode_speed_payload(
                                0.0, args.pitch_accel_byte, args.pitch_gear_ratio
                            ),
                            "expect_reply": False,
                            "expected_len": None,
                            "priority": "critical",
                            "target": args.serial_target,
                        },
                    ]
                )

                while not stop_event.is_set():
                    try:
                        joy_x = read_adc(adc_bus, 0)
                        joy_y = read_adc(adc_bus, 1)
                        log.info("ADC responsive again; resuming joystick control.")
                        break
                    except OSError:
                        time.sleep(0.5)
                        continue
                else:
                    break

            yaw_rate = map_value_to_rate(
                joy_x, deadzone=args.deadzone, max_rad_s=args.max_rate_rad_s
            )
            pitch_rate = map_value_to_rate(
                joy_y, deadzone=args.deadzone, max_rad_s=args.max_rate_rad_s
            )
            if args.invert_yaw:
                yaw_rate *= -1.0
            if args.invert_pitch:
                pitch_rate *= -1.0

            _send_update(
                [
                    {
                        "cmd_id": f"speed:yaw:{time.time_ns()}",
                        "func": "F6",
                        "addr": args.yaw_addr,
                        "payload": MksServo42Axis._encode_speed_payload(
                            yaw_rate, args.yaw_accel_byte, args.yaw_gear_ratio
                        ),
                        "expect_reply": False,
                        "expected_len": None,
                        "priority": "high",
                        "target": args.serial_target,
                    },
                    {
                        "cmd_id": f"speed:pitch:{time.time_ns()}",
                        "func": "F6",
                        "addr": args.pitch_group_addr,
                        "payload": MksServo42Axis._encode_speed_payload(
                            pitch_rate, args.pitch_accel_byte, args.pitch_gear_ratio
                        ),
                        "expect_reply": False,
                        "expected_len": None,
                        "priority": "high",
                        "target": args.serial_target,
                    },
                ]
            )

            now = time.time()
            if (now - last_log) >= 0.5:
                last_log = now
                log.info(
                    "joy raw yaw=%3d pitch=%3d | cmd yaw=%.3f rad/s pitch=%.3f rad/s",
                    joy_x,
                    joy_y,
                    yaw_rate,
                    pitch_rate,
                )

            time.sleep(0.05)
    except Exception as exc:  # noqa: BLE001
        log.error("Manual control failed: %s", exc)
        return 1
    finally:
        _send_update(
            [
                {
                    "cmd_id": "stop:yaw",
                    "func": "F6",
                    "addr": args.yaw_addr,
                    "payload": MksServo42Axis._encode_speed_payload(
                        0.0, args.yaw_accel_byte, args.yaw_gear_ratio
                    ),
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "stop:pitch",
                    "func": "F6",
                    "addr": args.pitch_group_addr,
                    "payload": MksServo42Axis._encode_speed_payload(
                        0.0, args.pitch_accel_byte, args.pitch_gear_ratio
                    ),
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "disable:yaw",
                    "func": "F3",
                    "addr": args.yaw_addr,
                    "payload": [0x00],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "disable:pitch",
                    "func": "F3",
                    "addr": args.pitch_group_addr,
                    "payload": [0x00],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
            ]
        )
        reply_sub.close()
        update_pub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
