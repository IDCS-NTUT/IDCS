"""Raspberry Pi joystick → MKS RS485 manual gimbal controller.

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
from typing import Any, Sequence

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
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config to honor gimbal.auto_control_enabled (default: manual mode)",
    )
    parser.add_argument(
        "--serial-update-endpoint",
        default="tcp://127.0.0.1:5571",
        help="Serial I/O service update PUB endpoint",
    )
    parser.add_argument(
        "--serial-reply-endpoint",
        default="tcp://127.0.0.1:5572",
        help="Serial I/O service reply SUB endpoint",
    )
    parser.add_argument(
        "--serial-target",
        default="gimbal",
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
    parser.add_argument(
        "--zmq-settle-ms",
        default=300,
        type=int,
        help="Wait this long after ZMQ socket connect to avoid PUB/SUB startup drops",
    )
    parser.add_argument(
        "--status-retries",
        default=3,
        type=int,
        help="How many times to retry startup status query batch before failing",
    )
    parser.add_argument(
        "--status-timeout-s",
        default=2.0,
        type=float,
        help="Timeout per startup status query attempt",
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
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("rpi.manual_control")

    stop_event = install_stop_event()
    adc_bus = smbus.SMBus(1)
    reply_sub = SerialReplySubscriber(
        args.serial_reply_endpoint,
        topics=[f"serial.reply.{args.serial_target}"],
    )
    update_pub = SerialUpdatePublisher(args.serial_update_endpoint)

    if args.zmq_settle_ms > 0:
        settle_s = args.zmq_settle_ms / 1000.0
        log.info("Waiting %.3fs for ZMQ PUB/SUB subscriptions to settle...", settle_s)
        time.sleep(settle_s)

    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            gimbal_cfg = cfg.get("gimbal") or {}
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

    def _send_update(commands: Sequence[dict[str, Any]], *, retries: int = 0) -> bool:
        update = {
            "type": "SerialUpdate",
            "source": "rpi.manual_control",
            "target": args.serial_target,
            "fields": {},
            "commands": list(commands),
            "update_ts_ms": int(time.time() * 1000),
        }
        attempts = max(retries, 0) + 1
        cmd_ids = [str(cmd.get("cmd_id")) for cmd in commands]
        for attempt in range(1, attempts + 1):
            if update_pub.send_update(update):
                log.debug(
                    "published SerialUpdate attempt=%d/%d cmds=%s",
                    attempt,
                    attempts,
                    cmd_ids,
                )
                return True
            log.debug(
                "publish SerialUpdate failed attempt=%d/%d cmds=%s",
                attempt,
                attempts,
                cmd_ids,
            )
            if attempt < attempts:
                time.sleep(0.05)
        return False

    def _status_addr_list() -> list[int]:
        return [args.yaw_addr, args.pitch_motor_a_addr, args.pitch_motor_b_addr]

    try:
        if not _send_update(
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
        ):
            raise SystemExit("failed to publish startup enable commands")

        if not _send_update(
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
        ):
            raise SystemExit("failed to publish startup zero commands")

        status_commands = [
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

        expected_all = set(_status_addr_list())
        status_ok = False
        for attempt in range(1, max(args.status_retries, 1) + 1):
            if not _send_update(status_commands, retries=2):
                log.warning("failed to publish status query batch (attempt %d)", attempt)
                continue
            deadline = time.monotonic() + max(args.status_timeout_s, 0.1)
            expected = set(expected_all)
            seen_f1_addrs: list[int] = []
            while expected and time.monotonic() < deadline:
                replies = reply_sub.recv_nowait()
                if replies:
                    log.debug("status wait received %d reply message(s)", len(replies))
                for reply in replies:
                    func = reply.get("func")
                    addr = reply.get("addr")
                    log.debug(
                        "status wait reply cmd_id=%s func=%s addr=%s parsed=%s",
                        reply.get("cmd_id"),
                        func,
                        addr,
                        reply.get("reply", {}).get("parsed"),
                    )
                    if func != "F1":
                        continue
                    if isinstance(addr, int):
                        seen_f1_addrs.append(addr)
                    if addr in expected:
                        status = reply.get("reply", {}).get("parsed", {}).get("status")
                        if status in (None, 0):
                            raise SystemExit(f"status query failed for addr={addr}")
                        log.info("axis addr=%s status=%s", addr, status)
                        expected.remove(addr)
                time.sleep(0.01)
            if not expected:
                status_ok = True
                break
            log.warning(
                "status attempt %d/%d timed out for addr(s): %s (seen F1 addr(s): %s)",
                attempt,
                max(args.status_retries, 1),
                sorted(expected),
                sorted(set(seen_f1_addrs)),
            )
            time.sleep(0.1)

        if not status_ok:
            raise SystemExit(f"status query timed out for addr(s): {sorted(expected_all)}")

        _send_update(
            [
                {
                    "cmd_id": f"speed:hold:yaw:{time.time_ns()}",
                    "func": "F6",
                    "addr": args.yaw_addr,
                    "payload": MksServo42Axis._encode_speed_payload(
                        0.0, args.yaw_accel_byte, args.yaw_gear_ratio
                    ),
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "high",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": f"speed:hold:pitch:{time.time_ns()}",
                    "func": "F6",
                    "addr": args.pitch_group_addr,
                    "payload": MksServo42Axis._encode_speed_payload(
                        0.0, args.pitch_accel_byte, args.pitch_gear_ratio
                    ),
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "high",
                    "target": args.serial_target,
                },
            ],
            retries=1,
        )

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
