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
import contextlib
import logging
import math
import signal
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Optional

import smbus
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.gimbal import GimbalInterface, MksServo42Axis, PitchAxisGroup, RS485Bus

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
        "--config",
        default=None,
        help="Optional YAML config to honor gimbal.auto_control_enabled (default: manual mode)",
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="RS485 serial port")
    parser.add_argument("--baud", default=115200, type=int, help="RS485 baudrate")
    parser.add_argument("--timeout", default=0.05, type=float, help="Serial timeout (s)")
    parser.add_argument("--retries", default=1, type=int, help="Command retry count")

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
    return parser


def install_stop_event() -> Event:
    stop_event = Event()

    def _handler(signum, _frame):
        stop_event.set()
        return None

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return stop_event


def _start_encoder_poller(
    gimbal: GimbalInterface,
    stop_event: Event,
    *,
    feedback_hz: float,
    log: logging.Logger,
) -> Optional[Thread]:
    period = 1.0 / max(feedback_hz, 0.1)
    log.info("Starting encoder poller at %.1f Hz (local confirmation only)", 1.0 / period)

    def _run() -> None:
        frame_id = 0
        while not stop_event.is_set():
            try:
                sample = gimbal.sample_state()
                log.debug(
                    "Encoder poll %d: pan=%.4f rad tilt=%.4f rad pan_rate=%s tilt_rate=%s",
                    frame_id,
                    float(sample.pan_rad),
                    float(sample.tilt_rad),
                    sample.pan_rate_rad_s,
                    sample.tilt_rate_rad_s,
                )
                frame_id += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("Encoder polling failed; continuing: %s", exc)
            stop_event.wait(period)

    thread = Thread(target=_run, name="encoder_poller", daemon=True)
    thread.start()
    return thread


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("rpi.manual_control")

    stop_event = install_stop_event()
    encoder_thread: Optional[Thread] = None
    adc_bus = smbus.SMBus(1)
    yaw_axis: Optional[MksServo42Axis] = None
    pitch_axis: Optional[PitchAxisGroup] = None
    gimbal: Optional[GimbalInterface] = None
    feedback_hz = 20.0

    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            gimbal_cfg = cfg.get("gimbal") or {}
            try:
                feedback_hz = float(gimbal_cfg.get("feedback_hz", feedback_hz))
            except Exception:  # noqa: BLE001
                feedback_hz = 20.0
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

    try:
        with RS485Bus(
            args.port,
            baudrate=args.baud,
            timeout=args.timeout,
            max_retries=max(args.retries, 0),
        ) as serial_bus:
            log.info("Opened RS485 bus on %s @ %d", args.port, args.baud)
            yaw_axis = MksServo42Axis(
                serial_bus,
                args.yaw_addr,
                group_addr=args.yaw_group_addr,
                counts_per_rev=args.counts_per_rev,
                gear_ratio=args.yaw_gear_ratio,
                use_group_writes=not args.no_group_writes,
            )
            pitch_axis = PitchAxisGroup(
                serial_bus,
                args.pitch_group_addr,
                motor_a=MksServo42Axis(
                    serial_bus,
                    args.pitch_motor_a_addr,
                    group_addr=args.pitch_group_addr,
                    counts_per_rev=args.counts_per_rev,
                    gear_ratio=args.pitch_gear_ratio,
                    use_group_writes=not args.no_group_writes,
                ),
                motor_b=MksServo42Axis(
                    serial_bus,
                    args.pitch_motor_b_addr,
                    group_addr=args.pitch_group_addr,
                    counts_per_rev=args.counts_per_rev,
                    gear_ratio=args.pitch_gear_ratio,
                    use_group_writes=not args.no_group_writes,
                ),
                authority=args.pitch_authority,
            )
            gimbal = GimbalInterface(
                yaw_axis,
                pitch_axis,
                max_rate_rad_s=args.max_rate_rad_s,
                yaw_accel_byte=args.yaw_accel_byte,
                pitch_accel_byte=args.pitch_accel_byte,
            )

            encoder_thread = _start_encoder_poller(
                gimbal,
                stop_event,
                feedback_hz=feedback_hz,
                log=log,
            )

            yaw_axis.enable(True)
            pitch_axis.enable(True)
            log.info(
                "Joystick control active (yaw addr=%d, pitch group=%s a=%d b=%d). Press Ctrl+C to stop.",
                yaw_axis.addr,
                pitch_axis.group_addr,
                pitch_axis.motor_a.addr,
                pitch_axis.motor_b.addr,
            )

            last_log = 0.0
            while not stop_event.is_set():
                joy_x = read_adc(adc_bus, 0)
                joy_y = read_adc(adc_bus, 1)

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

                gimbal.apply_rate_commands(yaw_rate, pitch_rate)

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
        if gimbal is not None:
            with contextlib.suppress(Exception):
                gimbal.stop()
        if pitch_axis is not None and yaw_axis is not None:
            with contextlib.suppress(Exception):
                pitch_axis.enable(False)
                yaw_axis.enable(False)
        stop_event.set()
        if encoder_thread is not None:
            encoder_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
