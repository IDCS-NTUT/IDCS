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
import importlib.util
import logging
import math
import signal
import sys
import time
from pathlib import Path
from threading import Event
from typing import Optional

import smbus
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.gimbal import (
    CONTROL_ADDR,
    CONTROL_FUNC,
    FLAG_RETURN,
    FLAG_TAKEOVER,
    PAYLOAD_LEN,
    ROLE_PI_ACTIVE,
    ControlPlaneFrame,
    GimbalInterface,
    HandshakeSchedule,
    MksServo42Axis,
    PiAuthorityState,
    PitchAxisGroup,
    RS485Bus,
    build_control_frame,
    next_ping_due,
    parse_control_frame,
)
from rpi.button_latch import ButtonLatch, ButtonLatchConfig

# ===============================
# PCF8591 joystick ADC
# ===============================
ADC_ADDR = 0x48
CONTROL_FRAME_LEN = 1 + 2 + PAYLOAD_LEN + 1
GPIOZERO_AVAILABLE = importlib.util.find_spec("gpiozero") is not None
if GPIOZERO_AVAILABLE:
    import gpiozero


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


def _try_read_control_frame(bus: RS485Bus) -> Optional[bytes]:
    serial_port = bus._serial
    previous_timeout = serial_port.timeout
    serial_port.timeout = 0
    try:
        start = serial_port.read(1)
        if not start:
            return None
        if start[0] != 0xFA:
            return None
        rest = serial_port.read(CONTROL_FRAME_LEN - 1)
        if len(rest) != (CONTROL_FRAME_LEN - 1):
            return None
        return bytes(start + rest)
    finally:
        serial_port.timeout = previous_timeout


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("rpi.manual_control")

    stop_event = install_stop_event()
    adc_bus = smbus.SMBus(1)
    yaw_axis: Optional[MksServo42Axis] = None
    pitch_axis: Optional[PitchAxisGroup] = None
    gimbal: Optional[GimbalInterface] = None

    cfg = {}
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

    authority_cfg = cfg.get("authority_handoff") or {}
    authority_enabled = bool(authority_cfg.get("enabled", False))
    control_plane_cfg = authority_cfg.get("control_plane") or {}
    button_cfg = authority_cfg.get("button") or {}
    schedule = HandshakeSchedule(
        ping_interval_s=float(control_plane_cfg.get("ping_interval_s", 0.5)),
        reply_window_s=float(control_plane_cfg.get("reply_window_s", 0.05)),
        bus_quiet_s=float(control_plane_cfg.get("bus_quiet_s", 0.0)),
    )
    schedule.validate()
    control_addr = int(control_plane_cfg.get("addr", CONTROL_ADDR))
    control_func = int(control_plane_cfg.get("func", CONTROL_FUNC))
    button_pin = int(button_cfg.get("gpio_pin", 17))
    button_latch = ButtonLatch(
        config=ButtonLatchConfig(
            debounce_s=float(button_cfg.get("debounce_s", 0.05)),
            cooldown_s=float(button_cfg.get("cooldown_s", 1.0)),
        )
    )
    button = gpiozero.Button(button_pin) if (authority_enabled and GPIOZERO_AVAILABLE) else None
    if authority_enabled and button is None:
        log.warning("authority handoff enabled but gpiozero not available; staying in standby")

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
                respond_on_writes=not args.no_respond_on_writes,
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
                    respond_on_writes=not args.no_respond_on_writes,
                ),
                motor_b=MksServo42Axis(
                    serial_bus,
                    args.pitch_motor_b_addr,
                    group_addr=args.pitch_group_addr,
                    counts_per_rev=args.counts_per_rev,
                    gear_ratio=args.pitch_gear_ratio,
                    use_group_writes=not args.no_group_writes,
                    respond_on_writes=not args.no_respond_on_writes,
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

            try:
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
                state = PiAuthorityState.STANDBY
                last_ping_ts: Optional[float] = None
                quiet_until_ts = 0.0
                ping_counter = 0
                takeover_pending = False
                return_pending = False
                while not stop_event.is_set():
                    now = time.monotonic()
                    pressed = bool(button.is_pressed) if button is not None else False
                    if button_latch.update(pressed=pressed, now=now):
                        if state == PiAuthorityState.STANDBY:
                            takeover_pending = True
                            log.info("takeover requested; waiting for next ping window")
                        elif state == PiAuthorityState.ACTIVE:
                            return_pending = True
                            log.info("return requested; sending return flag on next ping")
                    if authority_enabled:
                        if state == PiAuthorityState.STANDBY:
                            frame = _try_read_control_frame(serial_bus)
                            if frame is not None:
                                try:
                                    parsed = parse_control_frame(
                                        frame,
                                        expected_start=0xFA,
                                        expected_addr=control_addr,
                                        expected_func=control_func,
                                    )
                                except ValueError as exc:
                                    log.warning("invalid control-plane ping: %s", exc)
                                else:
                                    flags = FLAG_TAKEOVER if takeover_pending else 0
                                    reply = build_control_frame(
                                        ControlPlaneFrame(
                                            version=parsed.version,
                                            role=ROLE_PI_ACTIVE,
                                            flags=flags,
                                            counter=parsed.counter,
                                        ),
                                        start_byte=0xFB,
                                        addr=control_addr,
                                        func=control_func,
                                    )
                                    serial_bus._serial.write(reply)
                                    serial_bus._serial.flush()
                                    if takeover_pending:
                                        takeover_pending = False
                                        state = PiAuthorityState.ACTIVE
                                        quiet_until_ts = now + schedule.reply_window_s + schedule.bus_quiet_s
                                        log.info("takeover granted; entering ACTIVE mode")
                        elif state == PiAuthorityState.ACTIVE:
                            if next_ping_due(now=now, last_ping_ts=last_ping_ts, schedule=schedule):
                                flags = FLAG_RETURN if return_pending else 0
                                ping = build_control_frame(
                                    ControlPlaneFrame(
                                        version=0x01,
                                        role=ROLE_PI_ACTIVE,
                                        flags=flags,
                                        counter=ping_counter,
                                    ),
                                    start_byte=0xFA,
                                    addr=control_addr,
                                    func=control_func,
                                )
                                ping_counter = (ping_counter + 1) & 0xFFFF
                                serial_bus._serial.write(ping)
                                serial_bus._serial.flush()
                                last_ping_ts = now
                                quiet_until_ts = now + schedule.reply_window_s + schedule.bus_quiet_s
                                if return_pending:
                                    return_pending = False
                                    state = PiAuthorityState.STANDBY
                                    with contextlib.suppress(Exception):
                                        gimbal.stop()
                                    log.info("returned control; entering STANDBY mode")
                    try:
                        joy_x = read_adc(adc_bus, 0)
                        joy_y = read_adc(adc_bus, 1)
                    except OSError as exc:
                        log.warning(
                            "ADC read failed (%s). Stopping motors and waiting for recovery...",
                            exc,
                        )
                        with contextlib.suppress(Exception):
                            gimbal.stop()

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

                    can_send = True
                    if authority_enabled:
                        can_send = state == PiAuthorityState.ACTIVE and now >= quiet_until_ts
                    if can_send:
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
            finally:
                if gimbal is not None:
                    with contextlib.suppress(Exception):
                        gimbal.stop()
                if pitch_axis is not None and yaw_axis is not None:
                    with contextlib.suppress(Exception):
                        pitch_axis.enable(False)
                        yaw_axis.enable(False)
    except Exception as exc:  # noqa: BLE001
        log.error("Manual control failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
