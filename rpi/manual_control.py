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
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from threading import Event
from typing import Mapping, Optional, Tuple

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
    AuthoritySafetyConfig,
    AuthoritySafetyTracker,
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


def _parse_authority_handoff(cfg: Mapping[str, object]) -> Mapping[str, object]:
    authority_cfg = cfg.get("authority_handoff") or {}
    if not isinstance(authority_cfg, Mapping):
        raise SystemExit("authority_handoff must be a mapping if provided")
    return authority_cfg


def _drain_control_frames(inbound: "queue.Queue[bytes]") -> list[bytes]:
    frames: list[bytes] = []
    while True:
        try:
            frames.append(inbound.get_nowait())
        except queue.Empty:
            return frames


def _wait_until(ts: float) -> None:
    sleep_for = ts - time.monotonic()
    if sleep_for > 0:
        time.sleep(sleep_for)


def _should_send_servo_commands(*, authority_enabled: bool, state: PiAuthorityState) -> bool:
    if not authority_enabled:
        return True
    return state == PiAuthorityState.ACTIVE


@dataclass(frozen=True)
class _SerialCommand:
    name: str
    args: Tuple[object, ...]


class _SerialWorker(threading.Thread):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        stop_event: Event,
        inbound: "queue.Queue[bytes]",
    ) -> None:
        super().__init__(daemon=True)
        self._args = args
        self._stop_event = stop_event
        self._inbound = inbound
        self._commands: "queue.Queue[Tuple[_SerialCommand, queue.Queue[Optional[Exception]]]]" = (
            queue.Queue()
        )
        self._ready = threading.Event()
        self._startup_error: Optional[Exception] = None

    def submit(self, command: _SerialCommand) -> Optional[Exception]:
        response: "queue.Queue[Optional[Exception]]" = queue.Queue(maxsize=1)
        self._commands.put((command, response))
        return response.get()

    def wait_ready(self) -> Optional[Exception]:
        self._ready.wait()
        return self._startup_error

    def run(self) -> None:
        try:
            with RS485Bus(
                self._args.port,
                baudrate=self._args.baud,
                timeout=self._args.timeout,
                max_retries=max(self._args.retries, 0),
            ) as serial_bus:
                yaw_axis = MksServo42Axis(
                    serial_bus,
                    self._args.yaw_addr,
                    group_addr=self._args.yaw_group_addr,
                    counts_per_rev=self._args.counts_per_rev,
                    gear_ratio=self._args.yaw_gear_ratio,
                    use_group_writes=not self._args.no_group_writes,
                    respond_on_writes=not self._args.no_respond_on_writes,
                )
                pitch_axis = PitchAxisGroup(
                    serial_bus,
                    self._args.pitch_group_addr,
                    motor_a=MksServo42Axis(
                        serial_bus,
                        self._args.pitch_motor_a_addr,
                        group_addr=self._args.pitch_group_addr,
                        counts_per_rev=self._args.counts_per_rev,
                        gear_ratio=self._args.pitch_gear_ratio,
                        use_group_writes=not self._args.no_group_writes,
                        respond_on_writes=not self._args.no_respond_on_writes,
                    ),
                    motor_b=MksServo42Axis(
                        serial_bus,
                        self._args.pitch_motor_b_addr,
                        group_addr=self._args.pitch_group_addr,
                        counts_per_rev=self._args.counts_per_rev,
                        gear_ratio=self._args.pitch_gear_ratio,
                        use_group_writes=not self._args.no_group_writes,
                        respond_on_writes=not self._args.no_respond_on_writes,
                    ),
                    authority=self._args.pitch_authority,
                )
                gimbal = GimbalInterface(
                    yaw_axis,
                    pitch_axis,
                    max_rate_rad_s=self._args.max_rate_rad_s,
                    yaw_accel_byte=self._args.yaw_accel_byte,
                    pitch_accel_byte=self._args.pitch_accel_byte,
                )

                self._ready.set()
                while not self._stop_event.is_set():
                    try:
                        command, response = self._commands.get(timeout=0.01)
                    except queue.Empty:
                        frame = serial_bus.read_frame_with_timeout(
                            expected_start=0xFA,
                            expected_data_len=PAYLOAD_LEN,
                            timeout_s=0.01,
                        )
                        if frame is not None:
                            self._inbound.put(frame)
                        continue

                    try:
                        if command.name == "enable_axes":
                            enable = bool(command.args[0])
                            yaw_axis.enable(enable)
                            pitch_axis.enable(enable)
                        elif command.name == "stop":
                            gimbal.stop()
                        elif command.name == "apply_rates":
                            yaw_rate, pitch_rate = command.args
                            gimbal.apply_rate_commands(float(yaw_rate), float(pitch_rate))
                        elif command.name == "send_reply":
                            reply = command.args[0]
                            serial_bus._serial.write(reply)
                            serial_bus._serial.flush()
                        else:
                            raise ValueError(f"unknown serial command {command.name!r}")
                    except Exception as exc:  # noqa: BLE001
                        response.put(exc)
                    else:
                        response.put(None)
        except Exception as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("rpi.manual_control")

    stop_event = install_stop_event()
    adc_bus = smbus.SMBus(1)
    state = PiAuthorityState.STANDBY
    axes_enabled = False
    pending_enable = False
    inbound: "queue.Queue[bytes]" = queue.Queue()

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

    authority_cfg = _parse_authority_handoff(cfg)
    authority_enabled = bool(authority_cfg.get("enabled", False))
    control_plane_cfg = authority_cfg.get("control_plane") or {}
    button_cfg = authority_cfg.get("button") or {}
    safety_cfg = authority_cfg.get("safety") or {}
    if not isinstance(control_plane_cfg, Mapping):
        raise SystemExit("authority_handoff.control_plane must be a mapping")
    if not isinstance(button_cfg, Mapping):
        raise SystemExit("authority_handoff.button must be a mapping")
    if not isinstance(safety_cfg, Mapping):
        raise SystemExit("authority_handoff.safety must be a mapping")

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
    safety = AuthoritySafetyConfig(
        min_active_s=float(safety_cfg.get("min_active_s", 0.5)),
        min_standby_s=float(safety_cfg.get("min_standby_s", 0.5)),
        max_missing_pings=int(safety_cfg.get("max_missing_pings", 3)),
        max_missing_replies=int(safety_cfg.get("max_missing_replies", 3)),
        peer_timeout_s=float(safety_cfg.get("peer_timeout_s", 2.0)),
    )
    safety.validate()
    safety_tracker = AuthoritySafetyTracker(config=safety)
    button = gpiozero.Button(button_pin) if (authority_enabled and GPIOZERO_AVAILABLE) else None
    if authority_enabled and button is None:
        log.warning("authority handoff enabled but gpiozero not available; staying in standby")

    worker = _SerialWorker(args=args, stop_event=stop_event, inbound=inbound)
    worker.start()
    startup_error = worker.wait_ready()
    if startup_error is not None:
        log.error("failed to initialize RS485 worker: %s", startup_error)
        return 1
    log.info("RS485 worker online for %s @ %d", args.port, args.baud)
    try:
        if not authority_enabled:
            err = worker.submit(_SerialCommand("enable_axes", (True,)))
            if err:
                raise err
            axes_enabled = True
            log.info(
                "Joystick control active (yaw addr=%d, pitch group=%s a=%d b=%d). Press Ctrl+C to stop.",
                args.yaw_addr,
                args.pitch_group_addr,
                args.pitch_motor_a_addr,
                args.pitch_motor_b_addr,
            )
        else:
            log.info(
                "Authority handoff enabled; starting in STANDBY. "
                "Waiting for takeover before enabling servos.",
            )

        last_log = 0.0
        safety_tracker.record_state_change(now=time.monotonic())
        last_ping_ts: Optional[float] = None
        last_timeout_log_ts = 0.0
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
                    for frame in _drain_control_frames(inbound):
                        try:
                            parsed = parse_control_frame(
                                frame,
                                expected_start=0xFA,
                                expected_addr=control_addr,
                                expected_func=control_func,
                            )
                        except ValueError as exc:
                            message = str(exc)
                            if "address mismatch" not in message and "function mismatch" not in message:
                                log.warning("invalid control-plane ping: %s", exc)
                            continue
                        safety_tracker.record_ping_received(now=now)
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
                        err = worker.submit(_SerialCommand("send_reply", (reply,)))
                        if err:
                            raise err
                        if takeover_pending:
                            takeover_pending = False
                            if safety_tracker.can_transition_active(now=now):
                                state = PiAuthorityState.ACTIVE
                                safety_tracker.record_state_change(now=now)
                                quiet_until_ts = now + schedule.reply_window_s + schedule.bus_quiet_s
                                pending_enable = True
                                log.info("takeover granted; entering ACTIVE mode")
                            else:
                                takeover_pending = True
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
                        err = worker.submit(_SerialCommand("send_reply", (ping,)))
                        if err:
                            raise err
                        last_ping_ts = now
                        quiet_until_ts = now + schedule.reply_window_s + schedule.bus_quiet_s
                        _wait_until(quiet_until_ts)
                        if return_pending:
                            return_pending = False
                            if safety_tracker.can_transition_standby(now=now):
                                err = worker.submit(_SerialCommand("stop", ()))
                                if err:
                                    raise err
                                if axes_enabled:
                                    err = worker.submit(_SerialCommand("enable_axes", (False,)))
                                    if err:
                                        raise err
                                    axes_enabled = False
                                state = PiAuthorityState.STANDBY
                                safety_tracker.record_state_change(now=now)
                                log.info("returned control; entering STANDBY mode")
                            else:
                                return_pending = True
                if state == PiAuthorityState.STANDBY and safety_tracker.peer_unresponsive(now=now):
                    if (now - last_timeout_log_ts) >= safety.peer_timeout_s:
                        last_timeout_log_ts = now
                        log.warning(
                            "Jetson ping timeout detected; staying in standby until button request"
                        )
            if pending_enable and now >= quiet_until_ts and _should_send_servo_commands(
                authority_enabled=authority_enabled,
                state=state,
            ):
                err = worker.submit(_SerialCommand("enable_axes", (True,)))
                if err:
                    raise err
                axes_enabled = True
                pending_enable = False
            try:
                joy_x = read_adc(adc_bus, 0)
                joy_y = read_adc(adc_bus, 1)
            except OSError as exc:
                log.warning(
                    "ADC read failed (%s). Stopping motors and waiting for recovery...",
                    exc,
                )
                if _should_send_servo_commands(
                    authority_enabled=authority_enabled,
                    state=state,
                ):
                    err = worker.submit(_SerialCommand("stop", ()))
                    if err:
                        raise err

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
                err = worker.submit(_SerialCommand("apply_rates", (yaw_rate, pitch_rate)))
                if err:
                    raise err

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
        stop_event.set()
        worker.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
