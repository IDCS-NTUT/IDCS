"""Raspberry Pi joystick → MKS RS485 manual gimbal controller.

This script reuses the shared ``common.gimbal`` MKS driver so the Pi and Jetson
share identical serial framing, rate limits, and enable/stop behavior. It reads
an analog joystick via the PCF8591 ADC (I2C) and translates the two axes into
pan/tilt rate commands (rad/s) for a two-axis gimbal:

- Yaw: single motor (default addr=1)
- Pitch: dual motors with synchronized per-motor writes (default A addr=2, B addr=3)

Pitch mirroring is applied in software via per-motor command signs so opposing
motion does not depend on motor ``Dir`` menu configuration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any, Mapping, Sequence

import smbus
import yaml
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.serial_io import SerialUpdatePublisher, SerialReplySubscriber
from common.gimbal.mks_servo42_rs485 import MksServo42Axis

# ===============================
# PCF8591 joystick ADC
# ===============================
ADC_ADDR = 0x48


class ManualSwitchIO:
    """GPIO switch/emergency logic merged from manual_control_sw.

    Behavior:
    - S (active-low press) toggles ACTIVE state.
    - S2 (active-high level) forces emergency mode.
    - In emergency mode: L1 low, J low, OUT25 high.
    - In normal mode: L2/J follow ACTIVE, OUT25 follows S1 while ACTIVE.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        poll_dt: float,
        debounce_s: float,
        log: logging.Logger,
    ) -> None:
        self._enabled = enabled
        self._poll_dt = max(float(poll_dt), 0.001)
        self._debounce_s = max(float(debounce_s), 0.0)
        self._log = log

        self._gpio: Any | None = None
        self._ready = False

        self.S = 24
        self.S1 = 22
        self.S2 = 23

        self.L1 = 17
        self.L2 = 27
        self.J = 26
        self.OUT25 = 25

        self.active = False
        self.emergency = False
        self.saved_active = False
        self._prev_s_press: int | None = None
        self._last_s_ts = 0.0
        self._prev_in: dict[int, int] = {}
        self._last_out: dict[int, int] = {}

    @property
    def poll_dt(self) -> float:
        return self._poll_dt

    @property
    def ready(self) -> bool:
        return self._ready

    def setup(self) -> bool:
        if not self._enabled:
            self._log.info("switch GPIO integration disabled via CLI")
            return False
        try:
            gpio_mod = importlib.import_module("RPi.GPIO")
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                "RPi.GPIO unavailable (%s); continuing without switch GPIO integration",
                exc,
            )
            return False

        gpio = gpio_mod
        gpio.setwarnings(False)
        gpio.setmode(gpio.BCM)

        gpio.setup(self.S, gpio.IN, pull_up_down=gpio.PUD_UP)
        gpio.setup(self.S1, gpio.IN, pull_up_down=gpio.PUD_DOWN)
        gpio.setup(self.S2, gpio.IN, pull_up_down=gpio.PUD_DOWN)

        gpio.setup(self.L1, gpio.OUT, initial=gpio.HIGH)
        gpio.setup(self.L2, gpio.OUT, initial=gpio.HIGH)
        gpio.setup(self.J, gpio.OUT, initial=gpio.LOW)
        gpio.setup(self.OUT25, gpio.OUT, initial=gpio.HIGH)

        self._gpio = gpio
        self._ready = True
        self._prev_s_press = gpio.HIGH
        self._prev_in = {
            self.S: gpio.input(self.S),
            self.S1: gpio.input(self.S1),
            self.S2: gpio.input(self.S2),
        }
        self._last_out = {
            self.L1: gpio.HIGH,
            self.L2: gpio.HIGH,
            self.J: gpio.LOW,
            self.OUT25: gpio.HIGH,
        }
        self._apply_normal_outputs()
        self._log.info("switch GPIO integration enabled")
        return True

    def _set_out(self, pin: int, level: int) -> None:
        if not self._ready or self._gpio is None:
            return
        if self._last_out.get(pin) != level:
            self._gpio.output(pin, level)
            self._last_out[pin] = level
            self._log.info("GPIO output pin=%d level=%d", pin, level)

    def _apply_normal_outputs(self) -> None:
        if not self._ready or self._gpio is None:
            return
        gpio = self._gpio
        self._set_out(self.L1, gpio.HIGH)
        self._set_out(self.L2, gpio.LOW if self.active else gpio.HIGH)
        self._set_out(self.J, gpio.HIGH if self.active else gpio.LOW)
        if not self.active:
            self._set_out(self.OUT25, gpio.HIGH)
            return
        s1 = gpio.input(self.S1)
        self._set_out(self.OUT25, gpio.LOW if s1 == gpio.HIGH else gpio.HIGH)

    def _enter_emergency(self) -> None:
        if not self._ready or self._gpio is None:
            return
        if self.emergency:
            return
        gpio = self._gpio
        self.emergency = True
        self.saved_active = self.active
        self._log.warning("switch emergency triggered")
        self._set_out(self.L1, gpio.LOW)
        self._set_out(self.J, gpio.LOW)
        self._set_out(self.OUT25, gpio.HIGH)

    def _maintain_emergency(self) -> None:
        if not self._ready or self._gpio is None:
            return
        gpio = self._gpio
        self._set_out(self.L1, gpio.LOW)
        self._set_out(self.J, gpio.LOW)
        self._set_out(self.OUT25, gpio.HIGH)

    def _exit_emergency(self) -> None:
        if not self._ready:
            return
        self.emergency = False
        self.active = self.saved_active
        self._log.warning("switch emergency released")
        self._apply_normal_outputs()

    def update(self) -> dict[str, bool]:
        if not self._ready or self._gpio is None:
            return {
                "active": True,
                "active_changed": False,
                "emergency": False,
                "emergency_entered": False,
                "emergency_exited": False,
            }

        gpio = self._gpio
        active_before = self.active
        emergency_before = self.emergency

        s = gpio.input(self.S)
        s1 = gpio.input(self.S1)
        s2 = gpio.input(self.S2)
        for pin, value in ((self.S, s), (self.S1, s1), (self.S2, s2)):
            if value != self._prev_in.get(pin):
                self._log.info("GPIO input pin=%d changed -> %d", pin, value)
                self._prev_in[pin] = value

        if s2 == gpio.HIGH:
            self._enter_emergency()
        elif self.emergency:
            self._exit_emergency()

        if self.emergency:
            self._maintain_emergency()
            return {
                "active": self.active,
                "active_changed": self.active != active_before,
                "emergency": True,
                "emergency_entered": not emergency_before,
                "emergency_exited": False,
            }

        now = time.monotonic()
        if self._prev_s_press == gpio.HIGH and s == gpio.LOW:
            if now - self._last_s_ts >= self._debounce_s:
                self._last_s_ts = now
                self.active = not self.active
                self._log.info("switch ACTIVE -> %s", self.active)

        self._prev_s_press = s
        self._apply_normal_outputs()

        return {
            "active": self.active,
            "active_changed": self.active != active_before,
            "emergency": self.emergency,
            "emergency_entered": (not emergency_before) and self.emergency,
            "emergency_exited": emergency_before and (not self.emergency),
        }

    def cleanup(self) -> None:
        if not self._ready or self._gpio is None:
            return
        gpio = self._gpio
        try:
            gpio.output(self.L1, gpio.HIGH)
            gpio.output(self.L2, gpio.HIGH)
            gpio.output(self.J, gpio.LOW)
            gpio.output(self.OUT25, gpio.HIGH)
        finally:
            gpio.cleanup()


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
        "--serial-command-endpoint",
        default="tcp://127.0.0.1:5570",
        help="Serial I/O service command REP endpoint (used for readiness checks)",
    )
    parser.add_argument(
        "--serial-reply-endpoint",
        default="tcp://127.0.0.1:5572",
        help="Serial I/O service reply SUB endpoint",
    )
    parser.add_argument(
        "--serial-service-autostart",
        dest="serial_service_autostart",
        action="store_true",
        help="Auto-start tools.serial_io_service when command endpoint is not responding",
    )
    parser.add_argument(
        "--no-serial-service-autostart",
        dest="serial_service_autostart",
        action="store_false",
        help="Do not auto-start tools.serial_io_service",
    )
    parser.set_defaults(serial_service_autostart=True)
    parser.add_argument(
        "--serial-service-start-timeout-s",
        default=5.0,
        type=float,
        help="How long to wait for auto-started serial_io_service to become ready",
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
    parser.add_argument("--pitch-motor-a-addr", default=2, type=int, help="Pitch motor A address")
    parser.add_argument("--pitch-motor-b-addr", default=3, type=int, help="Pitch motor B address")
    parser.add_argument(
        "--pitch-motor-a-sign",
        default=1.0,
        type=float,
        help="Software command sign for pitch motor A (+1 or -1)",
    )
    parser.add_argument(
        "--pitch-motor-b-sign",
        default=-1.0,
        type=float,
        help="Software command sign for pitch motor B (+1 or -1)",
    )
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
    parser.add_argument(
        "--switch-io",
        dest="switch_io",
        action="store_true",
        help="Enable integrated RPi GPIO switch/emergency control (S/S1/S2/L1/L2/J/OUT25)",
    )
    parser.add_argument(
        "--no-switch-io",
        dest="switch_io",
        action="store_false",
        help="Disable integrated RPi GPIO switch/emergency control",
    )
    parser.set_defaults(switch_io=True)
    parser.add_argument(
        "--switch-poll-dt-s",
        default=0.005,
        type=float,
        help="GPIO switch polling interval in seconds",
    )
    parser.add_argument(
        "--switch-debounce-s",
        default=0.05,
        type=float,
        help="Debounce interval for S press toggle in seconds",
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


def _read_config(path: str | None) -> Mapping[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file {path} not found")
    with cfg_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"config file {path} must contain a mapping")
    return loaded


def _serial_service_responding(command_endpoint: str, *, timeout_s: float = 0.3) -> bool:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    timeout_ms = int(max(timeout_s, 0.05) * 1000)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.connect(command_endpoint)
    try:
        probe = {
            "type": "SerialServiceProbe",
            "probe_ts_ms": int(time.time() * 1000),
        }
        sock.send_string(json.dumps(probe))
        sock.recv_string()
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        sock.close(linger=0)


def _resolve_serial_service_args(
    cfg: Mapping[str, Any],
) -> tuple[str, int, float, int]:
    gimbal_cfg = cfg.get("gimbal") if isinstance(cfg, Mapping) else None
    if not isinstance(gimbal_cfg, Mapping):
        gimbal_cfg = {}
    port = str(gimbal_cfg.get("serial_port", "/dev/ttyUSB0"))
    baud = int(gimbal_cfg.get("baudrate", 115200))
    timeout = float(gimbal_cfg.get("timeout", 0.1))
    retries = int(gimbal_cfg.get("retries", 1))
    return port, baud, timeout, retries


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("rpi.manual_control")

    stop_event = install_stop_event()
    serial_service_proc: subprocess.Popen[Any] | None = None
    switch_io = ManualSwitchIO(
        enabled=args.switch_io,
        poll_dt=args.switch_poll_dt_s,
        debounce_s=args.switch_debounce_s,
        log=log,
    )

    loaded_cfg: Mapping[str, Any] = {}
    if args.config:
        try:
            loaded_cfg = _read_config(args.config)
        except FileNotFoundError:
            log.warning("config file %s not found; continuing with manual defaults", args.config)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "failed to read config %s (%s); continuing with manual defaults",
                args.config,
                exc,
            )

    if _serial_service_responding(args.serial_command_endpoint):
        log.info("Serial I/O service already responding at %s", args.serial_command_endpoint)
    elif args.serial_service_autostart:
        service_port, service_baud, service_timeout, service_retries = _resolve_serial_service_args(loaded_cfg)
        launch_cmd = [
            sys.executable,
            "-m",
            "tools.serial_io_service",
            "--command-endpoint",
            args.serial_command_endpoint,
            "--update-endpoint",
            args.serial_update_endpoint,
            "--reply-endpoint",
            args.serial_reply_endpoint,
            "--port",
            service_port,
            "--baud",
            str(service_baud),
            "--timeout",
            str(service_timeout),
            "--retries",
            str(service_retries),
        ]
        if args.config:
            launch_cmd.extend(["--config", args.config])
        if args.debug:
            launch_cmd.append("--debug")

        log.info("Starting serial I/O service: %s", " ".join(launch_cmd))
        launched_proc = subprocess.Popen(launch_cmd)
        serial_service_proc = launched_proc
        ready_deadline = time.monotonic() + max(args.serial_service_start_timeout_s, 0.5)
        while time.monotonic() < ready_deadline:
            if launched_proc.poll() is not None:
                raise SystemExit(
                    "serial_io_service exited during startup; run with --debug and check logs"
                )
            if _serial_service_responding(args.serial_command_endpoint):
                log.info("Serial I/O service became ready")
                break
            time.sleep(0.1)
        else:
            raise SystemExit(
                "serial_io_service did not become ready in time; "
                "check serial port permissions and endpoint settings"
            )
    else:
        log.warning(
            "Serial I/O service not responding at %s and auto-start disabled",
            args.serial_command_endpoint,
        )

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
            cfg = loaded_cfg or {}
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

    if args.pitch_motor_a_sign == 0.0 or args.pitch_motor_b_sign == 0.0:
        raise SystemExit("pitch motor signs must be non-zero")

    def _pitch_sync_enable_commands(enable: bool, *, priority: str) -> list[dict[str, Any]]:
        value = 0x01 if enable else 0x00
        return [
            {
                "cmd_id": f"sync:enable:pitch_a:{time.time_ns()}",
                "func": "0x4A",
                "addr": args.pitch_motor_a_addr,
                "payload": [value],
                "expect_reply": False,
                "expected_len": None,
                "priority": priority,
                "target": args.serial_target,
            },
            {
                "cmd_id": f"sync:enable:pitch_b:{time.time_ns()}",
                "func": "0x4A",
                "addr": args.pitch_motor_b_addr,
                "payload": [value],
                "expect_reply": False,
                "expected_len": None,
                "priority": priority,
                "target": args.serial_target,
            },
        ]

    def _pitch_sync_exec_command(*, priority: str) -> dict[str, Any]:
        return {
            "cmd_id": f"sync:exec:{time.time_ns()}",
            "func": "0x4B",
            "addr": 0x00,
            "payload": [],
            "expect_reply": False,
            "expected_len": None,
            "priority": priority,
            "target": args.serial_target,
        }

    def _pitch_speed_commands(rate_rad_s: float, *, cmd_prefix: str, priority: str) -> list[dict[str, Any]]:
        return [
            {
                "cmd_id": f"{cmd_prefix}:pitch_a:{time.time_ns()}",
                "func": "F6",
                "addr": args.pitch_motor_a_addr,
                "payload": MksServo42Axis._encode_speed_payload(
                    args.pitch_motor_a_sign * rate_rad_s,
                    args.pitch_accel_byte,
                    args.pitch_gear_ratio,
                ),
                "expect_reply": False,
                "expected_len": None,
                "priority": priority,
                "target": args.serial_target,
            },
            {
                "cmd_id": f"{cmd_prefix}:pitch_b:{time.time_ns()}",
                "func": "F6",
                "addr": args.pitch_motor_b_addr,
                "payload": MksServo42Axis._encode_speed_payload(
                    args.pitch_motor_b_sign * rate_rad_s,
                    args.pitch_accel_byte,
                    args.pitch_gear_ratio,
                ),
                "expect_reply": False,
                "expected_len": None,
                "priority": priority,
                "target": args.serial_target,
            },
        ]

    try:
        switch_io.setup()

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
                    "cmd_id": "enable:pitch_a",
                    "func": "F3",
                    "addr": args.pitch_motor_a_addr,
                    "payload": [0x01],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "enable:pitch_b",
                    "func": "F3",
                    "addr": args.pitch_motor_b_addr,
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
                    "cmd_id": "zero:pitch_a",
                    "func": "0x92",
                    "addr": args.pitch_motor_a_addr,
                    "payload": [],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "high",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "zero:pitch_b",
                    "func": "0x92",
                    "addr": args.pitch_motor_b_addr,
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

        if not _send_update(_pitch_sync_enable_commands(True, priority="high")):
            raise SystemExit("failed to enable pitch synchronization mode")

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
                *_pitch_speed_commands(0.0, cmd_prefix="speed:hold", priority="high"),
                _pitch_sync_exec_command(priority="high"),
            ],
            retries=1,
        )

        log.info(
            "Joystick control active (yaw addr=%d, pitch a=%d b=%d signs=(%.1f, %.1f)). Press Ctrl+C to stop.",
            args.yaw_addr,
            args.pitch_motor_a_addr,
            args.pitch_motor_b_addr,
            args.pitch_motor_a_sign,
            args.pitch_motor_b_sign,
        )

        last_log = 0.0
        last_manual_active = True
        emergency_sent = False
        while not stop_event.is_set():
            switch_state = switch_io.update()

            if switch_state["emergency"]:
                if switch_state["emergency_entered"] or not emergency_sent:
                    _send_update(
                        [
                            {
                                "cmd_id": f"estop:yaw:{time.time_ns()}",
                                "func": "F7",
                                "addr": args.yaw_addr,
                                "payload": [],
                                "expect_reply": False,
                                "expected_len": None,
                                "priority": "critical",
                                "target": args.serial_target,
                            },
                            {
                                "cmd_id": f"estop:pitch_a:{time.time_ns()}",
                                "func": "F7",
                                "addr": args.pitch_motor_a_addr,
                                "payload": [],
                                "expect_reply": False,
                                "expected_len": None,
                                "priority": "critical",
                                "target": args.serial_target,
                            },
                            {
                                "cmd_id": f"estop:pitch_b:{time.time_ns()}",
                                "func": "F7",
                                "addr": args.pitch_motor_b_addr,
                                "payload": [],
                                "expect_reply": False,
                                "expected_len": None,
                                "priority": "critical",
                                "target": args.serial_target,
                            },
                        ]
                    )
                    emergency_sent = True
                time.sleep(max(0.05, switch_io.poll_dt))
                continue

            if switch_state["emergency_exited"]:
                emergency_sent = False

            if not switch_state["active"]:
                if last_manual_active or switch_state["active_changed"]:
                    _send_update(
                        [
                            {
                                "cmd_id": f"speed:inactive:yaw:{time.time_ns()}",
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
                            *_pitch_speed_commands(
                                0.0,
                                cmd_prefix="speed:inactive",
                                priority="critical",
                            ),
                            _pitch_sync_exec_command(priority="critical"),
                        ]
                    )
                last_manual_active = False
                time.sleep(max(0.05, switch_io.poll_dt))
                continue

            last_manual_active = True

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
                        *_pitch_speed_commands(0.0, cmd_prefix="stop", priority="critical"),
                        _pitch_sync_exec_command(priority="critical"),
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
                    *_pitch_speed_commands(pitch_rate, cmd_prefix="speed", priority="high"),
                    _pitch_sync_exec_command(priority="high"),
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
                    "cmd_id": "stop:pitch_a",
                    "func": "F6",
                    "addr": args.pitch_motor_a_addr,
                    "payload": MksServo42Axis._encode_speed_payload(
                        0.0, args.pitch_accel_byte, args.pitch_gear_ratio
                    ),
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "stop:pitch_b",
                    "func": "F6",
                    "addr": args.pitch_motor_b_addr,
                    "payload": MksServo42Axis._encode_speed_payload(
                        0.0, args.pitch_accel_byte, args.pitch_gear_ratio
                    ),
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                _pitch_sync_exec_command(priority="critical"),
                *_pitch_sync_enable_commands(False, priority="critical"),
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
                    "cmd_id": "disable:pitch_a",
                    "func": "F3",
                    "addr": args.pitch_motor_a_addr,
                    "payload": [0x00],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "disable:pitch_b",
                    "func": "F3",
                    "addr": args.pitch_motor_b_addr,
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
        switch_io.cleanup()
        if serial_service_proc is not None:
            if serial_service_proc.poll() is None:
                serial_service_proc.terminate()
                try:
                    serial_service_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    serial_service_proc.kill()
                    serial_service_proc.wait(timeout=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
