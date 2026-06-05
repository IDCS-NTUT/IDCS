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

DEFAULT_GPIO_CONFIG: dict[str, Any] = {
    "inputs": {
        "fire": 4,
        "fire_control": 5,
        "safety": 6,
        "emergency": 16,
        "control_switch": 20,
    },
    "outputs": {
        "fire_control_light": 21,
        "safety_light": 22,
        "green_light": 23,
        "yellow_light": 24,
        "red_light": 25,
    },
    "input_pull": "up",
    "input_pulls": {},
    "output_active_level": "low",
}


def _coerce_gpio_pin_map(raw: Any, *, section: str) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"rpi.gpio.{section} must be a mapping of role names to BCM pins")

    pins: dict[str, int] = {}
    for raw_name, raw_pin in raw.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"rpi.gpio.{section} contains an empty role name")
        if raw_pin is None:
            continue
        try:
            pin = int(raw_pin)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rpi.gpio.{section}.{name} must be an integer BCM pin") from exc
        if pin < 0 or pin > 27:
            raise ValueError(f"rpi.gpio.{section}.{name} must be within BCM pin range [0, 27]")
        pins[name] = pin
    return pins


def _coerce_gpio_pull_map(
    raw: Any,
    *,
    section: str,
    input_roles: Mapping[str, int],
    default_pull: str,
) -> dict[str, str]:
    if raw is None:
        return {role: default_pull for role in input_roles}
    if not isinstance(raw, Mapping):
        raise ValueError(f"rpi.gpio.{section} must be a mapping of role names to pull modes")

    pulls: dict[str, str] = {}
    valid_roles = set(input_roles)
    for raw_name, raw_pull in raw.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"rpi.gpio.{section} contains an empty role name")
        if name not in valid_roles:
            raise ValueError(f"rpi.gpio.{section}.{name} does not match any configured input role")
        pull = str(raw_pull).strip().lower()
        if pull not in {"up", "down", "none"}:
            raise ValueError(f"rpi.gpio.{section}.{name} must be one of: up, down, none")
        pulls[name] = pull

    return {role: pulls.get(role, default_pull) for role in input_roles}


def _input_active_level_for_pull(pull: str) -> str:
    if pull == "up":
        return "low"
    if pull == "down":
        return "high"
    return "low"


def resolve_gpio_config(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve user GPIO config over defaults.

    Omitted roles are removed when a custom ``inputs`` or ``outputs`` mapping is
    supplied, which lets the panel be trimmed without code changes.
    """

    cfg = raw if isinstance(raw, Mapping) else {}
    input_pull = str(cfg.get("input_pull", DEFAULT_GPIO_CONFIG["input_pull"])).strip().lower()
    output_active_level = str(
        cfg.get("output_active_level", DEFAULT_GPIO_CONFIG["output_active_level"])
    ).strip().lower()

    if input_pull not in {"up", "down", "none"}:
        raise ValueError("rpi.gpio.input_pull must be one of: up, down, none")
    if output_active_level not in {"low", "high"}:
        raise ValueError("rpi.gpio.output_active_level must be one of: low, high")

    inputs = (
        _coerce_gpio_pin_map(cfg.get("inputs"), section="inputs")
        if "inputs" in cfg
        else dict(DEFAULT_GPIO_CONFIG["inputs"])
    )
    outputs = (
        _coerce_gpio_pin_map(cfg.get("outputs"), section="outputs")
        if "outputs" in cfg
        else dict(DEFAULT_GPIO_CONFIG["outputs"])
    )
    input_pulls = _coerce_gpio_pull_map(
        cfg.get("input_pulls") if "input_pulls" in cfg else None,
        section="input_pulls",
        input_roles=inputs,
        default_pull=input_pull,
    )
    input_active_levels = {
        role: _input_active_level_for_pull(pull) for role, pull in input_pulls.items()
    }

    used: dict[int, str] = {}
    for direction, pin_map in (("inputs", inputs), ("outputs", outputs)):
        for role, pin in pin_map.items():
            owner = used.get(pin)
            if owner is not None:
                raise ValueError(f"GPIO{pin} is assigned to both {owner} and {direction}.{role}")
            used[pin] = f"{direction}.{role}"

    return {
        "inputs": inputs,
        "outputs": outputs,
        "input_pull": input_pull,
        "input_pulls": input_pulls,
        "input_active_levels": input_active_levels,
        "output_active_level": output_active_level,
    }


class ManualSwitchIO:
    """GPIO switch/emergency logic for the RPi control panel.

    Behavior:
    - ``control_switch`` level drives manual ACTIVE state.
    - ``emergency`` level drives emergency mode.
    - ``fire_control`` level drives control-command enable state.
    - Configured outputs are driven from the matching logical panel states.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        poll_dt: float,
        debounce_s: float,
        control_toggle_pin: int = 16,
        gpio_config: Mapping[str, Any] | None = None,
        log: logging.Logger,
    ) -> None:
        self._enabled = enabled
        self._poll_dt = max(float(poll_dt), 0.001)
        self._debounce_s = max(float(debounce_s), 0.0)
        self._log = log

        self._gpio: Any | None = None
        self._ready = False

        resolved_cfg = resolve_gpio_config(gpio_config)
        self._inputs: dict[str, int] = dict(resolved_cfg["inputs"])
        self._outputs: dict[str, int] = dict(resolved_cfg["outputs"])
        self._input_pull = str(resolved_cfg["input_pull"])
        self._input_pulls: dict[str, str] = {
            str(role): str(pull) for role, pull in dict(resolved_cfg["input_pulls"]).items()
        }
        self._input_active_levels: dict[str, str] = {
            str(role): str(level)
            for role, level in dict(resolved_cfg["input_active_levels"]).items()
        }
        self._output_active_level = str(resolved_cfg["output_active_level"])
        if gpio_config is None and "fire_control" not in self._inputs:
            self._inputs["fire_control"] = int(control_toggle_pin)

        self.active = False
        self.emergency = False
        self.control_cmd_enabled = False
        self.fire = False
        self.safety = False
        self._prev_role_states: dict[str, bool] = {}
        self._prev_pin_levels: dict[str, int] = {}
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

        pull_modes = {
            "up": gpio.PUD_UP,
            "down": gpio.PUD_DOWN,
            "none": gpio.PUD_OFF,
        }
        inactive_level = self._inactive_output_level(gpio)
        for role, pin in self._inputs.items():
            pull_mode = pull_modes[self._input_pulls.get(role, self._input_pull)]
            gpio.setup(pin, gpio.IN, pull_up_down=pull_mode)
        for pin in self._outputs.values():
            gpio.setup(pin, gpio.OUT, initial=inactive_level)

        self._gpio = gpio
        self._ready = True
        self._prev_pin_levels = {
            role: gpio.input(pin)
            for role, pin in self._inputs.items()
        }
        self._prev_role_states = self._read_role_states()
        self.active = self._prev_role_states.get("control_switch", True)
        self.emergency = self._prev_role_states.get("emergency", False)
        self.control_cmd_enabled = self._prev_role_states.get("fire_control", False)
        self.fire = self._prev_role_states.get("fire", False)
        self.safety = self._prev_role_states.get("safety", False)
        self._last_out = {pin: inactive_level for pin in self._outputs.values()}
        self._apply_normal_outputs()
        self._log.info(
            "switch GPIO integration enabled inputs=%s pulls=%s outputs=%s active_levels=%s output_active=%s",
            self._inputs,
            self._input_pulls,
            self._outputs,
            self._input_active_levels,
            self._output_active_level,
        )
        return True

    def _active_output_level(self, gpio: Any) -> int:
        return gpio.LOW if self._output_active_level == "low" else gpio.HIGH

    def _inactive_output_level(self, gpio: Any) -> int:
        return gpio.HIGH if self._output_active_level == "low" else gpio.LOW

    def _read_role_states(self) -> dict[str, bool]:
        if not self._ready or self._gpio is None:
            return {}
        gpio = self._gpio
        states: dict[str, bool] = {}
        for role, pin in self._inputs.items():
            level = gpio.input(pin)
            if level != self._prev_pin_levels.get(role):
                self._log.info("GPIO input role=%s pin=%d changed -> %d", role, pin, level)
                self._prev_pin_levels[role] = level
            active_level = self._active_input_level_for_role(gpio, role)
            states[role] = level == active_level
        return states

    def _set_out(self, pin: int, level: int) -> None:
        if not self._ready or self._gpio is None:
            return
        if self._last_out.get(pin) != level:
            self._gpio.output(pin, level)
            self._last_out[pin] = level
            self._log.info("GPIO output pin=%d level=%d", pin, level)

    def _active_input_level_for_role(self, gpio: Any, role: str) -> int:
        level_name = self._input_active_levels.get(role)
        if level_name == "high":
            return gpio.HIGH
        return gpio.LOW

    def _set_role_out(self, role: str, active: bool) -> None:
        if not self._ready or self._gpio is None:
            return
        pin = self._outputs.get(role)
        if pin is None:
            return
        level = (
            self._active_output_level(self._gpio)
            if active
            else self._inactive_output_level(self._gpio)
        )
        self._set_out(pin, level)

    def _apply_normal_outputs(self) -> None:
        if not self._ready or self._gpio is None:
            return
        self._set_role_out("fire_control_light", self.control_cmd_enabled)
        self._set_role_out("safety_light", self.safety)
        self._set_role_out("green_light", True)
        self._set_role_out("yellow_light", self.fire or self.control_cmd_enabled)
        self._set_role_out("red_light", self.emergency)

    def _enter_emergency(self) -> None:
        if not self._ready or self._gpio is None:
            return
        if self.emergency:
            return
        self.emergency = True
        self._log.warning("switch emergency triggered")
        self._apply_normal_outputs()

    def _maintain_emergency(self) -> None:
        if not self._ready or self._gpio is None:
            return
        self._apply_normal_outputs()

    def _exit_emergency(self) -> None:
        if not self._ready:
            return
        self.emergency = False
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
                "control_cmd_enabled": False,
                "control_cmd_changed": False,
            }

        active_before = self.active
        emergency_before = self.emergency
        control_cmd_before = self.control_cmd_enabled

        role_states = self._read_role_states()
        self.fire = role_states.get("fire", False)
        self.safety = role_states.get("safety", False)
        self.active = role_states.get("control_switch", True)
        self.control_cmd_enabled = role_states.get("fire_control", False)
        emergency_active = role_states.get("emergency", False)

        if emergency_active:
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
                "control_cmd_enabled": self.control_cmd_enabled,
                "control_cmd_changed": self.control_cmd_enabled != control_cmd_before,
            }

        self._apply_normal_outputs()
        self._prev_role_states = role_states

        return {
            "active": self.active,
            "active_changed": self.active != active_before,
            "emergency": self.emergency,
            "emergency_entered": (not emergency_before) and self.emergency,
            "emergency_exited": emergency_before and (not self.emergency),
            "control_cmd_enabled": self.control_cmd_enabled,
            "control_cmd_changed": self.control_cmd_enabled != control_cmd_before,
        }

    def cleanup(self) -> None:
        if not self._ready or self._gpio is None:
            return
        gpio = self._gpio
        try:
            inactive_level = self._inactive_output_level(gpio)
            for pin in self._outputs.values():
                gpio.output(pin, inactive_level)
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


def _counts_to_rad(counts: int, *, counts_per_rev: int, gear_ratio: float) -> float:
    motor_revs = counts / float(counts_per_rev)
    axis_revs = motor_revs / gear_ratio
    return axis_revs * 2.0 * math.pi


def _reply_func_byte(reply: Mapping[str, Any]) -> int | None:
    func = reply.get("func")
    if isinstance(func, int):
        return func
    if isinstance(func, str):
        try:
            if func.lower().startswith("0x"):
                return int(func, 16)
            return int(func)
        except ValueError:
            return None
    return None


def _apply_hard_angle_limit(
    rate_cmd: float,
    current_angle: float | None,
    angle_min: float | None,
    angle_max: float | None,
    axis: str,
) -> float:
    if current_angle is None:
        return rate_cmd
    if angle_max is not None and current_angle >= angle_max and rate_cmd > 0.0:
        logging.getLogger("rpi.manual_control").debug(
            "hard angle limit: %s at %.4f rad >= max %.4f rad; blocking positive command %.4f rad/s",
            axis,
            current_angle,
            angle_max,
            rate_cmd,
        )
        return 0.0
    if angle_min is not None and current_angle <= angle_min and rate_cmd < 0.0:
        logging.getLogger("rpi.manual_control").debug(
            "hard angle limit: %s at %.4f rad <= min %.4f rad; blocking negative command %.4f rad/s",
            axis,
            current_angle,
            angle_min,
            rate_cmd,
        )
        return 0.0
    return rate_cmd


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
        help="Optional YAML config for serial defaults/service startup",
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
        help="Enable integrated RPi GPIO switch/emergency control from rpi.gpio config",
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
    baud = int(gimbal_cfg.get("baudrate", 256000))
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

    rpi_cfg = loaded_cfg.get("rpi") if isinstance(loaded_cfg, Mapping) else None
    gpio_cfg = rpi_cfg.get("gpio") if isinstance(rpi_cfg, Mapping) else None
    try:
        gpio_layout = resolve_gpio_config(gpio_cfg if isinstance(gpio_cfg, Mapping) else None)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    switch_io = ManualSwitchIO(
        enabled=args.switch_io,
        poll_dt=args.switch_poll_dt_s,
        debounce_s=args.switch_debounce_s,
        gpio_config=gpio_layout,
        log=log,
    )

    gimbal_cfg = loaded_cfg.get("gimbal") if isinstance(loaded_cfg, Mapping) else None
    if not isinstance(gimbal_cfg, Mapping):
        gimbal_cfg = {}
    yaw_min_raw = gimbal_cfg.get("yaw_min_rad")
    yaw_max_raw = gimbal_cfg.get("yaw_max_rad")
    pitch_min_raw = gimbal_cfg.get("pitch_min_rad")
    pitch_max_raw = gimbal_cfg.get("pitch_max_rad")
    camstate_yaw_sign = float(gimbal_cfg.get("camstate_yaw_sign", 1.0))
    camstate_pitch_sign = float(gimbal_cfg.get("camstate_pitch_sign", 1.0))
    yaw_min_rad = float(yaw_min_raw) if yaw_min_raw is not None else None
    yaw_max_rad = float(yaw_max_raw) if yaw_max_raw is not None else None
    pitch_min_rad = float(pitch_min_raw) if pitch_min_raw is not None else None
    pitch_max_rad = float(pitch_max_raw) if pitch_max_raw is not None else None
    if camstate_yaw_sign == 0.0:
        raise SystemExit("gimbal.camstate_yaw_sign must be non-zero")
    if camstate_pitch_sign == 0.0:
        raise SystemExit("gimbal.camstate_pitch_sign must be non-zero")
    if yaw_min_rad is not None and yaw_max_rad is not None and yaw_min_rad >= yaw_max_rad:
        raise SystemExit("gimbal.yaw_min_rad must be less than gimbal.yaw_max_rad")
    if pitch_min_rad is not None and pitch_max_rad is not None and pitch_min_rad >= pitch_max_rad:
        raise SystemExit("gimbal.pitch_min_rad must be less than gimbal.pitch_max_rad")
    if yaw_min_rad is not None or yaw_max_rad is not None:
        log.info("hard yaw angle limits: min=%s max=%s rad", yaw_min_rad, yaw_max_rad)
    if pitch_min_rad is not None or pitch_max_rad is not None:
        log.info("hard pitch angle limits: min=%s max=%s rad", pitch_min_rad, pitch_max_rad)

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

    respond_on_writes = not args.no_respond_on_writes

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
                    "expect_reply": respond_on_writes,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "enable:pitch_a",
                    "func": "F3",
                    "addr": args.pitch_motor_a_addr,
                    "payload": [0x01],
                    "expect_reply": respond_on_writes,
                    "expected_len": None,
                    "priority": "critical",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "enable:pitch_b",
                    "func": "F3",
                    "addr": args.pitch_motor_b_addr,
                    "payload": [0x01],
                    "expect_reply": respond_on_writes,
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
                    "expect_reply": respond_on_writes,
                    "expected_len": None,
                    "priority": "high",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "zero:pitch_a",
                    "func": "0x92",
                    "addr": args.pitch_motor_a_addr,
                    "payload": [],
                    "expect_reply": respond_on_writes,
                    "expected_len": None,
                    "priority": "high",
                    "target": args.serial_target,
                },
                {
                    "cmd_id": "zero:pitch_b",
                    "func": "0x92",
                    "addr": args.pitch_motor_b_addr,
                    "payload": [],
                    "expect_reply": respond_on_writes,
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
                *_pitch_speed_commands(0.0, cmd_prefix="speed:hold", priority="high"),
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
        yaw_counts: int | None = None
        pitch_counts: dict[int, int] = {}
        pitch_authority_addr = args.pitch_motor_a_addr if args.pitch_authority == "a" else args.pitch_motor_b_addr
        while not stop_event.is_set():
            for reply in reply_sub.recv_nowait():
                func = _reply_func_byte(reply)
                addr = reply.get("addr")
                if func != 0x31 or not isinstance(addr, int):
                    continue
                parsed = reply.get("reply", {}).get("parsed", {})
                if "counts" not in parsed:
                    continue
                counts = int(parsed["counts"])
                if addr == args.yaw_addr:
                    yaw_counts = counts
                elif addr in (args.pitch_motor_a_addr, args.pitch_motor_b_addr):
                    pitch_counts[addr] = counts

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

            current_yaw_rad = (
                camstate_yaw_sign
                * _counts_to_rad(
                    yaw_counts,
                    counts_per_rev=args.counts_per_rev,
                    gear_ratio=args.yaw_gear_ratio,
                )
                if yaw_counts is not None
                else None
            )
            current_pitch_rad = (
                camstate_pitch_sign
                * _counts_to_rad(
                    pitch_counts[pitch_authority_addr],
                    counts_per_rev=args.counts_per_rev,
                    gear_ratio=args.pitch_gear_ratio,
                )
                if pitch_authority_addr in pitch_counts
                else None
            )
            yaw_rate = _apply_hard_angle_limit(
                yaw_rate,
                current_yaw_rad,
                yaw_min_rad,
                yaw_max_rad,
                "yaw",
            )
            pitch_rate = _apply_hard_angle_limit(
                pitch_rate,
                current_pitch_rad,
                pitch_min_rad,
                pitch_max_rad,
                "pitch",
            )

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
