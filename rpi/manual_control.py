"""Shared Raspberry Pi panel GPIO and joystick ADC helpers.

The managed entry point is rpi.runtime_control. This module has no standalone
command loop and does not construct a serial or motor interface.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from typing import Any, Mapping

import smbus

from rpi.panel_indicators import IndicatorState

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
        "fire_control_lamp": 21,
        "safety_lamp": 22,
        "green_light": 23,
        "yellow_light": 24,
        "red_light": 25,
    },
    "input_pull": "up",
    "input_pulls": {},
    "output_active_level": "low",
    "input_modes": {},
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


def _coerce_input_modes(raw: Any, *, inputs: Mapping[str, int]) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("rpi.gpio.input_modes must be a mapping of role names to modes")

    modes: dict[str, str] = {}
    for raw_name, raw_mode in raw.items():
        name = str(raw_name).strip()
        if name not in inputs:
            raise ValueError(f"rpi.gpio.input_modes.{name} does not match a configured input")
        mode = str(raw_mode).strip().lower()
        if mode not in {"level", "latch"}:
            raise ValueError(f"rpi.gpio.input_modes.{name} must be one of: level, latch")
        if mode != "level":
            modes[name] = mode
    return modes


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


def _coerce_gpio_active_level_map(
    raw: Any,
    *,
    section: str,
    input_roles: Mapping[str, int],
    defaults: Mapping[str, str],
) -> dict[str, str]:
    if raw is None:
        return dict(defaults)
    if not isinstance(raw, Mapping):
        raise ValueError(f"rpi.gpio.{section} must be a mapping of role names to active levels")

    levels = dict(defaults)
    valid_roles = set(input_roles)
    for raw_name, raw_level in raw.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"rpi.gpio.{section} contains an empty role name")
        if name not in valid_roles:
            raise ValueError(f"rpi.gpio.{section}.{name} does not match any configured input role")
        level = str(raw_level).strip().lower()
        if level not in {"low", "high"}:
            raise ValueError(f"rpi.gpio.{section}.{name} must be one of: low, high")
        levels[name] = level
    return levels


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
    input_modes = _coerce_input_modes(cfg.get("input_modes"), inputs=inputs)
    shared_lamp_inputs = cfg.get("shared_lamp_inputs", {})
    if not isinstance(shared_lamp_inputs, Mapping):
        raise ValueError("rpi.gpio.shared_lamp_inputs must be a mapping")
    for role, lamp in shared_lamp_inputs.items():
        if (role not in {"fire_control", "safety"} or role not in inputs
                or lamp != f"{role}_lamp" or lamp not in outputs):
            raise ValueError("shared lamp inputs must name configured fire_control/safety inputs and lamp outputs")
        if input_modes.get(role) == "latch":
            raise ValueError("shared lamp inputs must use level mode, not latch")
    input_pulls = _coerce_gpio_pull_map(
        cfg.get("input_pulls") if "input_pulls" in cfg else None,
        section="input_pulls",
        input_roles=inputs,
        default_pull=input_pull,
    )
    derived_input_active_levels = {
        role: _input_active_level_for_pull(pull) for role, pull in input_pulls.items()
    }
    input_active_levels = _coerce_gpio_active_level_map(
        cfg.get("input_active_levels") if "input_active_levels" in cfg else None,
        section="input_active_levels",
        input_roles=inputs,
        defaults=derived_input_active_levels,
    )

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
        "input_modes": input_modes,
        "shared_lamp_inputs": dict(shared_lamp_inputs),
    }


class ManualSwitchIO:
    """GPIO switch/emergency logic for the RPi control panel.

    Behavior:
    - ``control_switch`` level drives manual ACTIVE state unless configured as
      ``input_modes.control_switch: latch``.
    - ``emergency`` level drives emergency mode.
    - ``fire_control`` level enables automatic controller commands only.
    - Target status drives the fire-control/safety lamps independently of inputs.
      Only these two contacts may share lamp power; unpowered inputs are invalid.
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
        self._input_modes: dict[str, str] = dict(resolved_cfg["input_modes"])
        self._shared_lamp_inputs = dict(resolved_cfg["shared_lamp_inputs"])
        self.indicators = IndicatorState()
        self.input_validity: dict[str, bool] = {}
        self._powered_last_read: dict[str, bool] = {}
        if gpio_config is None and "fire_control" not in self._inputs:
            self._inputs["fire_control"] = int(control_toggle_pin)

        self.active = False
        self.emergency = False
        self.control_cmd_enabled = False
        self.fire = False
        self.safety = False
        # Every configured input role's state as of the last update(), for
        # consumers beyond the safety logic (the panel display's menu buttons).
        self.role_states: dict[str, bool] = {}
        self._prev_role_states: dict[str, bool] = {}
        self._prev_raw_role_states: dict[str, bool] = {}
        self._latched_role_states: dict[str, bool] = {}
        self._last_latch_toggle_mono: dict[str, float] = {}
        self._prev_pin_levels: dict[str, int] = {}
        self._last_out: dict[int, int] = {}

    @property
    def poll_dt(self) -> float:
        return self._poll_dt

    @property
    def ready(self) -> bool:
        return self._ready

    def set_indicators(self, state: IndicatorState) -> None:
        self.indicators = state

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
        self._last_out = {pin: inactive_level for pin in self._outputs.values()}
        self._apply_normal_outputs()
        self._prev_pin_levels = {
            role: gpio.input(pin)
            for role, pin in self._inputs.items()
        }
        raw_role_states = self._read_raw_role_states()
        self._prev_raw_role_states = dict(raw_role_states)
        self._latched_role_states = {
            role: raw_role_states.get(role, False)
            for role, mode in self._input_modes.items()
            if mode == "latch"
        }
        self._prev_role_states = self._apply_input_modes(raw_role_states)
        self.active = self._prev_role_states.get("control_switch", False)
        self.emergency = self._prev_role_states.get("emergency", False)
        self.control_cmd_enabled = self._prev_role_states.get("fire_control", False)
        self.fire = self._prev_role_states.get("fire", False)
        self.safety = self._prev_role_states.get("safety", False)
        self._apply_normal_outputs()
        self._log.info(
            "switch GPIO integration enabled inputs=%s pulls=%s outputs=%s modes=%s active_levels=%s output_active=%s",
            self._inputs,
            self._input_pulls,
            self._outputs,
            self._input_modes,
            self._input_active_levels,
            self._output_active_level,
        )
        return True

    def _active_output_level(self, gpio: Any) -> int:
        return gpio.LOW if self._output_active_level == "low" else gpio.HIGH

    def _inactive_output_level(self, gpio: Any) -> int:
        return gpio.HIGH if self._output_active_level == "low" else gpio.LOW

    def _read_raw_role_states(self) -> dict[str, bool]:
        if not self._ready or self._gpio is None:
            return {}
        gpio = self._gpio
        states: dict[str, bool] = {}
        for role, pin in self._inputs.items():
            lamp = self._shared_lamp_inputs.get(role)
            powered = lamp is None or bool(getattr(self.indicators, lamp, False))
            valid = powered and (lamp is None or self._powered_last_read.get(role, False))
            self._powered_last_read[role] = powered
            self.input_validity[role] = valid
            level = gpio.input(pin)
            if level != self._prev_pin_levels.get(role):
                self._log.info("GPIO input role=%s pin=%d changed -> %d", role, pin, level)
                self._prev_pin_levels[role] = level
            active_level = self._active_input_level_for_role(gpio, role)
            states[role] = valid and level == active_level
        return states

    def _apply_input_modes(self, raw_states: Mapping[str, bool]) -> dict[str, bool]:
        states = dict(raw_states)
        now = time.monotonic()
        for role, mode in self._input_modes.items():
            if mode != "latch":
                continue
            raw_active = bool(raw_states.get(role, False))
            raw_was_active = bool(self._prev_raw_role_states.get(role, False))
            latched = bool(self._latched_role_states.get(role, False))
            if raw_active and not raw_was_active:
                last_toggle = self._last_latch_toggle_mono.get(role)
                if last_toggle is None or (now - last_toggle) >= self._debounce_s:
                    latched = not latched
                    self._latched_role_states[role] = latched
                    self._last_latch_toggle_mono[role] = now
                    self._log.info("GPIO latch role=%s toggled -> %s", role, latched)
            states[role] = latched
        self._prev_raw_role_states = dict(raw_states)
        return states

    def _set_out(self, pin: int, level: int, *, force: bool = False) -> None:
        if not self._ready or self._gpio is None:
            return
        changed = self._last_out.get(pin) != level
        if changed or force:
            self._gpio.output(pin, level)
            self._last_out[pin] = level
            if changed:
                self._log.info("GPIO output pin=%d level=%d", pin, level)

    def _active_input_level_for_role(self, gpio: Any, role: str) -> int:
        level_name = self._input_active_levels.get(role)
        if level_name == "high":
            return gpio.HIGH
        return gpio.LOW

    def _set_role_out(self, role: str, active: bool, *, force: bool = False) -> None:
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
        self._set_out(pin, level, force=force)

    def _apply_normal_outputs(self) -> None:
        if not self._ready or self._gpio is None:
            return
        self._set_role_out("fire_control_lamp", self.indicators.fire_control_lamp, force=True)
        self._set_role_out("safety_lamp", self.indicators.safety_lamp, force=True)
        self._set_role_out("green_light", True, force=True)
        self._set_role_out("yellow_light", self.indicators.yellow_light, force=True)
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
                "active": False,
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

        # Power first; a newly powered contact remains invalid until the next
        # poll. Never reuse a cached high or latch a lamp-generated edge.
        self._apply_normal_outputs()
        role_states = self._apply_input_modes(self._read_raw_role_states())
        self.role_states = dict(role_states)
        self.fire = role_states.get("fire", False)
        self.safety = role_states.get("safety", False)
        self.active = role_states.get("control_switch", False)
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
