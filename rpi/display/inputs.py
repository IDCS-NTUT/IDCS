"""Menu input sources: panel joystick and buttons, and a USB keyboard.

Every source turns its device into ``NavEvent``s. The panel's joystick and
GPIO roles arrive as ``PanelState`` messages from ``rpi.runtime_control``
(the process that owns the ADC and GPIO); the keyboard is read straight from
``/dev/input`` (the user is in the ``input`` group), so it works whichever
window has focus.

The joystick navigates only while manual control is off: while the operator
steers the gimbal with it, it is not a menu input.
"""

from __future__ import annotations

import errno
import glob
import os
import struct
from dataclasses import dataclass, field
from typing import Mapping

from rpi.display.menu import NavEvent

# GPIO roles (rpi.gpio.inputs) that act as menu buttons when a pin is configured.
BUTTON_ROLES: dict[str, NavEvent] = {
    "menu": NavEvent.MENU,
    "menu_select": NavEvent.SELECT,
    "menu_back": NavEvent.BACK,
}


@dataclass
class JoystickNavigator:
    """Joystick deflection -> discrete events with hysteresis and auto-repeat.

    ``x`` right-positive and ``y`` up-positive, both in [-1, 1]. Up/down repeat
    while held (scrolling a list); left/right fire once per push.
    """

    engage: float = 0.6
    release: float = 0.3
    repeat_delay_s: float = 0.5
    repeat_period_s: float = 0.2
    _held: NavEvent | None = None
    _next_repeat_s: float = 0.0

    def update(self, x: float, y: float, now_s: float) -> list[NavEvent]:
        if self._held is not None:
            if self._deflection(self._held, x, y) >= self.release:
                if self._held in (NavEvent.UP, NavEvent.DOWN) and now_s >= self._next_repeat_s:
                    self._next_repeat_s = now_s + self.repeat_period_s
                    return [self._held]
                return []
            self._held = None
        direction = self._direction(x, y)
        if direction is None:
            return []
        self._held = direction
        self._next_repeat_s = now_s + self.repeat_delay_s
        return [direction]

    def reset(self) -> None:
        self._held = None

    def _direction(self, x: float, y: float) -> NavEvent | None:
        if max(abs(x), abs(y)) < self.engage:
            return None
        if abs(y) >= abs(x):
            return NavEvent.UP if y > 0 else NavEvent.DOWN
        return NavEvent.RIGHT if x > 0 else NavEvent.LEFT

    @staticmethod
    def _deflection(event: NavEvent, x: float, y: float) -> float:
        return {NavEvent.UP: y, NavEvent.DOWN: -y, NavEvent.RIGHT: x, NavEvent.LEFT: -x}[event]


@dataclass
class PanelNavigator:
    """PanelState messages -> events (joystick + configured menu buttons)."""

    joystick: JoystickNavigator = field(default_factory=JoystickNavigator)
    _buttons: dict[str, bool] = field(default_factory=dict)

    def update(self, panel: Mapping, now_s: float) -> list[NavEvent]:
        events: list[NavEvent] = []
        inputs = panel.get("inputs") or {}
        for role, event in BUTTON_ROLES.items():
            pressed = bool(inputs.get(role, False))
            if pressed and not self._buttons.get(role, False):
                events.append(event)
            self._buttons[role] = pressed
        if panel.get("manual_active") or panel.get("emergency"):
            self.joystick.reset()
        else:
            x, y = panel.get("joystick") or (0.0, 0.0)
            events.extend(self.joystick.update(float(x), float(y), now_s))
        return events


# struct input_event on 64-bit Linux: timeval (2 x long), type, code, value.
_EVENT = struct.Struct("llHHi")
_EV_KEY = 0x01
KEYMAP: dict[int, NavEvent] = {
    103: NavEvent.UP,  # KEY_UP
    108: NavEvent.DOWN,  # KEY_DOWN
    105: NavEvent.LEFT,  # KEY_LEFT
    106: NavEvent.RIGHT,  # KEY_RIGHT
    28: NavEvent.SELECT,  # KEY_ENTER
    96: NavEvent.SELECT,  # KEY_KPENTER
    57: NavEvent.SELECT,  # KEY_SPACE
    1: NavEvent.BACK,  # KEY_ESC
    14: NavEvent.BACK,  # KEY_BACKSPACE
    50: NavEvent.MENU,  # KEY_M
    139: NavEvent.MENU,  # KEY_MENU
}


def key_events(data: bytes) -> list[NavEvent]:
    """Decode raw input_event records: key presses and auto-repeats."""
    events = []
    for offset in range(0, len(data) - _EVENT.size + 1, _EVENT.size):
        _, _, ev_type, code, value = _EVENT.unpack_from(data, offset)
        if ev_type == _EV_KEY and value in (1, 2) and code in KEYMAP:
            events.append(KEYMAP[code])
    return events


class KeyboardInput:
    """Non-blocking reader over every keyboard-like /dev/input/event* device.

    Devices are rescanned by ``rescan()`` so a keyboard plugged in later works.
    """

    def __init__(self, pattern: str = "/dev/input/event*") -> None:
        self._pattern = pattern
        self._fds: dict[str, int] = {}

    def rescan(self) -> None:
        for path in glob.glob(self._pattern):
            if path in self._fds or not _has_keys(path):
                continue
            try:
                self._fds[path] = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue

    def poll(self) -> list[NavEvent]:
        events: list[NavEvent] = []
        for path, fd in list(self._fds.items()):
            while True:
                try:
                    data = os.read(fd, _EVENT.size * 64)
                except BlockingIOError:
                    break
                except OSError as exc:  # unplugged
                    if exc.errno in (errno.ENODEV, errno.EIO):
                        os.close(fd)
                        del self._fds[path]
                    break
                if not data:
                    break
                events.extend(key_events(data))
        return events

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def _has_keys(event_path: str) -> bool:
    """True for devices with arrow keys (keyboards, remotes), not mice or HDMI-CEC."""
    name = os.path.basename(event_path)
    try:
        with open(f"/sys/class/input/{name}/device/capabilities/key") as handle:
            words = handle.read().split()
    except OSError:
        return False
    bits = 0
    for word in words:
        bits = (bits << 64) | int(word, 16)
    return all(bits >> code & 1 for code in (103, 108, 105, 106))
