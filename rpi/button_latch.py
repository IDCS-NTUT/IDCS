"""Debounced button latch helper for authority handoff on Raspberry Pi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ButtonLatchConfig:
    """Configuration for debounced button latching."""

    debounce_s: float = 0.05
    cooldown_s: float = 1.0

    def validate(self) -> None:
        if self.debounce_s <= 0:
            raise ValueError("debounce_s must be positive")
        if self.cooldown_s < 0:
            raise ValueError("cooldown_s must be non-negative")


@dataclass
class ButtonLatch:
    """Track debounced button presses and emit a single event per press."""

    config: ButtonLatchConfig
    last_state: bool = False
    last_change_ts: Optional[float] = None
    last_trigger_ts: Optional[float] = None

    def update(self, *, pressed: bool, now: float) -> bool:
        """Return True once per debounced press event.

        Args:
            pressed: Current raw button state.
            now: Monotonic timestamp in seconds.
        """

        self.config.validate()

        if pressed != self.last_state:
            self.last_state = pressed
            self.last_change_ts = now
            return False

        if not pressed:
            return False

        if self.last_change_ts is None:
            return False

        if (now - self.last_change_ts) < self.config.debounce_s:
            return False

        if self.last_trigger_ts is not None:
            if (now - self.last_trigger_ts) < self.config.cooldown_s:
                return False

        self.last_trigger_ts = now
        return True
