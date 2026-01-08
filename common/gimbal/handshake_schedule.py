"""Handshake window scheduling helpers for RS-485 control-plane traffic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HandshakeSchedule:
    """Defines timing parameters for control-plane ping/reply exchanges."""

    ping_interval_s: float
    reply_window_s: float
    bus_quiet_s: float = 0.0

    def validate(self) -> None:
        if self.ping_interval_s <= 0:
            raise ValueError("ping_interval_s must be positive")
        if self.reply_window_s <= 0:
            raise ValueError("reply_window_s must be positive")
        if self.bus_quiet_s < 0:
            raise ValueError("bus_quiet_s must be non-negative")
        if self.reply_window_s >= self.ping_interval_s:
            raise ValueError("reply_window_s must be shorter than ping_interval_s")


@dataclass
class HandshakeWindow:
    """Tracks the timing of a single ping and reply window."""

    ping_ts: float
    reply_deadline_ts: float
    quiet_until_ts: float

    def is_reply_open(self, now: float) -> bool:
        return self.ping_ts <= now <= self.reply_deadline_ts

    def is_quiet_period(self, now: float) -> bool:
        return now < self.quiet_until_ts


def open_window(*, now: float, schedule: HandshakeSchedule) -> HandshakeWindow:
    """Create a handshake window based on the current time and schedule."""

    schedule.validate()
    reply_deadline = now + schedule.reply_window_s
    quiet_until = reply_deadline + schedule.bus_quiet_s
    return HandshakeWindow(
        ping_ts=now,
        reply_deadline_ts=reply_deadline,
        quiet_until_ts=quiet_until,
    )


def next_ping_due(*, now: float, last_ping_ts: Optional[float], schedule: HandshakeSchedule) -> bool:
    """Return True if it's time to emit the next ping frame."""

    schedule.validate()
    if last_ping_ts is None:
        return True
    return (now - last_ping_ts) >= schedule.ping_interval_s
