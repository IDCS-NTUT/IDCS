"""Safety and robustness helpers for authority handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AuthoritySafetyConfig:
    """Configuration for authority handoff safety checks."""

    min_active_s: float = 0.5
    min_standby_s: float = 0.5
    max_missing_pings: int = 3
    max_missing_replies: int = 3
    peer_timeout_s: float = 2.0

    def validate(self) -> None:
        if self.min_active_s < 0:
            raise ValueError("min_active_s must be non-negative")
        if self.min_standby_s < 0:
            raise ValueError("min_standby_s must be non-negative")
        if self.max_missing_pings < 0:
            raise ValueError("max_missing_pings must be non-negative")
        if self.max_missing_replies < 0:
            raise ValueError("max_missing_replies must be non-negative")
        if self.peer_timeout_s < 0:
            raise ValueError("peer_timeout_s must be non-negative")


@dataclass
class AuthoritySafetyTracker:
    """Track authority handoff timing and peer health."""

    config: AuthoritySafetyConfig
    last_state_change_ts: Optional[float] = None
    last_ping_rx_ts: Optional[float] = None
    last_reply_rx_ts: Optional[float] = None
    missing_pings: int = 0
    missing_replies: int = 0

    def record_state_change(self, *, now: float) -> None:
        self.config.validate()
        self.last_state_change_ts = now

    def record_ping_received(self, *, now: float) -> None:
        self.config.validate()
        self.last_ping_rx_ts = now
        self.missing_pings = 0

    def record_reply_received(self, *, now: float) -> None:
        self.config.validate()
        self.last_reply_rx_ts = now
        self.missing_replies = 0

    def note_ping_missed(self) -> None:
        self.config.validate()
        self.missing_pings += 1

    def note_reply_missed(self) -> None:
        self.config.validate()
        self.missing_replies += 1

    def can_transition_active(self, *, now: float) -> bool:
        self.config.validate()
        if self.last_state_change_ts is None:
            return True
        return (now - self.last_state_change_ts) >= self.config.min_standby_s

    def can_transition_standby(self, *, now: float) -> bool:
        self.config.validate()
        if self.last_state_change_ts is None:
            return True
        return (now - self.last_state_change_ts) >= self.config.min_active_s

    def peer_unresponsive(self, *, now: float) -> bool:
        self.config.validate()
        if self.missing_pings >= self.config.max_missing_pings:
            return True
        if self.missing_replies >= self.config.max_missing_replies:
            return True
        last_heard = self._last_heard_ts()
        if last_heard is None:
            return False
        return (now - last_heard) >= self.config.peer_timeout_s

    def _last_heard_ts(self) -> Optional[float]:
        timestamps = [ts for ts in (self.last_ping_rx_ts, self.last_reply_rx_ts) if ts is not None]
        if not timestamps:
            return None
        return max(timestamps)
