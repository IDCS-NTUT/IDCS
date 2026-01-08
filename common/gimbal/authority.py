"""State machines for RS-485 control authority negotiation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class JetsonAuthorityState(str, Enum):
    ACTIVE = "active"
    YIELDING = "yielding"
    STANDBY = "standby"
    RECOVERING = "recovering"


class PiAuthorityState(str, Enum):
    STANDBY = "standby"
    REQUESTING_TAKEOVER = "requesting_takeover"
    ACTIVE = "active"
    RETURNING = "returning"


@dataclass(frozen=True)
class AuthorityTransition:
    """Represents a single authority state transition with a rationale."""

    source: Enum
    dest: Enum
    reason: str

    def __str__(self) -> str:
        return f"{self.source.value} -> {self.dest.value}: {self.reason}"
