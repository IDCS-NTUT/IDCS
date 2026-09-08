"""Deterministic, hardware-free replay primitives for controller traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from common.schemas import ControlIntent, ControlIntentLimits, ControlObservation


class ControlPolicy(Protocol):
    """Future controller interface: one atomic snapshot yields one intent."""

    def decide(self, observation: ControlObservation) -> ControlIntent: ...


class HoldPolicy:
    """Deterministic safe baseline used to validate replay infrastructure."""

    def __init__(self, *, valid_for_ns: int = 50_000_000) -> None:
        if valid_for_ns <= 0:
            raise ValueError("valid_for_ns must be > 0")
        self._valid_for_ns = valid_for_ns
        self._sequence = 0

    def decide(self, observation: ControlObservation) -> ControlIntent:
        self._sequence += 1
        issued = observation.created_monotonic_ns
        reason = "hold_valid" if observation.safety.auto_allowed else "hold_safety_disallowed"
        return ControlIntent(
            sequence=self._sequence,
            observation_sequence=observation.sequence,
            issued_monotonic_ns=issued,
            valid_until_monotonic_ns=issued + self._valid_for_ns,
            mode="shadow",
            yaw_rate_rad_s=0.0,
            pitch_rate_rad_s=0.0,
            limits=ControlIntentLimits(),
            reason=reason,
        )


@dataclass(frozen=True)
class ReplayResult:
    observations: int
    intents: int
    rejected: int


def replay_observations(
    observations: Iterable[ControlObservation], policy: ControlPolicy
) -> tuple[list[ControlIntent], ReplayResult]:
    """Run validated snapshots in recorded order without consulting a clock."""

    intents: list[ControlIntent] = []
    rejected = 0
    previous_sequence = -1
    previous_time = -1
    for observation in observations:
        if (
            observation.sequence <= previous_sequence
            or observation.created_monotonic_ns < previous_time
        ):
            rejected += 1
            continue
        previous_sequence = observation.sequence
        previous_time = observation.created_monotonic_ns
        intents.append(policy.decide(observation))
    return intents, ReplayResult(
        observations=len(intents) + rejected, intents=len(intents), rejected=rejected
    )
