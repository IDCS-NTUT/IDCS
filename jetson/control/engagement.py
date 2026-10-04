"""Fire-button engagement events: the operator's confirmation, recorded.

There is no effector yet: an engagement is a logged decision. A press (rising
edge of the panel's fire button) engages the controller's current target only
while auto control is armed (master arm, auto enabled, no manual, no E-stop)
and the controller is tracking or coasting on a track. Any other press is
recorded as refused, with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.schemas import ControlIntent, ControlObservation


@dataclass(frozen=True)
class Engagement:
    track_id: int
    monotonic_ns: int


class EngagementMonitor:
    def __init__(self) -> None:
        self._fire_was_pressed = False
        self.last: Engagement | None = None

    def update(self, observation: ControlObservation, intent: ControlIntent) -> dict[str, Any] | None:
        """A record for a press this tick (engage or engage_refused), else None."""
        safety = observation.safety
        pressed = bool(safety.valid and safety.fire)
        rising = pressed and not self._fire_was_pressed
        self._fire_was_pressed = pressed
        if not rising:
            return None
        target = observation.target
        now_ns = observation.created_monotonic_ns
        refused = None
        if safety.emergency_active:
            refused = "emergency"
        elif not safety.master_arm:
            refused = "master_arm_off"
        elif safety.manual_active:
            refused = "manual"
        elif not safety.auto_allowed:
            refused = "auto_not_armed"
        elif intent.reason not in ("tracking", "coasting") or target.track_id is None:
            refused = f"not_tracking:{intent.reason}"
        record: dict[str, Any] = {
            "monotonic_ns": now_ns,
            "track_id": target.track_id,
            "on_target": target.on_target,
            "bearing_error_rad": target.bearing_error_rad,
            "controller_reason": intent.reason,
        }
        if refused is not None:
            return {"type": "engage_refused", "why": refused, **record}
        assert target.track_id is not None
        self.last = Engagement(int(target.track_id), now_ns)
        return {"type": "engage", **record}
