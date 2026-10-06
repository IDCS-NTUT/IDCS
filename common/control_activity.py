"""Read-only evidence from the intent consumer, never command authority."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from common.schemas import ControlIntent


class AppliedControlActivity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    type: Literal["AppliedControlActivity"] = "AppliedControlActivity"
    version: Literal[1] = 1
    backend: Literal["sim", "gimbal"]
    session: str = Field(min_length=1, max_length=80)
    sequence: int = Field(ge=0)
    created_monotonic_ns: int = Field(ge=0)
    intent_sequence: int | None = Field(default=None, ge=0)
    mode: Literal["shadow", "live"] = "shadow"
    reason: str = Field(min_length=1, max_length=80)
    applied: bool = False


class ActivityEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    type: Literal["ActivityEnvelope"] = "ActivityEnvelope"
    activity: AppliedControlActivity
    # Accumulated same-host age before forwarding, not remote-clock subtraction.
    age_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class SimActivityRelay(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    type: Literal["SimActivityRelay"] = "SimActivityRelay"
    version: Literal[1] = 1
    activity: ActivityEnvelope | None = None


class IntentActivityMonitor:
    """Mark only after a sink applies/forwards an accepted intent.

    Zero rate at a centred target still counts as active tracking. A failed send,
    shadow request, expired lease or unavailable sink never counts.
    """

    def __init__(self, backend: Literal["sim", "gimbal"], *, session: str | None = None):
        self.backend = backend
        self.session = session or uuid.uuid4().hex
        self.sequence = 0
        self.intent: ControlIntent | None = None

    def mark_applied(self, intent: ControlIntent) -> None:
        self.intent = intent

    def clear(self) -> None:
        self.intent = None

    def sample(self, *, now_ns: int | None = None, ready: bool = True) -> AppliedControlActivity:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        self.sequence += 1
        intent = self.intent
        valid = (ready and intent is not None and intent.mode == "live"
                 and intent.issued_monotonic_ns <= now_ns <= intent.valid_until_monotonic_ns)
        return AppliedControlActivity(backend=self.backend, session=self.session, sequence=self.sequence,
            created_monotonic_ns=now_ns, intent_sequence=None if intent is None else intent.sequence,
            mode="shadow" if intent is None else intent.mode,
            reason="inactive" if not valid else intent.reason, applied=bool(valid))


class ActivityCache:
    """Fail-closed progress/freshness guard; duplicate relay polls do not renew it."""

    def __init__(self, backend: Literal["sim", "gimbal"], stale_s: float = .5):
        self.backend, self.stale_s = backend, stale_s
        self.envelope: ActivityEnvelope | None = None
        self.received_s: float | None = None
        self._retired: set[str] = set()
        self._invalidated = False

    def invalidate(self) -> None:
        self._invalidated = True

    def accept(self, envelope: ActivityEnvelope, now_s: float) -> bool:
        activity = envelope.activity
        if activity.backend != self.backend or activity.session in self._retired:
            return False
        if self.envelope is not None:
            old = self.envelope.activity
            if activity.session == old.session and activity.sequence <= old.sequence:
                return False
            if activity.session != old.session:
                self._retired.add(old.session)
        self.envelope, self.received_s = envelope, now_s
        self._invalidated = False
        return True

    def age_ms(self, now_s: float) -> float | None:
        if self.envelope is None or self.received_s is None or now_s < self.received_s:
            return None
        return self.envelope.age_ms + (now_s - self.received_s) * 1000

    def fresh(self, now_s: float) -> bool:
        age = self.age_ms(now_s)
        return not self._invalidated and age is not None and age < self.stale_s * 1000

    def tracking(self, now_s: float) -> bool:
        return self._active_reason(now_s, "tracking")

    def manual(self, now_s: float) -> bool:
        return self._active_reason(now_s, "manual")

    def _active_reason(self, now_s: float, reason: str) -> bool:
        if not self.fresh(now_s) or self.envelope is None:
            return False
        state = self.envelope.activity
        return state.applied and state.mode == "live" and state.reason == reason

    def forward(self, now_s: float) -> ActivityEnvelope | None:
        age = self.age_ms(now_s)
        if self._invalidated or age is None or self.envelope is None:
            return None
        return ActivityEnvelope(activity=self.envelope.activity, age_ms=age)
