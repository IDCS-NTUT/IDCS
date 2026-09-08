"""Fixed-cadence, latest-only controller boundary.

The production controller is deliberately kept independent of video-frame
arrival.  This module owns the small amount of scheduling policy needed by a
controller sidecar: detections are accepted at their local receipt time and
the existing :class:`ControlLoop` advances at a fixed monotonic cadence.
It contains no gimbal or serial access; a caller must explicitly choose its
own ControlCmd publisher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol

from common.schemas import CamState, DetectionMsg


class _Controller(Protocol):
    def update_detection(self, msg: DetectionMsg, *, received_at: Optional[float] = None) -> None: ...

    def update_cam_state(self, state: CamState) -> None: ...

    def tick(self, now: Optional[float] = None) -> None: ...


@dataclass(frozen=True)
class FixedRateStats:
    """Counters describing the scheduling boundary, suitable for health logs."""

    ticks: int
    detections: int
    cam_states: int
    missed_periods: int


class FixedRateController:
    """Drive one controller at a fixed rate using a monotonic clock.

    ``advance`` never emits a burst of stale commands when the process is
    delayed.  It runs one current tick, records skipped periods, then resumes
    from that tick.  The underlying controller retains target-age protection.
    """

    def __init__(self, controller: _Controller, *, loop_hz: float) -> None:
        if not math.isfinite(loop_hz) or loop_hz <= 0.0:
            raise ValueError("loop_hz must be finite and > 0")
        self._controller = controller
        self.period_s = 1.0 / float(loop_hz)
        self._next_tick: Optional[float] = None
        self._ticks = 0
        self._detections = 0
        self._cam_states = 0
        self._missed_periods = 0

    def update_detection(self, msg: DetectionMsg, *, received_at: float) -> None:
        if not math.isfinite(received_at):
            raise ValueError("received_at must be finite")
        self._controller.update_detection(msg, received_at=received_at)
        self._detections += 1

    def update_cam_state(self, state: CamState) -> None:
        self._controller.update_cam_state(state)
        self._cam_states += 1

    def advance(self, now: float) -> bool:
        """Run a due tick and return whether a command was advanced."""

        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if self._next_tick is None:
            self._next_tick = now
        if now + 1e-12 < self._next_tick:
            return False

        overdue = max(0.0, now - self._next_tick)
        skipped = int(overdue / self.period_s)
        self._missed_periods += skipped
        # Use the current time rather than replaying old periods: commands
        # based on an old target must never catch up in a burst.
        self._controller.tick(now)
        self._ticks += 1
        self._next_tick = now + self.period_s
        return True

    @property
    def stats(self) -> FixedRateStats:
        return FixedRateStats(
            ticks=self._ticks,
            detections=self._detections,
            cam_states=self._cam_states,
            missed_periods=self._missed_periods,
        )
