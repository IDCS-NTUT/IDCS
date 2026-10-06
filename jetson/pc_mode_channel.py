"""Acknowledged PC service actions over a PC-initiated polling connection."""

from __future__ import annotations

import json
import secrets
import threading
import time
from typing import Callable

from common.pc_mode import PcModeTask, parse_poll, sim_activity_units_running
from common.control_activity import ActivityCache


class PcModeChannel:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.active_units: frozenset[str] = frozenset()
        self.last_seen: float | None = None
        self.session: str | None = None
        self.activity = ActivityCache("sim")
        self._next_id = secrets.randbits(53) or 1
        self._pending: PcModeTask | None = None
        self._result: tuple[int, str] | None = None
        self._running = False
        self._condition = threading.Condition()

    def online(self) -> bool:
        return self.last_seen is not None and self.clock() - self.last_seen <= 3.0

    def pending(self) -> bool:
        with self._condition:
            return self._pending is not None

    def poll(self, raw: bytes | str) -> str:
        poll = parse_poll(raw)
        with self._condition:
            if self.session != poll.session:
                if self.online():
                    raise ValueError("another PC mode agent is already connected")
                # A restarted peer may have lost its task receipt. Fail the
                # waiter; do not replay a command into a different session.
                if self._pending is not None:
                    self._pending = None
                    self._result = (1, "PC agent restarted during the action")
                self.session = poll.session
            self.active_units, self.last_seen = poll.active_units, self.clock()
            if poll.activity is None or not sim_activity_units_running(poll.active_units):
                self.activity.invalidate()
            else:
                self.activity.accept(poll.activity, self.last_seen)
            if self._pending is not None and poll.completed_id == self._pending.task_id:
                self._result = (0 if poll.ok else 1, poll.detail)
                self._pending = None
            self._condition.notify_all()
            return (self._pending.to_json() if self._pending is not None
                    else json.dumps({"type": "PcModeTask", "id": None}))

    def run(self, args: list[str], *, timeout_s: float = 35.0) -> tuple[int, str]:
        verb, unit = args
        with self._condition:
            if not self.online():
                return 1, "PC mode agent is offline"
            if self._pending is not None or self._running:
                return 1, "PC action acknowledgement is still pending"
            self._next_id += 1
            self._pending = PcModeTask(self._next_id, verb, unit)
            self._result = None
            self._running = True
            try:
                if not self._condition.wait_for(lambda: self._result is not None, timeout_s):
                    # Retain the pending action until it is acknowledged. This
                    # prevents a delayed start overtaking a later stop.
                    return 1, "PC action acknowledgement timed out"
                assert self._result is not None
                return self._result
            finally:
                self._running = False
