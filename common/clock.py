"""Shared clock utilities for live and deterministic step modes."""

from __future__ import annotations

import threading
import time
from typing import Protocol


class MonotonicClock(Protocol):
    """Interface exposed by all pipeline clocks."""

    def now(self) -> float:
        """Return the current simulation time in seconds."""

    def now_ns(self) -> int:
        """Return the current simulation time in nanoseconds."""

    def now_ms(self) -> int:
        """Return the current simulation time in milliseconds."""

    def sleep(self, seconds: float) -> None:
        """Block until the given amount of simulation time has elapsed."""

    def sleep_until(self, target_time: float) -> None:
        """Block until the simulation clock reaches ``target_time``."""

    def wall_time(self) -> float:
        """Return the host wall-clock time in seconds."""

    def sleep_wall(self, seconds: float) -> None:
        """Block using wall-clock time irrespective of simulation progress."""


class RealClock:
    """Thin wrapper around :mod:`time` providing the :class:`MonotonicClock` API."""

    def now(self) -> float:
        return time.monotonic()

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def now_ms(self) -> int:
        return self.now_ns() // 1_000_000

    def sleep(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        time.sleep(seconds)

    def sleep_until(self, target_time: float) -> None:
        remaining = target_time - self.now()
        if remaining <= 0.0:
            return
        time.sleep(remaining)

    def wall_time(self) -> float:
        return time.time()

    def sleep_wall(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        time.sleep(seconds)


class StepClock:
    """Deterministic clock that advances only when commanded."""

    def __init__(self, *, start_time: float = 0.0) -> None:
        if start_time < 0.0:
            raise ValueError("start_time must be non-negative")
        self._current_ns = int(round(start_time * 1e9))
        self._lock = threading.Condition()

    def now(self) -> float:
        with self._lock:
            return self._current_ns / 1e9

    def now_ns(self) -> int:
        with self._lock:
            return self._current_ns

    def now_ms(self) -> int:
        return self.now_ns() // 1_000_000

    def sleep(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        self.sleep_until(self.now() + seconds)

    def sleep_until(self, target_time: float) -> None:
        target_ns = int(round(target_time * 1e9))
        with self._lock:
            while self._current_ns < target_ns:
                self._lock.wait()

    def advance(self, seconds: float) -> float:
        """Advance the clock by ``seconds`` and return the new time."""

        if seconds < 0.0:
            raise ValueError("cannot advance clock backwards")
        if seconds == 0.0:
            return self.now()
        delta_ns = int(round(seconds * 1e9))
        if delta_ns <= 0:
            delta_ns = 1  # ensure progress when rounding occurs
        with self._lock:
            self._current_ns += delta_ns
            self._lock.notify_all()
            return self._current_ns / 1e9

    def advance_to(self, target_time: float) -> float:
        """Advance the clock to ``target_time`` (seconds)."""

        target_ns = int(round(target_time * 1e9))
        with self._lock:
            if target_ns < self._current_ns:
                raise ValueError("cannot rewind clock")
            if target_ns == self._current_ns:
                return self._current_ns / 1e9
            self._current_ns = target_ns
            self._lock.notify_all()
            return self._current_ns / 1e9

    def wall_time(self) -> float:
        return time.time()

    def sleep_wall(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        time.sleep(seconds)


def create_clock(*, step_mode: bool = False) -> MonotonicClock:
    """Factory that returns a live or step-mode clock."""

    if step_mode:
        return StepClock()
    return RealClock()


__all__ = [
    "MonotonicClock",
    "RealClock",
    "StepClock",
    "create_clock",
]

