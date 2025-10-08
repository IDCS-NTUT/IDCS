"""Phase gate primitives for stepping the inference pipeline."""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import Iterator, Optional


class PipelinePhase(str, Enum):
    """Enumerates the ordered phases of the Jetson pipeline."""

    CAPTURE_DECODE = "capture"
    DETECT_POSTPROCESS = "detect"
    PREDICT_LEAD = "predict"
    PARALLAX_PROJECTION = "parallax"
    CONTROLLER = "controller"
    PUBLISH_OVERLAY = "publish"


class PhaseGateController:
    """Coordinates deterministic stepping across pipeline phases."""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._phases = list(PipelinePhase)
        self._phase_count = len(self._phases)
        self._lock = threading.Condition()
        self._next_token_to_assign = 0
        self._last_completed_token = -1
        self._completed_cycles = 0
        self._allowed_token_count: Optional[int]
        if self._enabled:
            self._allowed_token_count = 0
        else:
            self._allowed_token_count = None

    def phase(self, phase: PipelinePhase):
        """Return a context manager guarding ``phase`` progression."""

        if not self._enabled:
            return nullcontext()
        return self._phase_context(phase)

    def step(self, phase: PipelinePhase) -> bool:
        """Permit the next blocked ``phase`` to execute."""

        if not self._enabled:
            return True
        with self._lock:
            assert self._allowed_token_count is not None
            expected = self._phases[self._allowed_token_count % self._phase_count]
            if expected is not phase:
                return False
            self._allowed_token_count += 1
            self._lock.notify_all()
            return True

    def step_all(self) -> None:
        """Permit the remainder of the current cycle to execute."""

        if not self._enabled:
            return
        with self._lock:
            assert self._allowed_token_count is not None
            remaining = self._phase_count - (self._allowed_token_count % self._phase_count)
            if remaining <= 0:
                remaining = self._phase_count
            self._allowed_token_count += remaining
            self._lock.notify_all()

    @property
    def completed_cycles(self) -> int:
        """Return the number of fully completed cycles."""

        if not self._enabled:
            return self._next_token_to_assign // self._phase_count
        with self._lock:
            return self._completed_cycles

    @contextmanager
    def _phase_context(self, phase: PipelinePhase) -> Iterator[None]:
        token = self._acquire(phase)
        try:
            yield
        finally:
            self._release(token)

    def _acquire(self, phase: PipelinePhase) -> int:
        with self._lock:
            token = self._next_token_to_assign
            expected = self._phases[token % self._phase_count]
            if expected is not phase:
                raise RuntimeError(
                    f"phase order violation: expected {expected.value}, got {phase.value}"
                )
            self._next_token_to_assign += 1
            while (
                self._allowed_token_count is not None
                and token >= self._allowed_token_count
            ):
                self._lock.wait()
            return token

    def _release(self, token: int) -> None:
        with self._lock:
            self._last_completed_token = token
            if token % self._phase_count == self._phase_count - 1:
                self._completed_cycles = (token + 1) // self._phase_count
            self._lock.notify_all()


__all__ = ["PipelinePhase", "PhaseGateController"]
