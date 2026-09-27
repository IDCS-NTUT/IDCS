"""Timestamped camera-pose alignment for source-time target observations.

Each axis is interpolated on its own measurement times
(``CamState.pan_sample_monotonic_ns`` / ``tilt_sample_monotonic_ns``), not
the CamState publication time: yaw and pitch are polled separately and a
sample may be republished several times. A pose is returned only when the
whole mapped capture interval is bracketed by measured samples on both axes
and the motion uncertainty across that interval is bounded.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from common.schemas import CamState
from jetson.control_v3.timing import TimeInterval


@dataclass(frozen=True)
class AlignedPose:
    yaw_rad: float
    pitch_rad: float
    capture_midpoint_ns: int
    timing_width_ns: int
    yaw_uncertainty_rad: float
    pitch_uncertainty_rad: float


class _AxisSamples:
    def __init__(self, max_samples: int) -> None:
        self.samples: deque[tuple[int, float]] = deque(maxlen=max_samples)

    def add(self, timestamp_ns: int | None, value: float) -> bool:
        if timestamp_ns is None or timestamp_ns <= 0 or not math.isfinite(value):
            return False
        if self.samples and timestamp_ns <= self.samples[-1][0]:
            return False  # republished or out-of-order sample
        self.samples.append((int(timestamp_ns), float(value)))
        return True

    def bracket(self, t_ns: int) -> tuple[tuple[int, float], tuple[int, float]] | None:
        items = list(self.samples)
        for left, right in zip(items, items[1:]):
            if left[0] <= t_ns <= right[0]:
                return left, right
        return None

    def at(self, t_ns: int) -> float | None:
        pair = self.bracket(t_ns)
        if pair is None:
            return None
        (t0, v0), (t1, v1) = pair
        return v0 + (t_ns - t0) / (t1 - t0) * (v1 - v0) if t1 > t0 else v0

    def at_or_latest(self, t_ns: int) -> float | None:
        """Interpolated value at ``t_ns``, or the newest sample if ``t_ns`` is later."""
        if not self.samples:
            return None
        if t_ns >= self.samples[-1][0]:
            return self.samples[-1][1]
        return self.at(t_ns)


class CameraPoseHistory:
    def __init__(
        self, *, max_samples: int = 256, max_interval_width_ns: int = 20_000_000,
        max_motion_uncertainty_rad: float = 0.003,
        max_bracket_span_ns: int = 100_000_000,
    ) -> None:
        if (max_samples < 2 or max_interval_width_ns <= 0
                or max_motion_uncertainty_rad <= 0 or max_bracket_span_ns <= 0):
            raise ValueError("invalid pose-history limits")
        self._max_samples = max_samples
        self._yaw = _AxisSamples(max_samples)
        self._pitch = _AxisSamples(max_samples)
        self.max_interval_width_ns = max_interval_width_ns
        self.max_motion_uncertainty_rad = max_motion_uncertainty_rad
        self.max_bracket_span_ns = max_bracket_span_ns

    def reset(self) -> None:
        self._yaw = _AxisSamples(self._max_samples)
        self._pitch = _AxisSamples(self._max_samples)

    def observe(self, state: CamState) -> bool:
        """Record any new per-axis measurement; True if at least one was new."""
        yaw_new = self._yaw.add(state.pan_sample_monotonic_ns, state.pan)
        pitch_new = self._pitch.add(state.tilt_sample_monotonic_ns, state.tilt)
        return yaw_new or pitch_new

    def pose_at_or_latest(self, t_ns: int) -> tuple[float, float] | None:
        yaw = self._yaw.at_or_latest(t_ns)
        pitch = self._pitch.at_or_latest(t_ns)
        return None if yaw is None or pitch is None else (yaw, pitch)

    def at(self, interval: TimeInterval) -> tuple[AlignedPose | None, str]:
        width = interval.latest_ns - interval.earliest_ns
        if width < 0 or width > self.max_interval_width_ns:
            return None, "capture_clock_interval_too_wide"
        if len(self._yaw.samples) < 2 or len(self._pitch.samples) < 2:
            return None, "pose_history_warmup"
        midpoint = (interval.earliest_ns + interval.latest_ns) // 2
        values = []
        for axis in (self._yaw, self._pitch):
            if interval.earliest_ns < axis.samples[0][0] or interval.latest_ns > axis.samples[-1][0]:
                return None, "capture_not_bracketed_by_encoder"
            pair = axis.bracket(midpoint)
            if pair is None:
                return None, "pose_interpolation_unavailable"
            (t0, v0), (t1, v1) = pair
            if t1 - t0 > self.max_bracket_span_ns:
                return None, "pose_bracket_too_sparse"
            value = axis.at(midpoint)
            assert value is not None
            rate = 0.0 if t1 == t0 else abs(v1 - v0) / ((t1 - t0) / 1e9)
            values.append((value, rate * width / 2e9))
        (yaw, yaw_unc), (pitch, pitch_unc) = values
        if max(yaw_unc, pitch_unc) > self.max_motion_uncertainty_rad:
            return None, "pose_motion_uncertainty_exceeded"
        return AlignedPose(yaw, pitch, midpoint, width, yaw_unc, pitch_unc), "aligned"
