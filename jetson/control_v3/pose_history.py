"""Timestamped camera-pose alignment for source-time target observations.

Returns a pose only when the entire mapped capture-time interval is bracketed
by measured encoder samples and its motion uncertainty is bounded.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

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


class CameraPoseHistory:
    def __init__(
        self, *, max_samples: int = 256, max_interval_width_ns: int = 20_000_000,
        max_motion_uncertainty_rad: float = 0.003,
        max_bracket_span_ns: int = 100_000_000,
        pose_source: Literal["encoder", "render"] = "encoder",
    ) -> None:
        if (max_samples < 2 or max_interval_width_ns <= 0
                or max_motion_uncertainty_rad <= 0 or max_bracket_span_ns <= 0):
            raise ValueError("invalid pose-history limits")
        if pose_source not in {"encoder", "render"}:
            raise ValueError("pose source must be encoder or render")
        self._samples: deque[tuple[int, float, float]] = deque(maxlen=max_samples)
        self.max_interval_width_ns = max_interval_width_ns
        self.max_motion_uncertainty_rad = max_motion_uncertainty_rad
        self.max_bracket_span_ns = max_bracket_span_ns
        self.pose_source = pose_source

    def reset(self) -> None:
        self._samples.clear()

    def observe(self, state: CamState) -> bool:
        timestamp = state.state_monotonic_ns
        if self.pose_source == "render":
            if (
                state.render_pan is None or state.render_tilt is None
                or state.render_prediction_age_ms is None
                or state.render_prediction_age_ms > 100.0
            ):
                return False
            pan, tilt = state.render_pan, state.render_tilt
        else:
            pan, tilt = state.pan, state.tilt
        if timestamp is None or timestamp <= 0 or not all(
            math.isfinite(value) for value in (pan, tilt)
        ):
            return False
        if self._samples and timestamp <= self._samples[-1][0]:
            return False
        self._samples.append((timestamp, float(pan), float(tilt)))
        return True

    def at(self, interval: TimeInterval) -> tuple[AlignedPose | None, str]:
        width = interval.latest_ns - interval.earliest_ns
        if width < 0 or width > self.max_interval_width_ns:
            return None, "capture_clock_interval_too_wide"
        if len(self._samples) < 2:
            return None, "pose_history_warmup"
        if interval.earliest_ns < self._samples[0][0] or interval.latest_ns > self._samples[-1][0]:
            return None, "capture_not_bracketed_by_encoder"
        midpoint = (interval.earliest_ns + interval.latest_ns) // 2
        samples = list(self._samples)
        before = after = None
        for left, right in zip(samples, samples[1:]):
            if left[0] <= midpoint <= right[0]:
                before, after = left, right
                break
        if before is None or after is None:
            return None, "pose_interpolation_unavailable"
        dt_ns = after[0] - before[0]
        if dt_ns > self.max_bracket_span_ns:
            return None, "pose_bracket_too_sparse"
        fraction = (midpoint - before[0]) / dt_ns
        yaw = before[1] + fraction * (after[1] - before[1])
        pitch = before[2] + fraction * (after[2] - before[2])
        half_width_s = width / 2e9
        yaw_uncertainty = abs(after[1] - before[1]) / (dt_ns / 1e9) * half_width_s
        pitch_uncertainty = abs(after[2] - before[2]) / (dt_ns / 1e9) * half_width_s
        if max(yaw_uncertainty, pitch_uncertainty) > self.max_motion_uncertainty_rad:
            return None, "pose_motion_uncertainty_exceeded"
        return AlignedPose(yaw, pitch, midpoint, width,
                           yaw_uncertainty, pitch_uncertainty), "aligned"
