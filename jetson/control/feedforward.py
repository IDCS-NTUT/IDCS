"""Timestamped target-rate Kalman estimator; no PID or motor access."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetRateEstimate:
    valid: bool
    reason: str
    position_rad: float = 0.0
    rate_rad_s: float = 0.0
    sample_age_s: float = 0.0
    observations: int = 0


class TargetRateKalman:
    """One-axis constant-velocity filter in the observation clock domain."""

    def __init__(
        self, *, measurement_sigma_rad: float = 0.002,
        acceleration_sigma_rad_s2: float = 0.4,
        max_sample_age_s: float = 0.12,
        innovation_limit_rad: float = 0.06,
        min_observations: int = 4,
    ) -> None:
        if min(measurement_sigma_rad, acceleration_sigma_rad_s2,
               max_sample_age_s, innovation_limit_rad) <= 0:
            raise ValueError("Kalman noise and limits must be positive")
        if min_observations < 2:
            raise ValueError("min_observations must be at least two")
        self.measurement_variance = measurement_sigma_rad ** 2
        self.acceleration_variance = acceleration_sigma_rad_s2 ** 2
        self.max_sample_age_s = max_sample_age_s
        self.innovation_limit_rad = innovation_limit_rad
        self.min_observations = min_observations
        self.reset()

    def reset(self) -> None:
        self._track_id: int | None = None
        self._sample_ns: int | None = None
        self._position = 0.0
        self._rate = 0.0
        self._p00 = 1e-3
        self._p01 = 0.0
        self._p11 = 0.1
        self._observations = 0

    def observe(self, *, track_id: int, angle_rad: float, sample_ns: int) -> bool:
        if not math.isfinite(angle_rad) or sample_ns <= 0:
            self.reset()
            return False
        if self._track_id != track_id:
            self.reset()
            self._track_id = track_id
        if self._sample_ns is None:
            self._position = angle_rad
            self._sample_ns = sample_ns
            self._observations = 1
            return True
        if sample_ns <= self._sample_ns:
            return False
        dt = (sample_ns - self._sample_ns) / 1e9
        if dt > self.max_sample_age_s * 2:
            self.reset()
            self._track_id = track_id
            self._position = angle_rad
            self._sample_ns = sample_ns
            self._observations = 1
            return True
        position = self._position + self._rate * dt
        q = self.acceleration_variance
        p00 = self._p00 + 2 * dt * self._p01 + dt * dt * self._p11 + q * dt ** 4 / 4
        p01 = self._p01 + dt * self._p11 + q * dt ** 3 / 2
        p11 = self._p11 + q * dt * dt
        innovation = angle_rad - position
        if abs(innovation) > self.innovation_limit_rad:
            self.reset()
            self._track_id = track_id
            self._position = angle_rad
            self._sample_ns = sample_ns
            self._observations = 1
            return False
        denominator = p00 + self.measurement_variance
        k0, k1 = p00 / denominator, p01 / denominator
        self._position = position + k0 * innovation
        self._rate += k1 * innovation
        self._p00 = max((1 - k0) * p00, 0.0)
        self._p01 = (1 - k0) * p01
        self._p11 = max(p11 - k1 * p01, 0.0)
        self._sample_ns = sample_ns
        self._observations += 1
        return True

    def estimate(self, *, decision_ns: int, track_id: int) -> TargetRateEstimate:
        if self._track_id != track_id or self._sample_ns is None:
            return TargetRateEstimate(False, "target_uninitialized")
        age = (decision_ns - self._sample_ns) / 1e9
        if age < 0:
            return TargetRateEstimate(False, "sample_in_future")
        if age > self.max_sample_age_s:
            return TargetRateEstimate(False, "sample_stale", sample_age_s=age,
                                      observations=self._observations)
        if self._observations < self.min_observations:
            return TargetRateEstimate(False, "estimator_warmup", sample_age_s=age,
                                      observations=self._observations)
        return TargetRateEstimate(
            True, "ready", self._position + self._rate * age,
            self._rate, age, self._observations,
        )
