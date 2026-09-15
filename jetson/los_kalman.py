"""Timestamp-aware line-of-sight estimator with no transport dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LOSKalmanConfig:
    acceleration_spectral_density: float = 0.05
    measurement_variance_rad2: float = 9e-6
    initial_position_variance_rad2: float = 1e-4
    initial_rate_variance_rad2_s2: float = 0.25
    innovation_gate_nis: float = 16.0
    max_gap_s: float = 0.25
    max_consecutive_rejections: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.acceleration_spectral_density,
            self.measurement_variance_rad2,
            self.initial_position_variance_rad2,
            self.initial_rate_variance_rad2_s2,
            self.innovation_gate_nis,
            self.max_gap_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("all LOS Kalman configuration values must be finite and positive")
        if self.max_consecutive_rejections < 1:
            raise ValueError("max_consecutive_rejections must be >= 1")


@dataclass(frozen=True)
class LOSEstimate:
    angle_rad: float
    rate_rad_s: float
    sample_time_s: float
    query_time_s: float
    accepted_updates: int
    rejected_updates: int
    reinitialized_updates: int


class AxisLOSKalman:
    """Constant-velocity Kalman filter for one absolute target-bearing axis."""

    def __init__(self, config: LOSKalmanConfig) -> None:
        self._config = config
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._sample_time_s: Optional[float] = None
        self._accepted_updates = 0
        self._rejected_updates = 0
        self._reinitialized_updates = 0
        self._consecutive_rejections = 0

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._sample_time_s = None
        self._accepted_updates = 0
        self._rejected_updates = 0
        self._reinitialized_updates = 0
        self._consecutive_rejections = 0

    def _initialize(self, angle_rad: float, sample_time_s: float, *, preserve_counters: bool) -> None:
        if not preserve_counters:
            self._accepted_updates = 0
            self._rejected_updates = 0
            self._reinitialized_updates = 0
        self._x = np.array([angle_rad, 0.0], dtype=float)
        self._P = np.diag(
            [self._config.initial_position_variance_rad2, self._config.initial_rate_variance_rad2_s2]
        )
        self._sample_time_s = sample_time_s
        self._consecutive_rejections = 0

    @staticmethod
    def _transition(dt_s: float) -> np.ndarray:
        return np.array([[1.0, dt_s], [0.0, 1.0]], dtype=float)

    def _process_covariance(self, dt_s: float) -> np.ndarray:
        q = self._config.acceleration_spectral_density
        return q * np.array(
            [[dt_s**3 / 3.0, dt_s**2 / 2.0], [dt_s**2 / 2.0, dt_s]],
            dtype=float,
        )

    def _predict_values(self, at_s: float) -> tuple[np.ndarray, np.ndarray]:
        if self._x is None or self._P is None or self._sample_time_s is None:
            raise RuntimeError("LOS estimator is not initialized")
        dt_s = at_s - self._sample_time_s
        if dt_s < -1e-12:
            raise ValueError("LOS estimator cannot predict backward")
        dt_s = max(0.0, dt_s)
        transition = self._transition(dt_s)
        return transition @ self._x, transition @ self._P @ transition.T + self._process_covariance(dt_s)

    def update(self, angle_rad: float, *, sample_time_s: float) -> bool:
        if not math.isfinite(angle_rad) or not math.isfinite(sample_time_s):
            raise ValueError("LOS measurements and timestamps must be finite")
        if self._sample_time_s is not None and sample_time_s <= self._sample_time_s:
            raise ValueError("LOS measurement timestamps must increase")
        if self._sample_time_s is not None and sample_time_s - self._sample_time_s > self._config.max_gap_s:
            self.reset()
        if self._x is None:
            self._initialize(angle_rad, sample_time_s, preserve_counters=True)
            self._accepted_updates = 1
            return True

        predicted_x, predicted_P = self._predict_values(sample_time_s)
        H = np.array([[1.0, 0.0]], dtype=float)
        innovation = angle_rad - float((H @ predicted_x).item())
        innovation_variance = float((H @ predicted_P @ H.T).item()) + self._config.measurement_variance_rad2
        self._sample_time_s = sample_time_s
        if innovation_variance <= 0.0 or innovation * innovation / innovation_variance > self._config.innovation_gate_nis:
            self._x = predicted_x
            self._P = predicted_P
            self._rejected_updates += 1
            self._consecutive_rejections += 1
            if self._consecutive_rejections >= self._config.max_consecutive_rejections:
                self._initialize(angle_rad, sample_time_s, preserve_counters=True)
                self._accepted_updates += 1
                self._reinitialized_updates += 1
                return True
            return False

        gain = (predicted_P @ H.T) / innovation_variance
        self._x = predicted_x + gain[:, 0] * innovation
        identity = np.eye(2, dtype=float)
        residual_transform = identity - gain @ H
        # Joseph form remains symmetric positive semidefinite under rounding.
        self._P = (
            residual_transform @ predicted_P @ residual_transform.T
            + gain * self._config.measurement_variance_rad2 @ gain.T
        )
        self._P = 0.5 * (self._P + self._P.T)
        self._accepted_updates += 1
        self._consecutive_rejections = 0
        return True

    def estimate(self, *, query_time_s: Optional[float] = None) -> Optional[LOSEstimate]:
        if self._sample_time_s is None:
            return None
        at_s = self._sample_time_s if query_time_s is None else float(query_time_s)
        if not math.isfinite(at_s):
            raise ValueError("LOS query timestamp must be finite")
        predicted_x, _ = self._predict_values(at_s)
        return LOSEstimate(
            angle_rad=float(predicted_x[0]),
            rate_rad_s=float(predicted_x[1]),
            sample_time_s=self._sample_time_s,
            query_time_s=at_s,
            accepted_updates=self._accepted_updates,
            rejected_updates=self._rejected_updates,
            reinitialized_updates=self._reinitialized_updates,
        )


class TargetLOSKalman:
    """Two-axis estimator that resets atomically on selected-target changes."""

    def __init__(self, config: LOSKalmanConfig) -> None:
        self._yaw = AxisLOSKalman(config)
        self._pitch = AxisLOSKalman(config)
        self._track_id: Optional[int] = None

    def reset(self) -> None:
        self._yaw.reset()
        self._pitch.reset()
        self._track_id = None

    def update(
        self,
        *,
        track_id: int,
        absolute_bearing_rad: Tuple[float, float],
        sample_time_s: float,
    ) -> tuple[bool, bool]:
        if self._track_id is not None and track_id != self._track_id:
            self.reset()
        self._track_id = track_id
        return (
            self._yaw.update(float(absolute_bearing_rad[0]), sample_time_s=sample_time_s),
            self._pitch.update(float(absolute_bearing_rad[1]), sample_time_s=sample_time_s),
        )

    def estimate(self, *, query_time_s: float) -> Optional[Tuple[LOSEstimate, LOSEstimate]]:
        yaw = self._yaw.estimate(query_time_s=query_time_s)
        pitch = self._pitch.estimate(query_time_s=query_time_s)
        return None if yaw is None or pitch is None else (yaw, pitch)
