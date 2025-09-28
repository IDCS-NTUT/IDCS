"""Depth (Z-axis) tracking utilities for range prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TrackingZMeasurementNoise:
    base_m: float
    small_box_px: float


@dataclass(frozen=True)
class TrackingZProcessNoise:
    z: float


@dataclass(frozen=True)
class TrackingZConfig:
    enabled: bool
    meas_src_priority: Tuple[str, ...]
    meas_noise: TrackingZMeasurementNoise
    process_noise: TrackingZProcessNoise

    @classmethod
    def from_raw_config(cls, cfg: Mapping[str, Any]) -> "TrackingZConfig":
        section = cfg.get("tracking_z", {}) or {}
        if not isinstance(section, Mapping):
            section = {}

        enabled = bool(section.get("enabled", False))

        raw_priority: Sequence[Any] = section.get("meas_src_priority", ("known_size",)) or ()
        priority: Tuple[str, ...] = tuple(
            str(item).strip().lower()
            for item in raw_priority
            if isinstance(item, (str, bytes)) and str(item).strip()
        )
        if not priority:
            priority = ("known_size",)

        meas_section = section.get("meas_noise_m", {}) or {}
        try:
            base = float(meas_section.get("base", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError("tracking_z.meas_noise_m.base must be numeric") from exc
        if base <= 0.0:
            raise ValueError("tracking_z.meas_noise_m.base must be positive")

        try:
            small_box = float(meas_section.get("min_box_px", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("tracking_z.meas_noise_m.min_box_px must be numeric") from exc
        if small_box < 0.0:
            raise ValueError("tracking_z.meas_noise_m.min_box_px cannot be negative")

        proc_section = section.get("process_noise", {}) or {}
        try:
            q_z = float(proc_section.get("z", 0.3))
        except (TypeError, ValueError) as exc:
            raise ValueError("tracking_z.process_noise.z must be numeric") from exc
        if q_z < 0.0:
            raise ValueError("tracking_z.process_noise.z must be non-negative")

        return cls(
            enabled=enabled,
            meas_src_priority=priority,
            meas_noise=TrackingZMeasurementNoise(base_m=base, small_box_px=small_box),
            process_noise=TrackingZProcessNoise(z=q_z),
        )


@dataclass
class TrackingZMeasurement:
    value_m: float
    source: str
    box_size_px: Optional[Tuple[float, float]] = None
    confidence: Optional[float] = None


@dataclass
class TrackingZPrediction:
    distance_m: float
    velocity_mps: float
    source: str
    horizon_s: float


class ZTracker:
    """Constant-velocity tracker for target range (depth)."""

    def __init__(self, config: TrackingZConfig) -> None:
        self._cfg = config
        self._state: Optional[Tuple[float, float]] = None  # (z, z_dot)
        self._cov: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self._last_source: Optional[str] = None

    def reset(self) -> None:
        self._state = None
        self._cov = None
        self._last_source = None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, dt: float) -> None:
        if self._state is None:
            return
        if dt <= 0.0:
            return

        z, dz = self._state
        z += dz * dt
        self._state = (z, dz)

        if self._cov is None:
            return

        p00, p01 = self._cov[0]
        p10, p11 = self._cov[1]
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self._cfg.process_noise.z

        f00 = 1.0
        f01 = dt
        f10 = 0.0
        f11 = 1.0

        # F * P
        fp00 = f00 * p00 + f01 * p10
        fp01 = f00 * p01 + f01 * p11
        fp10 = f10 * p00 + f11 * p10
        fp11 = f10 * p01 + f11 * p11

        # P = (F * P) * F^T + Q
        new00 = fp00 * f00 + fp01 * f01 + q * dt4 * 0.25
        new01 = fp00 * f10 + fp01 * f11 + q * dt3 * 0.5
        new10 = fp10 * f00 + fp11 * f01 + q * dt3 * 0.5
        new11 = fp10 * f10 + fp11 * f11 + q * dt2

        self._cov = ((new00, new01), (new10, new11))

    # ------------------------------------------------------------------
    # Measurement update
    # ------------------------------------------------------------------
    def update(self, measurement: TrackingZMeasurement) -> bool:
        value = float(measurement.value_m)
        if not (value > 0.0):
            return False

        source = str(measurement.source).strip().lower() or "unknown"

        if self._state is None or self._cov is None:
            var = self._measurement_variance(measurement)
            self._state = (value, 0.0)
            self._cov = ((var, 0.0), (0.0, max(var, 1.0)))
            self._last_source = source
            return True

        var = self._measurement_variance(measurement)
        z, dz = self._state
        p00, p01 = self._cov[0]
        p10, p11 = self._cov[1]

        resid = value - z
        s = p00 + var
        if s <= 0.0:
            return False

        inv_s = 1.0 / s
        k0 = p00 * inv_s
        k1 = p10 * inv_s

        z += k0 * resid
        dz += k1 * resid
        self._state = (z, dz)

        new00 = (1.0 - k0) * p00
        new01 = (1.0 - k0) * p01
        new10 = p10 - k1 * p00
        new11 = p11 - k1 * p01

        self._cov = ((new00, new01), (new10, new11))
        self._last_source = source
        return True

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def project(self, horizon_s: float) -> Optional[TrackingZPrediction]:
        if self._state is None:
            return None

        horizon_s = max(0.0, float(horizon_s))
        z, dz = self._state
        z_pred = z + dz * horizon_s
        if not (z_pred > 0.0):
            return None

        source = self._last_source or "unknown"
        return TrackingZPrediction(
            distance_m=z_pred,
            velocity_mps=dz,
            source=source,
            horizon_s=horizon_s,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _measurement_variance(self, measurement: TrackingZMeasurement) -> float:
        sigma = self._cfg.meas_noise.base_m
        if measurement.source == "known_size" and measurement.box_size_px is not None:
            min_dim = max(1e-3, min(measurement.box_size_px))
            if self._cfg.meas_noise.small_box_px > 0.0 and min_dim < self._cfg.meas_noise.small_box_px:
                scale = self._cfg.meas_noise.small_box_px / min_dim
                sigma *= max(1.0, scale)
        if measurement.confidence is not None:
            conf = min(max(float(measurement.confidence), 0.0), 1.0)
            sigma *= 1.0 + (1.0 - conf)
        return sigma * sigma
