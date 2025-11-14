"""Reference and weighting utilities for the MPC gimbal controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from common.control import (
    AxisPair,
    ControlConfig,
    MpcHorizonConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.schemas import CamState


AxisName = str


@dataclass(frozen=True)
class AxisReferenceSequences:
    """Per-axis sequences consumed by :class:`MpcAxisController`."""

    theta: Tuple[float, ...]
    omega: Optional[Tuple[float, ...]]
    distance: Tuple[Optional[float], ...]
    lateral: Tuple[Optional[float], ...]
    radial: Tuple[Optional[float], ...]


class MpcReferenceBuilder:
    """Constructs per-axis reference and weighting sequences each tick."""

    def __init__(
        self,
        control_cfg: ControlConfig,
        horizon_cfg: MpcHorizonConfig,
        *,
        axes: Sequence[AxisName] = ("yaw", "pitch"),
    ) -> None:
        allowed = {"yaw", "pitch"}
        selected = []
        for axis in axes:
            axis_name = axis.lower()
            if axis_name not in allowed:
                raise ValueError(f"unsupported axis '{axis}'")
            if axis_name not in selected:
                selected.append(axis_name)
        if not selected:
            raise ValueError("at least one axis must be selected")

        self._cfg = control_cfg
        self._horizon = horizon_cfg
        self._axes = tuple(selected)
        self._default_distance = float(control_cfg.laser.default_distance_m)
        self._last_distance: Optional[float] = None
        self._last_distance_ts: Optional[float] = None

    @property
    def axes(self) -> Tuple[AxisName, ...]:
        return self._axes

    def build(
        self,
        *,
        target_uv: Tuple[float, float],
        aim_uv: Tuple[float, float],
        timestamp: float,
        cam_state: Optional[CamState] = None,
        theta_estimates: Optional[Mapping[AxisName, float]] = None,
        omega_estimates: Optional[Mapping[AxisName, float]] = None,
        distance_m: Optional[float] = None,
        target_velocity_px_s: Optional[Tuple[float, float]] = None,
    ) -> Dict[AxisName, AxisReferenceSequences]:
        """Return reference stacks for the configured axes."""

        cfg = self._cfg
        px_err = pixel_delta(
            target_uv[0],
            target_uv[1],
            aim_uv[0],
            aim_uv[1],
            cfg,
            apply_deadband=True,
        )
        err_rad = angular_error_from_pixel_delta(px_err, cfg)

        angular_vel, has_velocity = self._angular_velocity_from_pixels(target_velocity_px_s)
        distance = self._resolve_distance(distance_m)
        radial_vel = self._estimate_radial_velocity(distance, timestamp)
        distance_seq = self._project_distance(distance, radial_vel)
        lateral_seq = self._project_lateral(distance, angular_vel)
        radial_seq = self._repeat(radial_vel)

        references: Dict[AxisName, AxisReferenceSequences] = {}
        for axis in self._axes:
            theta0 = self._resolve_theta(axis, cam_state, theta_estimates)
            theta_err = err_rad.yaw if axis == "yaw" else err_rad.pitch
            target_rate = angular_vel.yaw if axis == "yaw" else angular_vel.pitch
            theta_seq = self._project_theta(theta0 + theta_err, target_rate)

            omega_seq: Optional[Tuple[float, ...]] = None
            if has_velocity:
                omega_base = self._resolve_omega(axis, cam_state, omega_estimates)
                omega_seq = self._repeat(target_rate + omega_base)

            references[axis] = AxisReferenceSequences(
                theta=theta_seq,
                omega=omega_seq,
                distance=distance_seq,
                lateral=lateral_seq,
                radial=radial_seq,
            )

        return references

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    def _project_theta(self, theta0: float, omega: float) -> Tuple[float, ...]:
        seq = []
        current = float(theta0)
        Ts = float(self._horizon.sample_time_s)
        for step in range(self._horizon.prediction_horizon):
            if step > 0:
                current = float(theta0 + omega * Ts * step)
            seq.append(current)
        return tuple(seq)

    def _project_distance(self, distance: float, radial_vel: float) -> Tuple[Optional[float], ...]:
        seq = []
        current = distance
        Ts = float(self._horizon.sample_time_s)
        for _ in range(self._horizon.prediction_horizon):
            seq.append(max(0.0, current))
            current += radial_vel * Ts
        return tuple(seq)

    def _project_lateral(self, distance: float, angular_vel: AxisPair) -> Tuple[Optional[float], ...]:
        tangential = abs(distance) * math.hypot(angular_vel.yaw, angular_vel.pitch)
        return tuple(tangential for _ in range(self._horizon.prediction_horizon))

    def _repeat(self, value: float) -> Tuple[float, ...]:
        return tuple(value for _ in range(self._horizon.prediction_horizon))

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------
    def _resolve_theta(
        self,
        axis: AxisName,
        cam_state: Optional[CamState],
        theta_estimates: Optional[Mapping[AxisName, float]],
    ) -> float:
        if theta_estimates and axis in theta_estimates:
            raw = theta_estimates[axis]
            if raw is not None and math.isfinite(raw):
                return float(raw)
        if cam_state is not None:
            value = cam_state.pan if axis == "yaw" else cam_state.tilt
            if value is not None and math.isfinite(value):
                return float(value)
        return 0.0

    def _resolve_omega(
        self,
        axis: AxisName,
        cam_state: Optional[CamState],
        omega_estimates: Optional[Mapping[AxisName, float]],
    ) -> float:
        if omega_estimates and axis in omega_estimates:
            raw = omega_estimates[axis]
            if raw is not None and math.isfinite(raw):
                return float(raw)
        if cam_state is not None:
            value = cam_state.pan_rate if axis == "yaw" else cam_state.tilt_rate
            if value is not None and math.isfinite(value):
                return float(value)
        return 0.0

    def _resolve_distance(self, measurement: Optional[float]) -> float:
        if measurement is None or not math.isfinite(measurement) or measurement <= 0.0:
            return self._default_distance
        return float(measurement)

    def _estimate_radial_velocity(self, distance: float, timestamp: float) -> float:
        if self._last_distance is None or self._last_distance_ts is None:
            self._last_distance = distance
            self._last_distance_ts = timestamp
            return 0.0
        dt = timestamp - self._last_distance_ts
        if dt <= 1e-6:
            return 0.0
        vel = (distance - self._last_distance) / dt
        self._last_distance = distance
        self._last_distance_ts = timestamp
        return float(vel)

    def _angular_velocity_from_pixels(
        self, velocity_px: Optional[Tuple[float, float]]
    ) -> Tuple[AxisPair, bool]:
        if velocity_px is None:
            return AxisPair(0.0, 0.0), False
        cfg = self._cfg
        yaw_rate = cfg.yaw_sign * float(velocity_px[0]) / cfg.fx_px
        pitch_rate = cfg.pitch_sign * float(velocity_px[1]) / cfg.fy_px
        return AxisPair(yaw=yaw_rate, pitch=pitch_rate), True


__all__ = ["AxisReferenceSequences", "MpcReferenceBuilder"]

