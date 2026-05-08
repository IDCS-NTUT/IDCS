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


@dataclass(frozen=True)
class AxisTargetPrediction:
    """Filtered target state for one MPC axis."""

    theta_base: float
    theta: float
    omega: float
    residual: float


@dataclass
class _AxisPredictorState:
    timestamp: float
    theta: float
    omega: float


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
        self._predictor_state: Dict[AxisName, _AxisPredictorState] = {}
        self._base_effect_delay_s = max(0.0, float(self._horizon.effect_delay_s))
        self._effect_delay_mode = str(self._horizon.effect_delay_mode).strip().lower()
        self._projectile_speed_m_s = (
            None
            if self._horizon.projectile_speed_m_s is None
            else float(self._horizon.projectile_speed_m_s)
        )
        self._impact_delay_bias_s = float(self._horizon.impact_delay_bias_s)
        self._adaptive_effect_delay_s: Dict[AxisName, float] = {
            axis: self._base_effect_delay_s for axis in self._axes
        }
        self._tuning_overrides: Dict[str, float] = {}

    @property
    def axes(self) -> Tuple[AxisName, ...]:
        return self._axes

    def effect_delay_for_axis(self, axis: AxisName) -> float:
        axis_name = axis.lower()
        if axis_name not in self._axes:
            return self._base_effect_delay_s
        return float(self._adaptive_effect_delay_s.get(axis_name, self._base_effect_delay_s))

    def set_tuning_overrides(self, overrides: Mapping[str, float]) -> None:
        allowed = {
            "predictor_alpha",
            "predictor_beta",
            "adaptive_effect_delay_alpha",
            "adaptive_effect_delay_gain",
            "adaptive_effect_delay_rate_eps",
        }
        for key, value in overrides.items():
            if key not in allowed:
                continue
            val = float(value)
            if not math.isfinite(val):
                continue
            self._tuning_overrides[key] = val

    def _resolve_tuning(self, key: str, default: float, *, minimum: float = 0.0) -> float:
        value = self._tuning_overrides.get(key, default)
        if not math.isfinite(value):
            value = default
        return max(minimum, float(value))

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
        nominal_delay = self._nominal_effect_delay(distance)
        for axis in self._axes:
            theta0 = self._resolve_theta(axis, cam_state, theta_estimates)
            theta_err = err_rad.yaw if axis == "yaw" else err_rad.pitch
            raw_target_rate = angular_vel.yaw if axis == "yaw" else angular_vel.pitch
            theta_target = theta0 + theta_err
            prediction = self._predict_target_axis(
                axis=axis,
                theta_base=theta0,
                theta_meas=theta_target,
                raw_rate=raw_target_rate,
                timestamp=timestamp,
            )
            theta_seed = prediction.theta
            target_rate = prediction.omega
            theta_residual = prediction.residual
            effect_delay = self._update_effect_delay(
                axis=axis,
                theta_residual=theta_residual,
                omega=target_rate,
                nominal_delay=nominal_delay,
            )
            theta_seq = self._project_theta(theta_seed, target_rate, effect_delay)

            omega_seq: Optional[Tuple[float, ...]] = None
            omega_base = self._resolve_omega(axis, cam_state, omega_estimates)
            if has_velocity and not self._horizon.predictor_enabled:
                omega_seq = self._repeat(target_rate + omega_base)
            elif self._horizon.predictor_enabled:
                omega_seq = self._repeat(target_rate + omega_base)

            references[axis] = AxisReferenceSequences(
                theta=theta_seq,
                omega=omega_seq,
                distance=distance_seq,
                lateral=lateral_seq,
                radial=radial_seq,
            )

        return references

    def preview_target_predictions(
        self,
        *,
        target_uv: Tuple[float, float],
        aim_uv: Tuple[float, float],
        timestamp: float,
        cam_state: Optional[CamState] = None,
        theta_estimates: Optional[Mapping[AxisName, float]] = None,
        target_velocity_px_s: Optional[Tuple[float, float]] = None,
    ) -> Dict[AxisName, AxisTargetPrediction]:
        """Return target-motion predictor output without mutating predictor state."""

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
        angular_vel, _ = self._angular_velocity_from_pixels(target_velocity_px_s)

        predictions: Dict[AxisName, AxisTargetPrediction] = {}
        for axis in self._axes:
            theta0 = self._resolve_theta(axis, cam_state, theta_estimates)
            theta_err = err_rad.yaw if axis == "yaw" else err_rad.pitch
            raw_target_rate = angular_vel.yaw if axis == "yaw" else angular_vel.pitch
            theta_target = theta0 + theta_err
            predictions[axis] = self._preview_target_axis(
                axis=axis,
                theta_base=theta0,
                theta_meas=theta_target,
                raw_rate=raw_target_rate,
                timestamp=timestamp,
            )
        return predictions

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    def _project_theta(
        self, theta0: float, omega: float, effect_delay_s: Optional[float] = None
    ) -> Tuple[float, ...]:
        seq = []
        lead_s = (
            max(0.0, float(effect_delay_s))
            if effect_delay_s is not None
            else self._base_effect_delay_s
        )
        current = float(theta0 + omega * lead_s)
        Ts = float(self._horizon.sample_time_s)
        for step in range(self._horizon.prediction_horizon):
            if step > 0:
                current = float(theta0 + omega * (lead_s + Ts * step))
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

    def _predict_target_axis(
        self,
        *,
        axis: AxisName,
        theta_base: float,
        theta_meas: float,
        raw_rate: float,
        timestamp: float,
    ) -> AxisTargetPrediction:
        if not self._horizon.predictor_enabled:
            prev = self._predictor_state.get(axis)
            theta_residual = 0.0
            if prev is not None:
                dt = float(timestamp - prev.timestamp)
                if math.isfinite(dt) and dt > 1e-6:
                    theta_pred = prev.theta + prev.omega * dt
                    theta_residual = float(theta_meas - theta_pred)
            self._predictor_state[axis] = _AxisPredictorState(
                timestamp=float(timestamp),
                theta=float(theta_meas),
                omega=float(raw_rate),
            )
            return AxisTargetPrediction(
                theta_base=float(theta_base),
                theta=float(theta_meas),
                omega=float(raw_rate),
                residual=float(theta_residual),
            )

        alpha = min(
            1.0,
            self._resolve_tuning(
                "predictor_alpha",
                float(self._horizon.predictor_alpha),
                minimum=0.0,
            ),
        )
        beta = min(
            1.0,
            self._resolve_tuning(
                "predictor_beta",
                float(self._horizon.predictor_beta),
                minimum=0.0,
            ),
        )

        prev = self._predictor_state.get(axis)
        if prev is None:
            state = _AxisPredictorState(
                timestamp=float(timestamp),
                theta=float(theta_meas),
                omega=float(raw_rate),
            )
            self._predictor_state[axis] = state
            return AxisTargetPrediction(
                theta_base=float(theta_base),
                theta=state.theta,
                omega=state.omega,
                residual=0.0,
            )

        dt = float(timestamp - prev.timestamp)
        if not math.isfinite(dt) or dt <= 1e-6:
            prev.timestamp = float(timestamp)
            prev.theta = float(theta_meas)
            prev.omega = float(raw_rate)
            return AxisTargetPrediction(
                theta_base=float(theta_base),
                theta=prev.theta,
                omega=prev.omega,
                residual=0.0,
            )

        theta_pred = prev.theta + prev.omega * dt
        omega_pred = prev.omega
        residual = float(theta_meas - theta_pred)

        theta_upd = theta_pred + alpha * residual
        omega_upd = omega_pred + (beta / dt) * residual

        prev.timestamp = float(timestamp)
        prev.theta = float(theta_upd)
        prev.omega = float(omega_upd)
        return AxisTargetPrediction(
            theta_base=float(theta_base),
            theta=prev.theta,
            omega=prev.omega,
            residual=float(residual),
        )

    def _preview_target_axis(
        self,
        *,
        axis: AxisName,
        theta_base: float,
        theta_meas: float,
        raw_rate: float,
        timestamp: float,
    ) -> AxisTargetPrediction:
        if not self._horizon.predictor_enabled:
            prev = self._predictor_state.get(axis)
            theta_residual = 0.0
            if prev is not None:
                dt = float(timestamp - prev.timestamp)
                if math.isfinite(dt) and dt > 1e-6:
                    theta_pred = prev.theta + prev.omega * dt
                    theta_residual = float(theta_meas - theta_pred)
            return AxisTargetPrediction(
                theta_base=float(theta_base),
                theta=float(theta_meas),
                omega=float(raw_rate),
                residual=float(theta_residual),
            )

        alpha = min(
            1.0,
            self._resolve_tuning(
                "predictor_alpha",
                float(self._horizon.predictor_alpha),
                minimum=0.0,
            ),
        )
        beta = min(
            1.0,
            self._resolve_tuning(
                "predictor_beta",
                float(self._horizon.predictor_beta),
                minimum=0.0,
            ),
        )

        prev = self._predictor_state.get(axis)
        if prev is None:
            return AxisTargetPrediction(
                theta_base=float(theta_base),
                theta=float(theta_meas),
                omega=float(raw_rate),
                residual=0.0,
            )

        dt = float(timestamp - prev.timestamp)
        if not math.isfinite(dt) or dt <= 1e-6:
            return AxisTargetPrediction(
                theta_base=float(theta_base),
                theta=float(theta_meas),
                omega=float(raw_rate),
                residual=0.0,
            )

        theta_pred = prev.theta + prev.omega * dt
        omega_pred = prev.omega
        residual = float(theta_meas - theta_pred)
        theta_upd = theta_pred + alpha * residual
        omega_upd = omega_pred + (beta / dt) * residual

        return AxisTargetPrediction(
            theta_base=float(theta_base),
            theta=float(theta_upd),
            omega=float(omega_upd),
            residual=float(residual),
        )

    def _nominal_effect_delay(self, distance_m: float) -> float:
        if self._effect_delay_mode == "time_to_impact":
            speed = self._projectile_speed_m_s
            if speed is None or speed <= 0.0:
                return self._base_effect_delay_s
            nominal = float(distance_m) / speed + self._impact_delay_bias_s
            return max(0.0, nominal)
        return self._base_effect_delay_s

    def _update_effect_delay(
        self,
        *,
        axis: AxisName,
        theta_residual: float,
        omega: float,
        nominal_delay: float,
    ) -> float:
        current = self._adaptive_effect_delay_s.get(axis, nominal_delay)
        if not self._horizon.adaptive_effect_delay_enabled:
            bounded_nominal = max(0.0, float(nominal_delay))
            self._adaptive_effect_delay_s[axis] = bounded_nominal
            return bounded_nominal

        min_s = max(0.0, float(self._horizon.adaptive_effect_delay_min_s))
        max_s = max(min_s, float(self._horizon.adaptive_effect_delay_max_s))
        alpha = min(
            1.0,
            self._resolve_tuning(
                "adaptive_effect_delay_alpha",
                float(self._horizon.adaptive_effect_delay_alpha),
                minimum=0.0,
            ),
        )
        gain = self._resolve_tuning(
            "adaptive_effect_delay_gain",
            float(self._horizon.adaptive_effect_delay_gain),
            minimum=0.0,
        )
        rate_eps = max(
            1e-9,
            self._resolve_tuning(
                "adaptive_effect_delay_rate_eps",
                float(self._horizon.adaptive_effect_delay_rate_eps),
                minimum=1e-9,
            ),
        )

        candidate = max(min_s, min(max_s, float(nominal_delay)))
        abs_rate = abs(float(omega))
        if math.isfinite(theta_residual) and abs_rate >= rate_eps:
            candidate += gain * (float(theta_residual) / abs_rate)

        candidate = min(max_s, max(min_s, candidate))
        updated = (1.0 - alpha) * current + alpha * candidate
        updated = min(max_s, max(min_s, updated))
        self._adaptive_effect_delay_s[axis] = float(updated)
        return float(updated)

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


__all__ = ["AxisReferenceSequences", "AxisTargetPrediction", "MpcReferenceBuilder"]

