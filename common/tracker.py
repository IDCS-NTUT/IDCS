"""Image-space tracking utilities and configuration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from common.control import AxisPair
from common.geometry import pixel_motion_from_camera_rotation


@dataclass(frozen=True)
class TrackingMeasurementNoise:
    base_px: float
    min_box_px: float


@dataclass(frozen=True)
class TrackingProcessNoise:
    u: float
    v: float


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    model: str
    predict_horizon_ms: float
    use_camera_derotation: bool
    meas_noise: TrackingMeasurementNoise
    process_noise: TrackingProcessNoise
    gate_chi2: float
    reset_on_target_switch: bool

    @classmethod
    def from_raw_config(cls, cfg: Mapping[str, Any]) -> "TrackingConfig":
        section = cfg.get("tracking", {}) or {}
        if not isinstance(section, Mapping):
            section = {}

        enabled = bool(section.get("enabled", False))
        model = str(section.get("model", "cv")).strip().lower()
        try:
            predict_horizon_ms = float(section.get("predict_horizon_ms", 120.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("tracking.predict_horizon_ms must be numeric") from exc
        if predict_horizon_ms < 0.0:
            raise ValueError("tracking.predict_horizon_ms cannot be negative")

        use_camera_derotation = bool(section.get("use_camera_derotation", False))

        meas_section = section.get("meas_noise_px", {}) or {}
        try:
            meas_base = float(meas_section.get("base", 2.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("tracking.meas_noise_px.base must be numeric") from exc
        if meas_base <= 0.0:
            raise ValueError("tracking.meas_noise_px.base must be positive")
        try:
            meas_min_box = float(meas_section.get("min_box_px", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("tracking.meas_noise_px.min_box_px must be numeric") from exc
        if meas_min_box < 0.0:
            raise ValueError("tracking.meas_noise_px.min_box_px cannot be negative")

        proc_section = section.get("process_noise", {}) or {}
        try:
            proc_u = float(proc_section.get("u", 1.0))
            proc_v = float(proc_section.get("v", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("tracking.process_noise values must be numeric") from exc
        if proc_u < 0.0 or proc_v < 0.0:
            raise ValueError("tracking.process_noise values must be non-negative")

        try:
            gate_chi2 = float(section.get("gate_chi2", 9.21))
        except (TypeError, ValueError) as exc:
            raise ValueError("tracking.gate_chi2 must be numeric") from exc
        if gate_chi2 < 0.0:
            raise ValueError("tracking.gate_chi2 cannot be negative")

        reset_on_switch = bool(section.get("reset_on_target_switch", True))

        return cls(
            enabled=enabled,
            model=model,
            predict_horizon_ms=predict_horizon_ms,
            use_camera_derotation=use_camera_derotation,
            meas_noise=TrackingMeasurementNoise(base_px=meas_base, min_box_px=meas_min_box),
            process_noise=TrackingProcessNoise(u=proc_u, v=proc_v),
            gate_chi2=gate_chi2,
            reset_on_target_switch=reset_on_switch,
        )


@dataclass
class TrackerMeasurement:
    uv: Tuple[float, float]
    box_size_px: Optional[Tuple[float, float]]
    confidence: Optional[float]


@dataclass
class TrackerPrediction:
    uv: Tuple[float, float]
    velocity: Tuple[float, float]
    horizon_s: float


class PixelTracker:
    """Constant-velocity pixel-space tracker with optional de-rotation."""

    def __init__(
        self,
        config: TrackingConfig,
        *,
        fx_px: float,
        fy_px: float,
        cx_px: float,
        cy_px: float,
    ) -> None:
        if config.model != "cv":
            raise ValueError(f"unsupported tracker model: {config.model!r}")
        self._cfg = config
        self._fx = float(fx_px)
        self._fy = float(fy_px)
        self._cx = float(cx_px)
        self._cy = float(cy_px)

        self._state: Optional[Tuple[float, float, float, float]] = None
        self._cov: Optional[Tuple[Tuple[float, ...], ...]] = None

    def reset(self) -> None:
        self._state = None
        self._cov = None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, dt: float, cam_rates: Optional[AxisPair]) -> None:
        if self._state is None:
            return
        if dt <= 0.0:
            return

        u, v, du, dv = self._state
        u += du * dt
        v += dv * dt

        if self._cfg.use_camera_derotation and cam_rates is not None:
            du_rot, dv_rot = pixel_motion_from_camera_rotation(
                u,
                v,
                yaw_rate_rad_s=cam_rates.yaw,
                pitch_rate_rad_s=cam_rates.pitch,
                dt_s=dt,
                fx_px=self._fx,
                fy_px=self._fy,
                cx_px=self._cx,
                cy_px=self._cy,
            )
            u += du_rot
            v += dv_rot

        self._state = (u, v, du, dv)

        if self._cov is None:
            return

        p = self._cov
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q_u = self._cfg.process_noise.u
        q_v = self._cfg.process_noise.v

        # State transition F assumes constant velocity.
        f00 = 1.0
        f02 = dt
        f11 = 1.0
        f13 = dt
        f22 = 1.0
        f33 = 1.0

        # F * P
        fp00 = f00 * p[0][0] + f02 * p[2][0]
        fp01 = f00 * p[0][1] + f02 * p[2][1]
        fp02 = f00 * p[0][2] + f02 * p[2][2]
        fp03 = f00 * p[0][3] + f02 * p[2][3]

        fp10 = f11 * p[1][0] + f13 * p[3][0]
        fp11 = f11 * p[1][1] + f13 * p[3][1]
        fp12 = f11 * p[1][2] + f13 * p[3][2]
        fp13 = f11 * p[1][3] + f13 * p[3][3]

        fp20 = p[2][0]
        fp21 = p[2][1]
        fp22 = p[2][2]
        fp23 = p[2][3]

        fp30 = p[3][0]
        fp31 = p[3][1]
        fp32 = p[3][2]
        fp33 = p[3][3]

        # (F * P) * F^T
        new00 = fp00 * f00 + fp02 * f02
        new01 = fp01 * f11 + fp03 * f13
        new02 = fp00 * 0.0 + fp02 * f22  # simplifies to fp02
        new03 = fp01 * 0.0 + fp03 * f33  # simplifies to fp03

        new10 = fp10 * f00 + fp12 * f02
        new11 = fp11 * f11 + fp13 * f13
        new12 = fp10 * 0.0 + fp12 * f22
        new13 = fp11 * 0.0 + fp13 * f33

        new20 = fp20 * f00 + fp22 * f02
        new21 = fp21 * f11 + fp23 * f13
        new22 = fp20 * 0.0 + fp22 * f22
        new23 = fp21 * 0.0 + fp23 * f33

        new30 = fp30 * f00 + fp32 * f02
        new31 = fp31 * f11 + fp33 * f13
        new32 = fp30 * 0.0 + fp32 * f22
        new33 = fp31 * 0.0 + fp33 * f33

        # Add process noise (discretised CV model driven by acceleration).
        q00 = 0.25 * q_u * dt4
        q02 = 0.5 * q_u * dt3
        q22 = q_u * dt2
        q11 = 0.25 * q_v * dt4
        q13 = 0.5 * q_v * dt3
        q33 = q_v * dt2

        self._cov = (
            (new00 + q00, new01, new02 + q02, new03),
            (new10, new11 + q11, new12, new13 + q13),
            (new20 + q02, new21, new22 + q22, new23),
            (new30, new31 + q13, new32, new33 + q33),
        )

    # ------------------------------------------------------------------
    # Measurement update
    # ------------------------------------------------------------------
    def update(
        self,
        measurement: TrackerMeasurement,
        *,
        rotation_correction_dt: float = 0.0,
        cam_rates: Optional[AxisPair] = None,
    ) -> bool:
        meas_u, meas_v = measurement.uv
        if self._cfg.use_camera_derotation and cam_rates is not None and rotation_correction_dt > 0.0:
            du_rot, dv_rot = pixel_motion_from_camera_rotation(
                meas_u,
                meas_v,
                yaw_rate_rad_s=cam_rates.yaw,
                pitch_rate_rad_s=cam_rates.pitch,
                dt_s=rotation_correction_dt,
                fx_px=self._fx,
                fy_px=self._fy,
                cx_px=self._cx,
                cy_px=self._cy,
            )
            meas_u -= du_rot
            meas_v -= dv_rot

        if self._state is None:
            self._state = (meas_u, meas_v, 0.0, 0.0)
            var_u, var_v = self._measurement_variance(measurement)
            # Provide a generous initial covariance so subsequent measurements
            # can quickly steer the estimate.
            self._cov = (
                (var_u, 0.0, 0.0, 0.0),
                (0.0, var_v, 0.0, 0.0),
                (0.0, 0.0, max(var_u, 1.0), 0.0),
                (0.0, 0.0, 0.0, max(var_v, 1.0)),
            )
            return True

        if self._cov is None:
            return False

        var_u, var_v = self._measurement_variance(measurement)
        u, v, du, dv = self._state
        resid_u = meas_u - u
        resid_v = meas_v - v

        p = self._cov
        s00 = p[0][0] + var_u
        s01 = p[0][1]
        s10 = p[1][0]
        s11 = p[1][1] + var_v
        det = s00 * s11 - s01 * s10
        if det <= 0.0:
            return False

        inv_s00 = s11 / det
        inv_s01 = -s01 / det
        inv_s10 = -s10 / det
        inv_s11 = s00 / det

        chi2 = resid_u * (inv_s00 * resid_u + inv_s01 * resid_v) + resid_v * (
            inv_s10 * resid_u + inv_s11 * resid_v
        )
        if self._cfg.gate_chi2 > 0.0 and chi2 > self._cfg.gate_chi2:
            return False

        # Kalman gain K = P H^T S^{-1}; measurement matrix H selects position.
        k00 = p[0][0] * inv_s00 + p[0][1] * inv_s10
        k01 = p[0][0] * inv_s01 + p[0][1] * inv_s11
        k10 = p[1][0] * inv_s00 + p[1][1] * inv_s10
        k11 = p[1][0] * inv_s01 + p[1][1] * inv_s11
        k20 = p[2][0] * inv_s00 + p[2][1] * inv_s10
        k21 = p[2][0] * inv_s01 + p[2][1] * inv_s11
        k30 = p[3][0] * inv_s00 + p[3][1] * inv_s10
        k31 = p[3][0] * inv_s01 + p[3][1] * inv_s11

        u += k00 * resid_u + k01 * resid_v
        v += k10 * resid_u + k11 * resid_v
        du += k20 * resid_u + k21 * resid_v
        dv += k30 * resid_u + k31 * resid_v

        self._state = (u, v, du, dv)

        # Update covariance: P = (I - K H) P
        k00_neg = 1.0 - k00
        k11_neg = 1.0 - k11

        new_p0 = (
            k00_neg * p[0][0] - k01 * p[1][0],
            k00_neg * p[0][1] - k01 * p[1][1],
            k00_neg * p[0][2] - k01 * p[1][2],
            k00_neg * p[0][3] - k01 * p[1][3],
        )
        new_p1 = (
            -k10 * p[0][0] + k11_neg * p[1][0],
            -k10 * p[0][1] + k11_neg * p[1][1],
            -k10 * p[0][2] + k11_neg * p[1][2],
            -k10 * p[0][3] + k11_neg * p[1][3],
        )
        new_p2 = (
            -k20 * p[0][0] - k21 * p[1][0] + p[2][0],
            -k20 * p[0][1] - k21 * p[1][1] + p[2][1],
            -k20 * p[0][2] - k21 * p[1][2] + p[2][2],
            -k20 * p[0][3] - k21 * p[1][3] + p[2][3],
        )
        new_p3 = (
            -k30 * p[0][0] - k31 * p[1][0] + p[3][0],
            -k30 * p[0][1] - k31 * p[1][1] + p[3][1],
            -k30 * p[0][2] - k31 * p[1][2] + p[3][2],
            -k30 * p[0][3] - k31 * p[1][3] + p[3][3],
        )

        self._cov = self._symmetrise((new_p0, new_p1, new_p2, new_p3))
        return True

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def project(
        self,
        horizon_s: float,
        cam_rates: Optional[AxisPair],
    ) -> Optional[TrackerPrediction]:
        if self._state is None:
            return None

        horizon_s = max(0.0, float(horizon_s))
        u, v, du, dv = self._state
        u_pred = u + du * horizon_s
        v_pred = v + dv * horizon_s

        if self._cfg.use_camera_derotation and cam_rates is not None and horizon_s > 0.0:
            du_rot, dv_rot = pixel_motion_from_camera_rotation(
                u_pred,
                v_pred,
                yaw_rate_rad_s=cam_rates.yaw,
                pitch_rate_rad_s=cam_rates.pitch,
                dt_s=horizon_s,
                fx_px=self._fx,
                fy_px=self._fy,
                cx_px=self._cx,
                cy_px=self._cy,
            )
            u_pred += du_rot
            v_pred += dv_rot

        return TrackerPrediction(uv=(u_pred, v_pred), velocity=(du, dv), horizon_s=horizon_s)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _measurement_variance(self, measurement: TrackerMeasurement) -> Tuple[float, float]:
        sigma = self._cfg.meas_noise.base_px
        if measurement.box_size_px is not None:
            min_dim = max(1e-3, min(measurement.box_size_px))
            if self._cfg.meas_noise.min_box_px > 0.0 and min_dim < self._cfg.meas_noise.min_box_px:
                scale = self._cfg.meas_noise.min_box_px / min_dim
                sigma *= max(1.0, scale)

        if measurement.confidence is not None:
            conf = min(max(measurement.confidence, 0.0), 1.0)
            sigma *= 1.0 + (1.0 - conf)

        var = sigma * sigma
        return (var, var)

    @staticmethod
    def _symmetrise(matrix: Tuple[Tuple[float, ...], ...]) -> Tuple[Tuple[float, ...], ...]:
        # Ensure covariance stays symmetric despite numerical round-off.
        return tuple(
            tuple((matrix[i][j] + matrix[j][i]) * 0.5 for j in range(4)) for i in range(4)
        )

