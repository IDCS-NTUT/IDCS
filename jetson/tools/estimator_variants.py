"""Estimator variants used by the MPC latency/effectiveness benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol

import numpy as np


class EstimatorVariant(Protocol):
    """Interface for benchmarked estimator variants."""

    def reset(self) -> None:
        ...

    def step(
        self,
        *,
        u_applied: float,
        theta_meas: Optional[float],
        omega_meas: Optional[float],
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class EstimatorVariantConfig:
    A: np.ndarray
    B: np.ndarray
    q_theta: float
    q_omega: float
    q_d: float
    r_theta: float
    r_omega: float = 0.04
    gate_nis: float = 9.21
    adapt_alpha: float = 0.05
    adapt_scale_min: float = 0.25
    adapt_scale_max: float = 8.0


class _LinearKalmanBase:
    def __init__(self, cfg: EstimatorVariantConfig) -> None:
        self._A = np.asarray(cfg.A, dtype=float)
        self._B = np.asarray(cfg.B, dtype=float)
        self._Q = np.diag(
            [
                max(0.0, float(cfg.q_theta)),
                max(0.0, float(cfg.q_omega)),
                max(0.0, float(cfg.q_d)),
            ]
        )
        self._x = np.zeros((self._A.shape[0],), dtype=float)
        self._P = np.eye(self._A.shape[0], dtype=float)

    def reset(self) -> None:
        self._x[:] = 0.0
        self._P = np.eye(self._A.shape[0], dtype=float)

    def _predict(self, u_applied: float) -> None:
        u_vec = np.array([[float(u_applied)]], dtype=float)
        self._x = self._A @ self._x + (self._B @ u_vec).reshape(self._x.shape)
        self._P = self._A @ self._P @ self._A.T + self._Q

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()


class BaselineThetaKF(_LinearKalmanBase):
    def __init__(self, cfg: EstimatorVariantConfig) -> None:
        super().__init__(cfg)
        self._H = np.array([[1.0, 0.0, 0.0]], dtype=float)
        self._R = max(1e-9, float(cfg.r_theta))

    def step(
        self,
        *,
        u_applied: float,
        theta_meas: Optional[float],
        omega_meas: Optional[float],
    ) -> np.ndarray:
        del omega_meas
        self._predict(u_applied)
        if theta_meas is None or not math.isfinite(theta_meas):
            return self.state
        z = float(theta_meas)
        innovation = z - float((self._H @ self._x.reshape(-1, 1)).item())
        S = float((self._H @ self._P @ self._H.T).item()) + self._R
        if S <= 1e-12:
            return self.state
        K = (self._P @ self._H.T) / S
        self._x = self._x + (K.flatten() * innovation)
        I = np.eye(self._A.shape[0], dtype=float)
        self._P = (I - K @ self._H) @ self._P
        return self.state


class GatedAdaptiveThetaKF(_LinearKalmanBase):
    def __init__(self, cfg: EstimatorVariantConfig) -> None:
        super().__init__(cfg)
        self._H = np.array([[1.0, 0.0, 0.0]], dtype=float)
        self._base_R = max(1e-9, float(cfg.r_theta))
        self._R = self._base_R
        self._gate_nis = max(0.0, float(cfg.gate_nis))
        self._adapt_alpha = min(1.0, max(0.0, float(cfg.adapt_alpha)))
        self._adapt_min = max(1e-4, float(cfg.adapt_scale_min))
        self._adapt_max = max(self._adapt_min, float(cfg.adapt_scale_max))

    def reset(self) -> None:
        super().reset()
        self._R = self._base_R

    def step(
        self,
        *,
        u_applied: float,
        theta_meas: Optional[float],
        omega_meas: Optional[float],
    ) -> np.ndarray:
        del omega_meas
        self._predict(u_applied)
        if theta_meas is None or not math.isfinite(theta_meas):
            return self.state
        z = float(theta_meas)
        innovation = z - float((self._H @ self._x.reshape(-1, 1)).item())
        pred_var = float((self._H @ self._P @ self._H.T).item())
        S = pred_var + self._R
        if S <= 1e-12:
            return self.state

        nis = (innovation * innovation) / S
        if nis > self._gate_nis:
            self._R = min(self._base_R * self._adapt_max, self._R * 1.2)
            return self.state

        residual_var = max(1e-9, innovation * innovation - pred_var)
        target_R = min(self._base_R * self._adapt_max, max(self._base_R * self._adapt_min, residual_var))
        self._R = (1.0 - self._adapt_alpha) * self._R + self._adapt_alpha * target_R

        K = (self._P @ self._H.T) / (pred_var + self._R)
        self._x = self._x + (K.flatten() * innovation)
        I = np.eye(self._A.shape[0], dtype=float)
        self._P = (I - K @ self._H) @ self._P
        return self.state


class ThetaOmegaFusionKF(_LinearKalmanBase):
    def __init__(self, cfg: EstimatorVariantConfig) -> None:
        super().__init__(cfg)
        self._H_theta = np.array([[1.0, 0.0, 0.0]], dtype=float)
        self._H_omega = np.array([[0.0, 1.0, 0.0]], dtype=float)
        self._R_theta = max(1e-9, float(cfg.r_theta))
        self._R_omega = max(1e-9, float(cfg.r_omega))

    def _update_scalar(self, H: np.ndarray, z: float, R: float) -> None:
        innovation = z - float((H @ self._x.reshape(-1, 1)).item())
        S = float((H @ self._P @ H.T).item()) + R
        if S <= 1e-12:
            return
        K = (self._P @ H.T) / S
        self._x = self._x + (K.flatten() * innovation)
        I = np.eye(self._A.shape[0], dtype=float)
        self._P = (I - K @ H) @ self._P

    def step(
        self,
        *,
        u_applied: float,
        theta_meas: Optional[float],
        omega_meas: Optional[float],
    ) -> np.ndarray:
        self._predict(u_applied)
        if theta_meas is not None and math.isfinite(theta_meas):
            self._update_scalar(self._H_theta, float(theta_meas), self._R_theta)
        if omega_meas is not None and math.isfinite(omega_meas):
            self._update_scalar(self._H_omega, float(omega_meas), self._R_omega)
        return self.state


EstimatorFactory = Callable[[EstimatorVariantConfig], EstimatorVariant]


def available_estimators() -> Dict[str, EstimatorFactory]:
    return {
        "baseline": BaselineThetaKF,
        "gated_adaptive": GatedAdaptiveThetaKF,
        "rate_fusion": ThetaOmegaFusionKF,
    }
