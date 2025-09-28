"""World-frame constant-velocity tracking utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from common.tracker import TrackingConfig, TrackingWorldParams


StateTuple = Tuple[float, float, float, float, float, float]
Covariance = Tuple[Tuple[float, ...], ...]


@dataclass
class WorldTrackerMeasurement:
    position_m: Tuple[float, float, float]
    position_std_m: Optional[float] = None


@dataclass
class WorldTrackerPrediction:
    position_m: Tuple[float, float, float]
    velocity_mps: Tuple[float, float, float]
    horizon_s: float


class WorldTracker:
    """Constant-velocity Kalman filter operating in world coordinates."""

    def __init__(self, config: TrackingConfig) -> None:
        if config.model != "world_cv":
            raise ValueError(f"WorldTracker requires model 'world_cv', got {config.model!r}")
        if config.world is None:
            raise ValueError("tracking.world parameters are required for world-frame tracking")
        self._cfg = config
        self._world_params: TrackingWorldParams = config.world
        self._state: Optional[StateTuple] = None
        self._cov: Optional[Covariance] = None

    def reset(self) -> None:
        self._state = None
        self._cov = None

    def predict(self, dt: float) -> None:
        if self._state is None or self._cov is None:
            if self._state is not None:
                x, y, z, vx, vy, vz = self._state
                self._state = (
                    x + vx * dt,
                    y + vy * dt,
                    z + vz * dt,
                    vx,
                    vy,
                    vz,
                )
            return

        if dt <= 0.0:
            return

        x, y, z, vx, vy, vz = self._state
        x += vx * dt
        y += vy * dt
        z += vz * dt
        self._state = (x, y, z, vx, vy, vz)

        f = (
            (1.0, 0.0, 0.0, dt, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, dt, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0, dt),
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )

        p = self._cov
        fp = self._matmul(f, p)
        new_cov = self._matmul(fp, self._transpose(f))

        q = self._process_noise(dt)
        self._cov = self._symmetrise(self._add_matrices(new_cov, q))

    def update(self, measurement: WorldTrackerMeasurement) -> bool:
        meas_var = self._measurement_variance(measurement)
        meas_x, meas_y, meas_z = measurement.position_m

        if self._state is None:
            vx = vy = vz = 0.0
            self._state = (meas_x, meas_y, meas_z, vx, vy, vz)
            vel_var = max(meas_var, 1.0)
            self._cov = (
                (meas_var, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, meas_var, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, meas_var, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, vel_var, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, vel_var, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, vel_var),
            )
            return True

        if self._cov is None:
            return False

        x, y, z, vx, vy, vz = self._state
        resid_x = meas_x - x
        resid_y = meas_y - y
        resid_z = meas_z - z

        p = self._cov
        s = self._innovation_covariance(p, meas_var)
        if s is None:
            return False

        inv_s = self._invert_3x3(s)
        if inv_s is None:
            return False

        chi2 = self._mahalanobis_sq((resid_x, resid_y, resid_z), inv_s)
        if self._cfg.gate_chi2 > 0.0 and chi2 > self._cfg.gate_chi2:
            return False

        k = self._kalman_gain(p, inv_s)

        updated = [x, y, z, vx, vy, vz]
        resid_vec = (resid_x, resid_y, resid_z)
        for i in range(6):
            k_row = k[i]
            updated[i] += (
                k_row[0] * resid_vec[0]
                + k_row[1] * resid_vec[1]
                + k_row[2] * resid_vec[2]
            )

        self._state = tuple(updated)  # type: ignore[assignment]

        hp = (p[0], p[1], p[2])
        new_p = [[0.0] * 6 for _ in range(6)]
        for i in range(6):
            for j in range(6):
                correction = (
                    k[i][0] * hp[0][j]
                    + k[i][1] * hp[1][j]
                    + k[i][2] * hp[2][j]
                )
                new_p[i][j] = p[i][j] - correction

        self._cov = self._symmetrise(tuple(tuple(row) for row in new_p))
        return True

    def project(self, horizon_s: float) -> Optional[WorldTrackerPrediction]:
        if self._state is None:
            return None

        horizon_s = max(0.0, float(horizon_s))
        x, y, z, vx, vy, vz = self._state
        pos = (
            x + vx * horizon_s,
            y + vy * horizon_s,
            z + vz * horizon_s,
        )
        vel = (vx, vy, vz)
        return WorldTrackerPrediction(position_m=pos, velocity_mps=vel, horizon_s=horizon_s)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _measurement_variance(self, measurement: WorldTrackerMeasurement) -> float:
        sigma = measurement.position_std_m
        if sigma is None or not math.isfinite(sigma) or sigma <= 0.0:
            sigma = self._world_params.meas_noise_pos_m
        return float(sigma) * float(sigma)

    def _process_noise(self, dt: float) -> Covariance:
        if dt <= 0.0:
            zeros = tuple((0.0,) * 6 for _ in range(6))
            return zeros

        q = float(self._world_params.process_noise_accel)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        q_pos = 0.25 * q * dt4
        q_cross = 0.5 * q * dt3
        q_vel = q * dt2

        q_mat = [[0.0] * 6 for _ in range(6)]
        axes = ((0, 3), (1, 4), (2, 5))
        for pos_idx, vel_idx in axes:
            q_mat[pos_idx][pos_idx] = q_pos
            q_mat[pos_idx][vel_idx] = q_cross
            q_mat[vel_idx][pos_idx] = q_cross
            q_mat[vel_idx][vel_idx] = q_vel

        return tuple(tuple(row) for row in q_mat)

    @staticmethod
    def _innovation_covariance(p: Covariance, meas_var: float) -> Optional[Tuple[Tuple[float, float, float], ...]]:
        a = p[0][0] + meas_var
        b = p[0][1]
        c = p[0][2]
        d = p[1][1] + meas_var
        e = p[1][2]
        f = p[2][2] + meas_var

        if not all(math.isfinite(v) for v in (a, b, c, d, e, f)):
            return None

        return (
            (a, b, c),
            (b, d, e),
            (c, e, f),
        )

    @staticmethod
    def _invert_3x3(m: Tuple[Tuple[float, float, float], ...]) -> Optional[Tuple[Tuple[float, float, float], ...]]:
        a, b, c = m[0]
        _, d, e = m[1]
        __, ___, f = m[2]

        det = (
            a * (d * f - e * e)
            - b * (b * f - e * c)
            + c * (b * e - d * c)
        )
        if det == 0.0:
            return None

        inv_det = 1.0 / det
        c00 = (d * f - e * e) * inv_det
        c01 = (c * e - b * f) * inv_det
        c02 = (b * e - c * d) * inv_det
        c11 = (a * f - c * c) * inv_det
        c12 = (c * b - a * e) * inv_det
        c22 = (a * d - b * b) * inv_det

        return (
            (c00, c01, c02),
            (c01, c11, c12),
            (c02, c12, c22),
        )

    @staticmethod
    def _mahalanobis_sq(residual: Tuple[float, float, float], inv: Tuple[Tuple[float, float, float], ...]) -> float:
        rx, ry, rz = residual
        term_x = inv[0][0] * rx + inv[0][1] * ry + inv[0][2] * rz
        term_y = inv[1][0] * rx + inv[1][1] * ry + inv[1][2] * rz
        term_z = inv[2][0] * rx + inv[2][1] * ry + inv[2][2] * rz
        return rx * term_x + ry * term_y + rz * term_z

    @staticmethod
    def _kalman_gain(p: Covariance, inv_s: Tuple[Tuple[float, float, float], ...]) -> Tuple[Tuple[float, float, float], ...]:
        gain = []
        for row in p:
            k0 = row[0] * inv_s[0][0] + row[1] * inv_s[1][0] + row[2] * inv_s[2][0]
            k1 = row[0] * inv_s[0][1] + row[1] * inv_s[1][1] + row[2] * inv_s[2][1]
            k2 = row[0] * inv_s[0][2] + row[1] * inv_s[1][2] + row[2] * inv_s[2][2]
            gain.append((k0, k1, k2))
        return tuple(gain)

    @staticmethod
    def _symmetrise(matrix: Covariance) -> Covariance:
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0
        sym = []
        for i in range(rows):
            row = []
            for j in range(cols):
                if i == j:
                    row.append(matrix[i][j])
                else:
                    avg = 0.5 * (matrix[i][j] + matrix[j][i])
                    row.append(avg)
            sym.append(tuple(row))
        return tuple(sym)

    @staticmethod
    def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Covariance:
        rows = len(a)
        cols = len(b[0]) if b else 0
        inner = len(b)
        result = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for k in range(inner):
                aik = a[i][k]
                if aik == 0.0:
                    continue
                row_k = b[k]
                for j in range(cols):
                    result[i][j] += aik * row_k[j]
        return tuple(tuple(row) for row in result)

    @staticmethod
    def _transpose(matrix: Sequence[Sequence[float]]) -> Covariance:
        if not matrix:
            return tuple()
        rows = len(matrix)
        cols = len(matrix[0])
        transposed = [[0.0] * rows for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                transposed[j][i] = matrix[i][j]
        return tuple(tuple(row) for row in transposed)

    @staticmethod
    def _add_matrices(a: Covariance, b: Covariance) -> Covariance:
        rows = len(a)
        cols = len(a[0]) if rows else 0
        out = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(a[i][j] + b[i][j])
            out.append(tuple(row))
        return tuple(out)
