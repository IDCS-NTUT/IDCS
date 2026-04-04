"""Axis-level building blocks for the MPC gimbal controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from common.control import (
    ControlConfig,
    MpcConfig,
    MpcConstraintConfig,
    MpcCostConfig,
    MpcEstimatorConfig,
    MpcHorizonConfig,
    MpcPlantConfig,
)

class MpcSolverError(RuntimeError):
    """Raised when the QP solver fails or is unavailable."""


SUCCESS_STATUSES = {"optimal", "solved", "solved inaccurate"}


@dataclass
class MpcQPSolution:
    """Container returned by :class:`MpcQPSolver` implementations."""

    status: str
    primal: Optional[np.ndarray]
    cost: Optional[float] = None
    info: Optional[Mapping[str, float]] = None

    @property
    def ok(self) -> bool:
        return self.primal is not None and self.status.lower() in SUCCESS_STATUSES


class MpcQPSolver(Protocol):
    """Abstract interface for quadratic-program solvers used by MPC."""

    def solve(
        self,
        H: np.ndarray,
        f: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        *,
        warm_start: Optional[np.ndarray] = None,
    ) -> MpcQPSolution:
        ...


class OsqpSolver:
    """Thin wrapper around :mod:`osqp` to satisfy :class:`MpcQPSolver`."""

    def __init__(self) -> None:
        try:  # pragma: no cover - optional dependency
            import osqp  # type: ignore
            from scipy import sparse  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional path
            raise MpcSolverError(
                "osqp (and scipy) are required for MPC but are not installed"
            ) from exc

        self._osqp = osqp
        self._sparse = sparse

    def solve(
        self,
        H: np.ndarray,
        f: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        *,
        warm_start: Optional[np.ndarray] = None,
    ) -> MpcQPSolution:  # pragma: no cover - exercised in hardware env
        P = self._sparse.csc_matrix(H)
        q = f
        A_mat = self._sparse.csc_matrix(A)
        solver = self._osqp.OSQP()
        solver.setup(P=P, q=q, A=A_mat, l=l, u=u, verbose=False)
        if warm_start is not None:
            solver.warm_start(x=warm_start)
        result = solver.solve()
        status = result.info.status.lower()
        solution = None if result.x is None else np.asarray(result.x, dtype=float)
        info = {
            "iter": float(result.info.iter),
            "status_val": float(result.info.status_val),
        }
        return MpcQPSolution(status=status, primal=solution, cost=result.info.obj_val, info=info)


class AxisKalmanFilter:
    """Linear Kalman filter for the 3-state gimbal axis model."""

    _R_OMEGA_HARDCODED = 0.04

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        estimator_cfg: MpcEstimatorConfig,
    ) -> None:
        self._A = A
        self._B = B
        self._C = C
        self._C_omega = np.array([[0.0, 1.0, 0.0]], dtype=float)
        self._Q = np.diag([estimator_cfg.q_theta, estimator_cfg.q_omega, estimator_cfg.q_d])
        self._R = float(estimator_cfg.r_theta)
        self._R_omega = float(self._R_OMEGA_HARDCODED)
        self._x = np.zeros((A.shape[0],), dtype=float)
        self._P = np.eye(A.shape[0], dtype=float)
        self._innovation = 0.0
        self._innovation_var = self._R

    def reset(self, state: Optional[Sequence[float]] = None) -> None:
        if state is not None:
            vec = np.asarray(state, dtype=float)
            if vec.shape != self._x.shape:
                raise ValueError("initial state must be length 3")
            self._x = vec.copy()
        else:
            self._x[:] = 0.0
        self._P = np.eye(self._A.shape[0], dtype=float)
        self._innovation = 0.0
        self._innovation_var = self._R

    def set_noise_covariances(
        self,
        *,
        q_theta: float,
        q_omega: float,
        q_d: float,
        r_theta: float,
    ) -> None:
        self._Q = np.diag([
            max(0.0, float(q_theta)),
            max(0.0, float(q_omega)),
            max(0.0, float(q_d)),
        ])
        self._R = max(1e-9, float(r_theta))

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def innovation(self) -> float:
        return self._innovation

    @property
    def innovation_var(self) -> float:
        return self._innovation_var

    def predict(self, u_k: float) -> np.ndarray:
        u_vec = np.array([[u_k]], dtype=float)
        self._x = self._A @ self._x + (self._B @ u_vec).reshape(self._x.shape)
        self._P = self._A @ self._P @ self._A.T + self._Q
        return self.state

    def _scalar_update(self, H: np.ndarray, z: float, R: float) -> Tuple[float, float]:
        y = z - float((H @ self._x.reshape(-1, 1)).item())
        S = float((H @ self._P @ H.T).item()) + R
        if S <= 0.0:
            raise RuntimeError("Kalman filter innovation covariance must be positive")
        K = self._P @ H.T / S
        self._x = self._x + (K.flatten() * y)
        I = np.eye(self._A.shape[0], dtype=float)
        self._P = (I - K @ H) @ self._P
        return y, S

    def update(
        self,
        theta_meas: Optional[float],
        omega_meas: Optional[float] = None,
    ) -> np.ndarray:
        if theta_meas is None and omega_meas is None:
            # Prediction-only update; track theta innovation variance for diagnostics.
            S = float((self._C @ self._P @ self._C.T).item()) + self._R
            self._innovation_var = S
            self._innovation = 0.0
            return self.state

        if theta_meas is not None:
            y, S = self._scalar_update(self._C, float(theta_meas), self._R)
            self._innovation = y
            self._innovation_var = S

        if omega_meas is not None:
            self._scalar_update(self._C_omega, float(omega_meas), self._R_omega)

        return self.state


@dataclass(frozen=True)
class MpcPredictionCache:
    """Pre-computed matrices used for horizon predictions."""

    Sx: np.ndarray
    Su: np.ndarray
    E_theta: np.ndarray
    E_omega: np.ndarray
    theta_projection: np.ndarray
    omega_projection: np.ndarray
    theta_input_map: np.ndarray
    omega_input_map: np.ndarray
    D: np.ndarray
    error_difference: np.ndarray


@dataclass(frozen=True)
class MpcAxisModel:
    """Encapsulates the per-axis plant and prediction matrices."""

    horizon: MpcHorizonConfig
    plant: MpcPlantConfig
    constraints: MpcConstraintConfig
    costs: MpcCostConfig
    predictions: MpcPredictionCache
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray

    @property
    def Np(self) -> int:
        return self.horizon.prediction_horizon

    @property
    def Nc(self) -> int:
        return self.horizon.control_horizon

    @property
    def Ts(self) -> float:
        return self.horizon.sample_time_s

    @classmethod
    def from_config(cls, cfg: MpcConfig) -> "MpcAxisModel":
        horizon = cfg.horizon
        if horizon.control_horizon > horizon.prediction_horizon:
            raise ValueError("control_horizon must not exceed prediction_horizon")

        Ts = float(horizon.sample_time_s)
        if Ts <= 0.0:
            raise ValueError("sample_time_s must be positive for MPC")

        a_u = float(cfg.plant.a_u)
        a_f = float(cfg.plant.a_f)

        A = np.array(
            [
                [1.0, Ts, 0.0],
                [0.0, 1.0 - Ts * a_f, Ts],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        B = np.array([[0.0], [Ts * a_u], [0.0]], dtype=float)
        C = np.array([[1.0, 0.0, 0.0]], dtype=float)

        predictions = _build_prediction_cache(
            A=A,
            B=B,
            Np=horizon.prediction_horizon,
            Nc=horizon.control_horizon,
            move_blocking=bool(horizon.move_blocking),
        )

        return cls(
            horizon=horizon,
            plant=cfg.plant,
            constraints=cfg.constraints,
            costs=cfg.costs,
            predictions=predictions,
            A=A,
            B=B,
            C=C,
        )


def _build_prediction_cache(
    *,
    A: np.ndarray,
    B: np.ndarray,
    Np: int,
    Nc: int,
    move_blocking: bool,
) -> MpcPredictionCache:
    nx = A.shape[0]
    if B.shape[0] != nx:
        raise ValueError("B must have the same number of rows as A")

    # Powers of A for horizon predictions.
    A_powers = [np.eye(nx, dtype=float)]
    for i in range(1, Np + 1):
        A_powers.append(A_powers[-1] @ A)

    Sx = np.zeros((nx * Np, nx), dtype=float)
    Su = np.zeros((nx * Np, Nc), dtype=float)

    for i in range(1, Np + 1):
        Sx_block = A_powers[i]
        Sx[(i - 1) * nx : i * nx, :] = Sx_block
        for j in range(min(i, Nc)):
            power = i - 1 - j
            Su_block = A_powers[power] @ B
            Su[(i - 1) * nx : i * nx, j : j + 1] += Su_block
        if move_blocking and i > Nc:
            # Sum_{m=0}^{i-1-Nc} A^m B to capture repeated tail input.
            tail_steps = i - 1 - Nc
            if tail_steps >= 0:
                tail_sum = np.zeros((nx, 1), dtype=float)
                for m in range(tail_steps + 1):
                    tail_sum += A_powers[m] @ B
                Su[(i - 1) * nx : i * nx, Nc - 1 : Nc] += tail_sum

    E_theta = _build_selector(Np, nx, component_index=0)
    E_omega = _build_selector(Np, nx, component_index=1)

    theta_projection = E_theta @ Sx
    omega_projection = E_omega @ Sx
    theta_input_map = E_theta @ Su
    omega_input_map = E_omega @ Su

    D = _first_difference_matrix(Nc)
    error_difference = _first_difference_matrix(Np)
    if error_difference.shape[0] > 1:
        error_difference = error_difference[1:, :]
    else:
        error_difference = np.zeros((0, Np), dtype=float)

    return MpcPredictionCache(
        Sx=Sx,
        Su=Su,
        E_theta=E_theta,
        E_omega=E_omega,
        theta_projection=theta_projection,
        omega_projection=omega_projection,
        theta_input_map=theta_input_map,
        omega_input_map=omega_input_map,
        D=D,
        error_difference=error_difference,
    )


def _build_selector(horizon: int, state_dim: int, *, component_index: int) -> np.ndarray:
    mat = np.zeros((horizon, horizon * state_dim), dtype=float)
    for i in range(horizon):
        mat[i, i * state_dim + component_index] = 1.0
    return mat


def _first_difference_matrix(nc: int) -> np.ndarray:
    mat = np.zeros((nc, nc), dtype=float)
    for i in range(nc):
        mat[i, i] = 1.0
        if i > 0:
            mat[i, i - 1] = -1.0
    return mat


@dataclass(frozen=True)
class MpcAxisDiagnostics:
    """Debug information returned after solving the QP."""

    status: str
    cost: Optional[float]
    u_sequence: np.ndarray
    theta_pred: np.ndarray
    omega_pred: np.ndarray
    weights: np.ndarray
    solver_info: Optional[Mapping[str, float]]
    slack: Optional[Dict[str, float]] = None
    cost_terms: Optional[Dict[str, float]] = None
    cost_term_directions: Optional[Dict[str, float]] = None


class MpcAxisController:
    """Stateful per-axis MPC controller with Kalman filtering and warm starts."""

    def __init__(
        self,
        axis: str,
        control_cfg: ControlConfig,
        mpc_cfg: MpcConfig,
        *,
        solver: Optional[MpcQPSolver] = None,
    ) -> None:
        if axis not in {"yaw", "pitch"}:
            raise ValueError("axis must be 'yaw' or 'pitch'")

        self.axis = axis
        self._control_cfg = control_cfg
        self._model = MpcAxisModel.from_config(mpc_cfg)
        self._use_rate_measurement = bool(getattr(mpc_cfg.estimator, "use_rate_measurement", True))
        self._filter = AxisKalmanFilter(
            A=self._model.A,
            B=self._model.B,
            C=self._model.C,
            estimator_cfg=mpc_cfg.estimator,
        )
        self._solver = solver or OsqpSolver()
        self._last_command = 0.0
        self._slack_indices = self._build_slack_indices(mpc_cfg.constraints)
        self._num_vars = self._model.Nc + len(self._slack_indices)
        self._warm_start = np.zeros((self._num_vars,), dtype=float)
        self._last_solution = np.zeros((self._model.Nc,), dtype=float)
        self._default_distance = float(control_cfg.laser.default_distance_m)
        self._theta_unit_scale = max(1e-9, float(mpc_cfg.costs.theta_unit_scale_rad))
        self._omega_unit_scale = max(1e-9, float(mpc_cfg.costs.omega_unit_scale_rad_s))
        self._effort_unit_scale = max(1e-9, float(mpc_cfg.costs.effort_unit_scale))
        self._slew_unit_scale = max(1e-9, float(mpc_cfg.costs.slew_unit_scale))
        self._base_costs: Dict[str, float] = {
            "q_theta": float(mpc_cfg.costs.q_theta),
            "q_omega": float(mpc_cfg.costs.q_omega),
            "q_dtheta": float(mpc_cfg.costs.q_dtheta),
            "r": float(mpc_cfg.costs.r),
            "s": float(mpc_cfg.costs.s),
            "terminal": float(mpc_cfg.costs.terminal or 0.0),
            "rho": float(mpc_cfg.costs.rho),
        }
        self._cost_overrides: Dict[str, float] = dict(self._base_costs)
        self._base_estimator: Dict[str, float] = {
            "q_theta": float(mpc_cfg.estimator.q_theta),
            "q_omega": float(mpc_cfg.estimator.q_omega),
            "q_d": float(mpc_cfg.estimator.q_d),
            "r_theta": float(mpc_cfg.estimator.r_theta),
        }
        self._estimator_overrides: Dict[str, float] = dict(self._base_estimator)
        self._base_constraints: Dict[str, Optional[float]] = {
            "u_min": float(mpc_cfg.constraints.u_min),
            "u_max": float(mpc_cfg.constraints.u_max),
            "du_max": float(mpc_cfg.constraints.du_max),
            "theta_min": (
                None
                if mpc_cfg.constraints.theta_min is None
                else float(mpc_cfg.constraints.theta_min)
            ),
            "theta_max": (
                None
                if mpc_cfg.constraints.theta_max is None
                else float(mpc_cfg.constraints.theta_max)
            ),
            "omega_min": (
                None
                if mpc_cfg.constraints.omega_min is None
                else float(mpc_cfg.constraints.omega_min)
            ),
            "omega_max": (
                None
                if mpc_cfg.constraints.omega_max is None
                else float(mpc_cfg.constraints.omega_max)
            ),
        }
        self._constraint_overrides: Dict[str, Optional[float]] = dict(self._base_constraints)

    def set_cost_overrides(self, overrides: Mapping[str, float]) -> None:
        """Apply runtime MPC cost overrides for supported non-negative weights."""

        for key, value in overrides.items():
            if key not in self._base_costs:
                continue
            val = float(value)
            if not math.isfinite(val) or val < 0.0:
                continue
            self._cost_overrides[key] = val

    def get_cost_overrides(self) -> Dict[str, float]:
        return dict(self._cost_overrides)

    def set_estimator_overrides(self, overrides: Mapping[str, float]) -> None:
        for key, value in overrides.items():
            if key not in self._base_estimator:
                continue
            val = float(value)
            if not math.isfinite(val):
                continue
            if key == "r_theta" and val <= 0.0:
                continue
            if key != "r_theta" and val < 0.0:
                continue
            self._estimator_overrides[key] = val
        self._filter.set_noise_covariances(
            q_theta=float(self._estimator_overrides["q_theta"]),
            q_omega=float(self._estimator_overrides["q_omega"]),
            q_d=float(self._estimator_overrides["q_d"]),
            r_theta=float(self._estimator_overrides["r_theta"]),
        )

    def set_constraint_overrides(self, overrides: Mapping[str, Optional[float]]) -> None:
        for key, value in overrides.items():
            if key not in self._base_constraints:
                continue
            if value is None:
                self._constraint_overrides[key] = None
                continue
            val = float(value)
            if not math.isfinite(val):
                continue
            if key in {"du_max"} and val <= 0.0:
                continue
            self._constraint_overrides[key] = val

    def _resolved_cost(self, key: str) -> float:
        base = self._base_costs[key]
        value = self._cost_overrides.get(key, base)
        if not math.isfinite(value) or value < 0.0:
            return base
        return float(value)

    def _effective_constraints(self) -> MpcConstraintConfig:
        def _resolve(key: str) -> Optional[float]:
            value = self._constraint_overrides.get(key, self._base_constraints.get(key))
            if value is None:
                return None
            val = float(value)
            if not math.isfinite(val):
                return self._base_constraints.get(key)
            return val

        u_min = _resolve("u_min")
        u_max = _resolve("u_max")
        du_max = _resolve("du_max")
        theta_min = _resolve("theta_min")
        theta_max = _resolve("theta_max")
        omega_min = _resolve("omega_min")
        omega_max = _resolve("omega_max")

        base = self._model.constraints
        if u_min is None:
            u_min = float(base.u_min)
        if u_max is None:
            u_max = float(base.u_max)
        if du_max is None:
            du_max = float(base.du_max)

        if not (u_min < u_max):
            u_min = float(base.u_min)
            u_max = float(base.u_max)
        du_max = max(1e-6, float(du_max))

        if theta_min is not None and theta_max is not None and theta_min > theta_max:
            theta_min = base.theta_min
            theta_max = base.theta_max
        if omega_min is not None and omega_max is not None and omega_min > omega_max:
            omega_min = base.omega_min
            omega_max = base.omega_max

        return MpcConstraintConfig(
            u_min=float(u_min),
            u_max=float(u_max),
            du_max=float(du_max),
            theta_min=None if theta_min is None else float(theta_min),
            theta_max=None if theta_max is None else float(theta_max),
            omega_min=None if omega_min is None else float(omega_min),
            omega_max=None if omega_max is None else float(omega_max),
        )

    @staticmethod
    def _slack_count(constraints: MpcConstraintConfig) -> int:
        count = 0
        if constraints.theta_min is not None:
            count += 1
        if constraints.theta_max is not None:
            count += 1
        if constraints.omega_min is not None:
            count += 1
        if constraints.omega_max is not None:
            count += 1
        return count

    def _build_slack_indices(self, constraints: MpcConstraintConfig) -> Dict[str, int]:
        indices: Dict[str, int] = {}
        next_idx = self._model.Nc
        if constraints.theta_min is not None:
            indices["theta_min"] = next_idx
            next_idx += 1
        if constraints.theta_max is not None:
            indices["theta_max"] = next_idx
            next_idx += 1
        if constraints.omega_min is not None:
            indices["omega_min"] = next_idx
            next_idx += 1
        if constraints.omega_max is not None:
            indices["omega_max"] = next_idx
        return indices

    def reset(self) -> None:
        self._filter.reset()
        self._last_command = 0.0
        self._warm_start[:] = 0.0
        self._last_solution[:] = 0.0

    @property
    def state(self) -> np.ndarray:
        return self._filter.state

    def step_estimator(
        self,
        u_applied: float,
        theta_measurement: Optional[float],
        omega_measurement: Optional[float] = None,
    ) -> np.ndarray:
        self._filter.predict(u_applied)
        omega_meas = omega_measurement if self._use_rate_measurement else None
        return self._filter.update(theta_measurement, omega_meas)

    def compute_control(
        self,
        theta_ref_seq: Sequence[float],
        omega_ref_seq: Optional[Sequence[float]] = None,
        *,
        distance_seq: Optional[Sequence[Optional[float]]] = None,
        lateral_seq: Optional[Sequence[Optional[float]]] = None,
        radial_seq: Optional[Sequence[Optional[float]]] = None,
        solver: Optional[MpcQPSolver] = None,
    ) -> Tuple[float, MpcAxisDiagnostics]:
        model = self._model
        xhat = self._filter.state
        theta_list = list(theta_ref_seq)
        if not theta_list:
            raise ValueError("theta_ref_seq must contain at least one element")
        theta_ref = _prepare_sequence(theta_list, model.Np, fill_value=theta_list[-1])
        omega_ref = (
            _prepare_sequence(list(omega_ref_seq), model.Np, fill_value=0.0)
            if omega_ref_seq is not None
            else _finite_difference_refs(theta_ref, self._filter.state[0], model.Ts)
        )
        distance_arr = _optional_sequence(
            distance_seq, model.Np, default=self._default_distance
        )
        weights = np.ones((model.Np,), dtype=float)
        gamma_vec = np.power(model.horizon.gamma, np.arange(model.Np, dtype=float))
        theta_norm = 1.0 / self._theta_unit_scale
        omega_norm = 1.0 / self._omega_unit_scale
        q_theta_weight = self._resolved_cost("q_theta")
        q_omega_weight = self._resolved_cost("q_omega")
        q_dtheta_weight = self._resolved_cost("q_dtheta")
        r_weight = self._resolved_cost("r")
        s_weight = self._resolved_cost("s")
        terminal_weight = self._resolved_cost("terminal")
        rho_weight = self._resolved_cost("rho")
        q_theta = (q_theta_weight * theta_norm**2) * weights * gamma_vec
        q_omega = (q_omega_weight * omega_norm**2) * weights * gamma_vec
        q_dtheta = (q_dtheta_weight * theta_norm**2) * weights[1:] * gamma_vec[1:]
        if terminal_weight > 0.0:
            q_theta[-1] += terminal_weight * theta_norm**2

        l_theta = np.zeros((model.Np,), dtype=float)
        l_dtheta = np.zeros((max(0, model.Np - 1),), dtype=float)
        l_du = np.zeros((model.Nc,), dtype=float)

        qp = self._assemble_qp(
            xhat,
            theta_ref,
            omega_ref,
            q_theta,
            q_omega,
            q_dtheta,
            l_theta,
            l_dtheta,
            l_du,
            distance_arr,
            r_weight,
            s_weight,
            rho_weight,
        )
        qp_solver = solver or self._solver
        solution = qp_solver.solve(*qp, warm_start=self._warm_start)
        u_cmd, diagnostics = self._post_process_solution(
            solution,
            theta_ref,
            omega_ref,
            q_theta,
            q_omega,
            q_dtheta,
            l_theta,
            l_dtheta,
            l_du,
            distance_arr,
            weights,
            r_weight,
            s_weight,
            rho_weight,
        )
        return u_cmd, diagnostics

    def _assemble_qp(
        self,
        xhat: np.ndarray,
        theta_ref: np.ndarray,
        omega_ref: np.ndarray,
        q_theta: np.ndarray,
        q_omega: np.ndarray,
        q_dtheta: np.ndarray,
        l_theta: np.ndarray,
        l_dtheta: np.ndarray,
        l_du: np.ndarray,
        distance: np.ndarray,
        r_weight: float,
        s_weight: float,
        rho_weight: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model = self._model
        constr = self._effective_constraints()
        preds = model.predictions
        Nc = model.Nc
        num_vars = self._num_vars

        theta_free = preds.theta_projection @ xhat
        omega_free = preds.omega_projection @ xhat
        theta_map = preds.theta_input_map
        omega_map = preds.omega_input_map
        delta_theta_map = preds.error_difference @ theta_map

        theta_err = theta_free - theta_ref
        omega_err = omega_free - omega_ref
        delta_theta_err = preds.error_difference @ theta_err
        distance = np.asarray(distance, dtype=float)

        H = np.zeros((num_vars, num_vars), dtype=float)
        f = np.zeros((num_vars,), dtype=float)

        def gram(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
            return matrix.T @ (weights[:, None] * matrix)

        def cross(matrix: np.ndarray, weights: np.ndarray, vec: np.ndarray) -> np.ndarray:
            return matrix.T @ (weights * vec)

        H[:Nc, :Nc] += 2.0 * gram(theta_map, q_theta)
        f[:Nc] += 2.0 * cross(theta_map, q_theta, theta_err)
        H[:Nc, :Nc] += 2.0 * gram(omega_map, q_omega)
        f[:Nc] += 2.0 * cross(omega_map, q_omega, omega_err)
        if q_dtheta.size:
            H[:Nc, :Nc] += 2.0 * gram(delta_theta_map, q_dtheta)
            f[:Nc] += 2.0 * cross(delta_theta_map, q_dtheta, delta_theta_err)

        # Positional tracking remains symmetric via q_theta; directional
        # l_theta bias is intentionally disabled.
        if l_theta.size:
            f[:Nc] += theta_map.T @ l_theta
        # Favour approaching the target more aggressively or gently depending on
        # delta_theta_err sign and the provided l_dtheta weight.
        if l_dtheta.size:
            f[:Nc] += delta_theta_map.T @ l_dtheta

        # Input and slew effort penalties.
        d_offset = np.zeros((Nc,), dtype=float)
        d_offset[0] = -self._last_command
        scaled_r = float(r_weight) / (self._effort_unit_scale ** 2)
        scaled_s = float(s_weight) / (self._slew_unit_scale ** 2)
        if scaled_r > 0.0:
            H[:Nc, :Nc] += 2.0 * scaled_r * np.eye(Nc)
        if scaled_s > 0.0:
            D = preds.D
            H[:Nc, :Nc] += 2.0 * scaled_s * (D.T @ D)
            f[:Nc] += 2.0 * scaled_s * (D.T @ d_offset)

        # Signed slew shaping (l_du) is intentionally disabled.
        if l_du.size:
            H[:Nc, :Nc] += 0.0
            f[:Nc] += preds.D.T @ l_du

        # Slack penalties
        for idx in self._slack_indices.values():
            H[idx, idx] += 2.0 * float(rho_weight)

        A_blocks: list[np.ndarray] = []
        l_blocks: list[np.ndarray] = []
        u_blocks: list[np.ndarray] = []

        def append_block(A_block: np.ndarray, l_block: np.ndarray, u_block: np.ndarray) -> None:
            if A_block.size == 0:
                return
            A_blocks.append(A_block)
            l_blocks.append(l_block)
            u_blocks.append(u_block)

        # Input bounds
        A_input = np.zeros((Nc, self._num_vars), dtype=float)
        A_input[:, :Nc] = np.eye(Nc)
        l_input = np.full((Nc,), constr.u_min, dtype=float)
        u_input = np.full((Nc,), constr.u_max, dtype=float)
        append_block(A_input, l_input, u_input)

        # Rate limits
        A_rate = np.zeros((Nc, self._num_vars), dtype=float)
        A_rate[:, :Nc] = preds.D
        d_offset = np.zeros((Nc,), dtype=float)
        d_offset[0] = -self._last_command
        du = float(constr.du_max)
        l_rate = -du - d_offset
        u_rate = du - d_offset
        append_block(A_rate, l_rate, u_rate)

        # Slack non-negativity
        for key, idx in self._slack_indices.items():
            row = np.zeros((1, self._num_vars), dtype=float)
            row[0, idx] = 1.0
            append_block(row, np.array([0.0]), np.array([np.inf]))

        # State constraints
        theta_lower_idx = self._slack_indices.get("theta_min")
        theta_upper_idx = self._slack_indices.get("theta_max")
        omega_lower_idx = self._slack_indices.get("omega_min")
        omega_upper_idx = self._slack_indices.get("omega_max")

        def add_state_constraints(
            rows: list,
            l_bounds: list,
            u_bounds: list,
            gain: np.ndarray,
            free_vec: np.ndarray,
            lower_val: Optional[float],
            upper_val: Optional[float],
            lower_slack_idx: Optional[int],
            upper_slack_idx: Optional[int],
        ) -> None:
            for i in range(model.Np):
                if lower_val is not None:
                    row = np.zeros((self._num_vars,), dtype=float)
                    row[:Nc] = -gain[i]
                    if lower_slack_idx is not None:
                        row[lower_slack_idx] = -1.0
                    rows.append(row)
                    l_bounds.append(-np.inf)
                    u_bounds.append(free_vec[i] - lower_val)
                if upper_val is not None:
                    row = np.zeros((self._num_vars,), dtype=float)
                    row[:Nc] = gain[i]
                    if upper_slack_idx is not None:
                        row[upper_slack_idx] = -1.0
                    rows.append(row)
                    l_bounds.append(-np.inf)
                    u_bounds.append(upper_val - free_vec[i])

        row_list: list[np.ndarray] = []
        l_list: list[float] = []
        u_list: list[float] = []
        add_state_constraints(
            row_list,
            l_list,
            u_list,
            theta_map,
            theta_free,
            constr.theta_min,
            constr.theta_max,
            theta_lower_idx,
            theta_upper_idx,
        )
        add_state_constraints(
            row_list,
            l_list,
            u_list,
            omega_map,
            omega_free,
            constr.omega_min,
            constr.omega_max,
            omega_lower_idx,
            omega_upper_idx,
        )
        if row_list:
            A_state = np.vstack(row_list)
            l_state = np.array(l_list, dtype=float)
            u_state = np.array(u_list, dtype=float)
            append_block(A_state, l_state, u_state)

        A = np.vstack(A_blocks) if A_blocks else np.zeros((0, self._num_vars), dtype=float)
        l = np.concatenate(l_blocks) if l_blocks else np.zeros((0,), dtype=float)
        u = np.concatenate(u_blocks) if u_blocks else np.zeros((0,), dtype=float)
        return H, f, A, l, u

    def _post_process_solution(
        self,
        solution: MpcQPSolution,
        theta_ref: np.ndarray,
        omega_ref: np.ndarray,
        q_theta: np.ndarray,
        q_omega: np.ndarray,
        q_dtheta: np.ndarray,
        l_theta: np.ndarray,
        l_dtheta: np.ndarray,
        l_du: np.ndarray,
        distance: np.ndarray,
        weights: np.ndarray,
        r_weight: float,
        s_weight: float,
        rho_weight: float,
    ) -> Tuple[float, MpcAxisDiagnostics]:
        Nc = self._model.Nc
        if not solution.ok:
            safe = self._apply_limits(self._last_command)
            diagnostics = MpcAxisDiagnostics(
                status=solution.status,
                cost=None,
                u_sequence=self._last_solution[:Nc],
                theta_pred=self._model.predictions.theta_projection @ self._filter.state,
                omega_pred=self._model.predictions.omega_projection @ self._filter.state,
                weights=weights,
                solver_info=solution.info,
                slack=None,
                cost_terms=None,
                cost_term_directions=None,
            )
            return safe, diagnostics

        primal = solution.primal
        if primal is None or primal.size < Nc or not np.all(np.isfinite(primal)):
            safe = self._apply_limits(self._last_command)
            diagnostics = MpcAxisDiagnostics(
                status="invalid_solution",
                cost=None,
                u_sequence=self._last_solution[:Nc],
                theta_pred=self._model.predictions.theta_projection @ self._filter.state,
                omega_pred=self._model.predictions.omega_projection @ self._filter.state,
                weights=weights,
                solver_info=solution.info,
                slack=None,
                cost_terms=None,
                cost_term_directions=None,
            )
            return safe, diagnostics

        self._warm_start = primal.copy()
        u_sequence = primal[:Nc]
        X = self._model.predictions.Sx @ self._filter.state + self._model.predictions.Su @ u_sequence
        theta_pred = self._model.predictions.E_theta @ X
        omega_pred = self._model.predictions.E_omega @ X

        prev_command = self._last_command
        raw_cmd = float(u_sequence[0])
        cmd = self._apply_limits(raw_cmd)
        self._last_command = cmd
        self._last_solution = u_sequence.copy()

        cost_terms, cost_term_directions = self._compute_cost_terms(
            primal,
            u_sequence,
            theta_pred,
            omega_pred,
            theta_ref,
            omega_ref,
            q_theta,
            q_omega,
            q_dtheta,
            l_theta,
            l_dtheta,
            l_du,
            distance,
            prev_command,
            r_weight,
            s_weight,
            rho_weight,
        )

        diagnostics = MpcAxisDiagnostics(
            status=solution.status,
            cost=solution.cost,
            u_sequence=u_sequence,
            theta_pred=theta_pred,
            omega_pred=omega_pred,
            weights=weights,
            solver_info=solution.info,
            slack=self._extract_slack_summary(primal),
            cost_terms=cost_terms,
            cost_term_directions=cost_term_directions,
        )
        return cmd, diagnostics

    def _apply_limits(self, candidate: float) -> float:
        last = self._last_command if math.isfinite(self._last_command) else 0.0
        if not math.isfinite(candidate):
            return float(last)
        constr = self._effective_constraints()
        limited = float(np.clip(candidate, constr.u_min, constr.u_max))
        if not math.isfinite(limited):
            return float(last)
        delta = limited - last
        if not math.isfinite(delta):
            return float(last)
        delta = float(np.clip(delta, -constr.du_max, constr.du_max))
        if not math.isfinite(delta):
            return float(last)
        return float(last + delta)

    def _extract_slack_summary(self, vector: np.ndarray) -> Optional[Dict[str, float]]:
        if not self._slack_indices:
            return None
        summary: Dict[str, float] = {}
        for key, idx in self._slack_indices.items():
            if idx >= vector.size:
                continue
            summary[key] = max(0.0, float(vector[idx]))
        return summary or None

    def _compute_cost_terms(
        self,
        full_solution: np.ndarray,
        u_sequence: np.ndarray,
        theta_pred: np.ndarray,
        omega_pred: np.ndarray,
        theta_ref: np.ndarray,
        omega_ref: np.ndarray,
        q_theta: np.ndarray,
        q_omega: np.ndarray,
        q_dtheta: np.ndarray,
        l_theta: np.ndarray,
        l_dtheta: np.ndarray,
        l_du: np.ndarray,
        distance: np.ndarray,
        prev_command: float,
        r_weight: float,
        s_weight: float,
        rho_weight: float,
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        terms: Dict[str, float] = {}
        term_directions: Dict[str, float] = {}
        preds = self._model.predictions
        theta_map = preds.theta_input_map
        omega_map = preds.omega_input_map
        delta_theta_map = preds.error_difference @ theta_map

        def _record_direction(term_name: str, grad_u0: float) -> None:
            if not math.isfinite(grad_u0):
                return
            if abs(grad_u0) <= 1e-12:
                return
            term_directions[term_name] = -1.0 if grad_u0 > 0.0 else 1.0

        theta_err = theta_pred - theta_ref
        if theta_pred.size and q_theta.size:
            theta_cost = float(np.dot(q_theta[: theta_err.size], theta_err**2))
            if math.isfinite(theta_cost):
                terms["theta"] = theta_cost
            if theta_map.shape[1] > 0:
                theta_grad_u0 = 2.0 * float(
                    np.dot(theta_map[: theta_err.size, 0], q_theta[: theta_err.size] * theta_err)
                )
                _record_direction("theta", theta_grad_u0)

        if theta_pred.size and l_theta.size and np.any(l_theta):
            theta_signed = float(np.dot(l_theta[: theta_err.size], theta_err))
            if math.isfinite(theta_signed):
                terms["theta_linear"] = theta_signed
            if theta_map.shape[1] > 0:
                theta_linear_grad_u0 = float(np.dot(theta_map[:, 0], l_theta))
                _record_direction("theta_linear", theta_linear_grad_u0)

        omega_err = omega_pred - omega_ref
        if omega_pred.size and q_omega.size:
            omega_cost = float(np.dot(q_omega[: omega_err.size], omega_err**2))
            if math.isfinite(omega_cost):
                terms["omega"] = omega_cost
            if omega_map.shape[1] > 0:
                omega_grad_u0 = 2.0 * float(
                    np.dot(omega_map[: omega_err.size, 0], q_omega[: omega_err.size] * omega_err)
                )
                _record_direction("omega", omega_grad_u0)

        if q_dtheta.size:
            delta_theta_err = preds.error_difference @ theta_err
            dtheta_cost = float(np.dot(q_dtheta[: delta_theta_err.size], delta_theta_err**2))
            if math.isfinite(dtheta_cost):
                terms["dtheta"] = dtheta_cost
            if delta_theta_map.shape[1] > 0:
                dtheta_grad_u0 = 2.0 * float(
                    np.dot(
                        delta_theta_map[: delta_theta_err.size, 0],
                        q_dtheta[: delta_theta_err.size] * delta_theta_err,
                    )
                )
                _record_direction("dtheta", dtheta_grad_u0)
            if l_dtheta.size and np.any(l_dtheta):
                dtheta_signed = float(np.dot(l_dtheta[: delta_theta_err.size], delta_theta_err))
                if math.isfinite(dtheta_signed):
                    terms["dtheta_linear"] = dtheta_signed
                if delta_theta_map.shape[1] > 0:
                    dtheta_linear_grad_u0 = float(np.dot(delta_theta_map[:, 0], l_dtheta))
                    _record_direction("dtheta_linear", dtheta_linear_grad_u0)

        if r_weight > 0.0 and u_sequence.size:
            effort_cost = float(r_weight * float(u_sequence @ u_sequence))
            if math.isfinite(effort_cost):
                terms["effort"] = effort_cost
            effort_grad_u0 = 2.0 * float(r_weight) * float(u_sequence[0])
            _record_direction("effort", effort_grad_u0)

        if s_weight > 0.0:
            D = preds.D
            if D.size:
                delta = D @ u_sequence
                if delta.size:
                    delta = delta.copy()
                    delta[0] += -prev_command
                    slew_cost = float(s_weight * float(delta @ delta))
                    if math.isfinite(slew_cost):
                        terms["slew"] = slew_cost
                    slew_grad = 2.0 * float(s_weight) * (D.T @ delta)
                    if slew_grad.size:
                        _record_direction("slew", float(slew_grad[0]))
                    if l_du.size and np.any(l_du):
                        slew_signed = float(np.dot(l_du[: delta.size], delta))
                        if math.isfinite(slew_signed) and abs(slew_signed) > 0.0:
                            terms["slew_linear"] = slew_signed
                        slew_linear_grad = D.T @ l_du
                        if slew_linear_grad.size:
                            _record_direction("slew_linear", float(slew_linear_grad[0]))

        if self._slack_indices and rho_weight > 0.0:
            total = 0.0
            for idx in self._slack_indices.values():
                if idx >= full_solution.size:
                    continue
                val = max(0.0, float(full_solution[idx]))
                total += val * val
            if total > 0.0:
                slack_cost = float(rho_weight * total)
                if math.isfinite(slack_cost):
                    terms["slack"] = slack_cost

        return terms or None, term_directions or None


def _prepare_sequence(
    seq: Sequence[float],
    length: int,
    *,
    fill_value: float,
) -> np.ndarray:
    arr = np.asarray(list(seq), dtype=float)
    if arr.size == 0:
        arr = np.array([fill_value], dtype=float)
    if arr.size < length:
        pad_value = arr[-1]
        pad = np.full((length - arr.size,), pad_value, dtype=float)
        arr = np.concatenate([arr, pad])
    elif arr.size > length:
        arr = arr[:length]
    return arr


def _finite_difference_refs(theta_ref: np.ndarray, theta0: float, Ts: float) -> np.ndarray:
    omega = np.zeros_like(theta_ref)
    prev = float(theta0)
    for i, target in enumerate(theta_ref):
        omega[i] = (target - prev) / Ts
        prev = target
    return omega


def _optional_sequence(
    seq: Optional[Sequence[Optional[float]]],
    length: int,
    *,
    default: float,
) -> np.ndarray:
    values = np.full((length,), default, dtype=float)
    if seq is None:
        return values
    last = default
    for i in range(length):
        if i < len(seq):
            val = seq[i]
            if val is not None:
                last = float(val)
        values[i] = last
    return values


__all__ = [
    "AxisKalmanFilter",
    "MpcAxisController",
    "MpcAxisDiagnostics",
    "MpcAxisModel",
    "MpcPredictionCache",
    "MpcQPSolution",
    "MpcQPSolver",
    "MpcSolverError",
    "OsqpSolver",
]

