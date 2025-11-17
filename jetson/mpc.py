"""Axis-level building blocks for the MPC gimbal controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from common.control import (
    ControlConfig,
    MpcAdaptiveWeightConfig,
    MpcApproachConfig,
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
        self._Q = np.diag([estimator_cfg.q_theta, estimator_cfg.q_omega, estimator_cfg.q_d])
        self._R = float(estimator_cfg.r_theta)
        self._x = np.zeros((A.shape[0],), dtype=float)
        self._P = np.eye(A.shape[0], dtype=float) * 1e-3
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
        self._P = np.eye(self._A.shape[0], dtype=float) * 1e-3
        self._innovation = 0.0
        self._innovation_var = self._R

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

    def update(self, theta_meas: Optional[float]) -> np.ndarray:
        if theta_meas is None:
            # Prediction-only update; track innovation variance for diagnostics.
            S = float(self._C @ self._P @ self._C.T + self._R)
            self._innovation_var = S
            self._innovation = 0.0
            return self.state

        z = float(theta_meas)
        y = z - float(self._C @ self._x)
        S = float(self._C @ self._P @ self._C.T + self._R)
        if S <= 0.0:
            raise RuntimeError("Kalman filter innovation covariance must be positive")
        K = self._P @ self._C.T / S
        self._x = self._x + (K.flatten() * y)
        I = np.eye(self._A.shape[0], dtype=float)
        self._P = (I - K @ self._C) @ self._P
        self._innovation = y
        self._innovation_var = S
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
    adaptive: MpcAdaptiveWeightConfig
    approach: MpcApproachConfig
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
            adaptive=cfg.adaptive,
            approach=cfg.approach,
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

    def step_estimator(self, u_applied: float, theta_measurement: Optional[float]) -> np.ndarray:
        self._filter.predict(u_applied)
        return self._filter.update(theta_measurement)

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
        weights = _compute_adaptive_weights(
            self._model.adaptive,
            distance_seq,
            lateral_seq,
            radial_seq,
            length=model.Np,
        )
        gamma_vec = np.power(model.horizon.gamma, np.arange(model.Np, dtype=float))
        q_theta = model.costs.q_theta_base * weights * gamma_vec
        q_omega = model.costs.q_omega_base * weights * gamma_vec
        if model.costs.terminal is not None:
            q_theta[-1] += model.costs.terminal

        qp = self._assemble_qp(
            xhat, theta_ref, omega_ref, q_theta, q_omega, distance_arr
        )
        qp_solver = solver or self._solver
        solution = qp_solver.solve(*qp, warm_start=self._warm_start)
        u_cmd, diagnostics = self._post_process_solution(
            solution,
            theta_ref,
            omega_ref,
            q_theta,
            q_omega,
            distance_arr,
            weights,
        )
        return u_cmd, diagnostics

    def _assemble_qp(
        self,
        xhat: np.ndarray,
        theta_ref: np.ndarray,
        omega_ref: np.ndarray,
        q_theta: np.ndarray,
        q_omega: np.ndarray,
        distance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model = self._model
        preds = model.predictions
        Nc = model.Nc
        num_vars = self._num_vars

        theta_free = preds.theta_projection @ xhat
        omega_free = preds.omega_projection @ xhat
        theta_map = preds.theta_input_map
        omega_map = preds.omega_input_map

        theta_err = theta_free - theta_ref
        omega_err = omega_free - omega_ref
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

        self._add_approach_cost(H, f, theta_err, theta_map, omega_ref, distance)

        # Input and slew effort penalties.
        d_offset = np.zeros((Nc,), dtype=float)
        d_offset[0] = -self._last_command
        if model.costs.r > 0.0:
            H[:Nc, :Nc] += 2.0 * model.costs.r * np.eye(Nc)
        if model.costs.s > 0.0:
            D = preds.D
            H[:Nc, :Nc] += 2.0 * model.costs.s * (D.T @ D)
            f[:Nc] += 2.0 * model.costs.s * (D.T @ d_offset)

        # Slack penalties
        for idx in self._slack_indices.values():
            H[idx, idx] += 2.0 * model.costs.rho

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
        constr = model.constraints
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

    def _add_approach_cost(
        self,
        H: np.ndarray,
        f: np.ndarray,
        theta_err: np.ndarray,
        theta_map: np.ndarray,
        omega_ref: np.ndarray,
        distance: np.ndarray,
    ) -> None:
        cfg = self._model.approach
        diff = self._model.predictions.error_difference
        if diff.size == 0 or cfg.w_base <= 0.0 or cfg.k_approach <= 0.0:
            return
        weights = _approach_weight_schedule(cfg, theta_err, distance)
        if weights.size == 0 or not np.any(weights > 0.0):
            return
        delta_map = diff @ theta_map
        delta_free = diff @ theta_err
        delta_ref = _approach_delta_reference(omega_ref, cfg.k_approach)
        if delta_ref.shape != delta_free.shape:
            delta_ref = np.zeros_like(delta_free)
        H[: self._model.Nc, : self._model.Nc] += 2.0 * (
            delta_map.T @ (weights[:, None] * delta_map)
        )
        bias = weights * (delta_free - delta_ref)
        f[: self._model.Nc] += 2.0 * (delta_map.T @ bias)

    def _post_process_solution(
        self,
        solution: MpcQPSolution,
        theta_ref: np.ndarray,
        omega_ref: np.ndarray,
        q_theta: np.ndarray,
        q_omega: np.ndarray,
        distance: np.ndarray,
        weights: np.ndarray,
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
            )
            return safe, diagnostics

        primal = solution.primal
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

        cost_terms = self._compute_cost_terms(
            primal,
            u_sequence,
            theta_pred,
            omega_pred,
            theta_ref,
            omega_ref,
            q_theta,
            q_omega,
            distance,
            prev_command,
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
        )
        return cmd, diagnostics

    def _apply_limits(self, candidate: float) -> float:
        constr = self._model.constraints
        limited = float(np.clip(candidate, constr.u_min, constr.u_max))
        delta = limited - self._last_command
        delta = float(np.clip(delta, -constr.du_max, constr.du_max))
        return self._last_command + delta

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
        distance: np.ndarray,
        prev_command: float,
    ) -> Optional[Dict[str, float]]:
        terms: Dict[str, float] = {}
        theta_err = theta_pred - theta_ref
        if theta_pred.size and q_theta.size:
            theta_cost = float(np.dot(q_theta[: theta_err.size], theta_err**2))
            if math.isfinite(theta_cost):
                terms["theta"] = theta_cost

        omega_err = omega_pred - omega_ref
        if omega_pred.size and q_omega.size:
            omega_cost = float(np.dot(q_omega[: omega_err.size], omega_err**2))
            if math.isfinite(omega_cost):
                terms["omega"] = omega_cost

        diff = self._model.predictions.error_difference
        approach_cfg = self._model.approach
        if (
            diff.size
            and approach_cfg.w_base > 0.0
            and approach_cfg.k_approach > 0.0
            and theta_err.size
        ):
            distance_vec = np.asarray(distance, dtype=float)
            weights = _approach_weight_schedule(approach_cfg, theta_err, distance_vec)
            if weights.size and np.any(weights > 0.0):
                delta_err = diff @ theta_err
                delta_ref = _approach_delta_reference(omega_ref, approach_cfg.k_approach)
                if delta_ref.shape != delta_err.shape:
                    delta_ref = np.zeros_like(delta_err)
                approach_cost = float(np.sum(weights * (delta_err - delta_ref) ** 2))
                if math.isfinite(approach_cost):
                    terms["approach"] = approach_cost

        if self._model.costs.r > 0.0 and u_sequence.size:
            effort_cost = float(self._model.costs.r * float(u_sequence @ u_sequence))
            if math.isfinite(effort_cost):
                terms["effort"] = effort_cost

        if self._model.costs.s > 0.0:
            D = self._model.predictions.D
            if D.size:
                delta = D @ u_sequence
                if delta.size:
                    delta = delta.copy()
                    delta[0] += -prev_command
                    slew_cost = float(self._model.costs.s * float(delta @ delta))
                    if math.isfinite(slew_cost):
                        terms["slew"] = slew_cost

        if self._slack_indices and self._model.costs.rho > 0.0:
            total = 0.0
            for idx in self._slack_indices.values():
                if idx >= full_solution.size:
                    continue
                val = max(0.0, float(full_solution[idx]))
                total += val * val
            if total > 0.0:
                slack_cost = float(self._model.costs.rho * total)
                if math.isfinite(slack_cost):
                    terms["slack"] = slack_cost

        return terms or None


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


def _compute_adaptive_weights(
    adaptive_cfg: MpcAdaptiveWeightConfig,
    distance_seq: Optional[Sequence[Optional[float]]],
    lateral_seq: Optional[Sequence[Optional[float]]],
    radial_seq: Optional[Sequence[Optional[float]]],
    *,
    length: int,
) -> np.ndarray:
    distances = _optional_sequence(distance_seq, length, default=5.0)
    lateral = _optional_sequence(lateral_seq, length, default=0.0)
    radial = _optional_sequence(radial_seq, length, default=0.0)
    weights = np.zeros((length,), dtype=float)
    eps = adaptive_cfg.eps
    for i in range(length):
        d = max(eps, abs(distances[i]))
        v_lat = abs(lateral[i])
        v_rad = abs(radial[i])
        tau = distances[i] / max(eps, v_rad)
        w = (
            adaptive_cfg.alpha_d * (1.0 / (d + eps)) ** adaptive_cfg.p
            + adaptive_cfg.alpha_v * (v_lat / (d + eps))
            + adaptive_cfg.alpha_tau * (1.0 / (abs(tau) + eps))
        )
        weights[i] = float(np.clip(w, adaptive_cfg.w_min, adaptive_cfg.w_max))
    return weights


def _approach_delta_reference(omega_ref: np.ndarray, k_approach: float) -> np.ndarray:
    if omega_ref.size <= 1 or k_approach <= 0.0:
        return np.zeros((max(0, omega_ref.size - 1),), dtype=float)
    return -k_approach * np.sign(omega_ref[1:])


def _approach_weight_schedule(
    cfg: MpcApproachConfig, theta_err: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    horizon = theta_err.size
    if horizon <= 1:
        return np.zeros((0,), dtype=float)
    err_mag = np.abs(theta_err[1:])
    if distance.size < horizon:
        pad_value = distance[-1] if distance.size else 0.0
        padded = np.pad(distance, (0, horizon - distance.size), constant_values=pad_value)
    else:
        padded = distance
    dist_vals = np.abs(padded[1:])
    weights = np.zeros((horizon - 1,), dtype=float)
    lower = max(0.0, cfg.e_gate_center - cfg.e_gate_width)
    upper = cfg.e_gate_center + cfg.e_gate_width
    span = max(1e-9, upper - lower)
    use_distance_gate = cfg.d_gate_near is not None and cfg.d_gate_far is not None
    for i in range(weights.size):
        mag = err_mag[i]
        if cfg.e_gate_width <= 0.0:
            g_e = 1.0 if mag <= cfg.e_gate_center else 0.0
        else:
            if mag <= lower:
                g_e = 1.0
            elif mag >= upper:
                g_e = 0.0
            else:
                g_e = 1.0 - (mag - lower) / span
        g_d = 1.0
        if use_distance_gate:
            near = float(cfg.d_gate_near)
            far = float(cfg.d_gate_far)
            dist = dist_vals[i]
            if dist <= near:
                g_d = 1.0
            elif dist >= far:
                g_d = 0.0
            else:
                span_d = max(1e-6, far - near)
                g_d = 1.0 - (dist - near) / span_d
        weights[i] = cfg.w_base * g_e * g_d
    return np.clip(weights, 0.0, cfg.w_max)


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

