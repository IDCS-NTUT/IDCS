import unittest
from typing import Optional

import pytest

np = pytest.importorskip("numpy")

from common.control import (
    AxisPair,
    ControlConfig,
    LaserAimingControlConfig,
    MpcAdaptiveWeightConfig,
    MpcApproachConfig,
    MpcConfig,
    MpcConstraintConfig,
    MpcCostConfig,
    MpcEstimatorConfig,
    MpcHorizonConfig,
    MpcPlantConfig,
)
from jetson.mpc import AxisKalmanFilter, MpcAxisController, MpcAxisDiagnostics, MpcAxisModel


def _make_control_config() -> ControlConfig:
    return ControlConfig(
        mode="rate",
        loop_hz=50.0,
        fx_px=800.0,
        fy_px=820.0,
        cx_px=640.0,
        cy_px=360.0,
        aim_mode="camera_center",
        kp=AxisPair(0.0, 0.0),
        kd=AxisPair(0.0, 0.0),
        ki=AxisPair(0.0, 0.0),
        rate_limits=AxisPair(10.0, 10.0),
        accel_limits=AxisPair(40.0, 40.0),
        deadband_px=0.0,
        smooth_px_alpha=0.0,
        lost_target_timeout_ms=100,
        reinit_on_lost=True,
        target_selector="max_conf",
        yaw_sign=1.0,
        pitch_sign=-1.0,
        frame_size=(1280, 720),
        fov_deg=None,
        laser=LaserAimingControlConfig(
            tolerance_px=3.0,
            use_range="known_size",
            default_distance_m=25.0,
        ),
    )


def _make_mpc_config(
    prediction: int = 3,
    control: int = 2,
    *,
    approach: Optional[MpcApproachConfig] = None,
) -> MpcConfig:
    approach_cfg = approach or MpcApproachConfig(
        k_approach=0.0,
        w_base=0.0,
        w_max=0.0,
        e_gate_center=0.2,
        e_gate_width=0.1,
        d_gate_near=None,
        d_gate_far=None,
    )
    return MpcConfig(
        horizon=MpcHorizonConfig(
            prediction_horizon=prediction,
            control_horizon=control,
            sample_time_s=0.1,
            gamma=0.95,
            move_blocking=True,
        ),
        plant=MpcPlantConfig(a_u=1.0, a_f=0.2),
        estimator=MpcEstimatorConfig(q_theta=1e-3, q_omega=5e-3, q_d=1e-4, r_theta=2e-3),
        costs=MpcCostConfig(q_theta_base=2.0, q_omega_base=0.8, r=0.05, s=0.1, terminal=0.5, rho=50.0),
        adaptive=MpcAdaptiveWeightConfig(
            alpha_d=0.3,
            alpha_v=0.2,
            alpha_tau=0.2,
            p=1.0,
            eps=1e-3,
            w_min=0.2,
            w_max=5.0,
        ),
        approach=approach_cfg,
        constraints=MpcConstraintConfig(
            u_min=-1.0,
            u_max=1.0,
            du_max=0.4,
            theta_min=-0.6,
            theta_max=0.6,
            omega_min=-3.0,
            omega_max=3.0,
        ),
    )


class DummySolver:
    def __init__(
        self,
        solution: np.ndarray,
        status: str = "optimal",
        provide_solution: bool = True,
    ) -> None:
        self.solution = solution
        self.status = status
        self.provide_solution = provide_solution
        self.calls = []

    def solve(self, H, f, A, l, u, *, warm_start=None):
        self.calls.append({
            "H": H,
            "f": f,
            "A": A,
            "l": l,
            "u": u,
            "warm_start": None if warm_start is None else warm_start.copy(),
        })
        cost = float(np.linalg.norm(f))
        primal = self.solution.copy() if self.provide_solution else None
        return MpcQPSolution(status=self.status, primal=primal, cost=cost, info={"iter": 5.0})


class PredictionMatrixTests(unittest.TestCase):
    def test_prediction_matrices_match_manual_values(self) -> None:
        cfg = _make_mpc_config(prediction=2, control=2)
        model = MpcAxisModel.from_config(cfg)
        A = model.A
        B = model.B

        self.assertEqual(model.predictions.Sx.shape, (6, 3))
        self.assertEqual(model.predictions.Su.shape, (6, 2))
        self.assertEqual(model.predictions.D.shape, (2, 2))

        np.testing.assert_allclose(model.predictions.Sx[:3], A)
        np.testing.assert_allclose(model.predictions.Sx[3:], A @ A)
        np.testing.assert_allclose(model.predictions.Su[:3, 0:1], B)

        # Tail move-blocking should add an extra B contribution for the last column
        self.assertGreater(model.predictions.Su[3:, 1].sum(), 0.0)


class KalmanFilterTests(unittest.TestCase):
    def test_filter_converges_to_measurement(self) -> None:
        cfg = _make_mpc_config()
        model = MpcAxisModel.from_config(cfg)
        kf = AxisKalmanFilter(A=model.A, B=model.B, C=model.C, estimator_cfg=cfg.estimator)
        state = kf.predict(0.2)
        self.assertAlmostEqual(state[0], 0.0)
        updated = kf.update(0.1)
        self.assertAlmostEqual(updated[0], 0.1, places=3)


class AxisControllerTests(unittest.TestCase):
    def test_controller_builds_qp_and_tracks_solution(self) -> None:
        cfg = _make_mpc_config()
        control_cfg = _make_control_config()
        model = MpcAxisModel.from_config(cfg)
        num_vars = model.Nc + 4  # theta_min/max + omega_min/max slack vars
        solution = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=float)
        solver = DummySolver(solution)
        controller = MpcAxisController("yaw", control_cfg, cfg, solver=solver)

        theta_refs = [0.05, 0.02, -0.01]
        command, diagnostics = controller.compute_control(
            theta_refs,
            distance_seq=[5.0, None, 7.0],
            lateral_seq=[0.1, 0.0, 0.2],
            radial_seq=[0.5, 0.5, 0.1],
        )

        self.assertIsInstance(diagnostics, MpcAxisDiagnostics)
        self.assertEqual(diagnostics.status, "optimal")
        np.testing.assert_allclose(diagnostics.u_sequence, solution[: model.Nc])
        self.assertAlmostEqual(command, 0.2)
        self.assertIsInstance(diagnostics.slack, dict)
        assert diagnostics.slack is not None
        self.assertIn("theta_min", diagnostics.slack)
        self.assertEqual(len(solver.calls), 1)
        np.testing.assert_allclose(solver.calls[0]["warm_start"], np.zeros_like(solution))

        theta_free = model.predictions.theta_projection @ controller.state
        expected_theta = theta_free + model.predictions.theta_input_map @ diagnostics.u_sequence
        np.testing.assert_allclose(diagnostics.theta_pred, expected_theta)

    def test_osqp_status_solved_counts_as_success(self) -> None:
        cfg = _make_mpc_config()
        control_cfg = _make_control_config()
        solution = np.array([0.15, 0.05, 0.0, 0.0, 0.0, 0.0], dtype=float)
        solver = DummySolver(solution, status="solved")
        controller = MpcAxisController("yaw", control_cfg, cfg, solver=solver)

        cmd, diagnostics = controller.compute_control([0.02, 0.01, 0.0])

        self.assertNotEqual(cmd, 0.0)
        self.assertEqual(diagnostics.status, "solved")
        np.testing.assert_allclose(diagnostics.u_sequence, solution[: cfg.horizon.control_horizon])

    def test_solver_failure_falls_back_to_previous_command(self) -> None:
        cfg = _make_mpc_config()
        control_cfg = _make_control_config()
        controller = MpcAxisController(
            "pitch",
            control_cfg,
            cfg,
            solver=DummySolver(
                np.zeros(cfg.horizon.control_horizon + 4),
                status="failed",
                provide_solution=False,
            ),
        )
        cmd, diagnostics = controller.compute_control([0.0, 0.0, 0.0])
        self.assertEqual(cmd, 0.0)
        self.assertEqual(diagnostics.status, "failed")

    def test_approach_cost_changes_qp_terms(self) -> None:
        control_cfg = _make_control_config()
        base_cfg = _make_mpc_config()
        approach_cfg = MpcApproachConfig(
            k_approach=0.1,
            w_base=0.5,
            w_max=1.0,
            e_gate_center=0.2,
            e_gate_width=0.05,
            d_gate_near=0.0,
            d_gate_far=10.0,
        )
        biased_cfg = _make_mpc_config(approach=approach_cfg)

        num_vars = biased_cfg.horizon.control_horizon + MpcAxisController._slack_count(
            biased_cfg.constraints
        )
        solution = np.zeros((num_vars,), dtype=float)

        solver_plain = DummySolver(solution.copy())
        plain = MpcAxisController("yaw", control_cfg, base_cfg, solver=solver_plain)
        theta_refs = [0.0, 0.0, 0.0]
        omega_refs = [0.5, 0.5, 0.5]
        plain.compute_control(
            theta_refs,
            omega_ref_seq=omega_refs,
            distance_seq=[2.0, 2.0, 2.0],
        )

        solver_biased = DummySolver(solution.copy())
        biased = MpcAxisController("yaw", control_cfg, biased_cfg, solver=solver_biased)
        biased.compute_control(
            theta_refs,
            omega_ref_seq=omega_refs,
            distance_seq=[2.0, 2.0, 2.0],
        )

        self.assertEqual(len(solver_plain.calls), 1)
        self.assertEqual(len(solver_biased.calls), 1)
        H_plain = solver_plain.calls[0]["H"]
        H_biased = solver_biased.calls[0]["H"]
        f_plain = solver_plain.calls[0]["f"]
        f_biased = solver_biased.calls[0]["f"]
        self.assertFalse(np.allclose(H_plain, H_biased))
        self.assertFalse(np.allclose(f_plain, f_biased))


if __name__ == "__main__":
    unittest.main()
