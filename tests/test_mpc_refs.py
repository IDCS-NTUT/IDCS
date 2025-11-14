import math
import unittest

from common.control import (
    AxisPair,
    ControlConfig,
    LaserAimingControlConfig,
    MpcAdaptiveWeightConfig,
    MpcConfig,
    MpcConstraintConfig,
    MpcCostConfig,
    MpcEstimatorConfig,
    MpcHorizonConfig,
    MpcPlantConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.schemas import CamState
from jetson.mpc_refs import AxisReferenceSequences, MpcReferenceBuilder


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


def _make_mpc_config(prediction: int = 4, control: int = 2) -> MpcConfig:
    return MpcConfig(
        horizon=MpcHorizonConfig(
            prediction_horizon=prediction,
            control_horizon=control,
            sample_time_s=0.1,
            gamma=0.95,
            move_blocking=False,
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


class ReferenceBuilderTests(unittest.TestCase):
    def test_builder_generates_sequences_with_velocity(self) -> None:
        control_cfg = _make_control_config()
        mpc_cfg = _make_mpc_config()
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)
        cam_state = CamState(
            frame_id=1,
            src_ts_ms=0,
            pan=0.1,
            tilt=-0.05,
            pan_rate=0.2,
            tilt_rate=-0.1,
        )
        refs = builder.build(
            target_uv=(660.0, 340.0),
            aim_uv=(640.0, 360.0),
            timestamp=1.0,
            cam_state=cam_state,
            distance_m=12.0,
            target_velocity_px_s=(6.0, -4.0),
        )

        yaw_refs = refs["yaw"]
        self.assertIsInstance(yaw_refs, AxisReferenceSequences)
        self.assertEqual(len(yaw_refs.theta), mpc_cfg.horizon.prediction_horizon)
        px_err = pixel_delta(660.0, 340.0, 640.0, 360.0, control_cfg, apply_deadband=True)
        err_rad = angular_error_from_pixel_delta(px_err, control_cfg)
        Ts = mpc_cfg.horizon.sample_time_s
        target_rate = control_cfg.yaw_sign * 6.0 / control_cfg.fx_px
        expected_theta = tuple(
            cam_state.pan + err_rad.yaw + target_rate * Ts * i
            for i in range(mpc_cfg.horizon.prediction_horizon)
        )
        self.assertTrue(math.isclose(yaw_refs.theta[0], expected_theta[0], rel_tol=1e-6))
        self.assertSequenceEqual(tuple(round(x, 6) for x in yaw_refs.theta), tuple(round(x, 6) for x in expected_theta))

        self.assertIsNotNone(yaw_refs.omega)
        assert yaw_refs.omega is not None
        expected_rate = target_rate + (cam_state.pan_rate or 0.0)
        self.assertTrue(all(math.isclose(val, expected_rate, rel_tol=1e-9) for val in yaw_refs.omega))

        lateral_expected = abs(12.0) * math.hypot(
            control_cfg.yaw_sign * 6.0 / control_cfg.fx_px,
            control_cfg.pitch_sign * (-4.0) / control_cfg.fy_px,
        )
        self.assertTrue(all(math.isclose(val or 0.0, lateral_expected, rel_tol=1e-9) for val in yaw_refs.lateral))
        self.assertTrue(all(val == 12.0 for val in yaw_refs.distance))

    def test_single_axis_builder_without_velocity(self) -> None:
        control_cfg = _make_control_config()
        mpc_cfg = _make_mpc_config()
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon, axes=("yaw",))
        refs = builder.build(
            target_uv=(640.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=2.0,
            theta_estimates={"yaw": 0.3},
        )

        self.assertIn("yaw", refs)
        self.assertNotIn("pitch", refs)
        yaw_refs = refs["yaw"]
        self.assertIsNone(yaw_refs.omega)
        self.assertTrue(all(math.isclose(val, 0.3, rel_tol=1e-9) for val in yaw_refs.theta))
        default_distance = control_cfg.laser.default_distance_m
        self.assertTrue(all(math.isclose(val or 0.0, default_distance, rel_tol=1e-9) for val in yaw_refs.distance))

    def test_distance_projection_tracks_radial_velocity(self) -> None:
        control_cfg = _make_control_config()
        mpc_cfg = _make_mpc_config()
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)
        base_kwargs = dict(
            target_uv=(640.0, 360.0),
            aim_uv=(640.0, 360.0),
        )
        builder.build(timestamp=0.0, distance_m=10.0, **base_kwargs)
        refs = builder.build(timestamp=0.5, distance_m=9.5, **base_kwargs)
        yaw_refs = refs["yaw"]
        self.assertTrue(all(math.isclose(val or 0.0, 9.5, rel_tol=1e-9) for val in yaw_refs.distance[:1]))
        expected_radial = (9.5 - 10.0) / 0.5
        self.assertTrue(all(math.isclose(val or 0.0, expected_radial, rel_tol=1e-9) for val in yaw_refs.radial))


if __name__ == "__main__":
    unittest.main()
