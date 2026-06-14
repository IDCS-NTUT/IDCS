import math
import unittest
from typing import Optional

from common.control import (
    AxisPair,
    ControlConfig,
    LaserAimingControlConfig,
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


def _make_control_config(
    *,
    rate_limit: float = 10.0,
    mpc: Optional[MpcConfig] = None,
) -> ControlConfig:
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
        rate_limits=AxisPair(rate_limit, rate_limit),
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
        mpc=mpc,
    )


def _make_mpc_config(
    prediction: int = 4,
    control: int = 2,
    *,
    effect_delay_s: float = 0.0,
    predictor_enabled: bool = False,
    predictor_alpha: float = 0.85,
    predictor_beta: float = 0.05,
    adaptive_effect_delay_enabled: bool = False,
    adaptive_effect_delay_min_s: float = 0.0,
    adaptive_effect_delay_max_s: float = 0.25,
    adaptive_effect_delay_alpha: float = 0.1,
    adaptive_effect_delay_gain: float = 0.2,
    adaptive_effect_delay_rate_eps: float = 1e-3,
    effect_delay_mode: str = "fixed",
    projectile_speed_m_s: Optional[float] = None,
    impact_delay_bias_s: float = 0.0,
) -> MpcConfig:
    return MpcConfig(
        horizon=MpcHorizonConfig(
            prediction_horizon=prediction,
            control_horizon=control,
            sample_time_s=0.1,
            gamma=0.95,
            move_blocking=False,
            effect_delay_mode=effect_delay_mode,
            effect_delay_s=effect_delay_s,
            projectile_speed_m_s=projectile_speed_m_s,
            impact_delay_bias_s=impact_delay_bias_s,
            predictor_enabled=predictor_enabled,
            predictor_alpha=predictor_alpha,
            predictor_beta=predictor_beta,
            adaptive_effect_delay_enabled=adaptive_effect_delay_enabled,
            adaptive_effect_delay_min_s=adaptive_effect_delay_min_s,
            adaptive_effect_delay_max_s=adaptive_effect_delay_max_s,
            adaptive_effect_delay_alpha=adaptive_effect_delay_alpha,
            adaptive_effect_delay_gain=adaptive_effect_delay_gain,
            adaptive_effect_delay_rate_eps=adaptive_effect_delay_rate_eps,
        ),
        plant=MpcPlantConfig(a_u=1.0, a_f=0.2),
        estimator=MpcEstimatorConfig(q_theta=1e-3, q_omega=5e-3, q_d=1e-4, r_theta=2e-3),
        costs=MpcCostConfig(
            q_theta=2.0,
            l_theta=0.0,
            q_omega=0.8,
            q_dtheta=0.0,
            l_dtheta=0.0,
            r=0.05,
            s=0.1,
            l_du=0.0,
            terminal=None,
            rho=50.0,
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

    def test_predictor_enabled_omega_includes_camera_base_rate(self) -> None:
        control_cfg = _make_control_config()
        mpc_cfg = _make_mpc_config(predictor_enabled=True)
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
            target_velocity_px_s=(6.0, -4.0),
        )

        yaw_refs = refs["yaw"]
        self.assertIsNotNone(yaw_refs.omega)
        assert yaw_refs.omega is not None
        target_rate = control_cfg.yaw_sign * 6.0 / control_cfg.fx_px
        expected_rate = target_rate + (cam_state.pan_rate or 0.0)
        self.assertTrue(all(math.isclose(val, expected_rate, rel_tol=1e-9) for val in yaw_refs.omega))

    def test_predictor_references_are_bounded_after_tiny_dt_jump(self) -> None:
        mpc_cfg = _make_mpc_config(
            predictor_enabled=True,
            predictor_alpha=0.05,
            predictor_beta=1.0,
        )
        control_cfg = _make_control_config(rate_limit=0.5, mpc=mpc_cfg)
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)
        cam_state = CamState(
            frame_id=1,
            src_ts_ms=0,
            pan=0.0,
            tilt=0.0,
            pan_rate=0.0,
            tilt_rate=0.0,
        )

        builder.build(
            target_uv=(640.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=1.0,
            cam_state=cam_state,
            target_velocity_px_s=(0.0, 0.0),
        )
        refs = builder.build(
            target_uv=(1280.0, 0.0),
            aim_uv=(640.0, 360.0),
            timestamp=1.0 + 1e-6,
            cam_state=cam_state,
            target_velocity_px_s=(0.0, 0.0),
        )

        for axis_refs in refs.values():
            self.assertTrue(all(math.isfinite(value) for value in axis_refs.theta))
            assert axis_refs.omega is not None
            self.assertTrue(all(math.isfinite(value) for value in axis_refs.omega))
            self.assertLessEqual(max(abs(value) for value in axis_refs.omega), 12.0)
            self.assertLessEqual(max(abs(value) for value in axis_refs.theta), math.pi)

    def test_reset_predictor_discards_previous_target_motion(self) -> None:
        mpc_cfg = _make_mpc_config(
            predictor_enabled=True,
            predictor_alpha=0.05,
            predictor_beta=1.0,
        )
        control_cfg = _make_control_config(rate_limit=0.5, mpc=mpc_cfg)
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)
        cam_state = CamState(
            frame_id=1,
            src_ts_ms=0,
            pan=0.0,
            tilt=0.0,
            pan_rate=0.0,
            tilt_rate=0.0,
        )

        builder.build(
            target_uv=(640.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=1.0,
            cam_state=cam_state,
            target_velocity_px_s=(0.0, 0.0),
        )
        builder.reset_predictor()
        refs = builder.build(
            target_uv=(1280.0, 0.0),
            aim_uv=(640.0, 360.0),
            timestamp=1.0 + 1e-6,
            cam_state=cam_state,
            target_velocity_px_s=(0.0, 0.0),
        )

        yaw_omega = refs["yaw"].omega
        pitch_omega = refs["pitch"].omega
        assert yaw_omega is not None
        assert pitch_omega is not None
        self.assertAlmostEqual(yaw_omega[0], 0.0)
        self.assertAlmostEqual(pitch_omega[0], 0.0)

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

    def test_effect_delay_shifts_theta_reference(self) -> None:
        control_cfg = _make_control_config()
        effect_delay_s = 0.2
        mpc_cfg = _make_mpc_config(effect_delay_s=effect_delay_s)
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)
        cam_state = CamState(
            frame_id=1,
            src_ts_ms=0,
            pan=0.1,
            tilt=0.0,
            pan_rate=0.0,
            tilt_rate=0.0,
        )

        refs = builder.build(
            target_uv=(660.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=1.0,
            cam_state=cam_state,
            target_velocity_px_s=(8.0, 0.0),
        )

        yaw_refs = refs["yaw"]
        px_err = pixel_delta(660.0, 360.0, 640.0, 360.0, control_cfg, apply_deadband=True)
        err_rad = angular_error_from_pixel_delta(px_err, control_cfg)
        target_rate = control_cfg.yaw_sign * 8.0 / control_cfg.fx_px
        Ts = mpc_cfg.horizon.sample_time_s

        expected_theta = tuple(
            cam_state.pan + err_rad.yaw + target_rate * (effect_delay_s + Ts * i)
            for i in range(mpc_cfg.horizon.prediction_horizon)
        )
        self.assertSequenceEqual(
            tuple(round(x, 6) for x in yaw_refs.theta),
            tuple(round(x, 6) for x in expected_theta),
        )

    def test_alpha_beta_predictor_estimates_rate_from_measurements(self) -> None:
        control_cfg = _make_control_config()
        mpc_cfg = _make_mpc_config(
            predictor_enabled=True,
            predictor_alpha=0.85,
            predictor_beta=0.05,
        )
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)

        builder.build(
            target_uv=(640.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=0.0,
            theta_estimates={"yaw": 0.0, "pitch": 0.0},
            target_velocity_px_s=None,
        )
        refs = builder.build(
            target_uv=(648.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=0.1,
            theta_estimates={"yaw": 0.0, "pitch": 0.0},
            target_velocity_px_s=None,
        )

        yaw_refs = refs["yaw"]
        assert yaw_refs.omega is not None
        self.assertGreater(yaw_refs.omega[0], 0.0)
        self.assertGreater(yaw_refs.theta[1], yaw_refs.theta[0])

    def test_target_predictions_expose_filtered_predictor_state(self) -> None:
        control_cfg = _make_control_config()
        mpc_cfg = _make_mpc_config(
            predictor_enabled=True,
            predictor_alpha=1.0,
            predictor_beta=0.0,
        )
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)

        builder.build(
            target_uv=(640.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=0.0,
            theta_estimates={"yaw": 0.0, "pitch": 0.0},
            target_velocity_px_s=None,
        )
        predictions = builder.preview_target_predictions(
            target_uv=(660.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=0.1,
            theta_estimates={"yaw": 0.0, "pitch": 0.0},
            target_velocity_px_s=None,
        )

        yaw_prediction = predictions["yaw"]
        expected_theta = math.atan(20.0 / control_cfg.fx_px)
        self.assertAlmostEqual(yaw_prediction.theta_base, 0.0)
        self.assertAlmostEqual(yaw_prediction.theta, expected_theta)
        self.assertAlmostEqual(yaw_prediction.omega, 0.0)
        self.assertGreater(yaw_prediction.residual, 0.0)
        self.assertAlmostEqual(builder._predictor_state["yaw"].timestamp, 0.0)

    def test_adaptive_effect_delay_increases_when_target_runs_ahead(self) -> None:
        control_cfg = _make_control_config()
        base_delay = 0.02
        mpc_cfg = _make_mpc_config(
            effect_delay_s=base_delay,
            adaptive_effect_delay_enabled=True,
            adaptive_effect_delay_min_s=0.0,
            adaptive_effect_delay_max_s=0.2,
            adaptive_effect_delay_alpha=1.0,
            adaptive_effect_delay_gain=0.1,
            adaptive_effect_delay_rate_eps=1e-6,
        )
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)

        builder.build(
            target_uv=(640.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=0.0,
            theta_estimates={"yaw": 0.0, "pitch": 0.0},
            target_velocity_px_s=(8.0, 0.0),
        )
        builder.build(
            target_uv=(700.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=0.1,
            theta_estimates={"yaw": 0.0, "pitch": 0.0},
            target_velocity_px_s=(8.0, 0.0),
        )

        self.assertGreater(builder.effect_delay_for_axis("yaw"), base_delay)

    def test_adaptive_effect_delay_clamps_to_max(self) -> None:
        control_cfg = _make_control_config()
        max_delay = 0.08
        mpc_cfg = _make_mpc_config(
            effect_delay_s=0.01,
            adaptive_effect_delay_enabled=True,
            adaptive_effect_delay_min_s=0.0,
            adaptive_effect_delay_max_s=max_delay,
            adaptive_effect_delay_alpha=1.0,
            adaptive_effect_delay_gain=10.0,
            adaptive_effect_delay_rate_eps=1e-6,
        )
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)

        builder.build(
            target_uv=(640.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=0.0,
            theta_estimates={"yaw": 0.0, "pitch": 0.0},
            target_velocity_px_s=(0.5, 0.0),
        )
        builder.build(
            target_uv=(900.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=0.1,
            theta_estimates={"yaw": 0.0, "pitch": 0.0},
            target_velocity_px_s=(0.5, 0.0),
        )

        self.assertAlmostEqual(builder.effect_delay_for_axis("yaw"), max_delay, places=9)

    def test_time_to_impact_delay_uses_distance_over_speed(self) -> None:
        control_cfg = _make_control_config()
        speed = 100.0
        mpc_cfg = _make_mpc_config(
            effect_delay_mode="time_to_impact",
            projectile_speed_m_s=speed,
            impact_delay_bias_s=0.0,
            predictor_enabled=False,
            adaptive_effect_delay_enabled=False,
        )
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)

        refs = builder.build(
            target_uv=(660.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=1.0,
            cam_state=CamState(
                frame_id=1,
                src_ts_ms=0,
                pan=0.1,
                tilt=0.0,
                pan_rate=0.0,
                tilt_rate=0.0,
            ),
            distance_m=50.0,
            target_velocity_px_s=(8.0, 0.0),
        )

        yaw_refs = refs["yaw"]
        px_err = pixel_delta(660.0, 360.0, 640.0, 360.0, control_cfg, apply_deadband=True)
        err_rad = angular_error_from_pixel_delta(px_err, control_cfg)
        target_rate = control_cfg.yaw_sign * 8.0 / control_cfg.fx_px
        expected_delay = 50.0 / speed
        expected_first = 0.1 + err_rad.yaw + target_rate * expected_delay
        self.assertAlmostEqual(yaw_refs.theta[0], expected_first, places=6)

    def test_time_to_impact_delay_applies_bias(self) -> None:
        control_cfg = _make_control_config()
        speed = 200.0
        bias = 0.03
        mpc_cfg = _make_mpc_config(
            effect_delay_mode="time_to_impact",
            projectile_speed_m_s=speed,
            impact_delay_bias_s=bias,
            predictor_enabled=False,
            adaptive_effect_delay_enabled=False,
        )
        builder = MpcReferenceBuilder(control_cfg, mpc_cfg.horizon)

        refs = builder.build(
            target_uv=(660.0, 360.0),
            aim_uv=(640.0, 360.0),
            timestamp=1.0,
            cam_state=CamState(
                frame_id=1,
                src_ts_ms=0,
                pan=0.1,
                tilt=0.0,
                pan_rate=0.0,
                tilt_rate=0.0,
            ),
            distance_m=20.0,
            target_velocity_px_s=(8.0, 0.0),
        )

        yaw_refs = refs["yaw"]
        px_err = pixel_delta(660.0, 360.0, 640.0, 360.0, control_cfg, apply_deadband=True)
        err_rad = angular_error_from_pixel_delta(px_err, control_cfg)
        target_rate = control_cfg.yaw_sign * 8.0 / control_cfg.fx_px
        expected_delay = 20.0 / speed + bias
        expected_first = 0.1 + err_rad.yaw + target_rate * expected_delay
        self.assertAlmostEqual(yaw_refs.theta[0], expected_first, places=6)


if __name__ == "__main__":
    unittest.main()
