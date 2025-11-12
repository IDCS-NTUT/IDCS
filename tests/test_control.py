import math
import unittest
from typing import Any, Mapping, Optional
import json
from unittest.mock import patch

from common.control import (
    AxisPair,
    ControlConfig,
    LaserAimingControlConfig,
    LaserMountConfig,
    MPCActuatorLimits,
    MPCAdaptiveWeightsConfig,
    MPCCostConfig,
    MPCConfig,
    MPCEstimatorConfig,
    MPCHorizonConfig,
    MPCPlantConfig,
    MPCStateConstraints,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.schemas import Box, CamState, DetectionMsg
from jetson.controller import ControlLoop, MPCControlLoop, PIDControlLoop


class ControllerExportsTests(unittest.TestCase):
    def test_control_loop_aliases_pid(self) -> None:
        self.assertIs(ControlLoop, PIDControlLoop)


class ControlConfigParsingTests(unittest.TestCase):
    def _base_config(self) -> Mapping[str, Any]:
        return {
            "control": {
                "mode": "rate",
                "loop_hz": 50,
                "fx_px": 750.0,
                "fy_px": 760.0,
                "kp": {"yaw": 1.0, "pitch": 1.0},
                "kd": {"yaw": 0.1, "pitch": 0.1},
                "ki": {"yaw": 0.0, "pitch": 0.0},
                "rate_limits": {"yaw": 6.0, "pitch": 6.0},
                "accel_limits": {"yaw": 15.0, "pitch": 15.0},
            }
        }

    def test_defaults_to_pid_without_mpc_section(self) -> None:
        raw = self._base_config()
        config = ControlConfig.from_raw_config(raw, (1280, 720))
        self.assertEqual(config.controller_type, "pid")
        self.assertIsNone(config.mpc)

    def test_parses_mpc_configuration(self) -> None:
        raw = self._base_config()
        raw["control"].update(
            {
                "controller_type": "mpc",
                "mpc": {
                    "horizon": {"steps": 12, "dt_s": 0.04, "control_steps": 6},
                    "cost": {
                        "input": {"yaw": 0.2, "pitch": 0.3},
                        "delta": {"yaw": 0.1, "pitch": 0.1},
                    },
                    "actuator_limits": {
                        "rate": {"yaw": 5.0, "pitch": 4.5},
                        "accel": {"yaw": 12.0, "pitch": 10.0},
                    },
                    "state_constraints": {
                        "error": {"yaw": 0.35, "pitch": 0.4},
                        "rate": {"yaw": 3.0, "pitch": 3.5},
                    },
                    "plant": {
                        "a_u": {"yaw": 1.2, "pitch": 1.1},
                        "a_f": {"yaw": 1.8, "pitch": 2.1},
                    },
                    "estimator": {
                        "q_theta": {"yaw": 1.0e-3, "pitch": 1.5e-3},
                        "q_omega": {"yaw": 1.0e-2, "pitch": 2.0e-2},
                        "q_disturbance": {"yaw": 5.0e-4, "pitch": 7.0e-4},
                        "r_theta": {"yaw": 2.5e-3, "pitch": 3.0e-3},
                    },
                    "adaptive_weights": {
                        "q_theta_base": {"yaw": 1.6, "pitch": 1.7},
                        "q_omega_base": {"yaw": 0.35, "pitch": 0.4},
                        "alpha_distance": 1.5,
                        "alpha_lateral_velocity": 0.75,
                        "alpha_time": 0.2,
                        "exponent": 1.2,
                        "epsilon": 1.0e-5,
                        "w_min": 0.2,
                        "w_max": 4.0,
                    },
                },
            }
        )

        config = ControlConfig.from_raw_config(raw, (1280, 720))

        self.assertEqual(config.controller_type, "mpc")
        self.assertIsNotNone(config.mpc)
        assert config.mpc is not None
        self.assertEqual(config.mpc.horizon.steps, 12)
        self.assertAlmostEqual(config.mpc.horizon.step_dt_s, 0.04)
        self.assertEqual(config.mpc.horizon.control_horizon_steps, 6)
        self.assertAlmostEqual(config.mpc.cost.input.yaw, 0.2)
        self.assertAlmostEqual(config.mpc.cost.delta.pitch, 0.1)
        self.assertAlmostEqual(
            config.mpc.actuator_limits.rate.pitch,
            4.5,
        )
        self.assertIsNotNone(config.mpc.state_constraints.accel)
        assert config.mpc.state_constraints.accel is not None
        self.assertAlmostEqual(config.mpc.state_constraints.accel.yaw, 12.0)
        self.assertIsNotNone(config.mpc.state_constraints.error)
        self.assertAlmostEqual(config.mpc.plant.a_u.yaw, 1.2)
        self.assertAlmostEqual(config.mpc.plant.a_f.pitch, 2.1)
        self.assertAlmostEqual(config.mpc.estimator.q_theta.pitch, 1.5e-3)
        self.assertAlmostEqual(config.mpc.estimator.q_disturbance.yaw, 5.0e-4)
        self.assertAlmostEqual(config.mpc.estimator.r_theta.pitch, 3.0e-3)
        self.assertAlmostEqual(config.mpc.adaptive.q_theta_base.yaw, 1.6)
        self.assertAlmostEqual(config.mpc.adaptive.q_omega_base.pitch, 0.4)
        self.assertAlmostEqual(config.mpc.adaptive.alpha_distance, 1.5)
        self.assertAlmostEqual(config.mpc.adaptive.alpha_time, 0.2)
        self.assertAlmostEqual(config.mpc.adaptive.exponent, 1.2)
        self.assertAlmostEqual(config.mpc.adaptive.epsilon, 1.0e-5)
        self.assertAlmostEqual(config.mpc.adaptive.w_min, 0.2)
        self.assertAlmostEqual(config.mpc.adaptive.w_max, 4.0)

    def test_requires_mpc_section_when_selected(self) -> None:
        raw = self._base_config()
        raw["control"]["controller_type"] = "mpc"

        with self.assertRaises(ValueError):
            ControlConfig.from_raw_config(raw, (640, 480))


class _DummyPub:
    def __init__(self) -> None:
        self.sent = []

    def send_string(self, payload, flags=0):  # pragma: no cover - unused in tests
        self.sent.append((payload, flags))


class PixelDeltaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ControlConfig(
            mode="rate",
            controller_type="pid",
            loop_hz=30.0,
            fx_px=800.0,
            fy_px=820.0,
            cx_px=640.0,
            cy_px=360.0,
            aim_mode="camera_center",
            kp=AxisPair(0.0, 0.0),
            kd=AxisPair(0.0, 0.0),
            ki=AxisPair(0.0, 0.0),
            rate_limits=AxisPair(1.0, 1.0),
            accel_limits=AxisPair(1.0, 1.0),
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
            mpc=None,
        )

    def test_delta_matches_expected_signs(self) -> None:
        delta = pixel_delta(660.0, 340.0, 640.0, 360.0, self.config, apply_deadband=False)
        self.assertAlmostEqual(delta.yaw, 20.0)
        # pitch_sign = -1 so moving upwards (v decreases) yields positive error
        self.assertAlmostEqual(delta.pitch, 20.0)

    def test_delta_with_non_centre_reference(self) -> None:
        delta = pixel_delta(640.0, 360.0, 600.0, 330.0, self.config, apply_deadband=False)
        self.assertAlmostEqual(delta.yaw, 40.0)
        self.assertAlmostEqual(delta.pitch, -30.0)

    def test_angular_error_from_delta(self) -> None:
        px_err = AxisPair(yaw=40.0, pitch=-20.0)
        ang_err = angular_error_from_pixel_delta(px_err, self.config)
        self.assertAlmostEqual(ang_err.yaw, math.atan(40.0 / 800.0))
        self.assertAlmostEqual(ang_err.pitch, math.atan(-20.0 / 820.0))

    def test_linearized_conversion(self) -> None:
        px_err = AxisPair(yaw=-16.0, pitch=8.0)
        ang_err = angular_error_from_pixel_delta(px_err, self.config, linearize=True)
        self.assertAlmostEqual(ang_err.yaw, -16.0 / 800.0)
        self.assertAlmostEqual(ang_err.pitch, 8.0 / 820.0)


class LaserRangePolicyTests(unittest.TestCase):
    def _make_config(self, *, use_range: str = "known_size") -> ControlConfig:
        return ControlConfig(
            mode="rate",
            controller_type="pid",
            loop_hz=30.0,
            fx_px=800.0,
            fy_px=820.0,
            cx_px=640.0,
            cy_px=360.0,
            aim_mode="laser_point",
            kp=AxisPair(0.0, 0.0),
            kd=AxisPair(0.0, 0.0),
            ki=AxisPair(0.0, 0.0),
            rate_limits=AxisPair(1.0, 1.0),
            accel_limits=AxisPair(1.0, 1.0),
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
                use_range=use_range,
                default_distance_m=25.0,
            ),
            mpc=None,
        )

    def _make_loop(self, *, use_range: str = "known_size", distance_alpha: float = 0.5) -> ControlLoop:
        mount = LaserMountConfig.from_raw_config(
            {
                "laser": {
                    "offset_m": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "dir_cam": {"x": 0.0, "y": 0.0, "z": 1.0},
                }
            }
        )
        return ControlLoop(
            self._make_config(use_range=use_range),
            _DummyPub(),
            laser_mount=mount,
            distance_alpha=distance_alpha,
        )

    def test_known_size_prefers_measurement(self) -> None:
        loop = self._make_loop(use_range="known_size", distance_alpha=0.4)

        distance, source, active = loop._resolve_laser_range(8.0)

        self.assertAlmostEqual(distance, 8.0)
        self.assertEqual(source, "known_size")
        self.assertTrue(active)

    def test_fallback_blends_to_default_distance(self) -> None:
        loop = self._make_loop(use_range="known_size", distance_alpha=0.5)

        loop._resolve_laser_range(10.0)
        distance, source, active = loop._resolve_laser_range(None)

        self.assertAlmostEqual(distance, 0.5 * 25.0 + 0.5 * 10.0)
        self.assertEqual(source, "default")
        self.assertTrue(active)

    def test_infinite_policy_ignores_measurement(self) -> None:
        loop = self._make_loop(use_range="infinite", distance_alpha=0.6)

        distance, source, active = loop._resolve_laser_range(6.0)

        self.assertAlmostEqual(distance, 25.0)
        self.assertEqual(source, "infinite")
        self.assertTrue(active)

    def test_ground_plane_policy_falls_back(self) -> None:
        loop = self._make_loop(use_range="ground_plane", distance_alpha=0.3)

        loop._resolve_laser_range(9.0)
        distance, source, active = loop._resolve_laser_range(None)

        self.assertAlmostEqual(distance, 0.3 * 25.0 + 0.7 * 9.0)
        self.assertEqual(source, "default")
        self.assertTrue(active)


class LaserMountConfigTests(unittest.TestCase):
    def test_config_interprets_y_as_up(self) -> None:
        mount = LaserMountConfig.from_raw_config(
            {
                "laser": {
                    "offset_m": {"x": 0.12, "y": 0.34, "z": 0.56},
                    "dir_cam": {"x": 0.0, "y": 1.0, "z": 1.0},
                }
            }
        )

        self.assertAlmostEqual(mount.offset_m.x, 0.12)
        self.assertAlmostEqual(mount.offset_m.y, -0.34)
        self.assertAlmostEqual(mount.offset_m.z, 0.56)

        self.assertAlmostEqual(mount.dir_cam.x, 0.0)
        # +Y up in config becomes -Y in the internal CV frame
        self.assertLess(mount.dir_cam.y, 0.0)
        self.assertGreater(mount.dir_cam.z, 0.0)
        norm = math.sqrt(mount.dir_cam.x ** 2 + mount.dir_cam.y ** 2 + mount.dir_cam.z ** 2)
        self.assertAlmostEqual(norm, 1.0)


class TargetLeadEstimationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ControlConfig(
            mode="rate",
            controller_type="pid",
            loop_hz=30.0,
            fx_px=800.0,
            fy_px=820.0,
            cx_px=640.0,
            cy_px=360.0,
            aim_mode="camera_center",
            kp=AxisPair(0.0, 0.0),
            kd=AxisPair(0.0, 0.0),
            ki=AxisPair(0.0, 0.0),
            rate_limits=AxisPair(1.0, 1.0),
            accel_limits=AxisPair(1.0, 1.0),
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
            mpc=None,
        )
        self.loop = ControlLoop(self.config, _DummyPub())

    def _make_detection(
        self,
        u_px: float,
        v_px: float,
        frame_id: int,
        *,
        src_ts_ms: Optional[int] = None,
        rx_ts_ms: Optional[int] = None,
        infer_ts_ms: Optional[int] = None,
    ) -> DetectionMsg:
        box_width = 40.0
        box_height = 30.0
        img_w, img_h = self.config.frame_size
        src = 0 if src_ts_ms is None else int(src_ts_ms)
        rx = src if rx_ts_ms is None else int(rx_ts_ms)
        infer = rx if infer_ts_ms is None else int(infer_ts_ms)
        return DetectionMsg(
            frame_id=frame_id,
            src_ts_ms=src,
            rx_ts_ms=rx,
            infer_ts_ms=infer,
            img_w=img_w,
            img_h=img_h,
            boxes=[
                Box(
                    x=(u_px - box_width / 2.0) / img_w,
                    y=(v_px - box_height / 2.0) / img_h,
                    w=box_width / img_w,
                    h=box_height / img_h,
                    conf=0.9,
                    cls="0",
                )
            ],
        )

    def test_velocity_compensates_camera_rotation(self) -> None:
        self.loop.update_cam_state(
            CamState(
                frame_id=0,
                src_ts_ms=0,
                pan=0.0,
                tilt=0.0,
                pan_rate=0.5,
                tilt_rate=0.0,
            )
        )
        first = self._make_detection(
            640.0,
            360.0,
            frame_id=1,
            src_ts_ms=960,
            rx_ts_ms=990,
            infer_ts_ms=998,
        )
        second = self._make_detection(
            600.0,
            360.0,
            frame_id=2,
            src_ts_ms=1060,
            rx_ts_ms=1090,
            infer_ts_ms=1098,
        )
        with patch("jetson.controller.time.monotonic", side_effect=[1.0, 1.1]):
            self.loop.update_detection(first)
            self.loop.update_detection(second)

        self.assertIsNotNone(second.target_velocity_px_s)
        vx, vy = second.target_velocity_px_s
        self.assertLess(abs(vx), 1.0)
        self.assertLess(abs(vy), 1.0)
        self.assertIsNotNone(second.target_lead_uv)
        self.assertLess(abs(second.target_lead_uv[0] - 600.0), 0.1)

    def test_lead_advances_toward_predicted_position(self) -> None:
        first = self._make_detection(
            640.0,
            360.0,
            frame_id=3,
            src_ts_ms=4960,
            rx_ts_ms=4990,
            infer_ts_ms=4998,
        )
        second = self._make_detection(
            660.0,
            360.0,
            frame_id=4,
            src_ts_ms=5055,
            rx_ts_ms=5085,
            infer_ts_ms=5095,
        )
        with patch("jetson.controller.time.monotonic", side_effect=[5.0, 5.1]):
            self.loop.update_detection(first)
            self.loop.update_detection(second)

        self.assertIsNotNone(second.target_velocity_px_s)
        vx, vy = second.target_velocity_px_s
        self.assertGreater(vx, 0.0)
        self.assertAlmostEqual(vy, 0.0, places=3)

        self.assertIsNotNone(second.target_lead_uv)
        lead_u, lead_v = second.target_lead_uv
        self.assertGreater(lead_u, 660.0)
        self.assertAlmostEqual(lead_v, 360.0, places=3)



        lookahead = getattr(self.loop, "_lead_time_s")
        expected_lead_u = 660.0 + vx * lookahead
        self.assertAlmostEqual(lead_u, expected_lead_u, places=3)
        self.assertAlmostEqual(second.target_lead_time_s, lookahead)
        self.assertGreater(second.target_lead_time_s, 0.03)
        self.assertLess(second.target_lead_time_s, 0.06)

    def test_lead_prediction_does_not_influence_control(self) -> None:
        first = self._make_detection(
            640.0,
            360.0,
            frame_id=5,
            src_ts_ms=6960,
            rx_ts_ms=6990,
            infer_ts_ms=6995,
        )
        second = self._make_detection(
            660.0,
            360.0,
            frame_id=6,
            src_ts_ms=7055,
            rx_ts_ms=7085,
            infer_ts_ms=7095,
        )
        with patch("jetson.controller.time.monotonic", side_effect=[7.0, 7.1]):
            self.loop.update_detection(first)
            self.loop.update_detection(second)

        with patch.object(self.loop, "_send_cmd") as send_mock, patch(
            "jetson.controller.time.monotonic", return_value=7.133
        ):
            self.loop.tick()

        send_mock.assert_called_once()
        cmd = send_mock.call_args[0][0]
        # Control commands should continue to use the measured centroid rather than the
        # predicted lead location that is only emitted for telemetry/overlay purposes.
        self.assertAlmostEqual(cmd.target_uv[0], 660.0)
        self.assertAlmostEqual(cmd.target_uv[1], 360.0)
        self.assertAlmostEqual(cmd.err_uv[0], 20.0)
        self.assertAlmostEqual(cmd.err_uv[1], 0.0)

    def test_predictive_command_follows_last_velocity(self) -> None:
        first = self._make_detection(
            640.0,
            360.0,
            frame_id=7,
            src_ts_ms=8960,
            rx_ts_ms=8990,
            infer_ts_ms=8998,
        )
        second = self._make_detection(
            660.0,
            360.0,
            frame_id=8,
            src_ts_ms=9055,
            rx_ts_ms=9085,
            infer_ts_ms=9095,
        )
        loss = DetectionMsg(
            frame_id=9,
            src_ts_ms=9155,
            rx_ts_ms=9185,
            infer_ts_ms=9195,
            img_w=self.config.frame_size[0],
            img_h=self.config.frame_size[1],
            boxes=[],
        )

        with patch("jetson.controller.time.monotonic", side_effect=[9.0, 9.05, 9.06]):
            self.loop.update_detection(first)
            self.loop.update_detection(second)
            self.loop.update_detection(loss)

        self.assertTrue(loss.predictive_active)
        with patch.object(self.loop, "_send_cmd") as send_mock, patch(
            "jetson.controller.time.monotonic", return_value=9.07
        ):
            self.loop.tick()

        send_mock.assert_called_once()
        cmd = send_mock.call_args[0][0]
        self.assertFalse(cmd.target_ok)
        desired_rate = (math.atan((660.0 - 640.0) / self.config.fx_px) - 0.0) / 0.05
        accel_limited = self.config.accel_limits.yaw / self.config.loop_hz
        expected_rate = max(-accel_limited, min(desired_rate, accel_limited))
        self.assertAlmostEqual(cmd.pan_rate_cmd, expected_rate, places=6)
        self.assertAlmostEqual(cmd.tilt_rate_cmd, 0.0, places=6)

    def test_predictive_timeout_returns_home(self) -> None:
        first = self._make_detection(
            640.0,
            360.0,
            frame_id=10,
            src_ts_ms=9960,
            rx_ts_ms=9990,
            infer_ts_ms=9998,
        )
        second = self._make_detection(
            660.0,
            360.0,
            frame_id=11,
            src_ts_ms=10055,
            rx_ts_ms=10085,
            infer_ts_ms=10095,
        )
        loss = DetectionMsg(
            frame_id=12,
            src_ts_ms=10155,
            rx_ts_ms=10185,
            infer_ts_ms=10195,
            img_w=self.config.frame_size[0],
            img_h=self.config.frame_size[1],
            boxes=[],
        )

        with patch("jetson.controller.time.monotonic", side_effect=[10.0, 10.05, 10.06]):
            self.loop.update_detection(first)
            self.loop.update_detection(second)
            self.loop.update_detection(loss)

        with patch.object(self.loop, "_send_cmd") as predictive_mock, patch(
            "jetson.controller.time.monotonic", return_value=10.07
        ):
            self.loop.tick()

        with patch.object(self.loop, "_send_cmd") as hold_mock, patch(
            "jetson.controller.time.monotonic", return_value=10.25
        ):
            self.loop.tick()

        predictive_mock.assert_called_once()
        hold_mock.assert_called_once()
        cmd = hold_mock.call_args[0][0]
        self.assertFalse(cmd.target_ok)
        self.assertAlmostEqual(cmd.pan_rate_cmd, 0.0, places=6)
        self.assertAlmostEqual(cmd.tilt_rate_cmd, 0.0, places=6)

    def test_predictive_overlay_includes_predicted_box(self) -> None:
        first = self._make_detection(
            640.0,
            360.0,
            frame_id=13,
            src_ts_ms=11960,
            rx_ts_ms=11990,
            infer_ts_ms=11998,
        )
        second = self._make_detection(
            660.0,
            360.0,
            frame_id=14,
            src_ts_ms=12055,
            rx_ts_ms=12085,
            infer_ts_ms=12095,
        )
        loss = DetectionMsg(
            frame_id=15,
            src_ts_ms=12155,
            rx_ts_ms=12185,
            infer_ts_ms=12195,
            img_w=self.config.frame_size[0],
            img_h=self.config.frame_size[1],
            boxes=[],
        )

        with patch("jetson.controller.time.monotonic", side_effect=[12.0, 12.05, 12.1]):
            self.loop.update_detection(first)
            self.loop.update_detection(second)
            self.loop.update_detection(loss)

        self.assertTrue(loss.predictive_active)
        self.assertIsNotNone(loss.predictive_target_uv)
        predicted = loss.predictive_target_uv
        assert predicted is not None
        self.assertGreater(predicted[0], 660.0)
        self.assertAlmostEqual(predicted[1], 360.0, places=3)

        self.assertIsNotNone(loss.predictive_box_px)
        box = loss.predictive_box_px
        assert box is not None
        x1, y1, x2, y2 = box
        self.assertGreater(x2 - x1, 35.0)
        self.assertGreater(y2 - y1, 25.0)
        self.assertAlmostEqual((x1 + x2) / 2.0, predicted[0], places=1)
        self.assertAlmostEqual((y1 + y2) / 2.0, predicted[1], places=1)


class MPCControlLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mpc_config = MPCConfig(
            horizon=MPCHorizonConfig(steps=6, step_dt_s=0.05, control_horizon_steps=6),
            cost=MPCCostConfig(
                input=AxisPair(yaw=0.3, pitch=0.3),
                delta=AxisPair(yaw=0.1, pitch=0.1),
            ),
            actuator_limits=MPCActuatorLimits(
                rate=AxisPair(yaw=1.5, pitch=1.2),
                accel=AxisPair(yaw=3.0, pitch=2.5),
            ),
            state_constraints=MPCStateConstraints(error=None, rate=None, accel=None),
            plant=MPCPlantConfig(
                a_u=AxisPair(yaw=1.0, pitch=1.0),
                a_f=AxisPair(yaw=1.2, pitch=1.0),
            ),
            estimator=MPCEstimatorConfig(
                q_theta=AxisPair(yaw=1e-3, pitch=1e-3),
                q_omega=AxisPair(yaw=1e-2, pitch=1e-2),
                q_disturbance=AxisPair(yaw=1e-4, pitch=1e-4),
                r_theta=AxisPair(yaw=5e-3, pitch=5e-3),
            ),
            adaptive=MPCAdaptiveWeightsConfig(
                q_theta_base=AxisPair(yaw=1.2, pitch=1.2),
                q_omega_base=AxisPair(yaw=0.2, pitch=0.2),
                alpha_distance=1.0,
                alpha_lateral_velocity=0.6,
                alpha_time=0.3,
                exponent=1.0,
                epsilon=1e-6,
                w_min=0.2,
                w_max=5.0,
            ),
        )
        self.config = ControlConfig(
            mode="rate",
            controller_type="mpc",
            loop_hz=20.0,
            fx_px=780.0,
            fy_px=790.0,
            cx_px=640.0,
            cy_px=360.0,
            aim_mode="camera_center",
            kp=AxisPair(0.0, 0.0),
            kd=AxisPair(0.0, 0.0),
            ki=AxisPair(0.0, 0.0),
            rate_limits=AxisPair(2.0, 2.0),
            accel_limits=AxisPair(4.0, 4.0),
            deadband_px=0.0,
            smooth_px_alpha=0.0,
            lost_target_timeout_ms=200,
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
            mpc=self.mpc_config,
        )
        self.pub = _DummyPub()
        self.loop = MPCControlLoop(self.config, self.pub)

    def _make_detection(self, u_px: float, v_px: float, frame_id: int) -> DetectionMsg:
        img_w, img_h = self.config.frame_size
        box_w = 60.0
        box_h = 40.0
        return DetectionMsg(
            frame_id=frame_id,
            src_ts_ms=1000,
            rx_ts_ms=1005,
            infer_ts_ms=1010,
            img_w=img_w,
            img_h=img_h,
            boxes=[
                Box(
                    x=(u_px - box_w / 2.0) / img_w,
                    y=(v_px - box_h / 2.0) / img_h,
                    w=box_w / img_w,
                    h=box_h / img_h,
                    conf=0.95,
                    cls="0",
                )
            ],
        )

    def test_mpc_respects_rate_and_accel_limits(self) -> None:
        detection = self._make_detection(self.config.cx_px + 80.0, self.config.cy_px, frame_id=21)
        with patch("jetson.controller.time.monotonic", return_value=1.0):
            self.loop.update_detection(detection)

        self.loop.tick(now=1.05)
        self.assertEqual(len(self.pub.sent), 1)
        first_payload = json.loads(self.pub.sent[-1][0])
        yaw_rate_1 = first_payload["pan_rate_cmd"]
        self.assertGreater(yaw_rate_1, 0.0)
        self.assertLessEqual(abs(yaw_rate_1), self.mpc_config.actuator_limits.rate.yaw + 1e-6)
        self.assertAlmostEqual(first_payload["tilt_rate_cmd"], 0.0, places=4)

        self.loop.tick(now=1.1)
        self.assertEqual(len(self.pub.sent), 2)
        second_payload = json.loads(self.pub.sent[-1][0])
        yaw_rate_2 = second_payload["pan_rate_cmd"]
        dt = 1.1 - 1.05
        self.assertLessEqual(
            abs(yaw_rate_2 - yaw_rate_1),
            self.mpc_config.actuator_limits.accel.yaw * dt + 1e-6,
        )

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
