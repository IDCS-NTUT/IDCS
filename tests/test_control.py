import math
import unittest
from types import SimpleNamespace
from typing import Optional, Sequence
from unittest.mock import patch

from common.control import (
    AxisPair,
    ControlConfig,
    ControlDebugOverlayConfig,
    LaserAimingControlConfig,
    LaserMountConfig,
    MpcConfig,
    MpcConstraintConfig,
    MpcCostConfig,
    MpcEstimatorConfig,
    MpcHorizonConfig,
    MpcPlantConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.schemas import Box, CamState, DetectionMsg
from jetson.controller import ControlLoop


class _DummyPub:
    def __init__(self) -> None:
        self.sent = []

    def send_string(self, payload, flags=0):  # pragma: no cover - unused in tests
        self.sent.append((payload, flags))


class _StubMpcAxis:
    def __init__(self, axis: str, command: float) -> None:
        self.axis = axis
        self.command = command
        self.state = [0.0, 0.0, 0.0]
        self.last_refs: Optional[tuple] = None
        self.calls = []

    def reset(self) -> None:  # pragma: no cover - unused
        self.state = [0.0, 0.0, 0.0]

    def step_estimator(
        self, u_applied: float, theta_measurement: Optional[float]
    ) -> Sequence[float]:
        self.calls.append(("est", u_applied, theta_measurement))
        theta = 0.0 if theta_measurement is None else float(theta_measurement)
        self.state = [theta, 0.0, 0.0]
        return list(self.state)

    def compute_control(self, theta_ref_seq, omega_ref_seq=None, **kwargs):
        self.calls.append(("ctrl", theta_ref_seq, omega_ref_seq, kwargs))
        self.last_refs = tuple(theta_ref_seq)
        diag = SimpleNamespace(
            status="optimal",
            cost=abs(float(self.command)),
            u_sequence=[float(self.command)],
            theta_pred=list(theta_ref_seq),
            omega_pred=list(omega_ref_seq or []),
            weights=[1.0 for _ in theta_ref_seq],
            solver_info={"iter": 1.0},
            slack={"theta_min": 0.0},
            cost_terms={"theta": 0.1, "omega": 0.0},
        )
        return self.command, diag


def _make_mpc_config_for_tests() -> MpcConfig:
    return MpcConfig(
        horizon=MpcHorizonConfig(
            prediction_horizon=3,
            control_horizon=2,
            sample_time_s=0.05,
            gamma=0.95,
            move_blocking=True,
        ),
        plant=MpcPlantConfig(a_u=1.0, a_f=0.2),
        estimator=MpcEstimatorConfig(q_theta=1e-3, q_omega=1e-3, q_d=1e-4, r_theta=1e-3),
        costs=MpcCostConfig(
            q_theta=1.0,
            l_theta=0.0,
            q_omega=0.5,
            q_dtheta=0.0,
            l_dtheta=0.0,
            r=0.05,
            s=0.05,
            l_du=0.0,
            terminal=None,
            rho=10.0,
        ),
        constraints=MpcConstraintConfig(
            u_min=-1.0,
            u_max=1.0,
            du_max=0.5,
            theta_min=None,
            theta_max=None,
            omega_min=None,
            omega_max=None,
        ),
    )


class DebugOverlayParsingTests(unittest.TestCase):
    def _base_raw_config(self) -> dict:
        return {
            "control": {
                "mode": "rate",
                "controller": "pid",
                "fx_px": 800.0,
                "fy_px": 820.0,
                "kp": {"yaw": 0.0, "pitch": 0.0},
                "kd": {"yaw": 0.0, "pitch": 0.0},
                "rate_limits": {"yaw": 1.0, "pitch": 1.0},
                "accel_limits": {"yaw": 1.0, "pitch": 1.0},
                "sign_convention": {"yaw_positive": "right", "pitch_positive": "up"},
                "laser": {
                    "tolerance_px": 3.0,
                    "use_range": "known_size",
                    "default_distance_m": 25.0,
                },
            }
        }

    def test_overlay_defaults_disabled(self) -> None:
        cfg = self._base_raw_config()
        config = ControlConfig.from_raw_config(cfg, (1280, 720))
        self.assertFalse(config.debug_overlay.enabled)
        self.assertEqual(
            config.debug_overlay.show_terms,
            ControlDebugOverlayConfig.DEFAULT_TERMS,
        )

    def test_overlay_customization(self) -> None:
        cfg = self._base_raw_config()
        cfg["control"]["debug_overlay"] = {
            "enabled": True,
            "history_window_s": 2.5,
            "opacity": 0.75,
            "bar_height_px": 60,
            "show_terms": ["theta", "omega", "effort"],
        }
        config = ControlConfig.from_raw_config(cfg, (1280, 720))
        self.assertTrue(config.debug_overlay.enabled)
        self.assertAlmostEqual(config.debug_overlay.history_window_s, 2.5)
        self.assertEqual(config.debug_overlay.bar_height_px, 60)
        self.assertEqual(config.debug_overlay.opacity, 0.75)
        self.assertEqual(config.debug_overlay.show_terms, ("theta", "omega", "effort"))


class PixelDeltaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ControlConfig(
            mode="rate",
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
        self.assertEqual(cmd.controller_mode, "pid")
        self.assertIsNone(cmd.mpc)

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


class MpcControlLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mpc_cfg = _make_mpc_config_for_tests()
        self.config = ControlConfig(
            mode="rate",
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
            controller="mpc",
            mpc=self.mpc_cfg,
        )
        self.pub = _DummyPub()
        self.axes = {}
        self.loop = ControlLoop(
            self.config,
            self.pub,
            mpc_axis_factory=self._axis_factory,
        )

    def _axis_factory(self, axis: str, *_args):
        command = 0.12 if axis == "yaw" else -0.07
        stub = _StubMpcAxis(axis, command)
        self.axes[axis] = stub
        return stub

    def _make_detection(
        self,
        u_px: float,
        v_px: float,
        *,
        frame_id: int,
        src_ts_ms: int,
        rx_ts_ms: int,
        infer_ts_ms: int,
    ) -> DetectionMsg:
        img_w, img_h = self.config.frame_size
        box_w = 80.0
        box_h = 90.0
        return DetectionMsg(
            frame_id=frame_id,
            src_ts_ms=src_ts_ms,
            rx_ts_ms=rx_ts_ms,
            infer_ts_ms=infer_ts_ms,
            img_w=img_w,
            img_h=img_h,
            boxes=[
                Box(
                    x=(u_px - box_w / 2.0) / img_w,
                    y=(v_px - box_h / 2.0) / img_h,
                    w=box_w / img_w,
                    h=box_h / img_h,
                    conf=0.9,
                    cls="0",
                )
            ],
        )

    def test_mpc_tracking_dispatches_to_axes(self) -> None:
        detection = self._make_detection(
            660.0,
            360.0,
            frame_id=42,
            src_ts_ms=100,
            rx_ts_ms=110,
            infer_ts_ms=120,
        )
        with patch("jetson.controller.time.monotonic", return_value=1.0):
            self.loop.update_detection(detection)
        self.loop.update_cam_state(
            CamState(
                frame_id=0,
                src_ts_ms=0,
                pan=0.02,
                tilt=-0.01,
                pan_rate=0.0,
                tilt_rate=0.0,
            )
        )
        with patch.object(self.loop, "_send_cmd") as send_mock:
            self.loop.tick(now=1.03)

        send_mock.assert_called_once()
        cmd = send_mock.call_args[0][0]
        self.assertTrue(cmd.target_ok)
        self.assertAlmostEqual(cmd.pan_rate_cmd, self.axes["yaw"].command, places=6)
        self.assertAlmostEqual(cmd.tilt_rate_cmd, self.axes["pitch"].command, places=6)
        self.assertEqual(cmd.controller_mode, "mpc")
        self.assertIsNotNone(cmd.mpc)
        diag = cmd.mpc.get("yaw") if cmd.mpc is not None else None
        self.assertIsNotNone(diag)
        if diag is not None:
            self.assertEqual(diag.status, "optimal")
            self.assertAlmostEqual(diag.u0, self.axes["yaw"].command)
            self.assertIsNotNone(diag.terms)
            assert diag.terms is not None
            self.assertIn("theta", diag.terms)
        yaw_refs = self.axes["yaw"].last_refs
        self.assertIsNotNone(yaw_refs)
        if yaw_refs is not None:
            self.assertEqual(
                len(yaw_refs), self.mpc_cfg.horizon.prediction_horizon
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
