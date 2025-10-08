import math
import unittest
from unittest.mock import patch

from common.control import (
    AxisPair,
    ControlConfig,
    LaserAimingControlConfig,
    LaserMountConfig,
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

    def _make_detection(self, u_px: float, v_px: float, frame_id: int) -> DetectionMsg:
        box_width = 40.0
        box_height = 30.0
        img_w, img_h = self.config.frame_size
        return DetectionMsg(
            frame_id=frame_id,
            src_ts_ms=0,
            rx_ts_ms=0,
            infer_ts_ms=0,
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
        first = self._make_detection(640.0, 360.0, frame_id=1)
        second = self._make_detection(600.0, 360.0, frame_id=2)
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
        first = self._make_detection(640.0, 360.0, frame_id=3)
        second = self._make_detection(660.0, 360.0, frame_id=4)
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

    def test_lead_prediction_does_not_influence_control(self) -> None:
        first = self._make_detection(640.0, 360.0, frame_id=5)
        second = self._make_detection(660.0, 360.0, frame_id=6)
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
