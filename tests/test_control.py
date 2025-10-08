import math
import unittest
from unittest.mock import patch

from common.control import (
    AxisPair,
    ControlConfig,
    LaserAimingControlConfig,
    LaserMountConfig,
    pixel_delta,
    angular_error_from_pixel_delta,
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
            loop_hz=20.0,
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

    def _make_loop(self) -> ControlLoop:
        return ControlLoop(self.config, _DummyPub())

    def _make_detection(self, frame_id: int, u_px: float, v_px: float) -> DetectionMsg:
        width, height = self.config.frame_size
        box_w = 0.1
        box_h = 0.1
        box = Box(
            x=(u_px / width) - (box_w / 2.0),
            y=(v_px / height) - (box_h / 2.0),
            w=box_w,
            h=box_h,
            cls="0",
            conf=0.9,
        )
        return DetectionMsg(
            frame_id=frame_id,
            src_ts_ms=frame_id * 10,
            rx_ts_ms=frame_id * 10,
            infer_ts_ms=frame_id * 10,
            img_w=width,
            img_h=height,
            boxes=[box],
        )

    def test_velocity_compensates_camera_rotation(self) -> None:
        loop = self._make_loop()
        loop.update_cam_state(
            CamState(frame_id=0, src_ts_ms=0, pan=0.0, tilt=0.0, pan_rate=0.0, tilt_rate=0.0)
        )

        first = self._make_detection(1, 640.0, 360.0)
        with patch("jetson.controller.time.monotonic", return_value=0.0):
            loop.update_detection(first)

        loop.update_cam_state(
            CamState(frame_id=2, src_ts_ms=50, pan=0.01, tilt=0.0, pan_rate=0.2, tilt_rate=0.0)
        )
        second = self._make_detection(2, 632.0, 360.0)
        with patch("jetson.controller.time.monotonic", return_value=0.05):
            loop.update_detection(second)

        self.assertIsNotNone(second.target_velocity_px_s)
        vx, vy = second.target_velocity_px_s
        self.assertAlmostEqual(vx, 0.0, places=3)
        self.assertAlmostEqual(vy, 0.0, places=3)
        self.assertIsNotNone(second.target_lead_uv)
        self.assertAlmostEqual(second.target_lead_uv[0], 624.0, places=1)

    def test_lead_overlay_projects_target_motion(self) -> None:
        loop = self._make_loop()
        loop.update_cam_state(
            CamState(frame_id=0, src_ts_ms=0, pan=0.0, tilt=0.0, pan_rate=0.0, tilt_rate=0.0)
        )

        first = self._make_detection(1, 640.0, 360.0)
        with patch("jetson.controller.time.monotonic", return_value=0.0):
            loop.update_detection(first)

        loop.update_cam_state(
            CamState(frame_id=2, src_ts_ms=50, pan=0.0, tilt=0.0, pan_rate=0.0, tilt_rate=0.0)
        )
        second = self._make_detection(2, 650.0, 360.0)
        with patch("jetson.controller.time.monotonic", return_value=0.05):
            loop.update_detection(second)

        self.assertIsNotNone(second.target_velocity_px_s)
        vx, vy = second.target_velocity_px_s
        self.assertAlmostEqual(vx, 200.0, places=1)
        self.assertAlmostEqual(vy, 0.0, places=3)
        self.assertIsNotNone(second.target_lead_uv)
        self.assertAlmostEqual(second.target_lead_uv[0], 660.0, places=1)
        self.assertAlmostEqual(second.target_lead_uv[1], 360.0, places=1)


if __name__ == "__main__":
    unittest.main()
