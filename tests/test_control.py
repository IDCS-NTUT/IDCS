import math
import unittest
from typing import Optional

from common.control import (
    AxisPair,
    ControlConfig,
    LaserAimingControlConfig,
    LaserMountConfig,
    MotionModelConfig,
    MotionModelDerotationConfig,
    MotionModelNoiseConfig,
    pixel_delta,
    angular_error_from_pixel_delta,
)
from common.schemas import CamState
from jetson.controller import ControlLoop, _CameraFrameMotionModel


def _make_motion_model_config() -> MotionModelConfig:
    return MotionModelConfig(
        mode="camera_frame",
        latency_ms_source="auto",
        prediction_horizon_ms=None,
        derotation=MotionModelDerotationConfig(enabled=True, rate_scale=1.0),
        noise=MotionModelNoiseConfig(
            process_px=AxisPair(0.0, 0.0),
            measurement_px=AxisPair(0.0, 0.0),
            auto_inflate_with_rates=False,
            max_inflate_scale=3.0,
        ),
    )


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
            motion_model=_make_motion_model_config(),
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
            motion_model=_make_motion_model_config(),
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


class CameraFrameMotionModelTests(unittest.TestCase):
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
            motion_model=_make_motion_model_config(),
        )
        self.model = _CameraFrameMotionModel(self.config)

    def _cam_state(
        self,
        *,
        pan_rate: Optional[float] = None,
        tilt_rate: Optional[float] = None,
    ) -> CamState:
        return CamState(
            frame_id=0,
            src_ts_ms=0,
            pan=0.0,
            tilt=0.0,
            pan_rate=pan_rate,
            tilt_rate=tilt_rate,
        )

    def test_velocity_matches_measurement_without_pose(self) -> None:
        self.model.update_target((640.0, 360.0), 0.0)
        state = self.model.update_target((660.0, 360.0), 0.1)
        self.assertAlmostEqual(state.velocity_px_s[0], 200.0)
        self.assertAlmostEqual(state.velocity_px_s[1], 0.0)

    def test_derotation_cancels_camera_motion(self) -> None:
        self.model.update_cam_state(self._cam_state(pan_rate=0.5), 0.0)
        self.model.update_target((640.0, 360.0), 0.0)
        state = self.model.update_target((600.0, 360.0), 0.1)
        self.assertAlmostEqual(state.velocity_px_s[0], 0.0, places=6)

    def test_prediction_applies_camera_shift(self) -> None:
        self.model.update_cam_state(self._cam_state(pan_rate=0.5), 0.0)
        self.model.update_target((640.0, 360.0), 0.0)
        self.model.update_target((600.0, 360.0), 0.1)
        prediction = self.model.predict(now=0.1, horizon_s=0.1)
        assert prediction is not None
        self.assertAlmostEqual(prediction.uv[0], 560.0)
        self.assertAlmostEqual(prediction.velocity_px_s[0], 0.0)

    def test_prediction_combines_target_and_camera_motion(self) -> None:
        self.model.update_cam_state(self._cam_state(pan_rate=0.5), 0.0)
        self.model.update_target((640.0, 360.0), 0.0)
        self.model.update_target((620.0, 360.0), 0.1)
        prediction = self.model.predict(now=0.1, horizon_s=0.1)
        assert prediction is not None
        self.assertAlmostEqual(prediction.uv[0], 600.0)
        self.assertAlmostEqual(prediction.velocity_px_s[0], 200.0)

    def test_pitch_derotation_respects_sign_convention(self) -> None:
        tilt_rate = 0.3
        dt = 0.1
        expected_shift = -self.config.fy_px * tilt_rate * dt / self.config.pitch_sign
        self.model.update_cam_state(self._cam_state(tilt_rate=tilt_rate), 0.0)
        self.model.update_target((640.0, 360.0), 0.0)
        state = self.model.update_target((640.0, 360.0 + expected_shift), dt)
        self.assertAlmostEqual(state.velocity_px_s[1], 0.0, places=6)


class LatencyMeasurementTests(unittest.TestCase):
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
            motion_model=_make_motion_model_config(),
        )

    def test_latency_measurement_updates_state(self) -> None:
        loop = ControlLoop(self.config, _DummyPub())
        metrics = {"ema_ms": 42.0, "camera_to_infer_ms": 48.0}
        loop.update_latency_measurement(selected_ms=42.0, source="ema_ms", metrics=metrics)

        self.assertAlmostEqual(loop.latency_ms, 42.0)
        self.assertIn("camera_to_infer_ms", loop.latency_metrics)
        self.assertAlmostEqual(loop.latency_metrics["ema_ms"], 42.0)


if __name__ == "__main__":
    unittest.main()
