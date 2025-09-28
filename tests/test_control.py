import json
import math
import unittest
from typing import Any, Dict, Optional, Tuple
from unittest import mock

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
from common.schemas import Box, CamState, DetectionMsg
from jetson.controller import (
    ControlLoop,
    _CameraFrameMotionModel,
    _MotionModelPrediction,
    _TargetEstimate,
)


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


class _StubMotionModel:
    def __init__(self, prediction: Optional[_MotionModelPrediction]) -> None:
        self._prediction = prediction
        self.latency_ms: Optional[float] = None
        self.mode = "stub"

    def reset(self) -> None:  # pragma: no cover - simple stub
        return

    def update_cam_state(self, cam_state: CamState, timestamp: float) -> None:  # pragma: no cover - simple stub
        return

    def update_target(
        self, uv: Tuple[float, float], timestamp: float
    ) -> _TargetEstimate:  # pragma: no cover - simple stub
        return _TargetEstimate(uv=uv, velocity_px_s=(0.0, 0.0), timestamp=timestamp)

    def update_latency(self, latency_ms: Optional[float]) -> None:  # pragma: no cover - simple stub
        self.latency_ms = latency_ms

    def predict(self, now: float) -> Optional[_MotionModelPrediction]:
        return self._prediction

    def diagnostics(self, now: float) -> Dict[str, Any]:  # pragma: no cover - simple stub
        return {
            "mode": self.mode,
            "horizon_s": 0.0 if self.latency_ms is None else max(self.latency_ms, 0.0) / 1000.0,
            "latency_ms": self.latency_ms,
            "has_state": self._prediction is not None,
            "state_uv": None,
            "velocity_px_s": None,
            "state_timestamp": None,
            "state_age_s": None,
            "residual_px": None,
            "camera_shift_px": None,
            "last_update_ts": None,
        }


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


class ControlLoopMotionModelIntegrationTests(unittest.TestCase):
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

    def _make_loop(self, *, cli_json_logs: bool = False) -> ControlLoop:
        return ControlLoop(self.config, _DummyPub(), cli_json_logs=cli_json_logs)

    def _make_detection(self) -> DetectionMsg:
        box = Box(
            x=0.45,
            y=0.4,
            w=0.1,
            h=0.1,
            cls="0",
            conf=0.9,
            distance_m=5.0,
            distance_src="height",
        )
        return DetectionMsg(
            frame_id=1,
            src_ts_ms=1000,
            rx_ts_ms=1005,
            infer_ts_ms=1010,
            img_w=1280,
            img_h=720,
            boxes=[box],
        )

    def test_detection_msg_populates_prediction_metadata(self) -> None:
        loop = self._make_loop()
        loop.update_latency_measurement(
            selected_ms=42.0,
            source="ema_ms",
            metrics={"ema_ms": 42.0},
        )
        loop.update_cam_state(
            CamState(
                frame_id=1,
                src_ts_ms=1000,
                pan=math.radians(10.0),
                tilt=math.radians(-5.0),
                pan_rate=math.radians(20.0),
                tilt_rate=math.radians(-10.0),
            )
        )

        msg = self._make_detection()
        loop.update_detection(msg)

        self.assertEqual(msg.track_mode, "camera_frame")
        self.assertAlmostEqual(msg.latency_ms_used_for_prediction, 42.0)
        self.assertIsNotNone(msg.pred_px)
        assert msg.pred_px is not None
        horizon_s = 42.0 / 1000.0
        yaw_delta = math.radians(20.0) * horizon_s * self.config.motion_model.derotation.rate_scale
        pitch_delta = math.radians(-10.0) * horizon_s * self.config.motion_model.derotation.rate_scale
        expected_u = 640.0 - self.config.fx_px * yaw_delta / self.config.yaw_sign
        expected_v = 324.0 - self.config.fy_px * pitch_delta / self.config.pitch_sign
        self.assertAlmostEqual(msg.pred_px[0], expected_u)
        self.assertAlmostEqual(msg.pred_px[1], expected_v)
        self.assertAlmostEqual(msg.cam_yaw_deg, 10.0)
        self.assertAlmostEqual(msg.cam_pitch_deg, -5.0)
        self.assertAlmostEqual(msg.cam_yaw_rate_dps, 20.0)
        self.assertAlmostEqual(msg.cam_pitch_rate_dps, -10.0)
        self.assertAlmostEqual(msg.pred_distance_m, 5.0)

    def test_tracking_cmd_uses_prediction(self) -> None:
        loop = self._make_loop()
        msg = self._make_detection()
        loop.update_detection(msg)

        assert loop._last_detection_ts is not None
        prediction = _MotionModelPrediction(
            uv=(600.0, 340.0),
            horizon_s=0.05,
            state_timestamp=loop._last_detection_ts,
            age_s=0.01,
            camera_shift_px=(-6.0, 3.0),
            velocity_px_s=(120.0, -45.0),
        )
        loop._motion_model = _StubMotionModel(prediction)

        loop.tick(now=loop._last_detection_ts + 0.05)

        self.assertTrue(loop._pub.sent)
        payload = json.loads(loop._pub.sent[-1][0])
        self.assertAlmostEqual(payload["target_uv"][0], 600.0)
        self.assertAlmostEqual(payload["target_uv"][1], 340.0)
        self.assertIs(loop._latest_prediction, prediction)
        assert loop._latest_detection is not None
        self.assertIs(loop._latest_detection.prediction, prediction)
        assert loop._latest_detection.predicted_uv is not None
        self.assertAlmostEqual(loop._latest_detection.predicted_uv[0], 600.0)
        self.assertAlmostEqual(loop._latest_detection.predicted_uv[1], 340.0)

    def test_tracking_cmd_falls_back_without_prediction(self) -> None:
        loop = self._make_loop()
        msg = self._make_detection()
        loop.update_detection(msg)

        assert loop._last_detection_ts is not None
        loop._motion_model = _StubMotionModel(None)

        loop.tick(now=loop._last_detection_ts + 0.05)

        self.assertTrue(loop._pub.sent)
        payload = json.loads(loop._pub.sent[-1][0])
        self.assertAlmostEqual(payload["target_uv"][0], 640.0)
        self.assertAlmostEqual(payload["target_uv"][1], 324.0)
        self.assertIsNone(loop._latest_prediction)
        assert loop._latest_detection is not None
        self.assertIsNone(loop._latest_detection.prediction)
        self.assertIsNone(loop._latest_detection.predicted_uv)

    def test_control_log_includes_motion_diagnostics_json(self) -> None:
        loop = self._make_loop(cli_json_logs=True)

        cam_state = CamState(
            frame_id=1,
            src_ts_ms=1000,
            pan=0.0,
            tilt=0.0,
            pan_rate=0.0,
            tilt_rate=0.0,
        )

        with mock.patch("jetson.controller.time.monotonic", return_value=1.0):
            loop.update_cam_state(cam_state)
            msg = self._make_detection()
            loop.update_detection(msg)

        shifted_box = Box(
            x=0.46,
            y=0.4,
            w=0.1,
            h=0.1,
            cls="0",
            conf=0.9,
            distance_m=5.0,
            distance_src="height",
        )
        msg2 = self._make_detection()
        msg2.frame_id = 2
        msg2.boxes = [shifted_box]

        with mock.patch("jetson.controller.time.monotonic", return_value=1.05):
            loop.update_detection(msg2)

        with self.assertLogs("jetson.control", level="INFO") as cm:
            loop.tick(now=1.06)

        self.assertTrue(cm.output)
        payload = json.loads(cm.output[-1].split(":", 2)[-1])
        motion = payload.get("motion")
        assert isinstance(motion, dict)
        self.assertEqual(motion.get("mode"), "camera_frame")
        self.assertTrue(motion.get("has_state"))
        horizon = motion.get("horizon_s")
        assert horizon is not None
        assert self.config.loop_dt is not None
        self.assertAlmostEqual(horizon, self.config.loop_dt, places=4)
        self.assertIsNotNone(motion.get("residual_px"))

    def test_cli_log_includes_motion_summary(self) -> None:
        loop = self._make_loop()

        cam_state = CamState(
            frame_id=1,
            src_ts_ms=1000,
            pan=0.0,
            tilt=0.0,
            pan_rate=0.0,
            tilt_rate=0.0,
        )

        with mock.patch("jetson.controller.time.monotonic", return_value=2.0):
            loop.update_cam_state(cam_state)
            msg = self._make_detection()
            loop.update_detection(msg)

        shifted_box = Box(
            x=0.44,
            y=0.4,
            w=0.1,
            h=0.1,
            cls="0",
            conf=0.9,
            distance_m=5.0,
            distance_src="height",
        )
        msg2 = self._make_detection()
        msg2.frame_id = 3
        msg2.boxes = [shifted_box]

        with mock.patch("jetson.controller.time.monotonic", return_value=2.05):
            loop.update_detection(msg2)

        with self.assertLogs("jetson.control", level="INFO") as cm:
            loop.tick(now=2.06)

        self.assertTrue(cm.output)
        line = cm.output[-1]
        self.assertIn("motion=camera_frame", line)
        self.assertIn("res=(", line)


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
