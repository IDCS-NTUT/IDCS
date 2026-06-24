import json
import sys
import types
import unittest

from common.control import AxisPair, ControlConfig, LaserAimingControlConfig
from common.schemas import DetectionMsg, ManualControlState


gi = types.ModuleType("gi")
gi.require_version = lambda *args, **kwargs: None
gi_repository = types.ModuleType("gi.repository")
gi_repository.Gst = types.SimpleNamespace(init=lambda *args, **kwargs: None)
sys.modules["gi"] = gi
sys.modules["gi.repository"] = gi_repository

yolo_engine = types.ModuleType("jetson.yolo_engine")
yolo_engine.YoloEngine = object
sys.modules["jetson.yolo_engine"] = yolo_engine

from jetson import server  # noqa: E402


class _DummyPub:
    def __init__(self) -> None:
        self.payloads = []

    def send_string(self, payload: str, flags: int = 0) -> None:
        self.payloads.append(payload)


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
        gimbal_accel_limits=AxisPair(4.2, 2.8),
    )


class ServerControlAccelTests(unittest.TestCase):
    def test_transition_command_uses_gimbal_accel_limits(self) -> None:
        pub = _DummyPub()
        msg = DetectionMsg(
            frame_id=7,
            src_ts_ms=100,
            rx_ts_ms=110,
            infer_ts_ms=120,
            img_w=1280,
            img_h=720,
            boxes=[],
        )

        server._send_transition_cmd(
            pub,
            msg=msg,
            target_uv=(700.0, 320.0),
            control_cfg=_make_control_config(),
            speed_rad_s=0.5,
        )

        payload = json.loads(pub.payloads[-1])
        self.assertAlmostEqual(payload["pan_accel_cmd"], 4.2)
        self.assertAlmostEqual(payload["tilt_accel_cmd"], 2.8)

    def test_server_hold_command_uses_gimbal_accel_limits(self) -> None:
        pub = _DummyPub()

        server._publish_hold_control_cmd(
            pub,
            frame_id=8,
            src_ts_ms=200,
            controller_mode="mpc",
            yaw_accel_limit_rad_s2=4.2,
            pitch_accel_limit_rad_s2=2.8,
        )

        payload = json.loads(pub.payloads[-1])
        self.assertAlmostEqual(payload["pan_accel_cmd"], 4.2)
        self.assertAlmostEqual(payload["tilt_accel_cmd"], 2.8)
        self.assertEqual(payload["pan_rate_cmd"], 0.0)
        self.assertEqual(payload["tilt_rate_cmd"], 0.0)

    def test_manual_passthrough_zeros_when_emergency_flag_is_set(self) -> None:
        pub = _DummyPub()
        manual_state = ManualControlState(
            src_ts_ms=300,
            source="test",
            active=True,
            emergency=True,
            joystick_raw=(128, 128),
            joystick_rate_cmd=(0.8, -0.4),
        )

        server._publish_manual_passthrough_control_cmd(
            pub,
            frame_id=9,
            src_ts_ms=300,
            controller_mode="mpc",
            manual_state=manual_state,
            max_yaw_rate=1.0,
            max_pitch_rate=1.0,
            yaw_accel_limit_rad_s2=4.2,
            pitch_accel_limit_rad_s2=2.8,
        )

        payload = json.loads(pub.payloads[-1])
        self.assertFalse(payload["target_ok"])
        self.assertEqual(payload["pan_rate_cmd"], 0.0)
        self.assertEqual(payload["tilt_rate_cmd"], 0.0)


if __name__ == "__main__":
    unittest.main()
