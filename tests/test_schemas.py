import json
import unittest

from common.schemas import (
    ControlCmd,
    DetectionMsg,
    ManualControlState,
    control_cmd_from_json,
    detection_msg_to_json,
    manual_control_state_from_json,
)


class DetectionMsgSchemaTests(unittest.TestCase):
    def test_detection_msg_includes_track_ids_when_present(self) -> None:
        msg = DetectionMsg(
            frame_id=12,
            src_ts_ms=100,
            rx_ts_ms=110,
            infer_ts_ms=120,
            img_w=640,
            img_h=480,
            boxes=[
                {
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.3,
                    "h": 0.4,
                    "cls": "drone",
                    "conf": 0.9,
                    "track_id": 17,
                }
            ],
            target_idx=0,
            target_track_id=17,
        )

        payload = json.loads(detection_msg_to_json(msg))

        self.assertEqual(payload.get("target_track_id"), 17)
        self.assertIn("boxes", payload)
        self.assertEqual(payload["boxes"][0].get("track_id"), 17)

    def test_detection_msg_omits_track_ids_when_absent(self) -> None:
        msg = DetectionMsg(
            frame_id=13,
            src_ts_ms=200,
            rx_ts_ms=210,
            infer_ts_ms=220,
            img_w=640,
            img_h=480,
            boxes=[
                {
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.3,
                    "h": 0.4,
                    "cls": "drone",
                    "conf": 0.9,
                }
            ],
        )

        payload = json.loads(detection_msg_to_json(msg))

        self.assertNotIn("target_track_id", payload)
        self.assertIn("boxes", payload)
        self.assertNotIn("track_id", payload["boxes"][0])

    def test_detection_msg_includes_laser_fields_when_present(self) -> None:
        msg = DetectionMsg(
            frame_id=42,
            src_ts_ms=1000,
            rx_ts_ms=1010,
            infer_ts_ms=1020,
            img_w=1280,
            img_h=720,
            boxes=[],
            laser_dot_px=(123.4, 567.8),
            laser_on_target=True,
            laser_range_m=12.5,
            laser_range_source="known_size",
            parallax_compensation_active=True,
        )

        payload = json.loads(detection_msg_to_json(msg))

        self.assertIn("laser_dot_px", payload)
        self.assertIn("laser_on_target", payload)
        self.assertIn("laser_range_m", payload)
        self.assertIn("laser_range_source", payload)
        self.assertIn("parallax_compensation_active", payload)
        self.assertNotIn("laser_origin_px", payload)

    def test_detection_msg_omits_none_laser_fields(self) -> None:
        msg = DetectionMsg(
            frame_id=7,
            src_ts_ms=2000,
            rx_ts_ms=2015,
            infer_ts_ms=2030,
            img_w=1920,
            img_h=1080,
            boxes=[],
        )

        payload = json.loads(detection_msg_to_json(msg))

        self.assertNotIn("laser_dot_px", payload)
        self.assertNotIn("laser_origin_px", payload)
        self.assertNotIn("laser_on_target", payload)
        self.assertNotIn("laser_range_m", payload)
        self.assertNotIn("laser_range_source", payload)
        self.assertNotIn("parallax_compensation_active", payload)

    def test_detection_msg_includes_predictive_fields_when_present(self) -> None:
        msg = DetectionMsg(
            frame_id=51,
            src_ts_ms=4000,
            rx_ts_ms=4010,
            infer_ts_ms=4020,
            img_w=640,
            img_h=480,
            boxes=[],
            predictive_active=True,
            predictive_target_uv=(120.0, 240.0),
            predictive_box_px=(100.0, 200.0, 140.0, 280.0),
            track_crop_box_px=(80.0, 160.0, 560.0, 460.0),
        )

        payload = json.loads(detection_msg_to_json(msg))

        self.assertTrue(payload.get("predictive_active"))
        self.assertIn("predictive_target_uv", payload)
        self.assertIn("predictive_box_px", payload)
        self.assertIn("track_crop_box_px", payload)

    def test_detection_msg_omits_predictive_fields_when_absent(self) -> None:
        msg = DetectionMsg(
            frame_id=52,
            src_ts_ms=4100,
            rx_ts_ms=4110,
            infer_ts_ms=4120,
            img_w=640,
            img_h=480,
            boxes=[],
        )

        payload = json.loads(detection_msg_to_json(msg))

        self.assertNotIn("predictive_active", payload)
        self.assertNotIn("predictive_target_uv", payload)
        self.assertNotIn("predictive_box_px", payload)
        self.assertNotIn("track_crop_box_px", payload)


class ControlCmdSchemaTests(unittest.TestCase):
    def test_control_cmd_accepts_optional_laser_fields(self) -> None:
        cmd = ControlCmd(
            frame_id=99,
            src_ts_ms=3000,
            cmd_ts_ms=3010,
            target_ok=True,
            target_uv=(640.0, 360.0),
            err_uv=(1.0, -2.0),
            err_rad=(0.01, -0.02),
            pan_rate_cmd=0.1,
            tilt_rate_cmd=-0.1,
            laser_dot_px=(640.0, 360.0),
            laser_origin_px=(600.0, 360.0),
            laser_on_target=True,
            laser_range_m=18.0,
            laser_range_source="default",
            parallax_compensation_active=True,
        )

        self.assertEqual(cmd.laser_dot_px, (640.0, 360.0))
        self.assertTrue(cmd.laser_on_target)
        self.assertTrue(cmd.parallax_compensation_active)
        self.assertEqual(cmd.laser_range_m, 18.0)
        self.assertEqual(cmd.laser_range_source, "default")

    def test_control_cmd_accepts_optional_accel_fields(self) -> None:
        cmd = ControlCmd(
            frame_id=100,
            src_ts_ms=3100,
            cmd_ts_ms=3110,
            target_ok=True,
            target_uv=(640.0, 360.0),
            err_uv=(0.0, 0.0),
            err_rad=(0.0, 0.0),
            pan_rate_cmd=0.2,
            tilt_rate_cmd=-0.2,
            pan_accel_cmd=3.5,
            tilt_accel_cmd=2.0,
        )

        payload = json.loads(cmd.model_dump_json(exclude_none=True))
        parsed = control_cmd_from_json(payload)

        self.assertAlmostEqual(parsed.pan_accel_cmd or 0.0, 3.5)
        self.assertAlmostEqual(parsed.tilt_accel_cmd or 0.0, 2.0)

    def test_control_cmd_accepts_mpc_diagnostics(self) -> None:
        cmd = ControlCmd(
            frame_id=101,
            src_ts_ms=4000,
            cmd_ts_ms=4010,
            target_ok=True,
            target_uv=(640.0, 360.0),
            err_uv=(0.0, 0.0),
            err_rad=(0.0, 0.0),
            pan_rate_cmd=0.0,
            tilt_rate_cmd=0.0,
            controller_mode="mpc",
            mpc={
                "yaw": {
                    "status": "optimal",
                    "cost": 0.5,
                    "u0": 0.1,
                    "slack": {"theta_min": 0.0},
                    "solver": {"iter": 5.0},
                    "terms": {"theta": 0.2, "omega": 0.1},
                }
            },
        )

        self.assertEqual(cmd.controller_mode, "mpc")
        self.assertIsNotNone(cmd.mpc)
        assert cmd.mpc is not None
        self.assertIn("yaw", cmd.mpc)
        diag = cmd.mpc["yaw"]
        self.assertEqual(diag.status, "optimal")
        self.assertAlmostEqual(diag.u0, 0.1)
        self.assertIn("theta_min", diag.slack)
        self.assertIsNotNone(diag.terms)
        assert diag.terms is not None
        self.assertIn("theta", diag.terms)

    def test_control_cmd_from_json_round_trip(self) -> None:
        cmd = ControlCmd(
            frame_id=5,
            src_ts_ms=111,
            cmd_ts_ms=222,
            target_ok=False,
            target_uv=(0.0, 0.0),
            err_uv=(0.0, 0.0),
            err_rad=(0.0, 0.0),
            pan_rate_cmd=0.0,
            tilt_rate_cmd=0.0,
            controller_mode="mpc",
            mpc={
                "yaw": {
                    "status": "optimal",
                    "cost": 1.0,
                    "u0": -0.2,
                    "terms": {"theta": 0.5},
                }
            },
        )

        serialized = cmd.model_dump_json(exclude_none=True)
        parsed = control_cmd_from_json(serialized.encode("utf-8"))

        self.assertEqual(parsed.frame_id, cmd.frame_id)
        self.assertEqual(parsed.controller_mode, "mpc")
        assert parsed.mpc is not None
        self.assertIn("yaw", parsed.mpc)
        self.assertIsNotNone(parsed.mpc["yaw"].terms)


class ManualControlStateSchemaTests(unittest.TestCase):
    def test_manual_control_state_includes_command_toggle_fields(self) -> None:
        state = ManualControlState(
            src_ts_ms=1234,
            source="rpi.runtime_control",
            active=True,
            emergency=False,
            control_cmd_enabled=True,
            control_cmd_changed=True,
            joystick_raw=(128, 129),
            joystick_rate_cmd=(0.0, 0.1),
        )

        payload = state.model_dump(exclude_none=True)

        self.assertTrue(payload.get("control_cmd_enabled"))
        self.assertTrue(payload.get("control_cmd_changed"))

    def test_manual_control_state_defaults_command_toggle_fields_when_missing(self) -> None:
        parsed = manual_control_state_from_json(
            {
                "src_ts_ms": 555,
                "source": "legacy.rpi.runtime_control",
                "active": True,
                "emergency": False,
                "joystick_raw": [127, 127],
                "joystick_rate_cmd": [0.0, 0.0],
            }
        )

        self.assertFalse(parsed.control_cmd_enabled)
        self.assertFalse(parsed.control_cmd_changed)

