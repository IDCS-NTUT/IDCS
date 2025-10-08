import json
import unittest

from common.schemas import (
    ControlCmd,
    DetectionMsg,
    detection_msg_to_json,
)


class DetectionMsgSchemaTests(unittest.TestCase):
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

    def test_detection_msg_includes_motion_fields_when_present(self) -> None:
        msg = DetectionMsg(
            frame_id=11,
            src_ts_ms=4000,
            rx_ts_ms=4010,
            infer_ts_ms=4020,
            img_w=1280,
            img_h=720,
            boxes=[],
            target_lead_uv=(512.0, 256.0),
            target_velocity_px_s=(25.0, -10.0),
        )

        payload = json.loads(detection_msg_to_json(msg))

        self.assertIn("target_lead_uv", payload)
        self.assertIn("target_velocity_px_s", payload)


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
            target_lead_uv=(650.0, 360.0),
            target_velocity_px_s=(15.0, -5.0),
        )

        self.assertEqual(cmd.laser_dot_px, (640.0, 360.0))
        self.assertTrue(cmd.laser_on_target)
        self.assertTrue(cmd.parallax_compensation_active)
        self.assertEqual(cmd.laser_range_m, 18.0)
        self.assertEqual(cmd.laser_range_source, "default")
        self.assertEqual(cmd.target_lead_uv, (650.0, 360.0))
        self.assertEqual(cmd.target_velocity_px_s, (15.0, -5.0))

