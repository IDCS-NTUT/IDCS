import json
import unittest

from jetson import gimbal_bridge
from tools import serial_io_service


class _FakePub:
    def __init__(self):
        self.sent = []

    def send_string(self, payload):
        self.sent.append(payload)


class SerialIoTimingTests(unittest.TestCase):
    def test_publish_reply_includes_bus_boundary_timing(self):
        pub = _FakePub()
        serial_io_service._reply_sequence = 0
        cmd = serial_io_service.SerialCommand(
            cmd_id="enc:yaw:1",
            func="0x31",
            addr=1,
            payload=(),
            expect_reply=True,
            expected_len=6,
            priority="high",
            target="gimbal",
            timeout_ms=None,
            retry=None,
            sent_ts_ms=1000,
            enqueued_monotonic_ns=1_000_000_000,
        )

        serial_io_service._publish_reply(
            pub,
            "serial.reply.gimbal",
            cmd,
            bytes([0, 0, 0, 0, 0, 1]),
            sent_ts_ms=1000,
            reply_ts_ms=1017,
            execute_start_monotonic_ns=1_012_000_000,
            reply_monotonic_ns=1_017_500_000,
        )

        _topic, body = pub.sent[0].split(" ", 1)
        payload = json.loads(body)
        self.assertEqual(1, payload["sequence"])
        timing = payload["timing"]
        self.assertEqual(1_000_000_000, timing["enqueued_monotonic_ns"])
        self.assertEqual(1_012_000_000, timing["execute_start_monotonic_ns"])
        self.assertEqual(1_017_500_000, timing["reply_monotonic_ns"])
        self.assertAlmostEqual(12.0, timing["queue_age_ms"])
        self.assertAlmostEqual(5.5, timing["bus_duration_ms"])

    def test_gimbal_bridge_reply_timing_uses_monotonic_fields(self):
        reply = {
            "timing": {
                "execute_start_monotonic_ns": 2_000_000_000,
                "reply_monotonic_ns": 2_030_000_000,
                "queue_age_ms": 14.0,
                "bus_duration_ms": 3.5,
            }
        }

        timing = gimbal_bridge._reply_timing(reply, fallback_mono=9.0)

        self.assertEqual(2.0, timing["execute_s"])
        self.assertEqual(2.03, timing["reply_s"])
        self.assertEqual(14.0, timing["queue_age_ms"])
        self.assertEqual(3.5, timing["bus_duration_ms"])

    def test_gimbal_bridge_reply_timing_falls_back_for_old_serial_service(self):
        timing = gimbal_bridge._reply_timing({"timing": {"duration_ms": 21}}, fallback_mono=7.5)

        self.assertEqual(7.5, timing["execute_s"])
        self.assertEqual(7.5, timing["reply_s"])
        self.assertEqual(0.0, timing["queue_age_ms"])
        self.assertEqual(21.0, timing["bus_duration_ms"])


if __name__ == "__main__":
    unittest.main()
