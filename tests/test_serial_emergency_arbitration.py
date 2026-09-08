import json
import socket
import time
import unittest
from collections import deque

from common.gimbal.mks_servo42_rs485 import RS485Bus
from tools import serial_io_service


def _command(
    cmd_id: str,
    func: str,
    payload=(),
    *,
    priority: str = "normal",
    addr: int = 1,
    expect_reply: bool = True,
):
    return serial_io_service.SerialCommand(
        cmd_id=cmd_id,
        func=func,
        addr=addr,
        payload=tuple(payload),
        expect_reply=expect_reply,
        expected_len=1 if expect_reply else None,
        priority=priority,
        target="gimbal",
        timeout_ms=None,
        retry=None,
        sent_ts_ms=int(time.time() * 1000),
        enqueued_monotonic_ns=time.monotonic_ns(),
    )


class _FakePub:
    def __init__(self):
        self.sent = []

    def send_string(self, payload):
        self.sent.append(payload)


class _FakeSerial:
    def __init__(self, timeout=0.0075):
        self.timeout = timeout
        self.write_timeout = timeout


class _FakeBus:
    def __init__(self):
        self._serial = _FakeSerial()
        self.max_retries = 1
        self.last_tx_monotonic_ns = None
        self.calls = []
        self.observed_timeout = None

    def send_command(self, addr, func, payload, **kwargs):
        self.observed_timeout = self._serial.timeout
        self.last_tx_monotonic_ns = time.monotonic_ns()
        self.calls.append((addr, func, tuple(payload), kwargs))
        return b""


class _ContinuousNoiseSerial(_FakeSerial):
    def __init__(self):
        super().__init__(timeout=0.005)
        self.is_open = True

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

    def write(self, _payload):
        return None

    def flush(self):
        pass

    def read(self, size):
        return b"\x00" * size


class SerialEmergencyArbitrationTests(unittest.TestCase):
    def test_emergency_preempts_and_discards_queued_motion(self):
        speed = _command("speed:yaw:1", "F6", (0, 10, 10), priority="high")
        encoder = _command("encoder:yaw:1", "0x31", (), priority="high")
        enable = _command("enable:yaw:1", "F3", (1,), priority="critical")
        estop = _command("estop:yaw:1", "F7", (), priority="critical", expect_reply=False)
        queue = deque([speed, encoder, enable, estop])

        dropped = serial_io_service._discard_motion_for_pending_emergency(queue)

        self.assertEqual(2, dropped)
        self.assertEqual(estop, serial_io_service._pop_next_command(queue))
        self.assertEqual([encoder], list(queue))

    def test_critical_zero_speed_is_an_emergency_command(self):
        stop = _command("stop:yaw", "F6", (0, 0, 10), priority="critical")
        self.assertTrue(serial_io_service._is_emergency_command(stop))

    def test_startup_order_is_preserved_below_emergency_rank(self):
        startup_stop = _command("startup:stop:0", "F6", (0, 0, 0), priority="high")
        startup_enable = _command("startup:enable:1", "F3", (1,), priority="critical")
        queue = deque([startup_stop, startup_enable])

        self.assertEqual(startup_stop, serial_io_service._pop_next_command(queue))
        self.assertEqual(startup_enable, serial_io_service._pop_next_command(queue))

    def test_emergency_is_written_without_waiting_for_reply_or_retry(self):
        bus = _FakeBus()
        pub = _FakePub()
        estop = _command("estop:yaw:2", "F7", (), priority="critical")
        estop.request_monotonic_ns = time.monotonic_ns()
        estop.request_host = socket.gethostname()

        serial_io_service._process_command(
            bus, estop, pub, critical_latency_budget_ms=25.0
        )

        self.assertEqual(1, len(bus.calls))
        self.assertFalse(bus.calls[0][3]["response_expected"])
        self.assertEqual(0, bus.calls[0][3]["retries"])
        topic, raw = pub.sent[0].split(" ", 1)
        message = json.loads(raw)
        self.assertEqual("serial.telemetry.gimbal", topic)
        self.assertEqual("SerialEmergencyTiming", message["type"])
        self.assertIsNotNone(message["timing"]["wire_monotonic_ns"])
        self.assertIsNotNone(message["timing"]["request_to_wire_ms"])
        self.assertEqual("same_host_request_to_wire", message["timing"]["budget_scope"])
        self.assertFalse(message["timing"]["budget_missed"])

    def test_client_timeout_and_retry_overrides_are_bounded(self):
        bus = _FakeBus()
        speed = _command(
            "speed:yaw:bounded",
            "F6",
            (0, 10, 10),
            priority="high",
            expect_reply=False,
        )
        speed.timeout_ms = 10_000
        speed.retry = 100

        serial_io_service._process_command(
            bus, speed, None, max_non_emergency_block_ms=20.0
        )

        self.assertEqual(1, bus.calls[0][3]["retries"])
        self.assertAlmostEqual(0.010, bus.observed_timeout)

    def test_continuous_noise_cannot_extend_transaction_past_absolute_deadline(self):
        bus = RS485Bus.__new__(RS485Bus)
        bus.port = "fake"
        bus.baudrate = 38400
        bus.timeout = 0.005
        bus.max_retries = 0
        bus.last_tx_monotonic_ns = None
        bus._serial = _ContinuousNoiseSerial()

        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            bus.send_command(1, 0x31, (), expected_response_len=6, retries=0)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.05)


if __name__ == "__main__":
    unittest.main()
