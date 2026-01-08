import unittest

from common.gimbal.authority_safety import AuthoritySafetyConfig, AuthoritySafetyTracker
from common.gimbal.control_plane import (
    FLAG_TAKEOVER,
    ControlPlaneFrame,
    build_ping,
    parse_control_frame,
)
from common.gimbal.handshake_schedule import HandshakeSchedule, next_ping_due, open_window
from common.gimbal.mks_servo42_rs485 import MASTER_START
from rpi.button_latch import ButtonLatch, ButtonLatchConfig


class ControlPlaneFrameTests(unittest.TestCase):
    def test_build_and_parse_ping(self) -> None:
        frame = build_ping(role=0x00, counter=42, flags=FLAG_TAKEOVER)
        parsed = parse_control_frame(frame, expected_start=MASTER_START)
        self.assertEqual(parsed.version, 0x01)
        self.assertEqual(parsed.role, 0x00)
        self.assertEqual(parsed.flags, FLAG_TAKEOVER)
        self.assertEqual(parsed.counter, 42)

    def test_parse_rejects_invalid_flags(self) -> None:
        payload = ControlPlaneFrame(version=0x01, role=0x00, flags=0x08, counter=1)
        frame = build_ping(role=payload.role, counter=payload.counter, flags=payload.flags)
        with self.assertRaises(ValueError):
            parse_control_frame(frame, expected_start=MASTER_START)


class HandshakeScheduleTests(unittest.TestCase):
    def test_window_and_due_logic(self) -> None:
        schedule = HandshakeSchedule(ping_interval_s=1.0, reply_window_s=0.2, bus_quiet_s=0.1)
        window = open_window(now=10.0, schedule=schedule)
        self.assertTrue(window.is_reply_open(10.1))
        self.assertTrue(window.is_quiet_period(10.25))
        self.assertFalse(window.is_quiet_period(10.5))
        self.assertTrue(next_ping_due(now=10.0, last_ping_ts=None, schedule=schedule))
        self.assertFalse(next_ping_due(now=10.5, last_ping_ts=10.0, schedule=schedule))
        self.assertTrue(next_ping_due(now=11.1, last_ping_ts=10.0, schedule=schedule))

    def test_invalid_schedule_rejected(self) -> None:
        schedule = HandshakeSchedule(ping_interval_s=1.0, reply_window_s=1.0)
        with self.assertRaises(ValueError):
            schedule.validate()


class ButtonLatchTests(unittest.TestCase):
    def test_debounce_and_cooldown(self) -> None:
        latch = ButtonLatch(config=ButtonLatchConfig(debounce_s=0.05, cooldown_s=0.2))
        self.assertFalse(latch.update(pressed=True, now=0.0))
        self.assertFalse(latch.update(pressed=True, now=0.02))
        self.assertTrue(latch.update(pressed=True, now=0.06))
        self.assertFalse(latch.update(pressed=True, now=0.1))
        self.assertTrue(latch.update(pressed=True, now=0.27))
        self.assertFalse(latch.update(pressed=False, now=0.3))


class AuthoritySafetyTrackerTests(unittest.TestCase):
    def test_transition_gates_and_peer_timeout(self) -> None:
        tracker = AuthoritySafetyTracker(config=AuthoritySafetyConfig(min_active_s=1.0, peer_timeout_s=0.5))
        tracker.record_state_change(now=0.0)
        self.assertFalse(tracker.can_transition_standby(now=0.5))
        self.assertTrue(tracker.can_transition_standby(now=1.0))
        tracker.record_ping_received(now=1.2)
        self.assertFalse(tracker.peer_unresponsive(now=1.5))
        self.assertTrue(tracker.peer_unresponsive(now=1.8))
