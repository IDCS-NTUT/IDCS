import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


control_plane = _load_module("control_plane", ROOT / "common" / "gimbal" / "control_plane.py")
handshake_schedule = _load_module(
    "handshake_schedule", ROOT / "common" / "gimbal" / "handshake_schedule.py"
)
authority_safety = _load_module(
    "authority_safety", ROOT / "common" / "gimbal" / "authority_safety.py"
)
button_latch = _load_module("button_latch", ROOT / "rpi" / "button_latch.py")

AuthoritySafetyConfig = authority_safety.AuthoritySafetyConfig
AuthoritySafetyTracker = authority_safety.AuthoritySafetyTracker
ControlPlaneFrame = control_plane.ControlPlaneFrame
FLAG_TAKEOVER = control_plane.FLAG_TAKEOVER
MASTER_START = control_plane.MASTER_START
build_ping = control_plane.build_ping
parse_control_frame = control_plane.parse_control_frame
HandshakeSchedule = handshake_schedule.HandshakeSchedule
next_ping_due = handshake_schedule.next_ping_due
open_window = handshake_schedule.open_window
ButtonLatch = button_latch.ButtonLatch
ButtonLatchConfig = button_latch.ButtonLatchConfig


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
