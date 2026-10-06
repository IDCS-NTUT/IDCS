"""Real menu construction/navigation, without opening a display or GPIO.

Uses unittest so this can run in the Pi runtime without installing pytest.
Optional drawing/GStreamer imports are available there, but not on Windows.
"""
import time
import unittest

from common.config import load_config_bundle, resolve_config_paths
from common.operator_commands import OperatorCommand
from rpi.display.menu import NavEvent, Settings
from rpi.display.status import SystemStatus


class OperatorMenuTests(unittest.TestCase):
    def setUp(self):
        try:
            from rpi.operator_display import DisplayConfig, SETTINGS, build_menu
        except ModuleNotFoundError as exc:
            if exc.name not in ("gi", "cairo"):
                raise
            self.skipTest(f"optional Pi display dependency: {exc.name}")
        cfg = load_config_bundle(resolve_config_paths("configs/base")).data
        self.status = SystemStatus()
        self.status.on_agent({
            "mode": "camera", "pc_online": True, "pc_action_pending": False,
            "modes": [{"name": name, "label": raw["label"]} for name, raw in cfg["operator"]["modes"].items()],
            "target_classes": ["drone"], "message": "mode Camera ready",
        }, time.monotonic())
        self.sent = []
        self.menu = build_menu(self.status, Settings(SETTINGS, None), DisplayConfig(cfg, None), self.sent.append)
        self.menu.handle(NavEvent.MENU)
        self.menu.handle(NavEvent.DOWN)
        self.menu.handle(NavEvent.RIGHT)  # System

    def test_mode_labels_and_current_status(self):
        self.assertIn("PC agent    online", self.menu.view().lines)
        self.menu.handle(NavEvent.RIGHT)  # Mode
        view = self.menu.view()
        self.assertEqual([r.label for r in view.rows], ["Camera", "Simulator", "Standby"])
        self.assertEqual(view.rows[0].value, "current")

    def test_sim_action_requires_confirm_and_sends_correct_command(self):
        self.menu.handle(NavEvent.RIGHT)
        self.menu.handle(NavEvent.DOWN)
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.menu.view().title, "Switch to Simulator?")
        self.assertEqual(self.sent, [])
        self.menu.handle(NavEvent.RIGHT)  # Cancel is first
        self.assertEqual(self.sent, [])
        self.menu.handle(NavEvent.RIGHT)
        self.menu.handle(NavEvent.DOWN)
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.sent, [OperatorCommand("set_mode", mode="sim")])

    def test_transition_disconnect_and_error_are_visible(self):
        self.status.on_agent({"mode": "mixed", "busy": "Simulator: waiting for video",
                              "pc_online": False, "pc_action_pending": True,
                              "message": "recovery required"}, time.monotonic())
        lines = self.menu.view().lines
        self.assertIn("Mode        Simulator: waiting for video", lines)
        self.assertIn("PC agent    offline", lines)
        self.assertIn("PC acknowledgement pending", lines)
        self.assertIn("recovery required", lines)
