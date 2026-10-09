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
    def test_clear_swarm_score_is_conditional_confirmed_and_session_scoped(self):
        self.assertNotIn('Clear Swarm score', [r.label for r in self.menu.view().rows])
        self.status.agent_state.update(mode='sim', sim_scene='single')
        self.assertNotIn('Clear Swarm score', [r.label for r in self.menu.view().rows])
        self.status.agent_state.update(sim_scene='swarm', sim_hud_session='sim-A')
        self.assertIn('Clear Swarm score', [r.label for r in self.menu.view().rows])
        for _ in range(3): self.menu.handle(NavEvent.DOWN)
        self.menu.handle(NavEvent.RIGHT)
        self.assertIn('Clear Swarm score?', self.menu.view().title)
        self.menu.handle(NavEvent.RIGHT)  # Default Cancel
        self.assertEqual(self.sent, [])
        self.menu.handle(NavEvent.RIGHT)
        self.status.agent_state['sim_hud_session'] = 'sim-B'  # Frozen confirmation cannot clear new session.
        self.menu.handle(NavEvent.DOWN)
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.sent, [OperatorCommand('clear_swarm_score', source_session='sim-A')])

    def test_simulator_scene_is_nested_and_confirmation_is_required(self):
        self.status.agent_state.update(mode='sim',sim_scene='single',sim_scenes=[
            {'name':'single','label':'Single'},{'name':'swarm','label':'Swarm'}])
        self.menu.handle(NavEvent.RIGHT)  # Mode
        self.menu.handle(NavEvent.DOWN)  # Simulator
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.menu.view().title,'Menu / System / Mode / Simulator')
        self.assertEqual([r.label for r in self.menu.view().rows],['Single','Swarm'])
        self.assertEqual(self.menu.view().rows[0].value,'current')
        self.menu.handle(NavEvent.DOWN)
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.menu.view().title,'Switch to Simulator / Swarm?')
        self.assertEqual(self.sent,[])
        self.menu.handle(NavEvent.DOWN)  # Confirm, not default Cancel
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.sent,[OperatorCommand('set_mode',mode='sim',scene='swarm')])

    def test_scene_cancel_and_current_state_do_not_send_commands(self):
        self.status.agent_state.update(mode='sim',sim_scene='swarm',sim_scenes=[
            {'name':'single','label':'Single'},{'name':'swarm','label':'Swarm'}],
            message='mode Simulator / Swarm ready')
        self.assertNotIn('mode Simulator / Swarm ready',self.menu.view().lines)
        self.menu.handle(NavEvent.RIGHT)
        self.menu.handle(NavEvent.DOWN)
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.menu.view().rows[1].value,'current')
        self.menu.handle(NavEvent.RIGHT)
        self.menu.handle(NavEvent.RIGHT)  # Cancel is selected initially
        self.assertEqual(self.sent,[])

    def test_steady_status_is_not_repeated_and_required_strip_has_no_hide_option(self):
        self.assertFalse(any(line.startswith('Mode ') for line in self.menu.view().lines))
        self.assertNotIn('mode Camera ready',self.menu.view().lines)
        self.menu.handle(NavEvent.LEFT)
        self.assertNotIn('Display',[row.label for row in self.menu.view().rows])
        self.menu.handle(NavEvent.DOWN)  # Status
        self.menu.handle(NavEvent.RIGHT)
        self.assertNotIn(f'Panel       {self.status.panel_mode(time.monotonic())}',self.menu.view().lines)
        self.assertFalse(any('fps' in line for line in self.menu.view().lines))
        self.assertEqual(self.sent,[])

    def test_hud_telemetry_page_is_on_demand_and_does_not_send_commands(self):
        from rpi.operator_display import build_menu,DisplayConfig,SETTINGS
        cfg = load_config_bundle(resolve_config_paths('configs/base')).data
        menu = build_menu(self.status,Settings(SETTINGS,None),DisplayConfig(cfg,None),
                          self.sent.append,lambda: ['Not joined to video','Yaw cmd +0.20 FF +0.12'])
        menu.handle(NavEvent.MENU)
        for _ in range(3): menu.handle(NavEvent.DOWN)
        menu.handle(NavEvent.RIGHT)
        self.assertEqual(menu.view().title,'Menu / HUD telemetry')
        self.assertIn('Not joined to video',menu.view().lines)
        self.assertEqual(self.sent,[])

    def test_applied_lock_has_no_second_box_but_unselected_cursor_has_preview(self):
        from types import SimpleNamespace as NS
        from rpi.operator_display import OperatorDisplay
        display = OperatorDisplay.__new__(OperatorDisplay)
        row = NS(track_id=7,selected=True,box=(.4,.4,.1,.1))
        display.status = NS(target_rows=lambda now:[row])
        display.menu = NS(highlighted=lambda:NS(key=('track',7)))
        self.assertEqual(display._boxes(),())
        row.selected=False
        preview = display._boxes()
        self.assertEqual(len(preview),1)
        self.assertEqual(preview[0].style,'cursor')

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
            "modes": [{"name": name, "label": raw["label"]} for name, raw in cfg["operator"]["modes"].items()
                      if name != 'sim_swarm'],
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
        self.assertIn("Transition  Simulator: waiting for video", lines)
        self.assertIn("PC agent    offline", lines)
        self.assertIn("PC acknowledgement pending", lines)
        self.assertIn("recovery required", lines)

    def test_physical_picker_uses_sim_catalog_identity_not_detector_track_id(self):
        self.menu._close()
        self.status.on_snapshot({"tracks": [{"track_id": 99, "class_id": "person"}], "selection": {}}, time.monotonic())
        catalog = dict(source="sim", session="sim-session", tracks=[dict(track_id=2, class_id="drone",
            confidence=1., missed_frames=0, box=dict(x=.7, y=.4, w=.05, h=.1))])
        self.status.on_agent({"selection_mode": "planner", "target_catalog": {"catalog": catalog, "age_ms": 0.}}, time.monotonic())
        self.menu.handle(NavEvent.LEFT)
        self.assertTrue(self.menu.picker_active)
        self.assertEqual([r.label for r in self.menu.view().rows], ["Use planner", "#2 drone 1.00"])
        self.menu.handle(NavEvent.DOWN)
        self.assertEqual(self.sent, [])
        # Reordering/replacing the live catalog must not silently retarget this confirmation.
        catalog["tracks"][0]["track_id"] = 3
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.sent, [OperatorCommand("lock", track_id=2, source="sim", source_session="sim-session")])
        self.assertFalse(self.menu.is_open)
