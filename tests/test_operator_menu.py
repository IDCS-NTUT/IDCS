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
    def test_roi_return_keeps_panel_overlay_without_a_second_reticle_or_status(self):
        from collections import deque
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        from common.hud_telemetry import HudView
        from rpi.operator_display import OperatorDisplay
        display = OperatorDisplay.__new__(OperatorDisplay)
        display.status = self.status
        display.menu = self.menu
        display.menu._close()
        display.visual_roi_trial = True
        display._sim_roi_source_available = lambda now:True
        display.roi_editor = NS(active=False)
        display.roi_workflow = NS(state='camera')
        display.hud_state = NS(view=lambda *args,**kwargs:HudView('UNAVAILABLE',''))
        display.hud_renderer = NS(render=Mock())
        display.video = NS(frame_size=(1280,720),set_overlay=Mock())
        display._boxes = lambda:()
        display._drawn_key = None
        display.render_ms = deque(maxlen=3)
        with patch('rpi.operator_display.render',return_value=['panel']) as drawing:
            display._draw(time.monotonic())
        display.hud_renderer.render.assert_not_called()
        self.assertEqual(display.video.set_overlay.call_args.args[0],['panel'])
        self.assertIsNotNone(drawing.call_args.kwargs['fire_state'])
        self.assertIsNotNone(drawing.call_args.kwargs['safety_state'])
        self.assertEqual(display.last_hud_source,'ISOLATED ROI')

    def test_manual_off_exits_roi_and_restores_the_joystick_menu(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        from rpi.display.roi_panel import RoiPanelWorkflow
        from rpi.operator_display import OperatorDisplay
        display = OperatorDisplay.__new__(OperatorDisplay)
        display._roi_manual_seen = True
        display.roi_workflow = RoiPanelWorkflow(active=True,state='tracking',request_id='request')
        display.status = NS(panel_state=dict(manual_active=False),show_notice=Mock())
        display.roi_editor = NS(active=True)
        display.menu = NS(_close=Mock())
        display._release_roi = Mock()
        display._update_roi_panel(1.)
        self.assertFalse(display.roi_editor.active)
        self.assertFalse(display.roi_workflow.active)
        display._release_roi.assert_called_once_with('request')
        display.menu._close.assert_called_once()

    def test_roi_lock_survives_late_image_metadata_but_releases_stale_receiver(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        from common.return_frame_identity import RoiSourceLease
        from rpi.display.roi_panel import RoiPanelWorkflow
        from rpi.operator_display import OperatorDisplay
        display = OperatorDisplay.__new__(OperatorDisplay)
        display.visual_roi_trial = True
        display._roi_request_session = 'run'
        display.roi_source = RoiSourceLease()
        display.roi_source.observe(dict(type='visual_roi_status_v1',source_mode='sim',
            source_ready=True,control_disabled=True,tracker_session='run',sequence=1),1.01)
        display.roi_workflow = RoiPanelWorkflow(active=True,state='tracking',request_id='request',previous_on=True)
        panel = dict(manual_active=True,emergency=False,inputs_valid=dict(fire_control=True),
                     inputs=dict(fire_control=True))
        display.status = NS(panel_state=panel,video=NS(live=lambda now:True),
                            panel=NS(live=lambda now:True),show_notice=Mock())
        display.video = NS(frame_identity=NS(latest=Mock(side_effect=AssertionError('already seeded'))))
        display.roi_editor = NS(_last_move_s=None)
        display._release_roi = Mock()
        with patch('rpi.operator_display.time.monotonic',return_value=1.02):
            display._update_roi_panel(1.0)  # status arrived during this tick
        self.assertEqual(display.roi_workflow.state,'tracking')
        display._release_roi.assert_not_called()
        with patch('rpi.operator_display.time.monotonic',return_value=1.6):
            display._update_roi_panel(1.6)
        self.assertEqual(display.roi_workflow.state,'camera')
        display._release_roi.assert_called_once_with('request')

    def test_isolated_roi_menu_uses_live_receiver_when_operator_agent_is_paused(self):
        from rpi.operator_display import DisplayConfig, SETTINGS, build_menu
        cfg = load_config_bundle(resolve_config_paths("configs/base")).data
        self.status.agent_state = {}
        ready = [True]
        menu = build_menu(self.status, Settings(SETTINGS, None), DisplayConfig(cfg, None),
                          visual_roi_trial=True, roi_available=lambda: ready[0])
        menu.handle(NavEvent.MENU)
        menu.handle(NavEvent.DOWN)
        menu.handle(NavEvent.RIGHT)
        self.assertIn('Visual ROI tracking (isolated)', [r.label for r in menu.view().rows])
        ready[0] = False
        self.assertNotIn('Visual ROI tracking (isolated)', [r.label for r in menu.view().rows])

    def test_root_separates_targets_operations_and_information(self):
        self.menu.handle(NavEvent.LEFT)
        self.assertEqual([row.label for row in self.menu.view().rows],
                         ['Targets', 'Operations', 'Information'])
        self.menu.handle(NavEvent.UP)
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual([row.label for row in self.menu.view().rows],
                         ['Planner selection', 'Eligible classes'])
        self.assertEqual(self.menu.view().rows[0].value, 'on')
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.sent, [OperatorCommand('selection_mode', mode='operator')])
        self.status.agent_state['selection_mode'] = 'operator'
        self.assertEqual(self.menu.view().rows[0].value, 'off')
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.sent[-1], OperatorCommand('selection_mode', mode='planner'))

    def test_clear_swarm_score_is_conditional_confirmed_and_session_scoped(self):
        self.assertNotIn('Clear Swarm score', [r.label for r in self.menu.view().rows])
        self.status.agent_state.update(mode='sim', sim_scene='single')
        self.assertNotIn('Clear Swarm score', [r.label for r in self.menu.view().rows])
        self.status.agent_state.update(sim_scene='swarm', sim_hud_session='sim-A')
        self.assertIn('Clear Swarm score', [r.label for r in self.menu.view().rows])
        for _ in range(2): self.menu.handle(NavEvent.DOWN)
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
        self.assertEqual(self.menu.view().title,'Menu / Operations / Source & scene / Simulator')
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
        self.menu.handle(NavEvent.DOWN)  # Information
        self.menu.handle(NavEvent.RIGHT)
        self.menu.handle(NavEvent.DOWN)  # Connections
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
        for _ in range(2): menu.handle(NavEvent.DOWN)
        menu.handle(NavEvent.RIGHT)
        for _ in range(2): menu.handle(NavEvent.DOWN)
        menu.handle(NavEvent.RIGHT)
        self.assertEqual(menu.view().title,'Menu / Information / Control')
        self.assertIn('Not joined to video',menu.view().lines)
        self.assertEqual(self.sent,[])

    def test_information_replaces_only_ordinary_hud_and_keeps_foreground_status(self):
        from collections import deque
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        from rpi.operator_display import OperatorDisplay

        self.menu.handle(NavEvent.LEFT)   # root
        self.menu.handle(NavEvent.DOWN)   # Information
        self.menu.handle(NavEvent.RIGHT)
        self.menu.handle(NavEvent.RIGHT)  # System information page
        self.assertEqual(self.menu.view().kind, 'info')
        display = OperatorDisplay.__new__(OperatorDisplay)
        display.status = self.status
        display.menu = self.menu
        display.roi_editor = NS(active=False)
        display.roi_workflow = NS(state='idle')
        display.visual_roi_trial = False
        display.hud_state = NS(view=lambda *args, **kwargs: NS(source='camera'))
        display.hud_renderer = NS(render=Mock(return_value=['ordinary HUD']))
        display.video = NS(frame_size=(1280, 720), set_overlay=Mock())
        display._boxes = lambda: ()
        display._drawn_key = None
        display.render_ms = deque(maxlen=3)
        with patch('rpi.operator_display.render', return_value=['menu', 'foreground']) as drawing:
            display._draw(time.monotonic())
            display.hud_renderer.render.assert_not_called()
            self.assertEqual(display.video.set_overlay.call_args.args[0],
                             ['menu', 'foreground'])
            shown = drawing.call_args.kwargs
            self.assertIn('camera', shown['bar'].fields[0])
            self.assertIsNotNone(shown['fire_state'])
            self.assertIsNotNone(shown['safety_state'])
            self.menu._close()
            display._draw(time.monotonic())
            display.hud_renderer.render.assert_called_once()
            self.assertEqual(display.video.set_overlay.call_args.args[0][0], 'ordinary HUD')

    def test_applied_lock_has_no_second_box_but_unselected_cursor_has_preview(self):
        from dataclasses import replace
        from types import SimpleNamespace as NS
        from rpi.display.status import TrackRow
        from rpi.operator_display import OperatorDisplay
        display = OperatorDisplay.__new__(OperatorDisplay)
        row = TrackRow(track_id=7,class_id='drone',confidence=.8,missed_frames=0,
                       selected=True,distance_m=None,box=(.4,.4,.1,.1),rank=1)
        display.status = NS(target_rows=lambda now:[row])
        display.menu = NS(highlighted=lambda:NS(key=('track',7)))
        display.roi_editor = NS(active=False)
        self.assertEqual(display._boxes(),())
        row = replace(row,selected=False)
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
        self.menu.handle(NavEvent.RIGHT)  # Operations

    def test_mode_labels_and_current_status(self):
        self.menu.handle(NavEvent.RIGHT)  # Source & scene
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
        self.menu.handle(NavEvent.LEFT)
        self.menu.handle(NavEvent.DOWN)
        self.menu.handle(NavEvent.RIGHT)
        self.menu.handle(NavEvent.RIGHT)
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
        self.assertEqual([r.label for r in self.menu.view().rows], ["drone 1.00"])
        self.assertEqual(self.menu.view().kind, 'picker')
        self.assertEqual(self.sent, [])
        # Reordering/replacing the live catalog must not silently retarget this confirmation.
        catalog["tracks"][0]["track_id"] = 3
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.sent, [OperatorCommand("lock", track_id=2, source="sim", source_session="sim-session")])
        self.assertFalse(self.menu.is_open)

    def test_picker_displays_rank_but_confirms_internal_id_and_session(self):
        self.menu._close()
        now = time.monotonic()
        tracks = [dict(track_id=i,class_id='drone',confidence=.8,missed_frames=0,
                       box=dict(x=.1+n*.2,y=.3,w=.05,h=.05)) for n,i in enumerate((90123,98765))]
        self.status.on_snapshot(dict(frame=dict(tracker_session='visual-A'),tracks=tracks,
            assessments=[dict(track_id=90123,engagement_rank=2),
                         dict(track_id=98765,engagement_rank=1,distance_m=12.4)]),now)
        self.status.on_agent(dict(selection_mode='planner',target_catalog=dict(age_ms=0,
            catalog=dict(source='camera',session='visual-A',tracks=tracks))),now)
        self.menu.handle(NavEvent.LEFT)
        self.assertEqual([r.label for r in self.menu.view().rows],
                         ['R1 drone 0.80  12.4 m','R2 drone 0.80'])
        self.menu.handle(NavEvent.DOWN)
        self.assertEqual(self.menu.highlighted().key,('track',90123))
        self.menu.handle(NavEvent.RIGHT)
        self.assertEqual(self.sent,[OperatorCommand('lock',track_id=90123,source='camera',source_session='visual-A')])


class TargetPresentationTests(unittest.TestCase):
    def test_rank_join_requires_fresh_matching_visual_source(self):
        status = SystemStatus()
        tracks = [dict(track_id=98765,class_id='drone',confidence=.8,missed_frames=0,
                       box=dict(x=.1,y=.3,w=.05,h=.05))]
        status.on_snapshot(dict(frame=dict(tracker_session='A'),tracks=tracks,
            assessments=[dict(track_id=98765,engagement_rank=0,distance_m=12.4)]),10.)
        def catalog(source,session):
            status.on_agent(dict(target_catalog=dict(age_ms=0,
                catalog=dict(source=source,session=session,tracks=tracks))),10.)
            return status.target_rows(10.1)[0]
        row = catalog('camera','A')
        self.assertEqual((row.track_id,row.rank,row.distance_m),(98765,0,12.4))
        self.assertEqual(row.display_name(True),'R0 drone')
        self.assertEqual(row.display_name(False),'drone')
        for source,session in (('camera','B'),('sim','A')):
            row = catalog(source,session)
            self.assertIsNone(row.rank)
            self.assertIsNone(row.distance_m)
            self.assertEqual(row.display_name(True),'drone')
        catalog('camera','A')
        status.perception.last_s = 8.
        self.assertIsNone(status.target_rows(10.1)[0].rank)

    def test_status_and_engagement_do_not_expose_tracker_ids(self):
        status = SystemStatus()
        status.on_snapshot(dict(tracks=[dict(track_id=90000,class_id='drone',confidence=.8),
                                       dict(track_id=95000,class_id='drone',confidence=.8)],
            assessments=[dict(track_id=90000,engagement_rank=1)],
            selection=dict(track_id=90000,policy='operator_lock')),10.)
        self.assertEqual(status.targets_text(10.1),'2 trk  LOCK R1')
        self.assertTrue(status.track_lines(10.1)[0].startswith('L R1 drone'))
        self.assertFalse(any('90000' in line or '#' in line for line in status.track_lines(10.1)))
        status.on_diagnostics(dict(engaged_track_id=90000,engaged_monotonic_ns=1),10.)
        status.on_diagnostics(dict(engaged_track_id=90000,engaged_monotonic_ns=2),10.1)
        self.assertIn('ENGAGE',status.alerts(10.2))
        self.assertEqual(status.engaged_track_id,90000)
