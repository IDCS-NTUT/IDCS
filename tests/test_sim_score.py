"""Deterministic score/error contracts; no GPU, detector, GPIO or motor owner."""
import copy
import json
import math
import unittest
import threading
import queue
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from common.sim_score import ScoreResetHandler, ScoreResetRequest, SwarmScore, simulation_metric
from common.hud_telemetry import HudCache, validate_message, HudView
from common.pc_mode import PcModeTask, parse_task
from common.operator_commands import OperatorCommand, parse_command
from pc.sim_camera import _PlannerEvalScenario
from tests.test_panel_hud import pose, diagnostic
from pc.hud_relay import HudSampler


def score():
    return dict(epoch=1, active=3, spawned=8, eliminated=4, breached=1)


class ScoreTests(unittest.TestCase):
    def test_bounded_loopback_reset_acknowledgement_and_retry(self):
        import zmq
        from pc.sim_score import clear_score
        ready, failures, stop = queue.Queue(), [], threading.Event()
        calls = []
        handler = ScoreResetHandler('sim-A', NS(clear_planner_eval_score=lambda: calls.append(1) or len(calls)))
        def serve():
            ctx = zmq.Context()
            socket = ctx.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, 1000)
            try:
                port = socket.bind_to_random_port('tcp://127.0.0.1')
                ready.put(f'tcp://127.0.0.1:{port}')
                while not stop.is_set():
                    if socket.poll(20):
                        socket.send_json(handler.handle(socket.recv()))
            except Exception as exc:
                failures.append(exc)
            finally:
                socket.close(0)
                ctx.term()
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            endpoint = ready.get(timeout=2.)
            task = PcModeTask(5, 'clear_score', 'idcs-sim-swarm-streamer.service', 'sim-A')
            self.assertEqual(clear_score(task, 'client-A', endpoint=endpoint), (0, 'Swarm score cleared (epoch 1)'))
            self.assertEqual(clear_score(task, 'client-A', endpoint=endpoint), (0, 'Swarm score cleared (epoch 1)'))
            wrong = PcModeTask(6, task.verb, task.unit, 'other-session')
            self.assertEqual(clear_score(wrong, 'client-A', endpoint=endpoint)[0], 1)
            self.assertEqual(calls, [1])
            with self.assertRaises(ValueError):
                clear_score(task, 'client-A', endpoint='tcp://192.168.0.5:5580')
        finally:
            stop.set()
            thread.join(3.)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])

    def test_clear_preserves_targets_motion_rng_schedule_and_recording_events(self):
        scenario = _PlannerEvalScenario({}, fps_hz=60.)
        targets = [NS(position=np.array([1., 2., 3.]), aim_dwell_s=.1, target_id=23)]
        scenario.active = targets
        scenario.total_spawned, scenario.eliminated_count, scenario.breach_count = 6, 3, 2
        scenario.events.append({'event': 'spawn', 'target_id': 23})
        rng = copy.deepcopy(scenario._rng.bit_generator.state)
        schedule, next_id = scenario._next_spawn_time_s, scenario._next_target_id
        self.assertEqual(scenario.clear_score(), 1)
        self.assertEqual(scenario.stats['spawned'], 1)
        self.assertEqual((scenario.stats['eliminated'], scenario.stats['breached']), (0, 0))
        self.assertIs(scenario.active, targets)
        self.assertEqual(targets[0].aim_dwell_s, .1)
        np.testing.assert_array_equal(targets[0].position, [1., 2., 3.])
        self.assertEqual(scenario._rng.bit_generator.state, rng)
        self.assertEqual((scenario._next_spawn_time_s, scenario._next_target_id), (schedule, next_id))
        self.assertEqual([e['event'] for e in scenario.events], ['spawn', 'score_reset'])

    def test_reset_session_and_duplicate_guard(self):
        calls = []
        camera = NS(clear_planner_eval_score=lambda: calls.append(1) or len(calls))
        handler = ScoreResetHandler('sim-A', camera)
        request = ScoreResetRequest(source_session='sim-A', client_session='pc-A', request_id=12)
        first = handler.handle(request.model_dump_json())
        self.assertTrue(first['ok'])
        self.assertEqual(first, handler.handle(request.model_dump_json()))
        self.assertEqual(calls, [1])
        for bad in (request.model_copy(update={'source_session': 'old'}),
                    request.model_copy(update={'request_id': 11})):
            self.assertFalse(handler.handle(bad.model_dump_json())['ok'])
        self.assertEqual(calls, [1])
        self.assertFalse(handler.handle('not json')['ok'])
        self.assertFalse(handler.handle(' ' * 4097)['ok'])
        self.assertEqual(handler.handle(request.model_copy(update={'request_id': 13}).model_dump_json())['epoch'], 2)

    def test_score_and_task_protocol_reject_invalid_or_unscoped_inputs(self):
        self.assertEqual(SwarmScore.model_validate(score()).active, 3)
        for changes in ({'active': -1}, {'active': True}, {'epoch': 1.5}, {'motor_rate': 1}):
            with self.assertRaises(ValueError):
                SwarmScore.model_validate(dict(score(), **changes))
        task = PcModeTask(5, 'clear_score', 'idcs-sim-swarm-streamer.service', 'sim-A')
        self.assertEqual(parse_task(task.to_json()), task)
        for unit, session in (('idcs-sim-swarm-streamer.service', None), ('idcs-hil-streamer.service', 's')):
            with self.assertRaises(ValueError):
                PcModeTask(5, 'clear_score', unit, session)
        command = OperatorCommand('clear_swarm_score', source_session='sim-A')
        self.assertEqual(parse_command(command.to_json()), command)
        with self.assertRaises(ValueError):
            parse_command(OperatorCommand('clear_swarm_score').to_json())

    def test_pose_wire_transports_score_but_does_not_renew_stale_data(self):
        sampler = HudSampler('relay-A')
        state = pose().model_copy(update={'sim_scene': 'swarm', 'swarm_score': SwarmScore.model_validate(score())})
        sampler.receive('pose', state.model_dump_json(), 1_000_000_010)
        value = sampler.message(1_000_000_020)
        self.assertEqual(value['pose']['data']['swarm_score'], score())
        cache = HudCache()
        cache.accept(value, 50.)
        view = cache.view(50.01, source_session='sim-A')
        self.assertEqual(simulation_metric(view.pose, None, None)[1], 'SWARM HIT 4  BREACH 1  LIVE 3')
        self.assertIsNone(cache.view(50.01, source_session='other').pose)
        later = copy.deepcopy(value)
        later['sequence'] += 1
        cache.accept(later, 50.2)
        self.assertIsNone(cache.view(50.26, source_session='sim-A').pose)
        for key, val in (('active', -1), ('eliminated', True), ('motor_rate', 1)):
            bad = copy.deepcopy(value)
            bad['pose']['data']['swarm_score'][key] = val
            with self.assertRaises(ValueError):
                validate_message(bad)

    def test_single_error_uses_raw_parallax_error_not_prediction(self):
        diag = diagnostic().model_dump(mode='json')
        diag['timing']['source_frame_age_ms'] = 20.
        diag['yaw'].update(raw_error_rad=math.radians(1.25), estimated_error_rad=9.)
        diag['pitch'].update(raw_error_rad=math.radians(-.5), estimated_error_rad=9.)
        self.assertEqual(simulation_metric({'sim_scene': 'single'}, diag, .01),
                         ('tracking_error', 'ERR +1.25\u00b0 / -0.50\u00b0'))
        for age in (None, -.1, .26):
            self.assertEqual(simulation_metric({'sim_scene': 'single'}, diag, age)[1], 'ERR --')
        for reason in ('coasting', 'target_invalid', 'manual', 'clock_unavailable'):
            self.assertEqual(simulation_metric({'sim_scene': 'single'}, dict(diag, reason=reason), .01)[1], 'ERR --')
        for capture_age in (None, -1., float('nan'), 245.):
            old = copy.deepcopy(diag)
            old['timing']['source_frame_age_ms'] = capture_age
            self.assertEqual(simulation_metric({'sim_scene': 'single'}, old, .01)[1], 'ERR --')
        diag['yaw']['raw_error_rad'] = float('nan')
        self.assertEqual(simulation_metric({'sim_scene': 'single'}, diag, .01)[1], 'ERR --')
        self.assertIsNone(simulation_metric({}, diag, .01))
        self.assertEqual(simulation_metric({'sim_scene': 'swarm', 'swarm_score': score()}, diag, .01)[0], 'swarm_score')

    def test_pc_metrics_have_one_owner_and_no_stale_or_camera_score(self):
        from pc.v2_hud import V2HudRenderer
        from common.ui_style import HudLayout
        renderer = V2HudRenderer(hfov_deg=135, vfov_deg=73, authority_label='SIM')
        for w, h in ((640, 360), (1280, 720), (1920, 1080)):
            layout = HudLayout(w, h)
            self.assertFalse(layout.simulation_metric.overlaps(layout.heading))
            self.assertFalse(layout.simulation_metric.overlaps(layout.elevation))
            state = pose().model_copy(update={'sim_scene': 'swarm', 'swarm_score': SwarmScore.model_validate(score())})
            labels = []
            with patch('pc.v2_hud.ui_text', side_effect=lambda frame, text, *a, **kw: labels.append(text)):
                result = renderer.render(np.zeros((h, w, 3), np.uint8), snapshot=None, cam_state=state, control_cmd=None, cam_state_age_s=.01)
            self.assertEqual(result.elements.count('swarm_score'), 1)
            self.assertEqual(sum(t.startswith('SWARM ') for t in labels), 1)
            stale = renderer.render(np.zeros((h, w, 3), np.uint8), snapshot=None, cam_state=state, control_cmd=None, cam_state_age_s=.3)
            self.assertNotIn('swarm_score', stale.elements)
            single = renderer.render(np.zeros((h, w, 3), np.uint8), snapshot=None,
                cam_state=pose().model_copy(update={'sim_scene': 'single'}), control_cmd=None, cam_state_age_s=.01)
            self.assertIn('tracking_error', single.elements)
            self.assertNotIn('swarm_score', single.elements)
        camera = V2HudRenderer(hfov_deg=135, vfov_deg=73)
        self.assertNotIn('swarm_score', camera.render(np.zeros((720, 1280, 3), np.uint8),
            snapshot=None, cam_state=state, control_cmd=None, cam_state_age_s=.01).elements)

    def test_pi_metrics_use_same_formatter_and_fit_reserved_strip(self):
        try:
            from rpi.display.hud import PiHudRenderer
        except ModuleNotFoundError as exc:
            if exc.name == 'cairo':
                self.skipTest('optional Cairo dependency')
            raise
        renderer = PiHudRenderer()
        for scene in ('swarm', 'single'):
            value = dict(sim_scene=scene, pan=0., tilt=0., home_pan=0., home_tilt=0.)
            if scene == 'swarm':
                value['swarm_score'] = score()
            labels = []
            with patch('rpi.display.hud._Tile.text', side_effect=lambda text, *a, **kw: labels.append(text)):
                images = renderer.render(1280, 720, HudView('SIM-ONLY', 'sim-A', pose=value))
            expected = 'swarm_score' if scene == 'swarm' else 'tracking_error'
            self.assertEqual(renderer.elements.count(expected), 1)
            self.assertIn(simulation_metric(value, None, None)[1], labels)
            self.assertTrue(all(i.x >= 0 and i.y >= 0 and i.x + i.width <= 1280 and i.y + i.height <= 720 for i in images))
        renderer.render(1280, 720, HudView('UNAVAILABLE', ''))
        self.assertNotIn('swarm_score', renderer.elements)


if __name__ == '__main__':
    unittest.main()
