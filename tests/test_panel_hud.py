"""No GPU, learned detector, GPIO, motor command or remote clock guesses."""
import copy
import json
import unittest

from common.hud_geometry import feedforward_state, heading_tape_ticks
from common.hud_telemetry import HudCache, validate_message
from pc.hud_relay import HudSampler
from common.schemas import CamState, ControlDiagnostics, ControlEstimatorAxisDiagnostics, ControlTimingDiagnostics
from common.target_authority import TargetCatalog
from common.perception import PerceptionSnapshotV2


def diagnostic(seq=1, stamp=1_000_000_010):
    return ControlDiagnostics(observation_sequence=seq, intent_sequence=seq,
        created_monotonic_ns=stamp, reason='tracking', track_id=7,
        timing=ControlTimingDiagnostics(source_clock_domain='pc_monotonic'),
        yaw=ControlEstimatorAxisDiagnostics(estimator_enabled=True, feedforward_term_rad_s=.12, final_rate_rad_s=.2),
        pitch=ControlEstimatorAxisDiagnostics(estimator_enabled=True, feedforward_term_rad_s=-.04, final_rate_rad_s=-.1),
        aim_reference_norm=(.52, .48), target_center_norm=(.6, .4))


def snapshot(seq=1, session='sim-A', stamp=1_000_000_020):
    return PerceptionSnapshotV2.model_validate(dict(sequence=seq,
        frame=dict(frame_id=seq, source_time_ns=stamp, observed_time_ns=stamp,
            source_clock_domain='pc_monotonic', observation_clock_domain='pc_monotonic', tracker_session=session,
            width=1280, height=720),
        tracks=[dict(track_id=7, class_id='drone', confidence=1., missed_frames=0, box=dict(x=.55, y=.35, w=.1, h=.1))],
        assessments=[dict(track_id=7, distance_m=12., distance_src='width', threat_level='suspicious', engagement_rank=1)],
        selection=dict(track_id=7, source_frame_id=seq, applied_frame_id=seq, selected_time_ns=stamp,
            selection_clock_domain='pc_monotonic', policy='test')))


def pose(seq=1, session='sim-A', stamp=1_000_000_000):
    return CamState(frame_id=seq, src_ts_ms=1, pan=.2, tilt=.1, home_pan=0., home_tilt=0.,
        state_monotonic_ns=stamp, pan_sample_monotonic_ns=stamp, tilt_sample_monotonic_ns=stamp,
        target_catalog=TargetCatalog(source='sim', session=session, sequence=seq, created_monotonic_ns=stamp))


def message():
    sampler = HudSampler('sender-A')
    for name, value in (('pose', pose()), ('diagnostics', diagnostic()), ('perception', snapshot())):
        sampler.receive(name, value.model_dump_json(), 1_000_000_030)
    return sampler.message(1_000_000_040)


class HudContractTests(unittest.TestCase):
    def test_wire_contains_no_authority_and_accumulates_same_host_age(self):
        data = message()
        cache = HudCache()
        self.assertTrue(cache.accept(json.dumps(data), 1000.))
        view = cache.view(1000.01, source_session='sim-A')
        self.assertAlmostEqual(view.pose_age_s, .010000040)
        self.assertEqual(view.diagnostics['reason'], 'tracking')
        self.assertIsNone(cache.view(1000.3, source_session='sim-A').diagnostics)
        self.assertIn('transit unknown', view.age_basis)
        self.assertNotIn('control_activity', data['pose']['data'])

    def test_duplicates_and_new_envelopes_cannot_renew_frozen_samples(self):
        data = message()
        cache = HudCache()
        cache.accept(data, 0)
        self.assertFalse(cache.accept(data, .2))
        data = copy.deepcopy(data)
        data['sequence'] += 1
        cache.accept(data, .2)
        self.assertIsNone(cache.view(.26, source_session='sim-A').pose)

    def test_session_switch_clears_old_data_and_rejects_retired_sender(self):
        data = message()
        cache = HudCache()
        cache.accept(data, 0)
        self.assertIsNone(cache.view(.01, source_session='other').pose)
        changed = copy.deepcopy(data)
        changed.update(session='sender-B', source_session='sim-B')
        cache.accept(changed, .1)
        self.assertFalse(cache.accept(data, .11))
        self.assertIsNone(cache.view(.11, source_session='sim-A').diagnostics)

    def test_malformed_authority_nan_and_overlarge_data_are_rejected(self):
        data = message()
        for mutate in (lambda d: d.update(intent={}), lambda d: d['pose']['data'].update(pan=float('nan')),
                       lambda d: d['pose']['data'].update(pan='bad'),
                       lambda d: d['diagnostics']['data'].update(aim_reference_norm=[2., 0.]),
                       lambda d: d['perception']['data']['tracks'][0]['box'].update(x=.99)):
            bad = copy.deepcopy(data)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_message(bad)
        with self.assertRaises(ValueError):
            validate_message(' ' * 65537)

    def test_sampler_uses_measurement_age_and_rejects_other_source_session(self):
        sampler = HudSampler('sender')
        state = pose().model_copy(update={'pan_sample_monotonic_ns': 900_000_000})
        sampler.receive('pose', state.model_dump_json(), 1_000_000_030)
        sampler.receive('perception', snapshot(session='old').model_dump_json(), 1_000_000_030)
        data = sampler.message(1_000_000_040)
        self.assertAlmostEqual(data['pose']['age_ms'], 100.000040)
        self.assertIsNone(data['perception'])
        self.assertIsNone(sampler.message(1_300_000_000)['pose'])

    def test_shared_tapes_translate_fractionally(self):
        a = {t.value_deg: t.position_px for t in heading_tape_ticks(30, 135, 1280)}
        b = {t.value_deg: t.position_px for t in heading_tape_ticks(30.25, 135, 1280)}
        deltas = [b[k] - a[k] for k in a.keys() & b.keys()]
        self.assertTrue(all(d<0 for d in deltas))
        self.assertGreater(max(deltas)-min(deltas), .1)

    def test_controller_counter_restart_keeps_new_display_samples(self):
        sampler = HudSampler('sender')
        sampler.receive('pose', pose().model_dump_json(), 1_000_000_040)
        sampler.receive('diagnostics', diagnostic(seq=50).model_dump_json(), 1_000_000_040)
        cache = HudCache()
        cache.accept(sampler.message(1_000_000_040), 0.)
        new = diagnostic(seq=1, stamp=1_100_000_000).model_copy(update={'reason': 'manual'})
        sampler.receive('diagnostics', new.model_dump_json(), 1_100_000_010)
        cache.accept(sampler.message(1_100_000_020), .1)
        self.assertEqual(cache.view(.11, source_session='sim-A').diagnostics['reason'], 'manual')

    def test_bounded_zmq_wire_roundtrip_is_display_only(self):
        import zmq
        ctx = zmq.Context()
        push, pull = ctx.socket(zmq.PUSH), ctx.socket(zmq.PULL)
        try:
            pull.bind('inproc://hud-fixture')
            push.connect('inproc://hud-fixture')
            push.send_json(message())
            self.assertTrue(pull.poll(500))
            cache = HudCache()
            self.assertTrue(cache.accept(pull.recv(), 10000.))
            self.assertIsNotNone(cache.view(10000.01, source_session='sim-A').diagnostics)
            self.assertIsNone(cache.view(10000.3, source_session='sim-A').diagnostics)
        finally:
            push.close(0)
            pull.close(0)
            ctx.term()

    def test_feedforward_states_and_signs(self):
        data = diagnostic().model_dump(mode='json')
        self.assertEqual(feedforward_state(data, .01), ('active', .12, -.04))
        self.assertEqual(feedforward_state(data, .26)[0], 'stale')
        data['yaw'].update(desired_rate_pre_limit_rad_s=.5, desired_rate_post_limit_rad_s=.2)
        self.assertEqual(feedforward_state(data, .01)[0], 'limited')
        data['reason'] = 'manual'
        self.assertEqual(feedforward_state(data, .01)[0], 'inhibited')


class HudRenderTests(unittest.TestCase):
    def setUp(self):
        try:
            from rpi.display.hud import PanelHudState, PiHudRenderer
        except ModuleNotFoundError as exc:
            if exc.name == 'cairo':
                self.skipTest('Pi Cairo not installed')
            raise
        self.State, self.Renderer = PanelHudState, PiHudRenderer

    def test_reticle_colour_cache_mode_change_and_stale_geometry(self):
        from common.hud_telemetry import HudView
        renderer = self.Renderer()
        def render(kind, perception=True):
            data = {'tracks':[dict(track_id=7, class_id=kind, missed_frames=0,
                box=dict(x=.55,y=.3,w=.1,h=.1))], 'selected_id':7, 'sequence':1}
            view = HudView('CAMERA','A',perception=data if perception else None,
                diagnostics=diagnostic().model_dump(mode='json'),diagnostics_age_s=.01)
            return renderer.render(1280,720,view)[-1]
        yolo = render('drone')
        self.assertEqual(renderer.reticle_mode, 'YOLO')
        self.assertIn('parallax_cue', renderer.elements)
        roi = render('visual')
        self.assertEqual(renderer.reticle_mode, 'ROI')
        self.assertNotIn('parallax_cue', renderer.elements)
        self.assertNotEqual(roi.data, yolo.data)
        self.assertIs(render('visual'), roi)
        offset = (360-roi.y)*roi.stride + (653-roi.x)*4
        b,g,r,a = roi.data[offset:offset+4]
        self.assertGreater(a, 0)
        self.assertLessEqual(abs(b - a), 1)
        self.assertLessEqual(abs(g - 217*a/255), 1)
        self.assertEqual(r, 0)  # Cairo stores premultiplied anti-aliased BGRA.
        stale = render('visual',False)
        self.assertEqual(renderer.reticle_state, 'stale')
        self.assertNotEqual(stale.data, roi.data)
        self.assertNotIn('selection', renderer.elements)

    def test_complete_inventory_clipped_rectangles_and_no_mpc(self):
        cache = HudCache()
        cache.accept(message(), 0.)
        renderer = self.Renderer()
        for width, height in ((1280, 720), (640, 360), (1920, 1080)):
            images = renderer.render(width, height, cache.view(.01, source_session='sim-A'))
            self.assertEqual(set(renderer.elements), {'center_reticle', 'tracking_mode', 'attitude',
                'control_status', 'parallax_cue', 'feedforward_indicator', 'freshness', 'source_status'})
            self.assertEqual(renderer.feedforward, 'active')
            for im in images:
                self.assertGreaterEqual(im.x, 0)
                self.assertGreaterEqual(im.y, 0)
                self.assertLessEqual(im.x + im.width, width)
                self.assertLessEqual(im.y + im.height, height)
                self.assertTrue(any(im.data[3::4]))

    def test_receiver_never_draws_target_trails_and_hides_stale_geometry(self):
        cache = HudCache()
        cache.accept(message(), 0.)
        renderer = self.Renderer()
        renderer.render(1280, 720, cache.view(.01, source_session='sim-A'))
        renderer.render(1280, 720, cache.view(.02, source_session='sim-A'))
        self.assertFalse(hasattr(renderer,'trails'))
        self.assertNotIn('track_history',renderer.elements)
        renderer.render(1280, 720, cache.view(.3, source_session='sim-A'))
        self.assertNotIn('parallax_cue', renderer.elements)
        self.assertNotIn('attitude', renderer.elements)
        self.assertNotIn('tracks',renderer.elements)

    def test_camera_and_sim_modes_never_mix(self):
        state = self.State()
        state.sim.accept(message(), 0.)
        state.on_snapshot(snapshot().model_dump(mode='json'), 0.)
        self.assertIsNotNone(state.view(.01, mode='sim', source_session='sim-A').pose)
        self.assertIsNone(state.view(.01, mode='camera').pose)
        self.assertIsNone(state.view(.01, mode='sim', source_session='wrong').diagnostics)
        self.assertIsNone(state.view(.01, mode='sim', source_session='sim-A', video_live=False).pose)
        camera = snapshot().model_dump(mode='json')
        camera['frame'].update(source_clock_domain='jetson_monotonic', tracker_session='cam-A')
        state.on_snapshot(camera, .1)
        self.assertIsNone(state.view(.11, mode='sim', source_session='sim-A').pose)
        self.assertIsNotNone(state.view(.11, mode='camera').perception)
        self.assertIsNone(state.view(.4, mode='camera').perception)

    def test_camera_full_hud_uses_real_sample_ages_not_publication_only(self):
        state = self.State()
        camera = snapshot().model_dump(mode='json')
        camera['frame'].update(source_clock_domain='jetson_monotonic', tracker_session='cam-A')
        state.on_snapshot(camera, 1000.)
        diag = diagnostic().model_dump(mode='json')
        diag['timing']['source_clock_domain'] = 'jetson_monotonic'
        state.on_camera('diagnostics', diag, 1000.)
        measured = pose().model_dump(mode='json')
        measured.update(hardware_available=True)
        state.on_camera('pose', measured, 1000.)
        renderer = self.Renderer()
        renderer.render(1280, 720, state.view(1000.01, mode='camera'))
        self.assertIn('attitude', renderer.elements)
        self.assertIn('parallax_cue', renderer.elements)
        self.assertEqual(renderer.feedforward, 'active')
        measured.update(state_monotonic_ns=2_000_000_000, pan_sample_monotonic_ns=1_000_000_000, tilt_sample_monotonic_ns=1_000_000_000)
        state.on_camera('pose', measured, 1000.02)
        self.assertIsNone(state.view(1000.03, mode='camera').pose)
        self.assertIsNone(state.view(1000.03, mode='camera', agent_fresh=False).diagnostics)

    def test_video_metadata_duplicates_do_not_renew_source_freshness(self):
        state = self.State()
        data = snapshot().model_dump(mode='json')
        state.on_snapshot(data, 0.)
        state.sim.accept(message(), 0.)
        state.on_snapshot(data, .24)
        self.assertEqual(state.view(.26, mode='sim', source_session='sim-A').source, 'UNAVAILABLE')


if __name__ == '__main__':
    unittest.main()
