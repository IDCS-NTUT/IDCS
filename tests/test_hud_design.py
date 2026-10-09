"""Design contracts independent of learned detection and live devices."""
import math
import unittest
from types import SimpleNamespace as NS

import numpy as np

from common import ui_style as theme
from common.hud_geometry import elevation_tape_ticks, heading_tape_ticks, tracking_reticle_style


class TapeProjectionTests(unittest.TestCase):
    def test_pitch_matches_the_actual_opengl_camera_projection(self):
        from pc.renderers._common import camera_basis_from_orientation,projection_matrix,view_matrix
        for pitch in (-20.,0.,20.):
            forward,right,up = camera_basis_from_orientation((0.,pitch,0.))
            matrix = projection_matrix(73.,1280/720,.1,1000.,fov_x_deg=135.) @ view_matrix((0,0,0),forward,up)
            ticks = {t.value_deg:t.position_px for t in elevation_tape_ticks(pitch,73,8,712,frame_height=720)}
            for elevation in (-10.,0.,10.):
                angle = math.radians(elevation)
                clip = matrix @ np.array((0,100*math.sin(angle),-100*math.cos(angle),1.))
                y = 360*(1-clip[1]/clip[3])
                self.assertAlmostEqual(ticks[elevation],y,places=4)

    def test_pitch_principal_point_and_horizon_at_every_resolution(self):
        for width,height in ((640,360),(1280,720),(1920,1080),(800,600)):
            layout = theme.HudLayout(width,height)
            for pitch in (-20.,-10.,0.,10.,20.):
                ticks = {t.value_deg:t.position_px for t in elevation_tape_ticks(
                    pitch,73,layout.elevation_top,layout.elevation_bottom,frame_height=height)}
                self.assertAlmostEqual(ticks[pitch],height/2)
                expected = height/2 + height/2*math.tan(math.radians(pitch))/math.tan(math.radians(73/2))
                self.assertAlmostEqual(ticks[0],expected)
                # Cropping the visible strip cannot rescale ticks or move zero.
                cropped = {t.value_deg:t.position_px for t in elevation_tape_ticks(
                    pitch,73,50,height-100,frame_height=height)}
                for value in ticks.keys() & cropped.keys():
                    self.assertAlmostEqual(ticks[value],cropped[value])

    def test_heading_center_wrap_and_corner_clipping_do_not_rescale(self):
        for yaw in (0.,10.,350.,360.):
            all_ticks = {t.value_deg:t.position_px for t in heading_tape_ticks(yaw,135,1280)}
            clipped = {t.value_deg:t.position_px for t in heading_tape_ticks(yaw,135,1280,right_px=1210)}
            self.assertAlmostEqual(all_ticks[yaw],640)
            for value in clipped:
                self.assertEqual(clipped[value],all_ticks[value])
                self.assertLessEqual(clipped[value],1210)
        near = {t.value_deg:t.position_px for t in heading_tape_ticks(359.9,135,1280)}
        self.assertLess(abs(near[360]-640),1)

    def test_full_axis_strips_and_corner_never_overlap(self):
        for w,h in ((640,360),(1280,720),(1920,1080),(800,600)):
            layout = theme.HudLayout(w,h)
            self.assertFalse(layout.heading.overlaps(layout.elevation))
            self.assertEqual(layout.heading.right,layout.elevation.x)
            self.assertEqual(layout.elevation.height,h)
            self.assertGreater(layout.heading.width,.93*w)
            for r in (layout.status,layout.receiver_status,layout.feedforward):
                self.assertFalse(r.overlaps(layout.elevation))
                self.assertFalse(r.overlaps(layout.heading))

    def test_invalid_projection_cannot_emit_bad_ticks(self):
        for fov in (0,180,float('nan')):
            with self.assertRaises(ValueError):
                elevation_tape_ticks(0,fov,8,712,frame_height=720)
            with self.assertRaises(ValueError):
                heading_tape_ticks(0,fov,1280)


class OwnershipTests(unittest.TestCase):
    def test_compact_summary_does_not_repeat_source_or_target_facts(self):
        from tests.test_panel_hud import diagnostic,message
        from common.hud_telemetry import HudCache
        cache = HudCache()
        cache.accept(message(),0)
        view = cache.view(.01,source_session='sim-A')
        style = tracking_reticle_style(view.perception,mode_hint='sim')
        text = theme.operational_summary(view.source,style,True,diagnostic().model_dump(mode='json'),.01)
        self.assertEqual(text,'SIM | TRACKING | PID+FF | LATEST HUD')
        for banned in ('#7','12.0m','rank','rad/s','Y ','P '):
            self.assertNotIn(banned,text)
        details = theme.hud_detail_lines(view)
        self.assertIn('NOT joined to video',details[0])
        self.assertTrue(any('cmd +0.200 FF +0.120' in line for line in details))
        self.assertEqual(theme.UI_OWNERS['target_range_threat_rank'],'jetson')

    def test_gpu_has_no_engineering_banner_and_owns_target_facts_once(self):
        from tests.test_panel_hud import snapshot
        from jetson.deepstream.pipeline import _decorate_osd_metadata
        # Dependency-free mock: Pi's unittest gate does not require pytest.
        class Colour:
            def set(self,*rgba): self.rgba=rgba
        added = []
        def acquire(batch):
            return NS(rect_params=[NS(border_color=Colour()) for _ in range(16)],
                text_params=[NS(font_params=NS(font_color=Colour()),text_bg_clr=Colour()) for _ in range(16)])
        pyds = NS(nvds_acquire_display_meta_from_pool=acquire,
            nvds_add_display_meta_to_frame=lambda frame,meta:added.append(meta))
        frame,raw = NS(source_frame_width=1280,source_frame_height=720),[]
        perception = snapshot()
        _decorate_osd_metadata(pyds,None,frame,raw,accepted_tracks=perception.tracks,
            target_track_id=7,target_assessment=perception.assessments[0],
            class_labels={0:'drone'},infer_stage_ms=13,pipeline_fps=60)
        labels = [meta.text_params[i].display_text for meta in added for i in range(meta.num_labels)]
        self.assertEqual(len(labels),1)
        self.assertEqual(labels[0].count('id=7'),1)
        self.assertIn('r=12.0m',labels[0])
        self.assertFalse(any('infer ' in t or ' fps' in t for t in labels))

    def test_pc_never_draws_a_second_target_and_central_video_is_clear(self):
        from pc.v2_hud import V2HudRenderer
        from tests.test_panel_hud import snapshot
        image = np.full((720,1280,3),40,np.uint8)
        before = image.copy()
        report = V2HudRenderer(hfov_deg=135,vfov_deg=73).render(image,
            snapshot=snapshot(),cam_state=None,control_cmd=None,perception_age_s=.01)
        self.assertFalse({'tracks','track_history','selection','range','threat','rank'} & set(report.elements))
        # Target occupies (.55,.35)..(.65,.45); receiver leaves it untouched.
        self.assertTrue(np.array_equal(image[250:330,700:840],before[250:330,700:840]))
        self.assertLess(np.count_nonzero(np.any(image!=before,axis=2))/image.shape[0]/image.shape[1],.04)

    def test_compatibility_command_status_preserves_its_controller_mode(self):
        from unittest.mock import patch
        from pc.v2_hud import V2HudRenderer
        from tests.test_v2_ui_hud import _command,_snapshot
        labels = []
        with patch('pc.v2_hud.ui_text',side_effect=lambda frame,value,*a,**k: labels.append(value)):
            V2HudRenderer(hfov_deg=135,vfov_deg=73).render(np.zeros((720,1280,3),np.uint8),
                snapshot=_snapshot(),cam_state=None,control_cmd=_command())
        self.assertTrue(any(' | MPC | ' in text for text in labels))
        self.assertFalse(any(' | PID | ' in text for text in labels))


class PiOwnershipTests(unittest.TestCase):
    def setUp(self):
        try:
            from rpi.display.hud import PiHudRenderer
        except ModuleNotFoundError as exc:
            if exc.name=='cairo':
                self.skipTest('Cairo is qualified on target Linux devices')
            raise
        self.Renderer = PiHudRenderer

    def test_pi_has_no_target_tiles_or_long_status_stack(self):
        from common.hud_telemetry import HudCache
        from tests.test_panel_hud import message
        cache = HudCache()
        cache.accept(message(),0)
        renderer = self.Renderer()
        images = renderer.render(1280,720,cache.view(.01,source_session='sim-A'))
        self.assertFalse({'tracks','track_history','selection','range','threat','rank','parallax_status'} & set(renderer.elements))
        layout = theme.HudLayout(1280,720)
        for tile in images[:-1]:
            # Fully transparent tile space does not count as obscuring video.
            self.assertLessEqual(tile.x+tile.width,1280)
            self.assertLessEqual(tile.y+tile.height,720)
        self.assertLess(layout.status.height+layout.receiver_status.height,50)
        self.assertFalse(hasattr(renderer,'trails'))

    def test_console_row_is_at_bottom_and_does_not_cross_pitch_strip(self):
        from rpi.display.render import render,StatusBar
        layout = theme.HudLayout(800,600)
        tiles = render(800,600,bar=StatusBar('ARMED',('30 fps','PLANNER REC')),
                       alerts=[],menu=None,fire_state='fire off')
        self.assertEqual(len(tiles),1)  # No extra fire chip with the same state.
        tile = tiles[0]
        self.assertGreater(tile.y,550)
        self.assertLessEqual(tile.x+tile.width,layout.feedforward.x)

    def test_single_fire_indicator_even_when_console_row_is_off(self):
        from rpi.display.render import render,StatusBar
        layout = theme.HudLayout(1280,720)
        images = render(1280,720,bar=None,alerts=[],menu=None,fire_state='fire ready')
        self.assertEqual(len(images),1)
        fire = theme.Rect(images[0].x,images[0].y,images[0].width,images[0].height)
        self.assertFalse(fire.overlaps(layout.status))
        self.assertFalse(fire.overlaps(layout.feedforward))
        images = render(1280,720,bar=StatusBar('ARMED',()),alerts=['FIRE BUTTON'],
                        menu=None,fire_state='FIRE BUTTON')
        self.assertEqual(len(images),2)  # Console state plus one alert, no fire chip.


if __name__=='__main__':
    unittest.main()
