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
    def test_rank_badges_are_corner_fills_only_for_multiple_accepted_targets(self):
        from common.perception import TrackAssessmentV2,PerceptionTrackV2,NormalizedBoxV2
        from jetson.deepstream.pipeline import _decorate_osd_metadata
        class Colour:
            def set(self,*rgba):self.rgba=rgba
        added=[]
        def acquire(batch):
            return NS(rect_params=[NS(border_color=Colour(),bg_color=Colour()) for _ in range(16)],
                text_params=[NS(font_params=NS(font_color=Colour()),text_bg_clr=Colour()) for _ in range(16)])
        pyds=NS(nvds_acquire_display_meta_from_pool=acquire,
            nvds_add_display_meta_to_frame=lambda frame,meta:added.append(meta))
        frame=NS(source_frame_width=1280,source_frame_height=720)
        tracks=tuple(PerceptionTrackV2(track_id=i,class_id='0',confidence=.8,missed_frames=0,
            box=NormalizedBoxV2(x=.1+i*.02,y=.3,w=.1,h=.1)) for i in range(35))
        assessments=tuple(TrackAssessmentV2(track_id=i,engagement_rank=i,
            threat_level=('benign','suspicious','threatening')[i%3]) for i in range(35))
        kwargs=dict(class_labels={0:'drone'},assessments=assessments)
        _decorate_osd_metadata(pyds,None,frame,[],accepted_tracks=tracks[:1],**kwargs)
        self.assertEqual(len(added),1)
        self.assertNotIn('R0',added[0].text_params[0].display_text)
        added.clear()
        _decorate_osd_metadata(pyds,None,frame,[],accepted_tracks=tracks,**kwargs)
        badges=added[3:]
        self.assertEqual([m.num_rects for m in badges],[16,16,3])
        for i in range(35):
            meta=badges[i//16];rect=meta.rect_params[i%16];text=meta.text_params[i%16]
            self.assertEqual(text.display_text,str(i))
            self.assertEqual((rect.border_width,rect.has_bg_color),(1,1))
            self.assertEqual(rect.border_color.rgba,rect.bg_color.rgba)
            self.assertEqual(rect.bg_color.rgba,(*theme.unit_rgb((theme.GREEN,theme.AMBER,theme.RED)[i%3]),1.))
            box=tracks[i].box
            self.assertAlmostEqual(rect.left+rect.width,(box.x+box.w)*1280)
            self.assertAlmostEqual(rect.top+rect.height,(box.y+box.h)*720)
            self.assertLessEqual(text.x_offset+len(str(i))*text.font_params.font_size*.7,rect.left+rect.width)
        added.clear()
        _decorate_osd_metadata(pyds,None,frame,[],accepted_tracks=tracks[:2],class_labels={0:'drone'})
        self.assertEqual(len(added),1)  # No invented rank for missing assessments.

        # Tiny/edge corners must never emit a number that cannot fit its fill.
        added.clear()
        tiny = tracks[0].model_copy(update={'box':NormalizedBoxV2(x=.1,y=0,w=.1,h=.014)})
        _decorate_osd_metadata(pyds,None,frame,[],accepted_tracks=(tiny,tracks[1]),**kwargs)
        self.assertEqual(sum(m.num_rects for m in added),3)  # 2 boxes, only 1 readable badge.
        self.assertEqual(added[-1].text_params[0].display_text,'1')

    def test_compact_target_labels_preserve_facts_and_coasting_identity(self):
        from common.perception import TrackAssessmentV2
        from jetson.deepstream.pipeline import _compact_track_label,_target_osd_suffix
        assessment = TrackAssessmentV2(track_id=7,distance_m=12.,distance_src='width',
            threat_level='suspicious',engagement_rank=1)
        original = assessment.model_dump()
        label = _compact_track_label('drone',7,.8,coasting=False,selected=True,
                                     assessment=assessment)
        self.assertEqual(label,'TGT drone 12 m 0.80')
        self.assertLess(len(label),len('TARGET drone 0.80 id=7 r=12.0m suspicious rank=1'))
        self.assertEqual(_compact_track_label('drone',7,.8,coasting=True,selected=True,
            assessment=assessment),'TGT drone 12 m COAST')
        self.assertEqual(_compact_track_label('person',0,.8,coasting=False,selected=False,
            assessment=assessment),'person 0.80')
        self.assertEqual(_compact_track_label('class=4',None,.8,coasting=False,selected=False),
            'class=4 0.80')
        self.assertEqual(_target_osd_suffix(None),'')
        self.assertEqual(_target_osd_suffix(TrackAssessmentV2(track_id=0,engagement_rank=0)),
            '')
        self.assertEqual(assessment.model_dump(),original)
        for risk in ('benign','suspicious','threatening'):
            changed = assessment.model_copy(update={'threat_level':risk})
            self.assertEqual(_target_osd_suffix(changed),' 12 m')
        fractional = assessment.model_copy(update={'distance_m':12.4})
        self.assertEqual(_target_osd_suffix(fractional),' 12 m')
        self.assertEqual(fractional.distance_m,12.4)
        self.assertNotIn('~',_target_osd_suffix(fractional))

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
        self.assertNotIn('#',labels[0])
        self.assertIn('12 m',labels[0])
        self.assertNotIn('R1',labels[0])
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

    def test_compatibility_command_status_uses_control_label(self):
        from unittest.mock import patch
        from pc.v2_hud import V2HudRenderer
        from tests.test_v2_ui_hud import _command,_snapshot
        labels = []
        with patch('pc.v2_hud.ui_text',side_effect=lambda frame,value,*a,**k: labels.append(value)):
            V2HudRenderer(hfov_deg=135,vfov_deg=73).render(np.zeros((720,1280,3),np.uint8),
                snapshot=_snapshot(),cam_state=None,control_cmd=_command())
        self.assertTrue(any(' | CONTROL | ' in text for text in labels))
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

    def test_safety_strip_and_faults_render_above_open_menu(self):
        from unittest.mock import patch
        from rpi.display import render as module
        from rpi.display.menu import MenuView
        order=[]
        image=module.OverlayImage(0,0,1,1,4,b'\x00\x00\x00\xff')
        def layer(name):
            def draw(*args):order.append(name);return image
            return draw
        with patch.object(module,'_menu',side_effect=layer('menu')),\
             patch.object(module,'_status_bar',side_effect=layer('safety')),\
             patch.object(module,'_alert',side_effect=layer('alert')):
            module.render(640,360,bar=None,alerts=['EMERGENCY STOP','VIDEO LOST'],
                menu=MenuView('Menu',(),None,(),1),fire_state='fire off',safety_state='E-stop ON')
        self.assertEqual(order,['menu','safety','alert'])


if __name__=='__main__':
    unittest.main()
