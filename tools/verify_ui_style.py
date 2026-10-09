"""Offline UI pictures and CPU render benchmark; no sockets/GPU/control owners.

python -m tools.verify_ui_style --role pc --output /outside/repo/ui-qa
Synthetic GPU metadata is painted into a fixture, NOT a native GPU qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ui_style as theme
from common.hud_telemetry import HudView
from common.perception import PerceptionSnapshotV2
from common.schemas import CamState, ControlDiagnostics, ControlEstimatorAxisDiagnostics, ControlTimingDiagnostics
from jetson.deepstream.pipeline import _decorate_osd_metadata

ROOT=Path(__file__).resolve().parents[1]
SCENARIOS=('sim','yolo','roi','coasting','stale','missing','manual','menu','emergency','edge',
           'pitch_up','pitch_down','heading_wrap','multi_target','slow_video','fire_menu','estop_menu')


def fixture(name,seq=1):
    roi=name=='roi'
    box=dict(x=.98 if name=='edge' else .55,y=.01 if name=='edge' else .43,w=.02 if name=='edge' else .06,h=.06)
    snapshot=PerceptionSnapshotV2.model_validate(dict(sequence=seq,
        frame=dict(frame_id=seq,source_time_ns=1_000_000_000+seq,
            observed_time_ns=1_000_000_000+seq,source_clock_domain='pc_monotonic' if name=='sim' else 'jetson_monotonic',
            observation_clock_domain='jetson_monotonic',width=1280,height=720,tracker_session='synthetic-style-v1'),
        tracks=[dict(track_id=7,class_id='visual' if roi else 'drone',confidence=.9,
            missed_frames=1 if name=='coasting' else 0,box=box)],
        assessments=[] if roi else [dict(track_id=7,distance_m=12.,distance_src='width',threat_level='suspicious',engagement_rank=1)],
        selection=dict(track_id=7,source_frame_id=seq,applied_frame_id=seq,selected_time_ns=1_000_000_000+seq,
            selection_clock_domain='jetson_monotonic',policy='synthetic_fixture')))
    if name=='multi_target':
        data=snapshot.model_dump(mode='json')
        data['tracks'].append(dict(track_id=8,class_id='person',confidence=.8,missed_frames=0,
            box=dict(x=.25,y=.4,w=.08,h=.15)))
        data['assessments'].append(dict(track_id=8,distance_m=20.,distance_src='height',
            threat_level='benign',engagement_rank=2))
        snapshot=PerceptionSnapshotV2.model_validate(data)
    pose=CamState(frame_id=seq,src_ts_ms=1,pan=.2+seq*.0001,tilt=.1+seq*.00007,home_pan=0,home_tilt=0)
    if name in ('pitch_up','pitch_down'):
        pose=pose.model_copy(update={'tilt':math.radians(20 if name=='pitch_up' else -20)})
    if name=='heading_wrap':
        pose=pose.model_copy(update={'pan':math.radians(359.9)})
    diag=ControlDiagnostics(observation_sequence=seq,intent_sequence=seq,created_monotonic_ns=1_000_000_000+seq,
        reason='manual' if name=='manual' else 'safety_hold' if name=='emergency' else 'tracking',track_id=7,
        timing=ControlTimingDiagnostics(source_clock_domain='pc_monotonic' if name=='sim' else 'jetson_monotonic'),
        yaw=ControlEstimatorAxisDiagnostics(estimator_enabled=True,feedforward_term_rad_s=.12,final_rate_rad_s=.20),
        pitch=ControlEstimatorAxisDiagnostics(estimator_enabled=True,feedforward_term_rad_s=-.04,final_rate_rad_s=-.1),
        aim_reference_norm=(.51,.5),target_center_norm=(.58,.46))
    age=.3 if name=='stale' else .01+(seq%20)/1000
    source='SIM-ONLY' if name=='sim' else 'CAMERA'
    perception=snapshot.model_dump(mode='json')
    compact=dict(sequence=seq,tracks=perception['tracks'],assessments=perception['assessments'],selected_id=7)
    view=HudView(source,'synthetic-style-v1',pose=None if name in ('stale','missing') else pose.model_dump(mode='json'),
        diagnostics=None if name in ('stale','missing') else diag.model_dump(mode='json'),
        perception=None if name in ('stale','missing') else compact,
        pose_age_s=age,diagnostics_age_s=age,perception_age_s=age)
    return snapshot,pose,diag,age,view


class _Colour:
    def set(self,*rgba): self.rgba=rgba


def annotated_background(width,height,snapshot,pose=None):
    """Paint mocked GPU metadata once. Never include it in receiver timings."""
    records=[]
    def acquire(batch):
        return NS(rect_params=[NS(border_color=_Colour(),bg_color=_Colour()) for _ in range(16)],
            text_params=[NS(font_params=NS(font_color=_Colour()),text_bg_clr=_Colour()) for _ in range(16)])
    pyds=NS(nvds_acquire_display_meta_from_pool=acquire,
        nvds_add_display_meta_to_frame=lambda frame,meta:records.append(meta))
    frame=NS(source_frame_width=width,source_frame_height=height)
    _decorate_osd_metadata(pyds,None,frame,[],accepted_tracks=snapshot.tracks,target_track_id=7,
                          assessments=snapshot.assessments,
                          target_assessment=next(iter(snapshot.assessments),None),
                          infer_stage_ms=14.,pipeline_fps=60.)
    picture=Image.new('RGB',(width,height),(60,68,75))
    paint=ImageDraw.Draw(picture)
    # Guaranteed zero-elevation horizon in a zero-roll rectilinear view.
    pitch=0 if pose is None else pose.tilt-(pose.home_tilt or 0)
    horizon=round(height/2+height/2*math.tan(pitch)/math.tan(math.radians(73/2)))
    horizon=max(0,min(height,horizon))
    paint.rectangle((0,horizon,width,height),fill=(90,94,98))
    for x in range(0,width,max(1,width//16)): paint.line((x,horizon,x,height),fill=(110,115,120))
    for record in records:
        for rect in record.rect_params[:record.num_rects]:
            bounds=(rect.left,rect.top,rect.left+rect.width,rect.top+rect.height)
            if rect.has_bg_color:
                paint.rectangle(bounds,fill=tuple(round(v*255) for v in rect.bg_color.rgba[:3]))
            else:
                colour=tuple(round(v*255) for v in rect.border_color.rgba[:3])
                paint.rectangle(bounds,outline=colour,width=rect.border_width)
        for label in record.text_params[:record.num_labels]:
            face=ImageFont.truetype(str(ROOT/'assets/fonts/DejaVuSans.ttf'),label.font_params.font_size,
                                   layout_engine=ImageFont.Layout.BASIC)
            x,y=label.x_offset,label.y_offset
            extent=paint.textbbox((x,y),label.display_text,font=face,anchor='lt')
            if label.set_bg_clr:
                paint.rectangle((extent[0]-3,extent[1]-3,extent[2]+3,extent[3]+3),fill=theme.PANEL)
            paint.text((x,y),label.display_text,font=face,fill=tuple(round(v*255) for v in label.font_params.font_color.rgba[:3]),anchor='lt')
    return np.asarray(picture)[:,:,::-1].copy()


def pi_images(renderer,width,height,view,name):
    from rpi.display.render import render,StatusBar
    from rpi.display.menu import MenuView,MenuRow
    menu=MenuView(title='System',rows=(MenuRow('Mode','>'),MenuRow('Recording','on'),MenuRow('Target type','>')),cursor=0,lines=(),depth=1) if name in ('menu','fire_menu','estop_menu') else None
    mode='E-STOP' if name in ('emergency','estop_menu') else 'MANUAL' if name=='manual' else 'NO PANEL' if name=='missing' else 'ARMED'
    alerts=['EMERGENCY STOP'] if name in ('emergency','estop_menu') else ['FIRE BUTTON'] if name=='fire_menu' else ['LOW VIDEO FPS'] if name=='slow_video' else []
    return renderer.render(width,height,view)+render(width,height,
        bar=StatusBar(mode,('PLANNER','REC')),alerts=alerts,menu=menu,
        fire_state='FIRE BUTTON' if name=='fire_menu' else 'fire off',
        safety_state='E-stop ON' if name in ('emergency','estop_menu') else 'E-stop clear')


def composite(background,images):
    frame=background.copy()
    for tile in images:
        rgba=np.frombuffer(tile.data,np.uint8).reshape(tile.height,tile.stride)[:,:tile.width*4].reshape(tile.height,tile.width,4)
        x0,y0=max(tile.x,0),max(tile.y,0)
        x1,y1=min(tile.x+tile.width,frame.shape[1]),min(tile.y+tile.height,frame.shape[0])
        if x0>=x1 or y0>=y1: continue
        layer=rgba[y0-tile.y:y1-tile.y,x0-tile.x:x1-tile.x]
        roi=frame[y0:y1,x0:x1]
        roi[:]=np.rint(layer[:,:,:3]+roi*(1-layer[:,:,3:4].astype(np.float32)/255)).clip(0,255).astype(np.uint8)
    return frame


def verify(args):
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    if args.role=='pi':
        from rpi.display.hud import PiHudRenderer
        make=lambda:PiHudRenderer(hfov_deg=135,vfov_deg=73)
    else:
        from pc.v2_hud import V2HudRenderer
        make=lambda:V2HudRenderer(hfov_deg=135,vfov_deg=73,authority_label='SIM')
    report=dict(role=args.role,baseline_commit=args.baseline_commit,dirty_source=True,
        fixture='synthetic-ui-v2 sparse ownership; no detector/no GPU/no GPIO/no sockets',measurements=[],screenshots=[],source_sha256={})
    for file in ('common/ui_style.py','common/hud_geometry.py','pc/ui_text.py','pc/v2_hud.py','pc/ui.py',
                 'rpi/display/render.py','rpi/display/hud.py','jetson/deepstream/pipeline.py','jetson/visual_lock/live_trial.py',
                 'common/ui_fonts.py','tools/verify_ui_style.py','assets/fonts/DejaVuSans.ttf','assets/fonts/DejaVuSans-Bold.ttf','assets/fonts/DejaVuSansMono.ttf'):
        report['source_sha256'][file]=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    if args.role=='pi':
        from common.ui_fonts import configure_fonts
        match=configure_fonts()['regular']
        report['process_font']=str(match)
        report['process_font_sha256']=hashlib.sha256(match.read_bytes()).hexdigest()
        if report['process_font_sha256']!=report['source_sha256']['assets/fonts/DejaVuSans.ttf']:
            raise RuntimeError('Cairo system font differs from bundled font; not qualified')
    for width,height in ((640,360),(1280,720),(1920,1080),(800,600)):
        renderer=make()
        for name in SCENARIOS:
            snapshot,pose,diag,age,view=fixture(name)
            background=annotated_background(width,height,snapshot,pose)
            if args.role=='pi':
                image=composite(background,pi_images(renderer,width,height,view,name))
                elements=renderer.elements
            else:
                from pc.ui_text import text
                renderer.authority_label=view.source
                image=background.copy()
                state=renderer.render(image,snapshot=None if name=='missing' else snapshot,
                    cam_state=None if name in ('stale','missing') else pose,control_cmd=None,
                    control_diagnostics=None if name=='missing' else diag,
                    cam_state_age_s=age,diagnostics_age_s=age,perception_age_s=age)
                r=theme.HudLayout(width,height).receiver_status
                if name=='slow_video':
                    text(image,'LOW VIDEO FPS',(r.x,r.y+15*height/720),theme.bgr(theme.AMBER),
                         size=theme.pixel_size(theme.SMALL_TEXT_PX,height),background=True,max_width=r.width)
                elements=state.elements
            filename=f'{name}-{width}x{height}.png'
            Image.fromarray(image[:,:,::-1]).save(output/filename)
            affected=float(np.mean(np.any(image!=background,axis=2)))
            report['screenshots'].append(dict(file=filename,elements=elements,receiver_affected_fraction=affected))
        snapshot,pose,diag,age,view=fixture('sim')
        background=annotated_background(width,height,snapshot,pose)
        renderer=make()
        timings=[]
        for i in range(args.frames+20):
            snapshot,pose,diag,age,view=fixture('sim',i+1)
            started=time.perf_counter_ns()
            if args.role=='pi':
                pi_images(renderer,width,height,view,'sim')  # Native compositor owns the video blend.
            else:
                image=background.copy()
                renderer.render(image,snapshot=snapshot,cam_state=pose,control_cmd=None,
                    control_diagnostics=diag,cam_state_age_s=age,diagnostics_age_s=age,perception_age_s=age)
                from pc.ui_text import text
                r=theme.HudLayout(width,height).receiver_status
            elapsed=(time.perf_counter_ns()-started)/1e6
            if i>=20: timings.append(elapsed)
        p95=float(np.percentile(timings,95))
        report['measurements'].append(dict(width=width,height=height,frames=args.frames,
            median_ms=statistics.median(timings),p95_ms=p95,max_ms=max(timings),
            gate_ms=15.,passed=p95<=15.))
    report['passed']=all(r['passed'] for r in report['measurements'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('screenshots','source_sha256')},indent=2))
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=('pc','pi'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=300)
    parser.add_argument('--baseline-commit',default='unrecorded')
    parser.add_argument('--tests-only',action='store_true',help='contract tests only; no benchmark or video owner')
    args=parser.parse_args()
    if args.tests_only:
        import importlib.util
        import types
        package=types.ModuleType('tests')
        package.__path__=[str(ROOT/'tests')]
        sys.modules['tests']=package
        args.output.mkdir(parents=True,exist_ok=False)
        if importlib.util.find_spec('pytest') is None:
            import unittest
            suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_panel_hud','tests.test_hud_design','tests.test_video_health','tests.test_operator_menu',
                                                                'tests.test_rpi_gpio_config'])
            with (args.output/'tests.txt').open('w') as log:
                result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
            print((args.output/'tests.txt').read_text())
            return 0 if result.wasSuccessful() else 1
        import pytest
        names=['test_ui_style','test_hud_design','test_video_health','test_tracking_reticle','test_v2_ui_hud','test_v2_hud_attitude',
            'test_panel_hud','test_panel_indicators','test_control_activity','test_deepstream_accepted_osd',
            'test_deepstream_metadata_adapter','test_deepstream_pipeline_split','test_deepstream_runtime',
            'test_ui_return_video','test_ui_overlay','test_operator_display','test_visual_lock_live_trial',
            'test_serial_io_ipc','test_rpi_gpio_config']
        if args.role=='pi': names.append('test_operator_menu')
        import contextlib
        with (args.output/'tests.txt').open('w') as log,contextlib.redirect_stdout(log):
            code=pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(args.output/'tmp'),
                              *[str(ROOT/'tests'/f'{n}.py') for n in names]])
        print((args.output/'tests.txt').read_text())
        return code
    if not 30<=args.frames<=3000: parser.error('frames must be 30..3000')
    return verify(args)


if __name__=='__main__': raise SystemExit(main())
