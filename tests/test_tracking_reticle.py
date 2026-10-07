"""Deterministic HUD identity/state tests: no learned detector or GPU owner."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from common.hud_geometry import perception_progresses, tracking_reticle_style
from tests.test_v2_ui_hud import _snapshot, _command, _diagnostics
from pc.v2_hud import V2HudRenderer


@pytest.mark.parametrize('kind,hint,mode,rgb', [
    ('drone', None, 'YOLO', (0,255,0)),
    ('visual', None, 'ROI', (0,217,255)),
    ('drone', 'sim', 'SIM', (0,255,0)),
    ('visual', 'sim', 'ROI', (0,217,255)),
])
def test_shared_identity(kind, hint, mode, rgb):
    data = {'tracks': [dict(track_id=7, class_id=kind, missed_frames=0)],
            'selection': {'track_id': 7}}
    style = tracking_reticle_style(data, mode_hint=hint)
    assert (style.mode, style.state, style.rgb) == (mode, 'tracking', rgb)
    compact = dict(data, selected_id=7)
    compact.pop('selection')
    assert tracking_reticle_style(compact, mode_hint=hint) == style


@pytest.mark.parametrize('state,rgb', [('acquiring',(255,191,0)), ('uncertain',(255,191,0)),
                                      ('idle',(140,140,140)), ('lost',(140,140,140))])
def test_explicit_roi_lifecycle_without_inventing_selection(state, rgb):
    style = tracking_reticle_style(None, mode_hint='roi', state_hint=state)
    assert (style.label, style.rgb) == (f'ROI {state.upper()}', rgb)
    assert tracking_reticle_style(None, mode_hint='roi', state_hint='tracking').state == 'uncertain'


def test_missed_stale_empty_and_invalid_hints():
    data = dict(tracks=[dict(track_id=7, class_id='visual', missed_frames=1)], selected_id=7)
    assert tracking_reticle_style(data).state == 'uncertain'
    assert tracking_reticle_style(data, fresh=False).rgb == (140,140,140)
    assert tracking_reticle_style({}).label == 'TRACK IDLE'
    with pytest.raises(ValueError):
        tracking_reticle_style({}, mode_hint='nvdcf')
    with pytest.raises(ValueError):
        tracking_reticle_style({}, state_hint='selected')


def test_duplicate_backward_frozen_frame_and_session_change():
    old = _snapshot()
    assert perception_progresses(None, old)
    assert not perception_progresses(old, old)
    assert not perception_progresses(old, old.model_copy(update={'sequence': 41}))
    assert not perception_progresses(old, old.model_copy(update={'sequence': 43}))
    newer = old.model_copy(update={'sequence': 43, 'frame': old.frame.model_copy(update={'frame_id': 43})})
    assert perception_progresses(old, newer)
    assert perception_progresses(old, old.model_copy(update={'frame': old.frame.model_copy(update={'tracker_session': 'new'})}))


@pytest.mark.parametrize('kind,age,expected', [('drone',.01,(0,255,0)),
    ('visual',.01,(255,217,0)), ('visual',.251,(140,140,140))])
def test_pc_pixels_labels_and_no_roi_or_stale_depth(kind, age, expected):
    snapshot = _snapshot()
    selected = snapshot.tracks[0].model_copy(update={'class_id':kind})
    snapshot = snapshot.model_copy(update={'tracks':(selected,)})
    image = np.zeros((720,1280,3), dtype=np.uint8)
    diag = _diagnostics().model_copy(update={'aim_reference_norm':(.52,.48), 'target_center_norm':(.6,.4)})
    report = V2HudRenderer(hfov_deg=90, vfov_deg=60).render(image, snapshot=snapshot,
        cam_state=None, control_cmd=None, control_diagnostics=diag, diagnostics_age_s=.01, perception_age_s=age)
    assert tuple(image[360,640]) == expected
    assert 'tracking_mode' in report.elements
    assert ('parallax_cue' in report.elements) == (kind == 'drone' and age <= .25)
    if age > .25:
        assert 'selection' not in report.elements
    # Legacy command geometry must obey the same identity/freshness gates.
    report = V2HudRenderer(hfov_deg=90, vfov_deg=60).render(image, snapshot=snapshot,
        cam_state=None, control_cmd=_command(), perception_age_s=age)
    assert ('parallax_cue' in report.elements) == (kind == 'drone' and age <= .25)


@pytest.mark.parametrize('state', ['tracking','acquiring','uncertain','lost','idle'])
def test_gpu_osd_reticle_metadata_without_gpu(state):
    from jetson.visual_lock.live_trial import decorate
    colour = lambda: NS(red=0,green=0,blue=0,alpha=0)
    label = lambda: NS(font_params=NS(font_color=colour()),text_bg_clr=colour())
    display = NS(text_params=[label(),label()], line_params=[NS(line_color=colour()) for _ in range(4)],
        rect_params=[NS(border_color=colour()) for _ in range(16)])
    added = []
    pyds = NS(nvds_acquire_display_meta_from_pool=lambda batch:display,
              nvds_add_display_meta_to_frame=lambda meta,data:added.append(data))
    snapshot = _snapshot()
    visual = snapshot.tracks[0].model_copy(update={'class_id':'visual'})
    snapshot = snapshot.model_copy(update={'tracks':(visual,), 'selection':snapshot.selection if state == 'tracking' else None})
    result = NS(snapshot=snapshot, decision=NS(state=state,box=(530,260,70,40)),proposal=None)
    decorate(pyds, None, NS(), result, 42, 60., 'NvDCF')
    style = tracking_reticle_style(snapshot.model_dump(mode='json'),mode_hint='roi',state_hint=state)
    assert display.text_params[1].display_text == style.label
    assert display.num_labels == 2 and display.num_lines == 4
    c = display.line_params[0].line_color
    assert (c.red,c.green,c.blue) == tuple(v/255 for v in style.rgb)
    assert display.num_rects == (0 if state in ('lost','idle') else 1)
    assert added == [display]


def test_pi_cache_transition_and_roi_no_depth():
    pytest.importorskip('cairo')
    from common.hud_telemetry import HudView
    from rpi.display.hud import PiHudRenderer
    from tests.test_panel_hud import diagnostic
    renderer = PiHudRenderer()
    def render(kind, perception=True):
        data = {'tracks':[dict(track_id=7,class_id=kind,missed_frames=0,
            box=dict(x=.55,y=.3,w=.1,h=.1))], 'selected_id':7, 'sequence':1}
        view = HudView('CAMERA','A',perception=data if perception else None,
            diagnostics=diagnostic().model_dump(mode='json'), diagnostics_age_s=.01)
        return renderer.render(1280,720,view)[-1]
    yolo = render('drone')
    assert renderer.reticle_mode == 'YOLO' and 'parallax_cue' in renderer.elements
    roi = render('visual')
    assert renderer.reticle_mode == 'ROI' and 'parallax_cue' not in renderer.elements
    assert roi.data != yolo.data
    assert render('visual') is roi
    stale = render('visual',False)
    assert renderer.reticle_state == 'stale' and stale.data != roi.data
    assert 'selection' not in renderer.elements
    # ARGB32 is BGRA on the target little-endian CPUs; sample an opaque arm.
    offset = (360-roi.y)*roi.stride + (653-roi.x)*4
    b,g,r,a = roi.data[offset:offset+4]
    assert a > 0 and abs(b-a) <= 1 and abs(g-217*a/255) <= 1 and r == 0
