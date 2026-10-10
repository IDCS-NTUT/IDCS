"""Deterministic presentation tests: no sockets, model, GPU or motor commands."""
import numpy as np
import pytest

from common import ui_style as theme
from common.hud_geometry import tracking_reticle_style


@pytest.mark.parametrize('width,height',[(640,360),(960,540),(1280,720),(1920,1080),(800,600)])
def test_regions_fit_and_mode_badge_is_not_at_target_centre(width,height):
    layout = theme.HudLayout(width,height)
    centre = theme.Rect(width/2-25*layout.scale,height/2-25*layout.scale,50*layout.scale,50*layout.scale)
    for region in (layout.mode_badge,layout.status,layout.feedforward):
        assert 0 <= region.x < region.right <= width
        assert 0 <= region.y < region.bottom <= height
    assert not layout.mode_badge.overlaps(centre)
    assert not layout.status.overlaps(layout.feedforward)
    assert layout.heading.right == layout.elevation.x
    assert not layout.heading.overlaps(layout.elevation)
    assert layout.elevation.y == 0 and layout.elevation.bottom == height
    assert layout.heading.height <= height*.05
    for r in (layout.status,layout.feedforward,layout.receiver_status):
        assert not r.overlaps(layout.heading) and not r.overlaps(layout.elevation)
    assert not layout.receiver_status.overlaps(layout.feedforward)
    assert not layout.status.overlaps(layout.receiver_status)


@pytest.mark.parametrize('width,height',[(640,360),(1280,720),(1920,1080),(800,600)])
@pytest.mark.parametrize('position',[(0,0),(.99,0),(.99,.99),(0,.99),(.48,.49)])
def test_target_labels_fit_reserved_regions_and_long_text_is_bounded(width,height,position):
    layout = theme.HudLayout(width,height)
    label,rect = layout.target_label((width*position[0],height*position[1],25,25),'long drone target '*30)
    assert len(label) < 256
    assert 0 <= rect.x < rect.right <= width
    assert 0 <= rect.y < rect.bottom <= height
    assert not any(rect.overlaps(r) for r in layout.reserved)


def test_adjacent_labels_use_free_rows_without_moving_boxes():
    layout = theme.HudLayout(1280,720)
    box=(620,310,20,20)
    _,first=layout.target_label(box,'drone id=1')
    _,second=layout.target_label(box,'drone id=2',occupied=(first,))
    assert not first.overlaps(second)
    assert box==(620,310,20,20)


def test_reticle_and_boxes_take_colours_from_the_same_theme():
    assert theme.track_rgb()==theme.GREEN
    assert theme.track_rgb(coasting=True)==theme.AMBER
    assert theme.track_rgb(selected=True,coasting=True)==theme.RED
    assert tracking_reticle_style(None).rgb==theme.GRAY
    assert theme.bgr(theme.CYAN)==(255,217,0)
    assert theme.unit_rgb(theme.AMBER)==(1,191/255,0)


def test_shared_status_strings_do_not_infer_motor_authority():
    assert theme.source_summary('SIM-ONLY',True,7,{'distance_m':3.25})=='SIM-ONLY | pose OK | SIM target 3.2m'
    assert theme.source_summary('CAMERA',False)=='CAMERA | pose --'
    assert theme.control_summary('SIM','tracking','PID',.1,-.2)=='SIM | TRACKING | PID | Y +0.100 P -0.200 rad/s'
    assert 'STALE' in theme.control_summary('HARDWARE','tracking','PID',0.,0.,stale=True)
    assert theme.control_rgb('manual')==theme.BLUE
    assert theme.control_rgb('tracking',stale=True)==theme.AMBER
    assert theme.parallax_summary(3.25)=='Parallax aim offset 3.2px'
    assert 'no fresh tracking geometry' in theme.parallax_summary()


def test_cached_font_is_bundled_and_roi_blending_does_not_touch_rest_of_frame():
    from pc.ui_text import font,glyph,text,panel
    assert font(13).path.endswith('DejaVuSans.ttf')
    assert glyph('SIM TRACKING',13) is glyph('SIM TRACKING',13)
    image=np.full((360,640,3),50,np.uint8)
    before=image.copy()
    text(image,'SIM TRACKING',(40,40),theme.bgr(theme.GREEN),size=13,background=True)
    assert np.array_equal(image[80:],before[80:])
    assert np.any(image[:80]!=before[:80])
    panel(image,(-10,-10,30,30))
    text(image,'edge',(639,359),theme.bgr(theme.WHITE),size=13)


def test_cache_bounds_and_clipped_long_text():
    from pc.ui_text import font,glyph,panel_mask,text
    assert font.cache_info().maxsize==64
    assert glyph.cache_info().maxsize==512
    assert panel_mask.cache_info().maxsize==128
    image=np.zeros((360,640,3),np.uint8)
    text(image,'very long '*200,(30,40),theme.bgr(theme.WHITE),size=13,max_width=100)
    assert not np.any(image[:,140:])


def test_gpu_metadata_uses_shared_font_colour_scale_and_label_placement():
    from tests.test_deepstream_accepted_osd import _fake_osd,_track,_obs
    from jetson.deepstream.pipeline import _decorate_osd_metadata
    for width,height in ((640,360),(1280,720),(1920,1080)):
        pyds,frame,raw,added=_fake_osd()
        frame.source_frame_width,frame.source_frame_height=width,height
        track=_track(_obs(7,.98,.01))
        _decorate_osd_metadata(pyds,None,frame,raw,accepted_tracks=(track,),class_labels={0:'drone'})
        rect=added[0].rect_params[0]
        label=added[0].text_params[0]
        assert rect.left==.98*width and rect.top==.01*height
        assert label.font_params.font_name==theme.FONT_FAMILY
        assert label.font_params.font_size==theme.pixel_size(theme.TARGET_TEXT_PX,height)
        assert label.text_bg_clr.rgba==theme.PANEL_RGBA
        assert label.x_offset<rect.left and label.y_offset>=38*height/720


def test_pi_theme_and_geometry_match_pc_and_gpu():
    pytest.importorskip('cairo')
    from rpi.display import render,hud
    assert render.SANS==theme.FONT_FAMILY
    assert render.PANEL_BG==theme.PANEL_RGBA
    assert hud.GREEN==theme.unit_rgb(theme.GREEN)
    assert hud.CYAN==theme.unit_rgb(theme.CYAN)
    assert render.BOX_COLOURS['lock']==theme.unit_rgb(theme.RED)
    from common.ui_fonts import configure_fonts,FONT_DIRECTORY
    assert configure_fonts()['regular']==(FONT_DIRECTORY/'DejaVuSans.ttf').resolve()
    assert configure_fonts()['mono']==(FONT_DIRECTORY/'DejaVuSansMono.ttf').resolve()
