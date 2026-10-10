"""Accepted V2 geometry is the only operational GPU bounding-box authority."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from common.perception import NormalizedBoxV2, PerceptionTrackV2
from common.ui_style import GREEN,AMBER,RED,unit_rgb
from jetson.deepstream.metadata_adapter import TrackIdStitcher
from jetson.deepstream.pipeline import _decorate_osd_metadata


def _obs(track_id,x,y,w=.02,h=.03,matched=True):
    from jetson.deepstream.metadata_adapter import ObjectObservationV2
    return ObjectObservationV2(box=NormalizedBoxV2(x=x,y=y,w=w,h=h),class_id='0',
        confidence=.8,track_id=track_id,detector_matched=matched)


class _Colour:
    def set(self,*rgba):
        self.rgba = rgba


def _osd_rect(**values):
    return NS(border_color=_Colour(),**values)


def _osd_text():
    return NS(font_params=NS(font_color=_Colour()),text_bg_clr=_Colour())


def _fake_osd():
    added = []
    def acquire(batch):
        return NS(rect_params=[_osd_rect() for _ in range(16)],
                  text_params=[_osd_text() for _ in range(16)])
    pyds = NS(nvds_acquire_display_meta_from_pool=acquire,
        nvds_add_display_meta_to_frame=lambda frame, meta:added.append(meta))
    frame = NS(source_frame_width=1280,source_frame_height=720)
    raw = [NS(rect_params=_osd_rect(left=100,top=100,width=100,height=100),
        text_params=_osd_text(),object_id=i,confidence=.5,class_id=0) for i in (1,2710)]
    return pyds,frame,raw,added


def _track(observation, missed=0):
    return PerceptionTrackV2(track_id=observation.track_id,box=observation.box,
        class_id=observation.class_id,confidence=observation.confidence,missed_frames=missed)


@pytest.mark.parametrize('missed,colour',[(0,unit_rgb(GREEN)),(1,unit_rgb(AMBER))])
def test_draws_only_accepted_geometry_and_suppresses_native_and_shadow(missed,colour):
    pyds,frame,raw,added = _fake_osd()
    track = _track(_obs(1,.45,.4,w=.08,h=.1),missed)
    _decorate_osd_metadata(pyds,None,frame,raw,accepted_tracks=(track,),
        shadow_boxes=(_obs(2710,.44,.38,matched=False),),class_labels={0:'drone'})
    assert all(obj.rect_params.border_width == 0 and obj.rect_params.has_bg_color == 0
        and obj.text_params.display_text == '' and obj.text_params.set_bg_clr == 0 for obj in raw)
    assert sum(meta.num_rects for meta in added) == 1
    rect = added[0].rect_params[0]
    assert (rect.left,rect.top,rect.width,rect.height) == (.45*1280,.4*720,.08*1280,.1*720)
    assert rect.border_color.rgba[:3] == colour
    assert added[0].text_params[0].display_text == f'drone {"COAST" if missed else "0.80"}'


def test_empty_accepted_set_cannot_resurrect_raw_or_shadow_box():
    pyds,frame,raw,added = _fake_osd()
    _decorate_osd_metadata(pyds,None,frame,raw,accepted_tracks=(),
        shadow_boxes=(_obs(2710,.4,.4,matched=False),))
    assert sum(meta.num_rects for meta in added) == 0
    assert all(obj.rect_params.border_width == 0 for obj in raw)


def test_selection_colour_and_all_tracks_beyond_one_display_chunk():
    pyds,frame,raw,added = _fake_osd()
    tracks = tuple(_track(_obs(i,.1,.1)) for i in range(35))
    _decorate_osd_metadata(pyds,None,frame,raw,accepted_tracks=tracks,target_track_id=17,
        class_labels={0:'drone'})
    assert [meta.num_rects for meta in added] == [16,16,3]
    assert added[1].text_params[1].display_text.startswith('TGT drone')
    assert added[1].rect_params[1].border_color.rgba[:3] == unit_rgb(RED)
    assert all('#' not in meta.text_params[i].display_text for meta in added for i in range(meta.num_labels))
    assert [t.track_id for t in tracks] == list(range(35))  # Internal IDs unchanged.


def test_nested_live_case_below_iou_gate_coalesces_and_retains_native_alias():
    # Frozen geometry from the live12s PC survey: tight match inside inflated coast.
    fresh = _obs(1,.48403539657592776,.5887680901421442,
        w=.03260869681835175,h=.04045893616146511)
    coast = _obs(2710,.47856931686401366,.5705507490370009,
        w=.04513864517211914,h=.07674592865837945,matched=False)
    stitcher = TrackIdStitcher()
    for _ in range(8):
        tracks = stitcher.apply([fresh,coast],img_w=1280,img_h=720)
        assert tracks == [fresh]
        assert stitcher.stable_id(2710) == 1
    # When only the formerly discarded native ID survives, stable identity remains.
    surviving = replace(fresh,track_id=2710)
    assert stitcher.apply([surviving],img_w=1280,img_h=720) == [fresh]
    # If two native hypotheses genuinely separate, preserve both geometries.
    far = replace(coast,box=_obs(2710,.8,.8).box,detector_matched=True)
    assert len(stitcher.apply([fresh,far],img_w=1280,img_h=720)) == 2


def test_duplicate_rules_preserve_different_classes_and_separate_close_targets():
    same = _obs(1,.4,.4,w=.02,h=.03)
    other_class = replace(same,track_id=2,class_id='1')
    stitcher = TrackIdStitcher()
    assert len(stitcher.apply([same,other_class],img_w=1280,img_h=720)) == 2
    nested = _obs(3,.4,.385,w=.04,h=.06)  # contained, but distinct centres: preserve both
    assert len(TrackIdStitcher().apply([same,nested],img_w=1280,img_h=720)) == 2
    coast = _obs(4,.42,.4,w=.02,h=.03,matched=False)
    assert len(TrackIdStitcher().apply([same,coast],img_w=1280,img_h=720)) == 2
    huge = _obs(4,.38,.2,w=.06,h=.43,matched=False)
    assert len(TrackIdStitcher().apply([same,huge],img_w=1280,img_h=720)) == 2


def test_frozen_thin_drone_inside_taller_coast_estimate():
    fresh = _obs(1,.4898097991943359,.5537439558241103,
        w=.026494568586349486,h=.03140096664428711)
    coast = _obs(2712,.48036870956420896,.5273975372314453,
        w=.04552602767944336,h=.08257540067036946,matched=False)
    assert TrackIdStitcher().apply([fresh,coast],img_w=1280,img_h=720) == [fresh]


@pytest.mark.parametrize('matched',[True,False])
def test_nested_same_provenance_prefers_confidence_preserves_older_identity(matched):
    large = replace(_obs(11,.48097829818725585,.5428743574354383,
        w=.042119568586349486,h=.07367149988810222,matched=matched),confidence=.153076171875)
    tight = replace(_obs(13,.4853940010070801,.5609903547498915,
        w=.029891306161880495,h=.038043477800157335,matched=matched),confidence=.291015625)
    stitcher = TrackIdStitcher()
    for _ in range(8):
        assert stitcher.apply([large,tight],img_w=1280,img_h=720) == [replace(tight,track_id=11)]
        assert stitcher.stable_id(13) == 11


def test_stitching_cannot_cross_classes_and_stable_zero_is_valid():
    stitcher = TrackIdStitcher()
    stitcher.apply([_obs(0,.4,.4)],img_w=1280,img_h=720)
    stitcher.apply([],img_w=1280,img_h=720)
    assert stitcher.apply([_obs(9,.4,.4)],img_w=1280,img_h=720)[0].track_id == 0
    stitcher = TrackIdStitcher()
    stitcher.apply([_obs(0,.4,.4)],img_w=1280,img_h=720)
    stitcher.apply([],img_w=1280,img_h=720)
    changed = replace(_obs(9,.4,.4),class_id='1')
    assert stitcher.apply([changed],img_w=1280,img_h=720)[0].track_id == 9
