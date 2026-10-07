from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.perception import (NormalizedBoxV2, PerceptionFrameV2, PerceptionSnapshotV2,
                               PerceptionTrackV2, TargetSelectionV2)
from jetson.visual_lock.fusion import Decision
from jetson.visual_lock.session import Output
from jetson.visual_lock.nvdcf_handoff import (NvDCFHandoff,NativeTrack,write_operator_profile,
                                             inject_region,VISUAL_CLASS_ID,UNTRACKED_ID)
from jetson.visual_lock.live_trial import pipeline_description


def frame(i=0, **changes):
    values = dict(tracker_session="source-A",frame_id=i,source_time_ns=1_000_000_000+i*16_666_667,
        observed_time_ns=5_000_000_000+i*16_666_667,source_clock_domain="pc_monotonic",
        observation_clock_domain="jetson_monotonic",source_identity_verified=True,width=320,height=240)
    values.update(changes)
    return PerceptionFrameV2(**values)


def vpi(f,box=(80.,60.,80.,60.),selected=True):
    x,y,w,h = box
    tracks = (PerceptionTrackV2(track_id=7,box=NormalizedBoxV2(x=x/f.width,y=y/f.height,w=w/f.width,h=h/f.height),
                               class_id="visual",confidence=1,age_frames=3,missed_frames=0),) if selected else ()
    selection = TargetSelectionV2(track_id=7,source_frame_id=f.frame_id,applied_frame_id=f.frame_id,
        selected_time_ns=f.observed_time_ns,selection_clock_domain=f.observation_clock_domain,policy="operator_visual_lock") if selected else None
    return Output(Decision("tracking" if selected else "acquiring","test",box,accepted=True),
        PerceptionSnapshotV2(sequence=f.frame_id,frame=f,tracks=tracks,selection=selection),None,1.)


def native(**changes):
    return replace(NativeTrack(123,(80.,60.,80.,60.),.8),**changes)


def acquired():
    h = NvDCFHandoff()
    for i in range(3):
        f = frame(i); h.begin(f)
        seed = h.initialize(f,vpi(f))
        assert (seed is not None) == (i == 0)
        result = h.finish(f,[native()],vpi(f))
        assert bool(result.snapshot.selection) == (i == 2)
    return h


def test_one_seed_then_native_only_stable_public_identity_and_no_detection():
    h = acquired()
    assert h.stage == "nvdcf" and not h.needs_vpi
    f = frame(3); h.begin(f)
    assert h.initialize(f,vpi(f)) is None
    result = h.finish(f,[native()])
    assert result.snapshot.selection.track_id == 7 and result.snapshot.selection.applied_frame_id == 3
    assert not result.snapshot.detections and not result.snapshot.assessments
    assert result.snapshot.tracks[0].class_id == "visual"
    assert h.report()["seed_count"] == 1 and h.report()["filter_state_transferred"] is False


@pytest.mark.parametrize("change",[dict(tracker_session="B"),dict(source_clock_domain="other"),
    dict(observation_clock_domain="other"),dict(width=321),dict(source_identity_verified=False),
    dict(frame_id=2),dict(source_time_ns=1),dict(source_time_ns=2_000_000_000),dict(observed_time_ns=1)])
def test_frame_gates_before_native_work(change):
    h = acquired(); f = frame(3,**change)
    assert h.begin(f) is None and h.stage == "lost"
    assert h.finish(f,[native()]).snapshot.selection is None


@pytest.mark.parametrize("candidate",[native(track_id=124),native(class_id=0),native(track_id=UNTRACKED_ID),
    native(box=(float("nan"),60,80,60)),native(box=(80,60,2,60)),native(box=(300,60,80,60))])
def test_invalid_native_identity_geometry_never_rebind(candidate):
    h = acquired(); f = frame(3); h.begin(f)
    result = h.finish(f,[candidate])
    assert result.snapshot.selection is None and h.stage == "lost"


@pytest.mark.parametrize("candidates",[[],[native(confidence=.1)],[native(confidence=float("nan"))],
    [native(box=(160,60,80,60))],[native(box=(80,60,120,60))]])
def test_quality_loss_suppresses_immediately_then_latches(candidates):
    h = acquired()
    for i in range(3,6):
        f = frame(i); h.begin(f)
        assert h.finish(f,candidates).snapshot.selection is None
    assert h.stage == "lost" and not h.needs_vpi
    f = frame(6); h.begin(f)
    assert h.initialize(f,vpi(f)) is None
    assert h.finish(f,[native()]).snapshot.selection is None


def test_confirmation_requires_same_frame_vpi_agreement_and_times_out():
    h = NvDCFHandoff(); f = frame(); h.begin(f); h.initialize(f,vpi(f))
    for i in range(13):
        f = frame(i)
        if i: h.begin(f)
        p = vpi(f) if i == 0 else None
        candidates = [native()] if i%3 != 2 else []
        assert h.finish(f,candidates,p).snapshot.selection is None
    assert h.stage == "lost" and h.reason == "native confirmation timeout"


def test_stale_vpi_seed_cannot_initialize():
    h = NvDCFHandoff(); h.begin(frame(1))
    assert h.initialize(frame(1),vpi(frame(0))) is None and h.stage == "lost"


def test_uncertain_native_recovery_needs_three_good_frames():
    h = acquired(); f = frame(3); h.begin(f)
    assert h.finish(f,[]).snapshot.selection is None
    for i in range(4,7):
        f = frame(i); h.begin(f)
        assert bool(h.finish(f,[native()]).snapshot.selection) == (i == 6)
    assert h.seed_count == 1 and not h.needs_vpi


def test_ambiguous_tracks_fail_closed():
    h = acquired(); f = frame(3); h.begin(f)
    assert h.finish(f,[native(),native(track_id=124)]).snapshot.selection is None
    assert h.stage == "lost"


def test_native_output_must_match_pending_frame():
    h = acquired(); h.begin(frame(3))
    assert h.finish(frame(4),[native()]).snapshot.selection is None and h.stage == "lost"


def test_async_native_frames_keep_exact_identity_and_single_vpi_seed():
    h = NvDCFHandoff()
    for i in range(4):
        f = frame(i); h.begin(f)
        assert (h.initialize(f,vpi(f)) is not None) == (i == 0)
    for i in range(4):
        f = frame(i)
        result = h.finish(f,[native()],vpi(f) if i == 0 else None)
        assert bool(result.snapshot.selection) == (i >= 2)
    assert not h.pending and h.seed_count == 1 and not h.needs_vpi


def test_async_pending_queue_is_bounded():
    h = NvDCFHandoff()
    for i in range(9): h.begin(frame(i))
    assert len(h.pending) == 8 and h.stage == "lost"


def test_delayed_preseed_native_output_cannot_reject_future_seed():
    h = NvDCFHandoff()
    h.begin(frame(0)); h.initialize(frame(0),vpi(frame(0),selected=False))
    h.begin(frame(1)); assert h.initialize(frame(1),vpi(frame(1))) is not None
    early = h.finish(frame(0),[],vpi(frame(0),selected=False))
    assert early.decision.state == "acquiring" and h.bad == 0 and h.stage == "confirming"
    assert h.finish(frame(1),[native()],vpi(frame(1))).snapshot.selection is None


def test_external_region_injected_once_and_labeled_visual():
    coords = SimpleNamespace()
    obj = SimpleNamespace(rect_params=SimpleNamespace(),detector_bbox_info=SimpleNamespace(org_bbox_coords=coords))
    added = []
    pyds = SimpleNamespace(nvds_acquire_obj_meta_from_pool=lambda b:obj,
        nvds_add_obj_meta_to_frame=lambda m,o,p:added.append(o))
    meta = SimpleNamespace(bInferDone=True)
    inject_region(pyds,None,meta,(80,60,80,60))
    assert len(added) == 1 and meta.bInferDone and obj.class_id == VISUAL_CLASS_ID and obj.object_id == UNTRACKED_ID
    assert obj.obj_label == "visual" and coords.width == 80
    inject_region(pyds,None,meta,None)
    assert len(added) == 1 and not meta.bInferDone


def test_profile_isolated_from_production_and_pipeline_has_no_yolo(tmp_path):
    source = Path(__file__).resolve().parents[1]/"configs/deepstream/tracker_nvdcf.yml"
    before = source.read_bytes()
    profile = write_operator_profile(source,tmp_path/"operator.yml")
    assert source.read_bytes() == before
    assert profile["TargetManagement"]["probationAge"] == 0
    assert profile["TargetManagement"]["outputShadowTracks"] == 0
    assert profile["TargetManagement"]["maxTargetsPerStream"] == 1
    assert profile["TargetManagement"]["minTrackerConfidence"] == .2
    pipeline = pipeline_description(5000,["127.0.0.1"],5002,30,tmp_path/"operator.yml")
    assert pipeline.count("nvtracker ") == 1 and "nvinfer" not in pipeline
    assert pipeline.index("visual_input") < pipeline.index("handoff_tracker") < pipeline.index("nvdsosd")
