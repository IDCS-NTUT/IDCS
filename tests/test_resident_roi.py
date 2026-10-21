"""Resident-mode/worker contracts; deterministic pixels, no GPU or devices."""
import threading
import time
import uuid

import pytest

from common.visual_roi_seed import VisualRoiMode, VisualRoiSelect, VisualRoiRelease
from jetson.deepstream.resident_roi import RoiWorker
from tests.test_seamless_roi import adapter, image_frame, seed, Pixels
from types import SimpleNamespace as NS
from unittest.mock import Mock


def wait_for(predicate):
    end=time.monotonic()+2
    while time.monotonic()<end:
        if predicate(): return
        time.sleep(.002)
    assert predicate(), 'bounded worker wait expired'


def command(roi, mode, revision=None, **kw):
    return VisualRoiMode(str(uuid.uuid4()),roi.tracker_session,mode,
        roi.mode_revision if revision is None else revision,**kw)


def test_mode_is_scoped_revisioned_and_idempotent(adapter):
    roi,*_=adapter
    roi.explicit_modes=True
    request=command(roi,'roi')
    roi.enqueue(request.as_message());roi.begin_frame()
    assert roi.processing_mode=='roi' and roi.mode_revision==1
    roi.enqueue(request.as_message());roi.begin_frame()
    assert roi.mode_revision==1
    roi.enqueue(command(roi,'yolo',revision=0).as_message());roi.begin_frame()
    assert roi.processing_mode=='roi' and roi.mode_result['mode_accepted'] is False
    foreign=VisualRoiMode(str(uuid.uuid4()),'other-source','yolo',1)
    roi.enqueue(foreign.as_message());roi.begin_frame()
    assert roi.processing_mode=='roi'
    roi.enqueue(command(roi,'yolo').as_message());roi.begin_frame()
    assert roi.processing_mode=='yolo' and roi.mode_revision==2


def test_release_and_loss_do_not_change_operator_mode(adapter):
    roi,*_=adapter
    roi.explicit_modes=True
    roi.enqueue(command(roi,'roi').as_message());roi.begin_frame()
    lease=VisualRoiSelect(str(uuid.uuid4()),roi.tracker_session,True)
    roi.enqueue(lease.as_message());roi.begin_frame()
    roi._cancel('texture lost')
    assert roi.processing_mode=='roi' and roi.begin_frame() is True
    assert roi.source_status()['selection_owner']=='visual_lock'


def test_worker_keeps_exact_identity_and_unique_target_on_relock(adapter):
    roi,*_=adapter
    roi.explicit_modes=True
    worker=RoiWorker(roi)
    try:
        roi.enqueue(command(roi,'roi').as_message());worker.wake()
        wait_for(lambda:roi.processing_mode=='roi')
        ids=[]
        index=0
        for _ in range(2):
            lease=VisualRoiSelect(str(uuid.uuid4()),roi.tracker_session,True)
            roi.enqueue(lease.as_message());worker.wake()
            wait_for(lambda:roi.cache_enabled)
            first=image_frame(index)
            worker.submit(first,Pixels(),1000+index)
            wait_for(lambda:len(roi.cache._frames)>0)
            request=seed(first)
            roi.enqueue(request.as_message())
            for offset in (1,2,3):
                worker.submit(image_frame(index+offset),Pixels(),1000+index+offset)
                wait_for(lambda:worker.processed>=index+offset+1)
            value=worker.latest()
            assert value is not None
            _,snapshot,result,completed=value
            assert snapshot.frame.frame_id==index+3
            assert snapshot.frame.source_time_ns==image_frame(index+3).source_time_ns
            assert snapshot.frame.observed_time_ns==completed
            assert snapshot.selection is not None
            ids.append(snapshot.selection.track_id)
            roi.enqueue(VisualRoiRelease(request.request_id,roi.tracker_session).as_message())
            worker.wake();wait_for(lambda:not roi.owns_selection)
            assert roi.processing_mode=='roi'
            index+=4
        assert ids[0]!=ids[1] and min(ids) >= 1<<60
    finally:worker.close()
    assert not worker.thread.is_alive() and worker.error is None


def test_slow_initialization_cannot_block_submission_and_backlog_is_bounded(adapter):
    roi,*_=adapter
    roi.explicit_modes=True
    entered,finish=threading.Event(),threading.Event()
    original=roi.begin_frame
    def slow(**kwargs):
        entered.set()
        assert finish.wait(2)
        return original(**kwargs)
    roi.begin_frame=slow
    worker=RoiWorker(roi,capacity=2)
    images=[Pixels() for _ in range(12)]
    try:
        assert entered.wait(1)
        for i,image in enumerate(images):worker.submit(image_frame(i),image,i)
        assert len(worker.pending)==2 and worker.dropped==10
        assert sum(p.closed for p in images)==10
    finally:
        finish.set();worker.close()
    assert all(p.closed for p in images)


def test_clone_pool_reuses_exclusive_allocation_and_releases_registration_once():
    from jetson.visual_lock.live_surface import NVMMClonePool
    def cuda(handle,pointer,pitch):
        pointer._obj.value=handle*100
        pitch._obj.value=5120
        return 0
    lib=NS(vll_clone_cuda=Mock(side_effect=cuda),vll_clone_copy=Mock(return_value=0))
    images=[]
    def clone(buffer):
        image=NS(handle=len(images)+1,pointer=1000+len(images),cuda=None,close=Mock())
        images.append(image);return image
    borrower=NS(lib=lib,clone=Mock(side_effect=clone))
    pool=NVMMClonePool(borrower,capacity=2)
    a,b=pool.acquire(object()),pool.acquire(object())
    assert a.cuda==(100,5120) and b.cuda==(200,5120)
    assert pool.acquire(object()) is None
    a.close();a.close()
    c=pool.acquire(object())
    assert c.cuda==(100,5120) and len(pool.images)==2
    lib.vll_clone_copy.assert_called_once()
    assert not any(image.close.called for image in images)
    b.close();c.close();pool.close()
    for image in images:image.close.assert_called_once()


def test_resident_sim_does_not_override_manual_with_legacy_auto_lease():
    import json
    from tools.sim_panel import RoiSimAuthority,armed_state
    authority=RoiSimAuthority()
    status=dict(type='visual_roi_status_v1',source_mode='sim',managed_mode='sim',roi_capable=True,
        source_ready=True,control_disabled=False,tracker_session='run',sequence=1,processing_mode='roi')
    assert authority.observe(json.dumps(status),1.,status=True)
    panel=armed_state().model_copy(update=dict(source='rpi.runtime_control',active=True,control_cmd_enabled=False))
    authority.engaged=True
    assert authority.apply(panel,1.1)==panel


def test_roi_range_does_not_invent_depth_for_parallax():
    from dataclasses import replace
    from common.aiming import solve_snapshot_aiming
    from common.control import LaserMountConfig
    from tests.test_control_observation import _config,_snapshot
    config=replace(_config(),aim_mode='laser_point')
    snap=_snapshot()
    snap=snap.model_copy(update={'tracks':(snap.tracks[0].model_copy(update={'class_id':'visual'}),)})
    mount=LaserMountConfig.from_raw_config({'laser':{'offset_m':[.1,0,0],'dir_cam':[0,0,1]}})
    result=solve_snapshot_aiming(snap,config,mount)
    assert result is not None and result.distance_m is None and not result.parallax_active


def test_all_operator_sim_controllers_consume_visual_not_truth():
    from pathlib import Path
    from common.config import load_config_bundle,resolve_config_paths
    from jetson.control.runtime_config import ControlRuntimeConfig
    root=Path(__file__).resolve().parents[1]
    for name in ('idcs-sim-controller','idcs-sim-swarm-controller','idcs-sim-visual-lock-controller'):
        text=(root/'deploy/systemd/pc'/f'{name}.service').read_text()
        for line in text.splitlines():
            if not line.startswith(('ExecStart=','ExecStartPre=')) or '--config-extra ' not in line:continue
            overlays=line.split('--config-extra ',1)[1].split()[0]
            config=load_config_bundle(resolve_config_paths(root/'configs/base',
                [root/p for p in overlays.split(',')])).data
            runtime=ControlRuntimeConfig.from_config(config)
            assert runtime.snapshot_endpoint==config['net']['zmq_perception_v2']
            assert runtime.local_clock=='pc'
            assert runtime.intent_bind=='tcp://127.0.0.1:5571'


def test_software_video_fixture_accepts_only_intents_and_owns_no_hardware():
    from pathlib import Path
    from common.config import load_config_bundle,resolve_config_paths
    from common.schemas import ControlIntent
    from tools.resident_sim_fixture import Capture
    root=Path(__file__).resolve().parents[1]
    cfg=load_config_bundle(resolve_config_paths(root/'configs/base',
        [root/'configs/sim/v2_scene.yaml',root/'configs/sim/mode_simulated_mount.yaml'])).mutable_copy()
    with pytest.raises(ValueError,match='explicit loopback'):
        Capture('sim',1280,720,60,cfg,sim_control_enabled=False)
    cap=Capture('sim',1280,720,60,cfg,sim_control_enabled=True)
    cap.handle_control_cmd({'type':'ControlCmd','pan_rate_cmd':1.})
    assert cap.intent is None
    stamp=time.monotonic_ns()
    intent=ControlIntent(sequence=1,observation_sequence=1,issued_monotonic_ns=stamp,
        valid_until_monotonic_ns=stamp+50_000_000,mode='live',yaw_rate_rad_s=.1,pitch_rate_rad_s=0.,reason='fixture')
    cap.handle_control_cmd(intent.model_dump(mode='json'))
    ok,pixels=cap.read()
    assert ok and pixels.shape==(720,1280,3)
    snap=cap.build_ground_truth_snapshot(5,cap.last_frame_source_ns)
    assert snap.selection is None and snap.tracks[0].class_id=='fixture'
    assert snap.frame.tracker_session==cap._control_activity.session
    assert cap.build_cam_state(5,0)['pan']>=0


def test_worker_fault_cannot_skip_pipeline_resource_cleanup():
    from jetson.deepstream.resident_roi import ResidentRoiPipeline
    pipeline=ResidentRoiPipeline.__new__(ResidentRoiPipeline)
    pipeline.worker=NS(close=Mock(side_effect=RuntimeError('native failed')))
    pipeline.pool=NS(close=Mock())
    pipeline.glib=NS(source_remove=Mock());pipeline.timer=42
    pipeline.pull=Mock();pipeline.pub=Mock();pipeline.context=Mock()
    pipeline.probes=[('queue','sink',7)]
    pipeline.detach=Mock();pipeline.active=pipeline.identity_live=True
    with pytest.raises(RuntimeError,match='native failed'):pipeline.close()
    pipeline.pool.close.assert_called_once()
    pipeline.glib.source_remove.assert_called_once_with(42)
    pipeline.pull.close.assert_called_once_with(0)
    pipeline.pub.close.assert_called_once_with(0)
    pipeline.context.term.assert_called_once()
    pipeline.detach.assert_called_once_with('queue','sink')
    assert pipeline.stopping and not pipeline.active and not pipeline.identity_live
