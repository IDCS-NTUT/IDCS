"""Camera contracts with deterministic metadata and GPU-backend fakes only."""
from dataclasses import asdict, replace
import json
import uuid

import pytest

from common.perception import NormalizedBoxV2, PerceptionFrameV2
from common.return_frame_identity import ReturnFrameIdentity, RoiSourceLease
from common.rtp_identity import RtpFrameKey, RtpPacketIdentity
from common.visual_roi_seed import VisualRoiSeed
from jetson.deepstream.argus_sensor_meta import SensorFrame
from jetson.visual_lock.acquisition import acquire_cached
from jetson.visual_lock.camera_identity import CameraIdentityIndex
from jetson.visual_lock.frame_cache import ExactFrameCache
from jetson.visual_lock.fusion import Gates, VisualLock
from jetson.visual_lock.gate_config import load_gates
from jetson.visual_lock.live_trial import pipeline_description, resolve_snapshot_bind
from jetson.visual_lock.return_identity import ReturnRtpMapper
from rpi.display.menu import NavEvent
from rpi.display.roi_editor import RoiEditor
from rpi.display.roi_panel import RoiPanelWorkflow
from tests.test_visual_lock_acquisition import coordinator


def camera_frame(index, pts, sensor):
    index.record(pts, sensor)
    source = index.pop(pts)
    return PerceptionFrameV2(**asdict(source), source_identity_verified=True,
        observed_time_ns=sensor.start_ns+1_000_000, observation_clock_domain="jetson_monotonic",
        width=1280, height=720)


def seed_for(frame):
    return VisualRoiSeed.from_mapping(dict(type="visual_roi_seed_v1",request_id=str(uuid.uuid4()),
        tracker_session=frame.tracker_session, frame_id=frame.frame_id,
        source_time_ns=frame.source_time_ns, source_clock_domain=frame.source_clock_domain,
        width=1280,height=720,x=320,y=180,w=320,h=180))


def test_camera_to_gpu_cache_return_pi_seed_v2_and_release():
    index, cache, mapper, display = CameraIdentityIndex("camera-A"), ExactFrameCache(), ReturnRtpMapper(), ReturnFrameIdentity()
    frames = [camera_frame(index, 1000+i, SensorFrame(80+i, 1_000_000_000+i*16_666_667)) for i in range(5)]
    for i, frame in enumerate(frames):
        cache.add(frame, object(), decoded_pts_ns=1000+i)
    # Encoder uses a different PTS. Only its retained native metadata joins it.
    mapper.record_verified_frame(4, source_for(frames[0]))
    assert not mapper.record_encoder_input(9000, 3)
    assert mapper.record_encoder_input(9000, 4)
    packet = RtpPacketIdentity(RtpFrameKey(42, 600), 1, True)
    payload = mapper.encoded_packet(9000, packet)
    assert payload["source_time_ns"] == frames[0].source_time_ns
    assert display.push_header(payload)
    display.push_marker(7000, packet.key)
    display.push_decoded(7000)
    displayed = display.displayed(7000, 1.0)
    lease = RoiSourceLease()
    assert lease.observe(dict(type="visual_roi_status_v1",source_mode="camera",control_disabled=True,
        source_ready=True,tracker_session="camera-A",sequence=1), 1.0)
    assert lease.live(1.1, displayed)
    request = seed_for(displayed)
    assert cache.replay_from(replace(request,source_clock_domain="pc_monotonic")) is None
    assert cache.replay_from(replace(request,tracker_session="camera-B")) is None
    assert cache.replay_from(replace(request,source_time_ns=request.source_time_ns+1)) is None
    session, backends = coordinator()
    session.lock = VisualLock(Gates())  # unchanged production acquisition/quality gates
    acquisition = acquire_cached(session, cache.replay_from(request), NormalizedBoxV2(x=.25,y=.25,w=.25,h=.25))
    assert acquisition.accepted and acquisition.result.snapshot.selection
    assert acquisition.result.snapshot.frame == frames[-1]
    assert [row['decoded_pts_ns'] for row in acquisition.audit(cache.decoded_pts)] == list(range(1000,1005))
    assert not acquisition.result.snapshot.detections
    session.close()
    assert backends[-1].closed
    assert not session.update(frames[-1],None).snapshot.selection
    assert RoiPanelWorkflow().sim_lease()["enabled"] is False
    cache.clear()
    assert cache.replay_from(request) is None


def source_for(frame):
    from jetson.visual_lock.return_identity import SourceIdentity
    return SourceIdentity(frame.tracker_session, frame.frame_id, frame.source_time_ns, frame.source_clock_domain)


def test_camera_identity_has_no_nearest_fallback_and_poison_is_sticky():
    index = CameraIdentityIndex("camera",capacity=2)
    sensor = SensorFrame(99,1000)
    index.record(1,sensor)
    index.record(1,SensorFrame(100,2000))
    index.record(1,sensor)
    assert index.pop(1) is None
    index.record(2,sensor)
    assert index.pop(3) is None
    assert index.pop(2).frame_id == 99
    assert index.pop(2) is None
    for pts in (3,4,5): index.record(pts,SensorFrame(100+pts,1000*pts))
    assert index.pop(3) is None
    assert index.pop(5).frame_id == 105
    with pytest.raises(ValueError,match="out of order"):
        index.pop(4)
    index.record(6,None)
    index.record((1 << 64)-1,sensor)
    assert index.pop(6) is None and index.missing == 2


def test_camera_refusal_keeps_reason_and_requires_explicit_new_seed():
    index = CameraIdentityIndex("camera")
    frames = [camera_frame(index,i,SensorFrame(i,1_000_000_000+i*16_666_667)) for i in range(3)]
    session, backends = coordinator()
    box = NormalizedBoxV2(x=.25,y=.25,w=.25,h=.25)
    refused = acquire_cached(session,[(frames[0],"blank"),(frames[1],object())],box)
    assert refused.reason == "seed: seed lacks texture"
    assert not refused.result.snapshot.selection and backends[-1].closed
    assert not session.update(frames[1],object()).snapshot.selection
    assert acquire_cached(session,[(frames[1],object()),(frames[2],object())],box).accepted
    session.close()


def test_camera_lease_rejects_wrong_clock_source_change_stale_and_control():
    from common.return_frame_identity import DisplayedSourceFrame
    lease = RoiSourceLease()
    status = dict(type="visual_roi_status_v1",source_mode="camera",control_disabled=True,
        source_ready=True,tracker_session="camera",sequence=1)
    frame = DisplayedSourceFrame("camera",99,1000,"jetson_monotonic",1.)
    assert lease.observe(status,1.) and lease.live(1.1,frame)
    assert not lease.live(1.1,replace(frame,source_clock_domain="pc_monotonic"))
    assert not lease.live(1.6,frame)
    assert not lease.observe(dict(status,source_mode="sim",sequence=2),1.1)
    assert not lease.live_session(1.1,"camera")
    assert not lease.observe(dict(status,control_disabled=False,sequence=3),1.2)


def test_camera_pipeline_owns_one_source_and_no_detector_or_control():
    text = pipeline_description(5000,["127.0.0.1"],5002,60,source="camera",flip_method=2)
    assert 'nvarguscamerasrc name=camera sensor-id=0 sensor-mode=4' in text
    assert 'flip-method=2' in text and 'width=1280,height=720,framerate=60/1' in text
    for forbidden in ('nvinfer','nvtracker','nvv4l2decoder','input_parse','rtp_jitter','controller'):
        assert forbidden not in text
    assert resolve_snapshot_bind(None,panel_roi=True).endswith(':5597')


def test_panel_can_enlarge_refused_texture_box_then_move_again():
    editor = RoiEditor()
    editor.open(1280,720,direct_select=True)
    old = (editor.w,editor.h)
    editor.handle(NavEvent.SELECT)
    editor.move_joystick(1.,1.,1.)
    editor.move_joystick(1.,1.,1.1)
    assert editor.w > old[0] and editor.h > old[1]
    editor.handle(NavEvent.SELECT)
    assert not editor.resizing
    for i in range(100): editor.move_joystick(1.,1.,2.+i)
    assert 0 <= editor.x <= editor.width-editor.w
    assert 0 <= editor.y <= editor.height-editor.h


def test_gate_overrides_are_explicit_validated_and_defaults_unchanged(tmp_path):
    assert load_gates() == Gates()
    path = tmp_path/'gates.json'
    path.write_text(json.dumps(dict(min_points=12,min_psr=7.5)))
    assert load_gates(path) == Gates(min_points=12,min_psr=7.5)
    for value in ([],dict(unknown=1),dict(min_points=2),dict(min_points=8.5),
                  dict(min_psr=float('nan')),dict(min_points=True),dict(min_psr="6")):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError): load_gates(path)


def test_camera_cli_defaults_to_isolated_output_before_native_startup(monkeypatch):
    from jetson.visual_lock import live_trial
    captured = []
    monkeypatch.setattr(live_trial,'run',lambda args:captured.append(args) or 0)
    monkeypatch.setattr('sys.argv',['live_trial','--source','camera','--library','unused',
        '--surface-library','unused','--output','unused','--return-host','127.0.0.1',
        '--seed-xywh','100','100','64','48','--provenance','test'])
    assert live_trial.main() == 0
    assert captured[0].source == 'camera'
    assert captured[0].snapshot_bind == 'tcp://0.0.0.0:5597'


@pytest.mark.parametrize('extra',[
    ['--snapshot-bind','tcp://0.0.0.0:5564','--seed-xywh','100','100','64','48'],
    ['--roi-bind','tcp://127.0.0.1:5596'],  # no exact return identity
    ['--roi-bind','tcp://127.0.0.1:5596','--return-identity-bind','tcp://127.0.0.1:5595','--handoff-nvdcf'],
])
def test_camera_cli_refuses_unsafe_or_unsupported_combinations_before_native_startup(monkeypatch,extra):
    from jetson.visual_lock import live_trial
    monkeypatch.setattr(live_trial,'run',lambda args:pytest.fail('native startup on refused configuration'))
    monkeypatch.setattr('sys.argv',['live_trial','--source','camera','--library','unused',
        '--surface-library','unused','--output','unused','--return-host','127.0.0.1',
        '--provenance','test',*extra])
    with pytest.raises(SystemExit) as error:
        live_trial.main()
    assert error.value.code == 2
