import pytest

from jetson.visual_lock.live_trial import LiveDeadline, StartupCadence, pipeline_description, resolve_snapshot_bind
from tools.run_visual_lock_trial_guard import JETSON_SOFTWARE_UNITS, video_owners


def test_roi_quality_does_not_overwrite_console_safety_row():
    from types import SimpleNamespace as NS
    from common.ui_style import HudLayout
    from jetson.visual_lock.live_trial import decorate
    metadata = NS(text_params=[NS(font_params=NS(font_color=NS()),text_bg_clr=NS()) for _ in range(2)],
                  line_params=[NS(line_color=NS()) for _ in range(4)],rect_params=[])
    added=[]
    pyds=NS(nvds_acquire_display_meta_from_pool=lambda _:metadata,
            nvds_add_display_meta_to_frame=lambda frame,display:added.append(display))
    decorate(pyds,None,NS(),None,1,30.)
    layout=HudLayout(1280,720)
    quality=metadata.text_params[0]
    assert (quality.x_offset,quality.y_offset)==(round(layout.roi_quality.x),round(layout.roi_quality.y))
    assert quality.y_offset != round(layout.receiver_status.y)
    assert metadata.text_params[1].y_offset==round(layout.mode_badge.y)
    assert added==[metadata]


def test_return_cadence_keeps_source_metadata_without_duplicating_images():
    gi=pytest.importorskip('gi')
    try:
        gi.require_version('Gst','1.0')
    except ValueError:
        pytest.skip('Gst typelib unavailable')
    from gi.repository import Gst
    Gst.init(None)
    if not Gst.ElementFactory.find('videorate'):
        pytest.skip('videorate unavailable')
    # Caps-only CPU fixture: no decoder, encoder, GPU, socket or NVMM allocation.
    pipeline=Gst.parse_launch('appsrc name=source format=time ! '
        'videorate drop-only=true max-rate=30 skip-to-first=true ! '
        'video/x-raw(memory:NVMM),framerate=30/1 ! '
        'appsink name=sink sync=false')
    source=pipeline.get_by_name('source')
    source.set_property('caps',Gst.Caps.from_string(
        'video/x-raw(memory:NVMM),format=GRAY8,width=1,height=1,framerate=60/1'))
    reference=Gst.Caps.from_string('timestamp/x-idcs-source')
    originals={}
    try:
        pipeline.set_state(Gst.State.PLAYING)
        for i in range(15):
            buffer=Gst.Buffer.new_wrapped(b'\x00')
            buffer.pts=123456789+i*16666667
            buffer.duration=16666667
            source_ns=1234567890000000+i*16666667
            originals[source_ns]=buffer.pts
            buffer.add_reference_timestamp_meta(reference,source_ns,buffer.duration)
            assert source.emit('push-buffer',buffer)==Gst.FlowReturn.OK
        source.emit('end-of-stream')
        selected=[]
        retimed=False
        while True:
            sample=pipeline.get_by_name('sink').emit('try-pull-sample',Gst.SECOND)
            if sample is None:
                break
            buffer=sample.get_buffer()
            meta=buffer.get_reference_timestamp_meta(reference)
            assert meta is not None and meta.timestamp in originals
            selected.append(meta.timestamp)
            retimed |= buffer.pts != originals[meta.timestamp]
            caps=sample.get_caps()
            assert caps.get_features(0).contains('memory:NVMM')
            assert tuple(caps.get_structure(0).get_fraction('framerate'))[-2:]==(30,1)
        assert 0<len(selected)<15
        assert len(selected)==len(set(selected))
    finally:
        pipeline.set_state(Gst.State.NULL)


def test_panel_roi_snapshots_are_isolated_from_ordinary_controller():
    assert resolve_snapshot_bind(None, panel_roi=True) == "tcp://0.0.0.0:5597"
    assert resolve_snapshot_bind(None, panel_roi=False) == "tcp://0.0.0.0:5597"
    with pytest.raises(ValueError):
        resolve_snapshot_bind("tcp://0.0.0.0:5564", panel_roi=True)


def test_trial_guard_names_both_jetson_video_service_variants():
    assert "idcs-deepstream-camera.service" in JETSON_SOFTWARE_UNITS
    assert "idcs-deepstream-video.service" in JETSON_SOFTWARE_UNITS
    assert {"idcs-visual-lock-camera.service", "idcs-visual-lock-sim.service"} <= set(JETSON_SOFTWARE_UNITS)


def test_pi_guard_checks_display_owner_without_stopping_gpio_owner(tmp_path):
    for pid,module in ((900005,'rpi.operator_display'),(900006,'rpi.runtime_control')):
        folder = tmp_path/str(pid)
        folder.mkdir()
        (folder/'cmdline').write_bytes(f'python\0-m\0{module}\0'.encode())
    assert video_owners('pi',tmp_path) == [dict(pid=900005,module='rpi.operator_display')]


def test_guard_identifies_video_child_not_guard_command_arguments(tmp_path):
    rows = {"900001":["python","trial_guard.py","--","python","-m","pc.streamer"],
            "900002":["python","-m","pc.streamer"],
            "900003":["python","-m","jetson.visual_lock.live_trial"]}
    for pid,argv in rows.items():
        folder = tmp_path/pid
        folder.mkdir()
        (folder/"cmdline").write_bytes("\0".join(argv).encode())
    assert video_owners("pc",tmp_path) == [dict(pid=900002,module="pc.streamer")]
    assert video_owners("jetson",tmp_path) == [dict(pid=900003,module="jetson.visual_lock.live_trial")]


def test_guard_also_refuses_standalone_gpu_replay_owner(tmp_path):
    folder = tmp_path/"900004"; folder.mkdir()
    (folder/"cmdline").write_bytes(b"python\0-m\0tools.replay_visual_lock_capture\0")
    assert video_owners("jetson",tmp_path) == [dict(pid=900004,module="tools.replay_visual_lock_capture")]


def test_initial_seed_waits_for_cadence_after_startup_gap():
    gate = StartupCadence(required=3)
    assert not gate.observe(1)
    assert not gate.observe(2_400_000_001)
    assert not gate.observe(2_416_000_001)
    assert not gate.observe(2_432_000_001)
    assert gate.observe(2_448_000_001)


def test_startup_gate_resets_on_duplicate_backwards_or_gap():
    gate = StartupCadence(required=2)
    for stamp in (100,101,101,100,100_000_100):
        assert not gate.observe(stamp)
    assert not gate.observe(100_000_101)
    assert gate.observe(100_000_102)


def test_live_duration_starts_at_first_verified_frame_not_process_start():
    timer = LiveDeadline(100,30,60)
    assert timer.reason(155) is None
    timer.observe(155)
    timer.observe(170)
    assert timer.reason(184.9) is None
    assert timer.reason(185) == "duration complete"


def test_missing_source_has_separate_bounded_timeout():
    timer = LiveDeadline(100,900,60)
    assert timer.reason(159.9) is None
    assert timer.reason(160) == "source startup timeout"


def test_live_trial_is_gpu_video_only_and_replaces_detector_workload():
    text = pipeline_description(5000,["192.168.0.1","192.168.0.3"],5002,30)
    assert "payload=96" in text and "pt=97" in text
    assert "format=RGBA ! identity name=visual_input ! nvdsosd process-mode=1" in text
    assert 'identity name=return_rate' in text and 'videorate' not in text
    assert "copy-timestamp=true" in text and "192.168.0.1:5002,192.168.0.3:5002" in text
    assert 'framerate=30/1' in text
    assert text.index('name=return_rate') < text.index('name=return_encoder')
    assert 'iframeinterval=1 idrinterval=1' in text
    assert 'rtp-profile=avpf' in text and 'name=return_wire' in text
    predictive=pipeline_description(5000,['127.0.0.1'],5002,30,return_keyframe_interval=15)
    assert 'iframeinterval=15 idrinterval=15' in predictive
    for forbidden in ("nvinfer","nvtracker","appsrc","appsink","videoconvert ! video/x-raw,", "controller"):
        assert forbidden not in text
    assert "overlay_queue" not in text and "output-buffers" not in text


def test_overlay_queue_decouples_return_path_after_tracker_probe():
    text = pipeline_description(5000,["127.0.0.1"],5002,30,overlay_queue=True)
    assert ("nvvideoconvert output-buffers=8 ! video/x-raw(memory:NVMM),format=RGBA ! identity name=visual_input ! "
            "queue name=overlay_queue max-size-buffers=2 leaky=downstream ! nvdsosd") in text
    camera = pipeline_description(5000,["127.0.0.1"],5002,60,source="camera",flip_method=2,overlay_queue=True)
    assert "identity name=visual_input ! queue name=overlay_queue" in camera


@pytest.mark.parametrize("args",[(0,["127.0.0.1"],5002,30),(5000,["bad host"],5002,30),
                                 (5000,[],5002,30),(5000,["127.0.0.1"],5002,61)])
def test_live_pipeline_rejects_invalid_parameters(args):
    with pytest.raises(ValueError): pipeline_description(*args)
