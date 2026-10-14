import pytest

from jetson.visual_lock.live_trial import LiveDeadline, StartupCadence, pipeline_description, resolve_snapshot_bind
from tools.run_visual_lock_trial_guard import JETSON_SOFTWARE_UNITS, video_owners


def test_panel_roi_snapshots_are_isolated_from_ordinary_controller():
    assert resolve_snapshot_bind(None, panel_roi=True) == "tcp://0.0.0.0:5597"
    assert resolve_snapshot_bind(None, panel_roi=False) == "tcp://0.0.0.0:5564"
    with pytest.raises(ValueError):
        resolve_snapshot_bind("tcp://0.0.0.0:5564", panel_roi=True)


def test_trial_guard_names_both_jetson_video_service_variants():
    assert "idcs-deepstream-camera.service" in JETSON_SOFTWARE_UNITS
    assert "idcs-deepstream-video.service" in JETSON_SOFTWARE_UNITS


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
    assert "framerate=30/1" in text and "192.168.0.1:5002,192.168.0.3:5002" in text
    for forbidden in ("nvinfer","nvtracker","appsrc","appsink","videoconvert ! video/x-raw,", "controller"):
        assert forbidden not in text


@pytest.mark.parametrize("args",[(0,["127.0.0.1"],5002,30),(5000,["bad host"],5002,30),
                                 (5000,[],5002,30),(5000,["127.0.0.1"],5002,61)])
def test_live_pipeline_rejects_invalid_parameters(args):
    with pytest.raises(ValueError): pipeline_description(*args)
