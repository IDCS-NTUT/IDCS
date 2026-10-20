"""Normal-runtime ROI contracts. No sockets to hosts, GPU or hardware."""
import json
import threading
import uuid
from pathlib import Path
from queue import Empty
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from common.perception import PerceptionSnapshotV2, RoiStateV2
from common.return_frame_identity import DisplayedSourceFrame, RoiSourceLease
from common.visual_roi_seed import VisualRoiRelease, VisualRoiSeed, VisualRoiSelect
from jetson.deepstream.roi_pipeline import RoiPipeline
from jetson.deepstream.roi_runtime import RoiRuntime
from jetson.visual_lock.fusion import Proposal
from pc.roi_selection import SimRoiSelection
from tests.test_operator_agent import _agent, config, _snapshot
from tests.test_visual_lock import frame, proposal
from tests.test_managed_visual_lock import display_module
from tools.sim_panel import RoiSimAuthority, armed_state


def image_frame(i=0, **kw):
    return frame(i, width=1280, height=720, **kw)


def seed(f, **kw):
    return VisualRoiSeed(str(uuid.uuid4()), f.tracker_session, f.frame_id,
        f.source_time_ns, f.source_clock_domain, 320, 180, 160, 120, **kw)


class Pixels:
    def __init__(self):
        self.closed = False
    def close(self):
        self.closed = True
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    # Production rotation is tested separately. Never prune existing evidence.
    from jetson.deepstream import roi_runtime
    from jetson.visual_lock.managed import evidence_directory
    monkeypatch.setattr(roi_runtime, "evidence_directory", lambda root:
        evidence_directory(root, min_free_bytes=0))
    borrower = NS(clone=Mock(side_effect=lambda _: Pixels()),
                  borrow=Mock(side_effect=lambda _: Pixels()))
    backends = []
    class Backend:
        def __init__(self, *_):
            self.box = None
            self.closed = False
            backends.append(self)
        def process(self, image, seed=None):
            if seed is not None:
                self.box = tuple(seed)
                return Proposal(self.box, 0., 0., np.zeros((0, 2)),
                                np.zeros((0, 2)), np.zeros(0), seed_features=16)
            return proposal(self, dx=0, dy=0)
        def commit(self, box, **_):
            self.box = box
        def discard(self):
            pass
        def close(self):
            self.closed = True
    clock = NS(t=100.)
    roi = RoiRuntime("sim", "pixels-A", build=tmp_path, evidence_root=tmp_path/"evidence",
                     factory=lambda _: (borrower, Backend), clock=lambda: clock.t)
    return roi, borrower, backends, clock


def lock(adapter):
    roi, _, _, _ = adapter
    select(roi)
    assert not roi.begin_frame()
    assert roi.process(image_frame(), object()) is None
    request = seed(image_frame())
    roi.enqueue(request.as_message())
    assert roi.begin_frame()
    result = roi.process(image_frame(1), object())
    assert result.decision.state == "acquiring"
    return request, result


def select(roi):
    request = VisualRoiSelect(str(uuid.uuid4()), roi.tracker_session, True)
    roi.enqueue(request.as_message())
    return request


def router(roi):
    wire = RoiPipeline.__new__(RoiPipeline)
    wire.roi, wire.pipeline, wire.gst = roi, Mock(), NS(
        PadProbeReturn=NS(OK="ok", DROP="drop"), FlowReturn=NS(OK="flow-ok"),
        EventType=NS(EOS="eos"))
    wire.pipeline.get_by_name.return_value.get_static_pad.return_value.chain.return_value = "flow-ok"
    wire.done = threading.Event()
    wire.done.set()
    wire.frames, wire.frames_lock = {}, threading.Lock()
    wire.barrier_engaged = wire.metadata_skipped = wire.selector_paused = False
    wire.barrier_timeout_s = .5
    wire.stopping = False
    wire.stage_clock, wire.fail, wire.selector = Mock(), Mock(), Mock()
    wire.pyds = Mock()
    wire.closed = threading.Event()
    wire.close_error = None
    return wire


def test_no_idle_vpi_exact_seed_and_release_resume_yolo(adapter):
    roi, borrower, backends, _ = adapter
    for i in range(40):
        assert not roi.begin_frame()
        assert roi.process(image_frame(i), object(), decoded_pts_ns=i*100) is None
    assert not backends and borrower.borrow.call_count == 0
    assert not roi.cache._frames and borrower.clone.call_count == 0
    assert not roi.timings
    select(roi)
    for i in range(40, 60):
        assert not roi.begin_frame()
        roi.process(image_frame(i), object(), decoded_pts_ns=i*100)
    assert len(roi.cache._frames) == 16
    request = seed(image_frame(56))
    roi.enqueue(request.as_message())
    assert roi.begin_frame()  # Stops detector BEFORE acquisition.
    result = roi.process(image_frame(60), object())
    assert result.snapshot.frame.frame_id == 60
    assert roi.active == request and backends
    assert not roi.cache._frames
    assert roi.snapshot(image_frame(60), result).frame.frame_id == 60
    roi.enqueue(VisualRoiRelease(request.request_id, request.tracker_session).as_message())
    assert not roi.begin_frame()  # Release frame itself uses YOLO.
    roi.process(image_frame(61), object())
    assert backends[-1].closed and roi.session is None and roi.active is None
    assert roi.terminal["request_id"] == request.request_id
    assert "released" in roi.terminal["reason"]
    assert roi.source_status()["selection_owner"] == "detector"
    assert list(roi.evidence_root.glob("session-*/report.json"))
    roi.close()


def test_acquiring_and_tracking_skip_then_loss_resumes_next_frame(adapter):
    roi, _, _, _ = adapter
    request, result = lock(adapter)
    for i in (2, 3, 4):
        assert roi.begin_frame()
        result = roi.process(image_frame(i), object())
    assert result.snapshot.selection.policy == "operator_visual_lock"
    assert roi.begin_frame()
    lost = roi.process(image_frame(30), object())  # Existing source-gap gate.
    assert lost.decision.state == "lost" and lost.snapshot.selection is None
    assert not roi.begin_frame()
    assert roi.source_status()["terminal_request_id"] == request.request_id
    assert roi.source_status()["active_request_id"] is None
    assert roi.process(image_frame(31), object()) is None
    assert roi.active is None  # Never reseed.
    roi.close()


@pytest.mark.parametrize("fault", ("old_frame", "session", "stale", "unverified"))
def test_seed_refusal_and_missing_identity_return_box(adapter, fault):
    roi, _, backends, clock = adapter
    select(roi)
    roi.begin_frame()
    roi.process(image_frame(), object())
    f = image_frame(99) if fault == "old_frame" else image_frame()
    if fault == "session":
        f = image_frame(tracker_session="other")
    request = seed(f)
    roi.enqueue(request.as_message())
    if fault == "stale":
        clock.t += .6
    roi.begin_frame()
    roi.process(image_frame(1, source_identity_verified=fault != "unverified"), object())
    assert not roi.owns_selection and not backends
    assert not roi.begin_frame()
    assert roi.terminal["request_id"] == request.request_id
    roi.close()


def test_release_cancels_pending_and_mismatched_release_cannot_cancel(adapter):
    roi, _, _, _ = adapter
    request, _ = lock(adapter)
    roi.enqueue(VisualRoiRelease(str(uuid.uuid4()), request.tracker_session).as_message())
    assert roi.begin_frame()
    roi.process(image_frame(2), object())
    roi.enqueue(VisualRoiRelease(request.request_id, request.tracker_session).as_message())
    new = seed(image_frame(2))
    roi.enqueue(new.as_message())
    roi.enqueue(VisualRoiRelease(new.request_id, new.tracker_session).as_message())
    assert not roi.begin_frame()
    roi.process(image_frame(3), object())
    assert roi.session is None
    roi.close()


def test_overflow_cancels_authority_and_missing_native_keeps_yolo(adapter, tmp_path):
    roi, _, _, _ = adapter
    request, _ = lock(adapter)
    for _ in range(10):
        roi.enqueue(VisualRoiRelease(request.request_id, request.tracker_session).as_message())
    assert not roi.begin_frame()
    roi.process(image_frame(2), object())
    roi.close()
    missing = RoiRuntime("sim", "pixels-A", build=tmp_path/"missing", evidence_root=tmp_path)
    assert not missing.capable and "libraries missing" in missing.error
    missing.enqueue(seed(image_frame()).as_message())
    assert not missing.begin_frame()
    assert missing.process(image_frame(), object()) is None
    assert missing.source_status()["source_ready"] is False
    missing.close()


def test_evidence_bounds_and_cache_memory_are_reported(adapter):
    roi, _, _, _ = adapter
    lock(adapter)
    assert roi.logs["frames"].max_bytes == 32_000_000
    assert roi.logs["acquisitions"].max_bytes == 8_000_000
    assert roi.session.seed_diagnostics.max_writes == 64
    log = roi.logs["frames"]
    log.max_bytes = log.bytes
    log.write("large row\n")
    assert log.dropped == 1
    assert roi.status()["roi_cache_rgba_bytes"] == 58_982_400
    roi.close()
    report = json.loads(next(roi.evidence_root.glob("session-*/report.json")).read_text())
    assert report["logs"]["frames"]["dropped"] == 1


def test_router_really_bypasses_both_gpu_elements(adapter):
    roi, _, _, _ = adapter
    lock(adapter)
    wire = router(roi)
    pipeline = wire.pipeline
    buffer = Mock(pts=1)
    assert wire.route(None, NS(get_buffer=lambda: buffer), None) == "drop"
    pipeline.get_by_name.assert_called_once_with("roi_owner")
    pipeline.get_by_name.return_value.get_static_pad.return_value.chain.assert_called_once_with(buffer)
    assert not wire.done.is_set()
    wire.selector.pause_for_roi.assert_called_once()
    # Completion is the barrier, including failed metadata. No second input
    # may enter inference before the frame owner finishes VPI work.
    wire.done.set()
    roi.enqueue(VisualRoiRelease(roi.active.request_id, roi.active.tracker_session).as_message())
    pipeline.reset_mock()
    assert wire.route(None, NS(get_buffer=lambda: buffer), None) == "ok"
    pipeline.get_by_name.assert_not_called()
    wire.selector.resume_from_roi.assert_called_once()
    roi.close()


def test_idle_router_allows_many_outstanding_frames_without_any_wait(adapter):
    roi, borrower, backends, _ = adapter
    wire = router(roi)
    wire.done = Mock(wait=Mock(side_effect=AssertionError("idle barrier")))
    for i in range(120):
        assert wire.route(None, NS(get_buffer=lambda: Mock(pts=i)), None) == "ok"
        roi.process(image_frame(i), object())
    assert len(wire.frames) == 120  # No artificial one-frame limit.
    assert roi.counts["barrier_waits"] == roi.counts["cache_copies"] == 0
    assert not borrower.clone.called and not borrower.borrow.called and not backends
    assert not roi.timings and not roi.pipeline_timings
    wire.selector.pause_for_roi.assert_not_called()
    wire.selector.resume_from_roi.assert_not_called()
    roi.close()


def test_selection_lease_expiry_and_cancel_stop_copies_and_cannot_restart_same_lease(adapter):
    roi, borrower, _, clock = adapter
    lease = select(roi)
    roi.begin_frame()
    roi.process(image_frame(), object())
    pixels = roi.cache._frames[0][1]
    clock.t += .51
    roi.begin_frame()
    roi.process(image_frame(1), object())
    assert pixels.closed and not roi.cache._frames
    roi.enqueue(lease.as_message())
    roi.begin_frame()
    roi.process(image_frame(2), object())
    assert not roi.cache_enabled and borrower.clone.call_count == 1
    lease = select(roi)
    roi.begin_frame()
    roi.process(image_frame(3), object())
    roi.enqueue(VisualRoiSelect(lease.selection_id, lease.tracker_session, False).as_message())
    roi.begin_frame()
    roi.process(image_frame(4), object())
    assert not roi.cache._frames and borrower.clone.call_count == 2
    roi.close()


def test_selecting_has_absolute_deadline_even_with_fresh_heartbeats(adapter):
    roi, borrower, _, clock = adapter
    lease = select(roi)
    for i in range(61):
        clock.t = 100.+i*.25
        roi.enqueue(lease.as_message())
        roi.begin_frame()
        roi.process(image_frame(i), object())
    assert roi.cache_enabled
    clock.t += .25
    roi.enqueue(lease.as_message())
    roi.begin_frame()
    roi.process(image_frame(61), object())
    assert not roi.cache_enabled and not roi.cache._frames
    copies = borrower.clone.call_count
    roi.enqueue(seed(image_frame(61)).as_message())
    assert not roi.begin_frame()
    roi.process(image_frame(62), object())
    assert borrower.clone.call_count == copies
    assert "not selecting" in roi.terminal["reason"]
    roi.close()


def test_seed_without_selecting_refuses_even_the_current_frame(adapter):
    roi, borrower, backends, _ = adapter
    request = seed(image_frame())
    roi.enqueue(request.as_message())
    assert not roi.begin_frame()
    roi.process(image_frame(), object())
    assert "cache was not selecting" in roi.terminal["reason"]
    assert not borrower.clone.called and not backends
    roi.close()


def test_pending_barrier_drains_all_earlier_yolo_frames(adapter, monkeypatch):
    from jetson.deepstream import pipeline as module
    roi, _, _, _ = adapter
    select(roi)
    roi.begin_frame()
    roi.process(image_frame(), object())
    wire = router(roi)
    buffers = [Mock(pts=i) for i in range(3)]
    infos = [NS(get_buffer=lambda b=b: b) for b in buffers]
    for info in infos[:2]:
        assert wire.route(None, info, None) == "ok"
    roi.enqueue(seed(image_frame()).as_message())
    entered, finished = threading.Event(), threading.Event()
    real_wait = wire.done.wait
    wire.done.wait = lambda timeout: (entered.set(), real_wait(timeout))[1]
    outputs = []
    thread = threading.Thread(target=lambda: (outputs.append(wire.route(None, infos[2], None)), finished.set()))
    thread.start()
    assert entered.wait(1.)
    monkeypatch.setattr(module, "_metadata_probe", lambda *_: "ok")
    wire.metadata_probe(None, infos[0], None)
    assert not finished.is_set() and not wire.done.is_set()
    wire.metadata_probe(None, infos[1], None)
    assert finished.wait(1.)
    thread.join(1.)
    assert outputs == ["drop"] and wire.frames[2][1] is True
    assert roi.pending is not None
    wire.fail.assert_not_called()
    roi.close()


def test_seed_arriving_after_route_decision_waits_for_next_drained_boundary(adapter):
    roi, _, _, _ = adapter
    select(roi)
    roi.begin_frame()
    roi.process(image_frame(), object())
    wire = router(roi)
    original = roi.begin_frame
    def arrive(**kwargs):
        roi.enqueue(seed(image_frame()).as_message())
        return original(**kwargs)
    roi.begin_frame = arrive
    assert wire.route(None, NS(get_buffer=lambda: Mock(pts=1)), None) == "ok"
    assert roi.pending is None and roi.barrier_requested.is_set()
    assert roi.counts["barrier_waits"] == 0
    roi.begin_frame = original
    # Explicit completion of all preceding fake YOLO work.
    wire.frames.clear()
    wire.done.set()
    assert wire.route(None, NS(get_buffer=lambda: Mock(pts=2)), None) == "drop"
    assert roi.pending is not None and roi.counts["barrier_waits"] == 1
    roi.close()


def test_slow_metadata_releases_roi_without_failing_or_dropping_yolo(adapter, monkeypatch):
    from jetson.deepstream import pipeline as module
    roi, _, backends, _ = adapter
    lock(adapter)
    wire = router(roi)
    infos = [NS(get_buffer=lambda i=i: Mock(pts=i)) for i in (10, 11, 12)]
    assert wire.route(None, infos[0], None) == "drop"  # Real ROI bypass.
    wire.barrier_timeout_s = 0.
    assert wire.route(None, infos[1], None) == "ok"  # Slow ROI metadata.
    assert roi.counts["barrier_timeouts"] == 1 and roi.abort_reason is not None
    status = roi.source_status()
    assert status["active_request_id"] is None and status["roi_active"] is False
    assert status["terminal_request_id"] is not None and "delayed" in status["reason"]
    seen = []
    def metadata(*_):
        seen.append(wire.metadata_skipped)
        roi.process(image_frame(2), object())
        return "ok"
    monkeypatch.setattr(module, "_metadata_probe", metadata)
    assert wire.metadata_probe(None, infos[0], None) == "ok"
    assert not roi.owns_selection and backends[-1].closed
    assert "delayed" in roi.terminal["reason"]
    assert wire.metadata_probe(None, infos[1], None) == "ok"
    assert seen == [True, False]  # Late metadata retains its own route decision.
    assert wire.route(None, infos[2], None) == "ok"
    assert not wire.barrier_engaged
    wire.fail.assert_not_called()
    roi.close()


def test_eos_during_lock_closes_on_owner_and_wakes_barrier(adapter):
    roi, borrower, backends, _ = adapter
    request, _ = lock(adapter)
    wire = router(roi)
    wire.done.clear()
    event = NS(type="eos", get_structure=lambda: None)
    assert wire.owner_event(None, NS(get_event=lambda: event), None) == "ok"
    assert wire.closed.is_set() and wire.done.is_set()
    assert not roi.owns_selection and not roi.cache._frames and backends[-1].closed
    assert roi.terminal["request_id"] == request.request_id
    assert wire.close_error is None
    wire.fail.assert_not_called()


def test_cache_timeout_event_releases_pixels_without_another_video_frame(adapter):
    roi, _, _, clock = adapter
    select(roi)
    roi.begin_frame()
    roi.process(image_frame(), object())
    pixels = roi.cache._frames[0][1]
    clock.t += .6
    wire = router(roi)
    wire.cache_expiry_sent = True
    event = NS(type="custom", get_structure=lambda: NS(get_name=lambda: "idcs-roi-expire-cache"))
    assert wire.owner_event(None, NS(get_event=lambda: event), None) == "ok"
    assert pixels.closed and not roi.cache._frames and not roi.cache_enabled
    assert not wire.cache_expiry_sent
    roi.close()


def test_timeout_waits_only_for_executing_native_work_before_yolo(adapter, monkeypatch):
    from jetson.deepstream import pipeline as module
    roi, borrower, backends, _ = adapter
    setup, process_now, native_started, native_finish = [threading.Event() for _ in range(4)]
    metadata_finished, routed = threading.Event(), threading.Event()
    errors, outputs = [], []
    wire = router(roi)
    roi_info = NS(get_buffer=lambda: Mock(pts=10))
    yolo_info = NS(get_buffer=lambda: Mock(pts=11))
    def metadata(*_):
        roi.process(image_frame(2), object())
        return "ok"
    monkeypatch.setattr(module, "_metadata_probe", metadata)
    class SlowPixels(Pixels):
        def __enter__(self):
            native_started.set()
            assert native_finish.wait(2.)
            return self
    def owner():
        try:
            lock(adapter)
            borrower.borrow.side_effect = lambda _: SlowPixels()
            setup.set()
            assert process_now.wait(2.)
            wire.metadata_probe(None, roi_info, None)
            metadata_finished.set()
            roi.close()
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=owner)
    thread.start()
    route_thread = None
    try:
        assert setup.wait(2.)
        assert wire.route(None, roi_info, None) == "drop"
        process_now.set()
        assert native_started.wait(2.)
        wire.barrier_timeout_s = 0.
        aborted = threading.Event()
        abort = roi.abort
        roi.abort = lambda reason: (abort(reason), aborted.set())
        route_thread = threading.Thread(target=lambda: (outputs.append(wire.route(None, yolo_info, None)), routed.set()))
        route_thread.start()
        assert aborted.wait(1.) and not routed.is_set()
        native_finish.set()
        assert metadata_finished.wait(2.) and routed.wait(2.)
    finally:
        native_finish.set()
        process_now.set()
        thread.join(2.)
        if route_thread is not None:
            route_thread.join(2.)
    assert not errors and outputs == ["ok"]
    assert not thread.is_alive() and not route_thread.is_alive()
    assert not roi.owns_selection and backends[-1].closed
    wire.fail.assert_not_called()


@pytest.mark.parametrize("mode", ("camera", "sim", "sim_visual_lock"))
def test_display_sends_select_lease_only_for_normal_mode_editor(display_module, mode):
    display = display_module.OperatorDisplay.__new__(display_module.OperatorDisplay)
    display.visual_roi_trial = False
    display.roi_seed = Mock()
    display._roi_select, display._roi_select_sent_s = None, 0.
    display._roi_open_cache = True
    display.roi_workflow = NS(active=False, state="camera")
    display.roi_editor = NS(active=False)
    display.video = NS(frame_identity=NS(latest=lambda: NS(tracker_session="session")))
    display.status = NS(agent_state=dict(mode=mode), video=NS(live=lambda _: True),
                        panel=NS(live=lambda _: True), panel_state=dict(emergency=False))
    display._sim_roi_source_available = lambda _: True
    display._publish_roi_select(1.)
    display.roi_seed.send_json.assert_not_called()
    display.roi_workflow.active = display.roi_editor.active = True
    display._publish_roi_select(2.)
    if mode == "sim_visual_lock":
        display.roi_seed.send_json.assert_not_called()
        return
    first = display.roi_seed.send_json.call_args.args[0]
    assert VisualRoiSelect.from_mapping(first).enabled
    display._publish_roi_select(2.2)
    assert display.roi_seed.send_json.call_args.args[0] == first
    display.roi_workflow.state = "pending"
    display._publish_roi_select(2.3)
    assert display.roi_seed.send_json.call_args.args[0] == dict(first, enabled=False)
    display.roi_workflow.state = "selecting"
    display._publish_roi_select(2.5)
    assert display.roi_seed.send_json.call_args.args[0]["selection_id"] != first["selection_id"]
    display.roi_editor.active = False
    display._publish_roi_select(2.6)
    assert display.roi_seed.send_json.call_args.args[0]["enabled"] is False


def test_agent_authority_uses_runtime_state_and_does_not_expire_lock(config):
    from common.operator_commands import OperatorCommand
    agent, _, clock = _agent(config)
    snap = _snapshot([]).model_dump(mode="json")
    snap["frame"]["source_identity_verified"] = True
    snap["roi_state"] = dict(capable=True, owns_selection=True, request_id="box")
    agent.on_snapshot(snap)
    assert agent.state()["roi_capable"]
    clock.t += config.lock_lost_s+1
    sequence = agent.sequence
    agent.expire_lock()
    assert agent.sequence == sequence and agent.active_catalog() is None
    for command in (OperatorCommand("release"), OperatorCommand("selection_mode", mode="planner"),
                    OperatorCommand("clear_swarm_score", source_session="old")):
        assert not agent.handle(command)[0]
    snap["roi_state"]["owns_selection"] = False
    agent.on_snapshot(snap)
    assert not agent.visual_lock_active()


@pytest.mark.parametrize("mode", ("camera", "sim"))
def test_normal_mode_lease_requires_capability_and_matching_session(mode):
    status = dict(type="visual_roi_status_v1", managed_mode=mode, source_mode=mode,
        tracker_session="session", sequence=1, source_ready=True, control_disabled=False,
        roi_capable=True)
    lease = RoiSourceLease()
    assert lease.observe(status, 1., managed_mode=mode)
    domain = "pc_monotonic" if mode == "sim" else "jetson_monotonic"
    assert lease.live(1.1, DisplayedSourceFrame("session", 1, 2, domain, 1.))
    assert not lease.live(1.1, DisplayedSourceFrame("other", 1, 2, domain, 1.))
    assert not lease.observe(dict(status, sequence=2, roi_capable=False), 1.2, managed_mode=mode)


def test_single_sim_relay_holds_on_stale_roi_and_returns_to_ordinary_input():
    relay = SimRoiSelection()
    ordinary = "ordinary truth input"
    assert relay.payload(ordinary, 0.) == ordinary
    snap = PerceptionSnapshotV2(sequence=1, frame=image_frame(),
        roi_state=RoiStateV2(capable=True, owns_selection=True, request_id="roi"))
    relay.observe(snap.model_dump_json(), 1.)
    assert PerceptionSnapshotV2.model_validate_json(relay.payload(ordinary, 1.1)) == snap
    assert json.loads(relay.payload(ordinary, 2.)).get("selection") is None
    assert relay.payload(ordinary, 2.) != ordinary
    returned = snap.model_copy(update={"frame": image_frame(1),
        "roi_state": RoiStateV2(capable=True, owns_selection=False)})
    relay.observe(returned.model_dump_json(), 2.1)
    assert relay.payload(ordinary, 2.1) == ordinary


@pytest.mark.parametrize("mode", ("camera", "sim"))
def test_camera_sim_menu_capability_and_selection_authority(display_module, mode):
    from rpi.display.menu import Settings
    from rpi.display.roi_editor import ROI_SETTINGS
    from rpi.display.status import SystemStatus
    status = SystemStatus()
    status.agent_state = dict(mode=mode, roi_capable=True, visual_lock=False)
    select, release = Mock(), Mock()
    menu = display_module.build_menu(status, Settings(ROI_SETTINGS, None), NS(),
                                     roi_preview=select, roi_release=release)
    targets = menu.root.current_items()[0]
    def labels():
        return [getattr(item, "label", getattr(item, "title", None))
                for item in targets.current_items()]
    assert labels() == ["Select box", "Release", "Box size", "Planner selection", "Eligible classes"]
    targets.current_items()[0].run()
    targets.current_items()[1].run()
    select.assert_called_once()
    release.assert_called_once()
    status.agent_state["visual_lock"] = True
    assert labels() == ["Select box", "Release", "Box size"]
    status.agent_state.update(visual_lock=False, roi_capable=False)
    assert labels() == ["Planner selection", "Eligible classes"]


def test_normal_sim_lease_is_still_sim_only_and_fail_closed():
    roi = RoiSimAuthority()
    panel = armed_state().model_copy(update=dict(source="rpi.runtime_control", active=True,
                                                 control_cmd_enabled=False))
    lease = dict(type="roi_sim_lease_v1", enabled=True, managed_mode="sim",
                 request_id="box", tracker_session="run")
    status = dict(type="visual_roi_status_v1", source_mode="sim", managed_mode="sim",
        roi_capable=True, source_ready=True, control_disabled=False,
        active_request_id="box", tracker_session="run", sequence=1)
    assert roi.observe(json.dumps(lease), 1.)
    assert roi.observe(json.dumps(status), 1., status=True)
    assert roi.apply(panel, 1.01).control_cmd_enabled
    assert not roi.apply(panel, 1.26).control_cmd_enabled
    assert not roi.apply(panel.model_copy(update={"emergency": True}), 1.01).control_cmd_enabled
    status.update(source_mode="camera", managed_mode="camera", sequence=2)
    assert not roi.observe(json.dumps(status), 1.02, status=True)
    assert not roi.apply(panel, 1.03).control_cmd_enabled


def test_acquisition_sideband_blocks_agent_before_first_vpi_snapshot(config):
    agent, _, _ = _agent(config)
    snap = _snapshot([]).model_dump(mode="json")
    snap["frame"].update(tracker_session="session", frame_id=10, source_identity_verified=True)
    agent.on_snapshot(snap)
    status = dict(type="visual_roi_status_v1", roi_capable=True, source_ready=True,
        selection_owner="visual_lock", tracker_session="session", managed_mode="sim",
        sequence=1, roi_frame_id=11)
    agent.on_roi_status(status)
    assert agent.visual_lock_active()
    agent.on_snapshot(snap)  # Earlier detector frame cannot release the latch.
    assert agent.visual_lock_active()
    snap["frame"]["frame_id"] = 12
    agent.on_snapshot(snap)
    assert not agent.visual_lock_active()
    agent.on_roi_status(status)  # Neither duplicate nor older status resurrects it.
    assert not agent.visual_lock_active()


def test_gpu_entry_violation_and_frame_owner_teardown_are_explicit(adapter):
    roi, _, _, _ = adapter
    lock(adapter)
    wire = router(roi)
    wire.frames[1] = (None, True)
    pad = NS(get_parent_element=lambda: NS(get_name=lambda: "primary"))
    assert wire.gpu_entry(pad, NS(get_buffer=lambda: Mock(pts=1)), None) == "drop"
    assert roi.counts["unexpected_gpu_input"] == 1
    wire.fail.assert_called_once()
    errors = []
    def other_thread():
        try:
            roi.close()
        except RuntimeError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join(1.)
    assert errors == ["ROI teardown moved off its frame owner"]
    roi.close()


def test_native_dependency_load_error_and_abi_error_withdraw_capability(tmp_path, monkeypatch):
    from jetson.deepstream import roi_runtime
    for name in ("libidcs_visual_lock.so", "libidcs_visual_lock_live_surface.so"):
        (tmp_path/name).touch()
    monkeypatch.setattr(roi_runtime.ctypes, "CDLL", Mock(side_effect=OSError("dependency missing")))
    missing = RoiRuntime("sim", "pixels-A", build=tmp_path, evidence_root=tmp_path)
    assert not missing.capable and "dependency missing" in missing.error
    bad = Mock()
    bad.vl_abi_version.return_value = 3
    monkeypatch.setattr(roi_runtime.ctypes, "CDLL", Mock(return_value=bad))
    missing = RoiRuntime("sim", "pixels-A", build=tmp_path, evidence_root=tmp_path)
    assert not missing.capable and "ABI4 required" in missing.error


def test_socket_adapter_seed_result_and_release_flow(adapter):
    import zmq
    roi, _, _, clock = adapter
    select(roi)
    roi.begin_frame()
    roi.process(image_frame(), object())
    request = seed(image_frame())
    wire = RoiPipeline.__new__(RoiPipeline)
    wire.roi, wire.pub, wire.pull = roi, Mock(), Mock()
    wire.messages = __import__("queue").Queue(maxsize=128)
    wire.last_status_s = 0.
    wire.pull.recv.side_effect = [json.dumps(request.as_message()).encode(), zmq.Again()]
    assert wire.poll()
    assert roi.begin_frame()
    roi.process(image_frame(1), object())
    wire.pull.recv.side_effect = zmq.Again()
    wire.poll()
    results = [call.args[0] for call in wire.pub.send_json.call_args_list
               if call.args[0].get("type") == "visual_roi_result_v1"]
    assert results[-1]["accepted"] is True
    release = VisualRoiRelease(request.request_id, request.tracker_session)
    wire.pull.recv.side_effect = [json.dumps(release.as_message()).encode(), zmq.Again()]
    wire.poll()
    assert not roi.begin_frame()
    roi.process(image_frame(2), object())
    assert roi.source_status()["terminal_request_id"] == request.request_id
    roi.close()


def test_idle_capability_preserves_normal_hud_and_active_roi_names_its_owner(display_module):
    from collections import deque
    from common.hud_telemetry import HudView
    from rpi.display.status import SystemStatus
    status = SystemStatus()
    status.on_agent(dict(mode="camera", roi_capable=True, visual_lock=False,
                         selection_mode="planner"), 1.)
    display = display_module.OperatorDisplay.__new__(display_module.OperatorDisplay)
    display.status = status
    display.visual_roi_trial = False
    display.roi_editor = NS(active=False)
    display.roi_workflow = NS(state="camera")
    display.menu = NS(view=lambda: NS(kind="menu"))
    display.hud_state = NS(view=lambda *a, **kw: HudView("CAMERA", ""))
    display.hud_renderer = NS(render=Mock(return_value=[]))
    display.video = NS(frame_size=(1280, 720), set_overlay=Mock())
    display._sim_roi_source_available = lambda _: True
    display._boxes = lambda: ()
    display._drawn_key = None
    display.render_ms = deque(maxlen=4)
    display_module.render.return_value = []
    display._draw(1.1)
    display.hud_renderer.render.assert_called_once()
    status.agent_state["visual_lock"] = True
    assert "VISUAL LOCK" in status.agent_text(1.1)
    display._drawn_key = None
    display.hud_renderer.render.reset_mock()
    display._draw(1.2)
    display.hud_renderer.render.assert_not_called()


def test_selector_pause_waits_for_inflight_job_and_invalidates_old_epoch():
    from jetson.deepstream.async_target_selection import AsyncDeepStreamTargetSelector
    selector = AsyncDeepStreamTargetSelector.__new__(AsyncDeepStreamTargetSelector)
    paused, work = threading.Event(), threading.Lock()
    epoch = NS(value=0)
    selector._roi_gate = (paused, work, epoch)
    selector._latest = {"old": "result"}
    selector._latest_received_ns = 1
    finished = threading.Event()
    work.acquire()
    thread = threading.Thread(target=lambda: (selector.pause_for_roi(), finished.set()))
    thread.start()
    assert paused.wait(1.) and not finished.is_set()
    work.release()
    thread.join(1.)
    assert finished.is_set() and epoch.value == 1 and selector._latest is None
    selector.resume_from_roi()
    assert not paused.is_set() and selector._roi_epoch() == 1


def test_selector_worker_skips_paused_and_old_jobs_before_any_policy_work(monkeypatch):
    from queue import Queue
    from jetson.deepstream import async_target_selection as module
    paused, processed = threading.Event(), threading.Event()
    calls = []
    class Lock:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            processed.set()
    class Selector:
        def __init__(self, config):
            pass
        def reconfigure(self, *_):
            calls.append("reconfigure")
        def set_operator(self, **_):
            calls.append("operator")
        def select_snapshot(self, snapshot, **_):
            calls.append("select")
            return snapshot
    monkeypatch.setattr(module, "DeepStreamTargetSelector", Selector)
    requests, results = Queue(), Queue()
    gate = (paused, Lock(), NS(value=1))
    paused.set()
    thread = threading.Thread(target=module._worker, args=({}, "a", requests, results, gate))
    thread.start()
    assert results.get(timeout=1.)["type"] == "ready"
    message = dict(snapshot=PerceptionSnapshotV2(sequence=0, frame=image_frame()).model_dump(mode="json"),
                   config_digest="a", roi_epoch=1)
    requests.put(message)
    assert processed.wait(1.) and not calls
    processed.clear()
    paused.clear()
    requests.put(dict(message, roi_epoch=0))
    assert processed.wait(1.) and not calls
    processed.clear()
    requests.put(message)
    assert processed.wait(1.) and calls == ["operator", "select"]
    assert results.get(timeout=1.)["roi_epoch"] == 1
    requests.put(dict(type="stop"))
    thread.join(1.)
    assert not thread.is_alive()
