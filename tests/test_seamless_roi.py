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
from common.visual_roi_seed import VisualRoiRelease, VisualRoiSeed
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
    assert not roi.begin_frame()
    assert roi.process(image_frame(), object()) is None
    request = seed(image_frame())
    roi.enqueue(request.as_message())
    assert roi.begin_frame()
    result = roi.process(image_frame(1), object())
    assert result.decision.state == "acquiring"
    return request, result


def test_no_idle_vpi_exact_seed_and_release_resume_yolo(adapter):
    roi, borrower, backends, _ = adapter
    for i in range(40):
        assert not roi.begin_frame()
        assert roi.process(image_frame(i), object(), decoded_pts_ns=i*100) is None
    assert not backends and borrower.borrow.call_count == 0
    assert len(roi.cache._frames) == 16
    request = seed(image_frame(36))
    roi.enqueue(request.as_message())
    assert roi.begin_frame()  # Stops detector BEFORE acquisition.
    result = roi.process(image_frame(40), object())
    assert result.snapshot.frame.frame_id == 40
    assert roi.active == request and backends
    assert not roi.cache._frames
    assert roi.snapshot(image_frame(40), result).frame.frame_id == 40
    roi.enqueue(VisualRoiRelease(request.request_id, request.tracker_session).as_message())
    assert not roi.begin_frame()  # Release frame itself uses YOLO.
    roi.process(image_frame(41), object())
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
    pipeline = Mock()
    gst = NS(PadProbeReturn=NS(OK="ok", DROP="drop"), FlowReturn=NS(OK="flow-ok"))
    pipeline.get_by_name.return_value.get_static_pad.return_value.chain.return_value = "flow-ok"
    wire = RoiPipeline.__new__(RoiPipeline)
    wire.roi, wire.pipeline, wire.gst = roi, pipeline, gst
    wire.done = threading.Event()
    wire.done.set()
    wire.stopping = False
    wire.stage_clock, wire.fail = Mock(), Mock()
    wire.selector = Mock()
    buffer = object()
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
    wire = RoiPipeline.__new__(RoiPipeline)
    wire.roi, wire.fail = roi, Mock()
    wire.gst = NS(PadProbeReturn=NS(OK="ok", DROP="drop"))
    pad = NS(get_parent_element=lambda: NS(get_name=lambda: "primary"))
    assert wire.gpu_entry(pad, NS(get_buffer=lambda: object()), None) == "drop"
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
