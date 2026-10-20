"""Normal-runtime ROI contracts. No sockets to hosts, GPU or hardware."""
import json
import struct
import threading
import time
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
from jetson.visual_lock.return_identity import SourceIdentity
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
    wire.transition = None
    wire.stage_clock, wire.fail, wire.selector = Mock(), Mock(), Mock()
    wire.pyds = Mock()
    wire.closed = threading.Event()
    wire.close_error = None
    return wire


@pytest.fixture
def pipeline_wire(adapter, monkeypatch):
    """Gst scheduling double: real lifecycle, no sockets, threads or devices."""
    import zmq
    from jetson.deepstream import pipeline as module
    from jetson.deepstream.pipeline import VerificationStats
    gst = NS(PadProbeReturn=NS(OK="ok", DROP="drop"),
        PadProbeType=NS(BUFFER=1, BUFFER_LIST=2, EVENT_DOWNSTREAM=4, IDLE=8),
        PadLinkReturn=NS(OK="linked"), State=NS(NULL="null"),
        StateChangeReturn=NS(FAILURE="failed"), FlowReturn=NS(OK="flow-ok"),
        MapFlags=NS(READ="read"),
        EventType=NS(EOS="eos", CUSTOM_DOWNSTREAM="custom"))
    normal = Mock(return_value="ok")
    monkeypatch.setattr(module, "_metadata_probe", normal)
    class Pad:
        def __init__(self, element, name):
            self.element, self.name = element, name
            self.probes, self.peer = {}, None
            self.next_id = 0
        def add_probe(self, flags, callback, data):
            self.next_id += 1
            self.probes[self.next_id] = (flags, callback, data)
            if flags == gst.PadProbeType.IDLE:
                callback(self, None, data)
            return self.next_id
        def remove_probe(self, probe_id):
            del self.probes[probe_id]
        def link(self, peer):
            assert self.peer is None and peer.peer is None
            self.peer, peer.peer = peer, self
            return gst.PadLinkReturn.OK
        def unlink(self, peer):
            assert self.peer is peer
            self.peer = peer.peer = None
        def send_event(self, event):
            # Model queue serialized delivery explicitly through its owner pad.
            elements["osd"].pads["sink"].dispatch(gst.PadProbeType.EVENT_DOWNSTREAM,
                NS(get_event=lambda: event))
            return True
        def dispatch(self, flag, info):
            for flags, callback, data in list(self.probes.values()):
                if flags & flag:
                    callback(self, info, data)
        def chain(self, buffer):
            return "flow-ok"
    class Element:
        def __init__(self, name):
            self.name = name
            self.pads = {name: Pad(self, name) for name in ("src", "sink")}
            self.properties, self.states = {}, []
        def get_static_pad(self, name):
            return self.pads[name]
        def set_property(self, key, value):
            self.properties[key] = value
        def sync_state_with_parent(self):
            return True
        def set_state(self, state):
            self.states.append(state)
            return "success"
    elements = {name: Element(name) for name in (
        "mux", "primary", "tracker", "osd_rgba_convert", "osd", "encoder", "return_wire")}
    upstream = elements["tracker"].pads["src"]
    upstream.link(elements["osd_rgba_convert"].pads["sink"])
    pipeline = NS(get_by_name=elements.get, add=lambda e: elements.update({e.name: e}),
                  remove=lambda e: elements.pop(e.name))
    gst.ElementFactory = NS(make=lambda kind, name: Element(name))
    gst.Structure = NS(new_empty=lambda name: NS(get_name=lambda: name))
    gst.Event = NS(new_custom=lambda kind, structure: NS(type=kind, get_structure=lambda: structure))
    pub, pull = Mock(), Mock()
    pull.recv.side_effect = zmq.Again()
    context = Mock()
    context.socket.side_effect = [pub, pull]
    monkeypatch.setattr(zmq, "Context", lambda: context)
    stats = VerificationStats()
    wire = RoiPipeline(adapter[0], pipeline, gst, Mock(), Mock(),
        identity_bind="inproc://identity", seed_bind="inproc://seed", fail=Mock(),
        stage_clock=Mock(), selector=Mock(), metadata_pad=upstream,
        metadata_data=(gst, stats, Mock(), None, False, None, {}))
    stats.roi, stats.roi_pipeline = adapter[0], wire
    return wire, elements, normal


def start_wire(wire):
    lease = select(wire.roi)
    wire.lifecycle()  # Consume idle control intent; block normal metadata src.
    assert wire.transition == "starting"
    wire.lifecycle()  # Install queue/probes while blocked, then resume.
    assert wire.active and wire.identity_live and not wire.blocks
    return lease


def stop_wire(wire):
    for _ in range(8):
        wire.poll()
        if not wire.active and wire.transition is None:
            return
    pytest.fail(f"lifecycle did not detach: {wire.transition}")


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


def test_idle_and_restored_graph_have_only_the_ordinary_metadata_callback(pipeline_wire, adapter):
    wire, elements, normal = pipeline_wire
    roi, borrower, _, _ = adapter
    baseline = set(elements)
    assert not wire.active and not wire.probes and not wire.blocks
    for _ in range(120):
        for element in elements.values():
            for pad in element.pads.values():
                pad.dispatch(wire.gst.PadProbeType.BUFFER, NS(get_buffer=lambda: Mock(pts=1)))
        wire.poll()
    assert normal.call_count == 120
    assert not roi.counts and wire.mapper.mapped == 0
    assert not borrower.clone.called and not borrower.borrow.called
    lease = start_wire(wire)
    assert set(elements) == baseline | {"roi_owner"}
    roi.enqueue(VisualRoiSelect(lease.selection_id, lease.tracker_session, False).as_message())
    roi.begin_frame()
    stop_wire(wire)
    assert set(elements) == baseline
    assert not wire.probes and not wire.blocks and not wire.identity_live
    assert wire.metadata_pad.peer is elements["osd_rgba_convert"].pads["sink"]
    assert len(wire.metadata_pad.probes) == 1
    assert next(iter(wire.metadata_pad.probes.values()))[1] is normal
    assert roi.owner_ident is None
    wire.close()
    wire.fail.assert_not_called()


@pytest.mark.parametrize("expiry", ("heartbeat", "absolute", "cancel"))
def test_attached_lease_expiry_without_video_releases_owner_and_all_probes(pipeline_wire, adapter, expiry):
    wire, elements, _ = pipeline_wire
    roi, _, _, clock = adapter
    lease = start_wire(wire)
    roi.process(image_frame(), object())
    pixels = roi.cache._frames[0][1]
    owner = wire.owner
    if expiry == "heartbeat":
        clock.t += .51
    elif expiry == "absolute":
        clock.t += 15.01
        roi.selection_seen_s = clock.t
    else:
        wire.receive(VisualRoiSelect(lease.selection_id, lease.tracker_session, False).as_message())
        roi.begin_frame()
    # No route or metadata callback is needed to acknowledge cache expiry.
    stop_wire(wire)
    assert pixels.closed and not roi.cache._frames
    assert not wire.active and not wire.probes and not wire.identity_live
    assert "roi_owner" not in elements and owner.states == ["null"]
    assert roi.owner_ident is None
    assert roi.counts["probe_activations"] == roi.counts["probe_deactivations"] == 1
    wire.receive(lease.as_message())
    wire.lifecycle()
    assert not roi.cache_enabled and not wire.active  # Expired lease cannot resurrect.
    wire.close()
    wire.fail.assert_not_called()


def test_warmup_seed_refused_and_a_new_lease_requires_its_own_exact_history(pipeline_wire, adapter):
    wire, _, _ = pipeline_wire
    roi, _, backends, clock = adapter
    request = seed(image_frame())
    wire.receive(request.as_message())
    assert "identity history warming" in roi.terminal["reason"]
    assert not roi.commands.qsize() and not wire.active and not backends
    lease = start_wire(wire)
    roi.process(image_frame(), object())
    wire.receive(request.as_message())  # Cached, but never joined to return video.
    assert "identity history warming" in roi.terminal["reason"]
    stop_wire(wire)
    start_wire(wire)
    roi.process(image_frame(), object())
    identity = SourceIdentity(request.tracker_session, request.frame_id,
        request.source_time_ns, request.source_clock_domain)
    wire.identity_frames.append(identity)
    wire.receive(request.as_message())
    assert roi.barrier_requested.is_set()
    roi.begin_frame()
    assert roi.pending is not None
    # Accepted seed/catch-up/lock keeps route and owner probes, but no packet
    # or encoder identity probes once the selecting lease ends.
    roi.process(image_frame(1), object())
    assert roi.owns_selection
    wire.lifecycle()
    assert wire.active and not wire.identity_live
    assert ("return_wire", "src", "return_rtp") not in wire.probes
    assert ("encoder", "sink", "encoder_input") not in wire.probes
    assert ("mux", "src", "route") in wire.probes
    wire.receive(VisualRoiRelease(request.request_id, request.tracker_session).as_message())
    roi.begin_frame()
    roi.process(image_frame(2), object())
    stop_wire(wire)
    assert backends[-1].closed and roi.session is None
    wire.close()


def test_expiry_with_queued_heartbeat_and_seed_still_detaches_without_video(pipeline_wire, adapter):
    wire, _, _ = pipeline_wire
    roi, _, backends, clock = adapter
    lease = start_wire(wire)
    roi.process(image_frame(), object())
    request = seed(image_frame())
    wire.identity_frames.append(SourceIdentity(request.tracker_session, request.frame_id,
        request.source_time_ns, request.source_clock_domain))
    wire.receive(lease.as_message())
    wire.receive(request.as_message())
    assert not roi.commands.empty() and roi.barrier_requested.is_set()
    clock.t += .6
    stop_wire(wire)
    assert roi.terminal["request_id"] == request.request_id
    assert "identity history retired" in roi.terminal["reason"]
    wire.lifecycle()  # Consume the expired heartbeat with no router present.
    assert not wire.active and roi.commands.empty() and not roi.barrier_requested.is_set()
    assert not backends and not roi.cache._frames
    wire.close()


def test_detach_waits_for_outstanding_frames_and_native_work(pipeline_wire, adapter):
    wire, _, _ = pipeline_wire
    roi, _, _, clock = adapter
    start_wire(wire)
    roi.process(image_frame(), object())
    clock.t += .6
    wire.poll()  # Serialized expiry.
    wire.frames[12] = (None, False)
    wire.done.clear()
    wire.lifecycle()
    assert wire.transition == "draining"
    wire.lifecycle()
    assert wire.transition == "draining" and wire.owner is not None
    wire.transition_started_s -= 3.
    roi.native_idle.clear()
    wire.lifecycle()
    assert wire.transition == "draining"
    roi.native_idle.set()
    stop_wire(wire)
    assert roi.counts["probe_drain_timeouts"] == 1
    assert not wire.frames and wire.done.is_set()
    wire.fail.assert_not_called()
    wire.close()


def test_new_lease_queued_during_removal_starts_only_after_original_graph_restored(pipeline_wire, adapter):
    wire, _, _ = pipeline_wire
    roi, _, _, clock = adapter
    start_wire(wire)
    clock.t += .6
    wire.poll()
    wire.lifecycle()
    assert wire.transition == "draining"
    next_lease = VisualRoiSelect(str(uuid.uuid4()), roi.tracker_session, True)
    wire.receive(next_lease.as_message())
    info = NS(get_buffer=lambda: Mock(pts=21))
    wire.route(None, info, None)  # A router already entering the mux block.
    wire.metadata_probe(None, info, None)
    assert roi.selecting is None and not roi.commands.empty()
    stop_wire(wire)
    assert roi.selecting is None
    wire.lifecycle()
    wire.lifecycle()
    assert wire.active and wire.identity_live and roi.selecting == next_lease.selection_id
    assert not wire.identity_frames
    wire.close()


def test_loss_detaches_without_waiting_for_another_video_frame(pipeline_wire, adapter):
    wire, _, _ = pipeline_wire
    roi, _, backends, _ = adapter
    start_wire(wire)
    lock(adapter)
    info = NS(get_buffer=lambda: Mock(pts=10))
    assert wire.route(None, info, None) == "drop"
    assert roi.process(image_frame(30), object()).decision.state == "lost"
    wire.metadata_probe(None, info, None)
    assert roi._reset and wire.barrier_engaged and wire.selector_paused
    stop_wire(wire)
    assert not roi._reset and not wire.barrier_engaged and not wire.selector_paused
    assert backends[-1].closed
    wire.selector.resume_from_roi.assert_called_once()
    wire.close()


def test_idle_close_needs_no_owner_event_and_does_not_create_queue(pipeline_wire):
    wire, elements, _ = pipeline_wire
    wire.close()
    assert "roi_owner" not in elements and not wire.probes
    assert wire.closed.is_set()
    wire.glib.source_remove.assert_called_once_with(wire.timer)


def test_return_identity_joins_only_during_live_lease_and_late_callback_cannot_publish(pipeline_wire, adapter):
    wire, _, _ = pipeline_wire
    roi, _, _, clock = adapter
    packet = struct.pack("!BBHII", 0x80, 0x80 | 96, 1, 123, 456)
    buffer = Mock(pts=100)
    buffer.map.return_value = (True, NS(data=packet))
    info = NS(type=wire.gst.PadProbeType.BUFFER, get_buffer=lambda: buffer)
    wire.return_rtp(None, info, None)
    buffer.map.assert_not_called()
    start_wire(wire)
    f = image_frame()
    wire.record_frame(NS(frame_num=7), f)
    assert wire.mapper.record_encoder_input(100, 7)
    wire.return_rtp(None, info, None)
    message = wire.messages.get_nowait()
    assert message["frame_id"] == f.frame_id and message["rtp_timestamp"] == 123
    assert list(wire.identity_frames) == [SourceIdentity(f.tracker_session, f.frame_id,
        f.source_time_ns, f.source_clock_domain)]
    wire.record_frame(NS(frame_num=8), image_frame(1))
    assert wire.mapper.record_encoder_input(100, 8)
    def expire_during_map(*_):
        clock.t += .6
        wire.sync_identity()
        return True, NS(data=packet)
    buffer.map.side_effect = expire_during_map
    wire.return_rtp(None, info, None)
    assert not wire.identity_frames and wire.messages.empty() and not wire.identity_live
    assert buffer.unmap.call_count == 2
    stop_wire(wire)
    wire.close()


def test_idle_source_status_bootstraps_from_ordinary_metadata_and_expires(pipeline_wire, adapter):
    wire, _, _ = pipeline_wire
    roi, _, _, clock = adapter
    wire.metadata_data[1].latest_source_frame = (image_frame(), clock.t)
    wire.poll()
    assert roi.source_status()["source_ready"] is True
    assert not wire.active and not wire.probes
    clock.t += .26
    wire.poll()
    assert roi.source_status()["source_ready"] is False
    wire.close()


@pytest.mark.parametrize("missing_header", (False, True))
def test_real_ordinary_metadata_probe_never_calls_roi_handlers_while_idle(monkeypatch, missing_header):
    import sys
    from jetson.deepstream import pipeline as module
    meta = NS(frame_num=1, buf_pts=100, source_frame_width=1280,
              source_frame_height=720, obj_meta_list=None)
    batch = NS(frame_meta_list=object())
    pyds = NS(gst_buffer_get_nvds_batch_meta=lambda _: batch,
        NvDsFrameMeta=NS(cast=lambda x: x), NvDsObjectMeta=NS(cast=lambda x: x))
    monkeypatch.setitem(sys.modules, "pyds", pyds)
    monkeypatch.setattr(module, "_iterate_meta", lambda _p, head, _cast:
        [meta] if head is batch.frame_meta_list else [])
    monkeypatch.setattr(module, "perception_snapshot_from_metadata", lambda *_a, **_k:
        PerceptionSnapshotV2(sequence=0, frame=image_frame()))
    stats = module.VerificationStats()
    stats.roi = Mock()
    stats.roi.process.side_effect = AssertionError("idle ROI process callback")
    stats.roi_pipeline = Mock(active=False, metadata_skipped=True)
    stats.roi_pipeline.record_frame.side_effect = AssertionError("idle identity callback")
    stats.roi_idle_state = RoiStateV2(capable=True, owns_selection=False)
    transport = Mock(requires_headers=missing_header)
    transport.next_header.return_value = None
    gst = NS(PadProbeReturn=NS(OK="ok"))
    stage = Mock(consume=Mock(return_value=(1., 2., 3.)))
    assert module._metadata_probe(None, NS(get_buffer=lambda: Mock(pts=100)),
        (gst, stats, stage, transport, False, None, {})) == "ok"
    stats.roi.process.assert_not_called()
    stats.roi_pipeline.record_frame.assert_not_called()
    if not missing_header:
        published = transport.publish.call_args.args[0]
        assert published.roi_state == stats.roi_idle_state
        assert stats.latest_source_frame[0] == published.frame


@pytest.mark.parametrize("phase", ("starting", "draining", "retiring", "isolating", "detaching"))
def test_close_during_graph_transition_unblocks_pads_and_closes_owner(pipeline_wire, adapter, phase):
    wire, _, _ = pipeline_wire
    roi, _, _, clock = adapter
    if phase == "starting":
        select(roi)
        wire.lifecycle()
    else:
        start_wire(wire)
        clock.t += .6
        wire.poll()
        for _ in range(6):
            wire.lifecycle()
            if wire.transition == phase:
                break
    assert wire.transition == phase and wire.blocks
    wire.close()
    assert not wire.blocks and not wire.probes and wire.closed.is_set()
    wire.fail.assert_not_called()


@pytest.mark.parametrize("tracker", ("none", "nvdcf"))
@pytest.mark.parametrize("source", ("camera", "rtp"))
def test_enabled_idle_pipeline_description_is_identical_to_off(tmp_path, tracker, source):
    from jetson.deepstream.pipeline import _pipeline_description
    kwargs = dict(input_file=None, live_argus=source == "camera",
        rtp_input_port=5000 if source == "rtp" else None,
        argus_sensor_id=0, argus_sensor_mode=4, argus_width=1280, argus_height=720,
        argus_fps=60, nvinfer_config=tmp_path/"infer.txt", paced=False, tracker=tracker,
        gpu_osd=True, return_h264=True, return_udp_host="127.0.0.1", return_udp_port=5002,
        return_h264_file=None)
    assert _pipeline_description(**kwargs, seamless_roi=True) == _pipeline_description(**kwargs, seamless_roi=False)


def test_gstreamer_live_queue_insert_remove_and_reopen_without_gpu(adapter, monkeypatch):
    """Portable real-Gst integration; no DeepStream, GPU, network or hardware."""
    gi = pytest.importorskip("gi", reason="PyGObject/GStreamer unavailable on this machine")
    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
    import zmq
    from jetson.deepstream import pipeline as module
    from jetson.deepstream.pipeline import VerificationStats
    Gst.init(None)
    if Gst.ElementFactory.find("videotestsrc") is None:
        pytest.skip("GStreamer videotestsrc plugin unavailable")
    pipeline = Gst.parse_launch("videotestsrc is-live=true ! video/x-raw,width=16,height=16,framerate=60/1 "
        "! identity name=mux ! identity name=primary ! identity name=tracker "
        "! identity name=osd_rgba_convert ! identity name=osd ! identity name=encoder "
        "! identity name=return_wire ! fakesink sync=false")
    roi, _, _, clock = adapter
    pub, pull = Mock(), Mock()
    pull.recv.side_effect = zmq.Again()
    context = Mock()
    context.socket.side_effect = [pub, pull]
    monkeypatch.setattr(zmq, "Context", lambda: context)
    frames = []
    wire = None
    def metadata(_pad, info, _data):
        f = image_frame(len(frames))
        frames.append(f)
        if wire.active:
            roi.process(f, info.get_buffer())
        return Gst.PadProbeReturn.OK
    monkeypatch.setattr(module, "_metadata_probe", metadata)
    stats = VerificationStats()
    pyds = Mock()
    pyds.gst_buffer_get_nvds_batch_meta.return_value = NS(frame_meta_list=None)
    wire = RoiPipeline(roi, pipeline, Gst, GLib, pyds, identity_bind="inproc://identity",
        seed_bind="inproc://seed", fail=Mock(), stage_clock=Mock(),
        metadata_pad=pipeline.get_by_name("tracker").get_static_pad("src"),
        metadata_data=(Gst, stats, Mock(), None, False, None, {}))
    def until(predicate):
        deadline = time.monotonic()+3.
        while not predicate() and time.monotonic() < deadline:
            wire.poll()
            time.sleep(.005)
        assert predicate(), wire.transition
    try:
        assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
        until(lambda: len(frames) >= 3)
        for _ in range(2):
            lease = VisualRoiSelect(str(uuid.uuid4()), roi.tracker_session, True)
            wire.receive(lease.as_message())
            until(lambda: wire.active and len(roi.cache._frames) >= 3)
            owner_ident = roi.owner_ident
            assert owner_ident is not None and owner_ident != threading.get_ident()
            assert pipeline.get_by_name("roi_owner") is not None
            clock.t += .6
            until(lambda: not wire.active and wire.transition is None)
            assert pipeline.get_by_name("roi_owner") is None
            assert not wire.probes and not wire.blocks and roi.owner_ident is None
            count = len(frames)
            until(lambda: len(frames) >= count+3)
        wire.fail.assert_not_called()
        assert pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR) is None
    finally:
        wire.close()
        pipeline.set_state(Gst.State.NULL)


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
    display.roi_source = NS(session="session")
    display._roi_select, display._roi_select_sent_s = None, 0.
    display._roi_open_cache = True
    display.roi_workflow = NS(active=False, state="camera")
    display.roi_editor = NS(active=False)
    display.video = NS(frame_identity=NS(latest=lambda: None))
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


@pytest.mark.parametrize("mode", ("camera", "sim", "sim_visual_lock"))
def test_normal_editor_bootstraps_without_identity_but_seed_still_requires_it(display_module, monkeypatch, mode):
    from rpi.display.roi_editor import RoiEditor
    from rpi.display.roi_panel import RoiPanelWorkflow
    display = display_module.OperatorDisplay.__new__(display_module.OperatorDisplay)
    monkeypatch.setattr(display_module.time, "monotonic", lambda: 100.)
    display.visual_roi_trial = False
    display.status = NS(agent_state=dict(mode=mode, roi_capable=True), _agent_recent=lambda _: True,
        panel_state=dict(manual_active=True, emergency=False),
        video=NS(live=lambda _: True), panel=NS(live=lambda _: True), show_notice=Mock())
    display.roi_source = RoiSourceLease()
    assert display.roi_source.observe(dict(type="visual_roi_status_v1", source_mode=mode.removesuffix("_visual_lock"),
        managed_mode=mode, roi_capable=True, source_ready=True, control_disabled=False,
        tracker_session="run", sequence=1), 100., managed_mode=mode)
    display._roi_request_session = None
    display.video = NS(frame_identity=NS(latest=lambda: None), frame_size=(1280, 720))
    display.menu, display.settings = NS(_close=Mock()), dict(roi_box_size=120)
    display.roi_editor, display.roi_workflow = RoiEditor(), RoiPanelWorkflow()
    display.roi_seed, display.roi_lamp_pub = Mock(), None
    display._roi_select, display._roi_select_sent_s = None, 0.
    display._start_roi_preview()
    if mode == "sim_visual_lock":
        assert not display.roi_editor.active
        display.roi_seed.send_json.assert_not_called()
    else:
        assert display.roi_editor.active and display.roi_workflow.active
        sent = display.roi_seed.send_json.call_args.args[0]
        assert sent["type"] == "visual_roi_select_v1" and sent["tracker_session"] == "run"
        display._request_roi(100.)
        assert display.roi_seed.send_json.call_count == 1
        assert "displayed frame identity unavailable" in display.status.show_notice.call_args.args[0]
        # Locked mode has no identity packet probes. After loss, preview must
        # remain available for lamp-powered readable OFF to select again.
        display.roi_workflow.state, display.roi_workflow.request_id = "tracking", "lost-box"
        display._roi_request_session = "run"
        display._roi_select = None
        display._roi_terminal("lost-box", "lost texture", 100.)
        assert display._sim_roi_source_available(100.)
        display.roi_workflow.update(dict(manual_active=True, emergency=False,
            inputs_valid=dict(fire_control=True), inputs=dict(fire_control=False)),
            available=display._sim_roi_source_available(100.))
        assert display.roi_workflow.state == "selecting"
        display._publish_roi_select(100.1)
        assert display.roi_seed.send_json.call_args.args[0]["enabled"] is True
        monkeypatch.setattr(display_module.time, "monotonic", lambda: 100.6)
        assert not display._sim_roi_source_available(100.6)


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


def test_socket_adapter_seed_result_and_release_flow(adapter, pipeline_wire):
    import zmq
    roi, _, _, clock = adapter
    wire, _, _ = pipeline_wire
    start_wire(wire)
    roi.process(image_frame(), object())
    request = seed(image_frame())
    wire.identity_frames.append(SourceIdentity(request.tracker_session, request.frame_id,
        request.source_time_ns, request.source_clock_domain))
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
