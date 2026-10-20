"""GStreamer routing/sideband adapter. Imported only by the live pipeline."""
import json
import threading
import time
from collections import deque
from queue import Empty, Full, Queue

from common.gst_buffer_probe import probe_buffers
from common.perception import RoiStateV2
from common.rtp_identity import parse_rtp_identity
from common.visual_roi_seed import VisualRoiRelease, VisualRoiSeed, VisualRoiSelect
from jetson.visual_lock.return_identity import ReturnRtpMapper, SourceIdentity


class RoiPipeline:
    def __init__(self, roi, pipeline, gst, glib, pyds, *, identity_bind, seed_bind, fail, stage_clock,
                 selector=None, metadata_pad, metadata_data):
        import zmq
        from jetson.deepstream.pipeline import _metadata_probe
        self.roi, self.pipeline, self.gst, self.pyds = roi, pipeline, gst, pyds
        self.fail = fail
        self.stage_clock = stage_clock
        self.selector = selector
        self.done = threading.Event()
        self.done.set()
        self.frames = {}
        self.frames_lock = threading.Lock()
        self.barrier_engaged = False
        self.barrier_timeout_s = 2.0
        self.metadata_skipped = False
        self.selector_paused = False
        self.cache_expiry_sent = False
        self.closed = threading.Event()
        self.stopping = False
        self.close_error = None
        self.mapper = ReturnRtpMapper()
        self.active = False
        self.probes = {}
        self.identity_live = False
        self.identity_generation = 0
        self.identity_frames = deque(maxlen=128)
        self.owner = None
        self.transition = None
        self.blocks = []
        self.block_ready = threading.Event()
        self.retired = threading.Event()
        self.metadata_pad, self.metadata_data = metadata_pad, metadata_data
        self.normal_metadata = _metadata_probe
        self.metadata_id = metadata_pad.add_probe(gst.PadProbeType.BUFFER, _metadata_probe, metadata_data)
        metadata_data[1].roi_idle_state = RoiStateV2(capable=roi.capable, owns_selection=False)
        self.messages = Queue(maxsize=128)
        self.context = zmq.Context()
        self.pub = self.context.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 128)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(identity_bind)
        self.pull = self.context.socket(zmq.PULL)
        self.pull.setsockopt(zmq.RCVHWM, 4)
        self.pull.setsockopt(zmq.MAXMSGSIZE, 1024)
        self.pull.setsockopt(zmq.LINGER, 0)
        self.pull.bind(seed_bind)
        self.timer = glib.timeout_add(10, self.poll)
        self.glib = glib
        self.last_status_s = 0.0

    def attach(self, name, pad_name, callback, flags=None, data=None):
        element = self.pipeline.get_by_name(name)
        if element is None:
            raise RuntimeError(f"ROI pipeline element missing: {name}")
        pad = element.get_static_pad(pad_name)
        probe_id = pad.add_probe(flags or self.gst.PadProbeType.BUFFER, callback, data)
        self.probes[name, pad_name, callback.__name__] = (pad, probe_id)

    def detach(self, name, pad_name):
        for key in [key for key in self.probes if key[:2] == (name, pad_name)]:
            probe = self.probes.pop(key)
            probe[0].remove_probe(probe[1])

    def block(self, pad):
        # IDLE is blocking even after this callback returns OK. Do graph/state
        # changes only on GLib, never on the queue's own streaming task.
        self.block_ready.clear()
        def ready(*_):
            self.block_ready.set()
            return self.gst.PadProbeReturn.OK
        self.blocks.append((pad, pad.add_probe(self.gst.PadProbeType.IDLE, ready, None)))

    def unblock(self):
        for pad, probe_id in self.blocks:
            pad.remove_probe(probe_id)
        self.blocks.clear()

    def start(self):
        self.transition = "starting"
        self.block(self.metadata_pad)

    def activate(self):
        # The tracker/primary src pad is idle and blocked. The base graph has
        # no extra element; create the single RGBA owner only for a workflow.
        sink = self.pipeline.get_by_name("osd_rgba_convert").get_static_pad("sink")
        owner = self.gst.ElementFactory.make("queue", "roi_owner")
        if owner is None:
            raise RuntimeError("ROI owner queue unavailable")
        owner.set_property("max-size-buffers", 1)
        owner.set_property("max-size-bytes", 0)
        owner.set_property("max-size-time", 0)
        self.owner = owner
        self.pipeline.add(owner)
        self.metadata_pad.unlink(sink)
        if (self.metadata_pad.link(owner.get_static_pad("sink")) != self.gst.PadLinkReturn.OK
                or owner.get_static_pad("src").link(sink) != self.gst.PadLinkReturn.OK):
            raise RuntimeError("ROI owner queue link failed")
        self.metadata_pad.remove_probe(self.metadata_id)
        self.attach("osd", "sink", self.owner_event, self.gst.PadProbeType.EVENT_DOWNSTREAM)
        self.attach("osd", "sink", self.metadata_probe, data=self.metadata_data)
        self.attach("primary", "sink", self.gpu_entry)
        if self.pipeline.get_by_name("tracker") is not None:
            self.attach("tracker", "sink", self.gpu_entry)
        self.active = True
        self.attach("mux", "src", self.route)
        if not owner.sync_state_with_parent():
            raise RuntimeError("ROI owner queue state sync failed")
        self.roi.counts["probe_activations"] += 1
        self.transition = None
        self.unblock()

    def sync_identity(self):
        wanted = self.active and self.roi.cache_enabled and self.transition is None
        if wanted == self.identity_live:
            return
        with self.frames_lock:
            self.identity_live = wanted
            self.identity_generation += 1
            self.identity_frames.clear()
        if wanted:
            # An old callback retains its old mapper, so it cannot contaminate
            # a new lease even if a packet map released the GIL during removal.
            self.mapper = ReturnRtpMapper()
            self.attach("encoder", "sink", self.encoder_input)
            self.attach("return_wire", "src", self.return_rtp,
                        self.gst.PadProbeType.BUFFER | self.gst.PadProbeType.BUFFER_LIST)
        else:
            self.detach("return_wire", "src")
            self.detach("encoder", "sink")
            while True:
                try:
                    self.messages.get_nowait()
                except Empty:
                    break

    def lifecycle(self):
        if self.transition == "starting":
            if self.block_ready.is_set():
                self.activate()
        elif self.transition == "draining":
            # Dropped metadata must not turn lease cleanup into a fatal or
            # unbounded video stall. No lock/native work is admitted here.
            drained = self.done.is_set()
            if not drained and time.monotonic()-self.transition_started_s >= self.barrier_timeout_s:
                drained = self.roi.native_idle.is_set()
                if drained:
                    self.roi.counts["probe_drain_timeouts"] += 1
            if self.block_ready.is_set() and drained:
                self.transition = "isolating"
                self.block(self.metadata_pad)
        elif self.transition == "isolating":
            if self.block_ready.is_set():
                self.retired.clear()
                if self.send_owner_event("idcs-roi-retire"):
                    self.transition = "retiring"
        elif self.transition == "retiring":
            if self.retired.is_set():
                self.transition = "detaching"
                self.block(self.owner.get_static_pad("src"))
        elif self.transition == "detaching":
            if self.block_ready.is_set():
                self.deactivate()
        elif not self.active:
            if not self.roi.commands.empty():
                self.roi.begin_frame(allow_seed=False)
            if self.roi.cache_enabled:
                self.start()
        elif (not self.roi.cache_enabled and not self.roi.owns_selection
                and not self.roi.selecting and not self.roi.cache._frames
                and not self.roi.abort_reason):
            self.transition = "draining"
            self.transition_started_s = time.monotonic()
            self.block(self.pipeline.get_by_name("mux").get_static_pad("src"))
        self.sync_identity()
        self.roi.probe_status = dict(roi_probes_active=self.active,
            roi_identity_probes_active=self.identity_live, roi_probe_transition=self.transition,
            roi_owner_present=self.owner is not None, roi_identity_history_frames=len(self.identity_frames))

    def deactivate(self):
        # Mux admission is blocked and every admitted frame has finished. The
        # owner's src IDLE block additionally drains the complete return tail.
        self.active = False
        for name, pad_name, _ in list(self.probes):
            self.detach(name, pad_name)
        owner = self.owner
        # Stop the task while its pads remain blocked/linked. Deactivation
        # flushes pad waits; nothing can wake and push into an unlinked tail.
        if owner.set_state(self.gst.State.NULL) == self.gst.StateChangeReturn.FAILURE:
            raise RuntimeError("ROI owner queue stop failed")
        sink = self.pipeline.get_by_name("osd_rgba_convert").get_static_pad("sink")
        self.metadata_pad.unlink(owner.get_static_pad("sink"))
        owner.get_static_pad("src").unlink(sink)
        if self.metadata_pad.link(sink) != self.gst.PadLinkReturn.OK:
            raise RuntimeError("ROI ordinary path restore failed")
        self.metadata_id = self.metadata_pad.add_probe(
            self.gst.PadProbeType.BUFFER, self.normal_metadata, self.metadata_data)
        self.pipeline.remove(owner)
        self.owner = None
        with self.frames_lock:
            self.frames.clear()
            self.done.set()
        self.metadata_skipped = False
        self.barrier_engaged = False
        if self.selector_paused:
            self.selector.resume_from_roi()
            self.selector_paused = False
        self.roi.counts["probe_deactivations"] += 1
        self.transition = None
        self.unblock()

    def receive(self, message):
        kind = message.get("type")
        cls = {"visual_roi_release_v1": VisualRoiRelease,
               "visual_roi_select_v1": VisualRoiSelect}.get(kind, VisualRoiSeed)
        request = cls.from_mapping(message)
        if isinstance(request, VisualRoiSeed):
            if not self.roi.capable or request.tracker_session != self.roi.tracker_session:
                self.roi._notify(request, False, (self.roi.error or "ROI unavailable")
                    if not self.roi.capable else "source session mismatch")
                return
            identity = SourceIdentity(request.tracker_session, request.frame_id,
                                      request.source_time_ns, request.source_clock_domain)
            with self.frames_lock:
                ready = identity in self.identity_frames
            if not self.identity_live or not ready or self.transition is not None:
                self.roi._notify(request, False,
                    "ROI identity history warming or frame unavailable; open Select box again")
                return
        self.roi.enqueue(message)
        if not self.active and self.transition is None:
            # No streaming router exists yet (or it has drained). Process only
            # routing/control state here, never native/cache resources.
            self.roi.begin_frame(allow_seed=False)

    def send_owner_event(self, name):
        event = self.gst.Event.new_custom(self.gst.EventType.CUSTOM_DOWNSTREAM,
                                         self.gst.Structure.new_empty(name))
        return self.owner.get_static_pad("sink").send_event(event)

    def route(self, _pad, info, _data):
        buffer = info.get_buffer()
        if buffer is None:
            return self.gst.PadProbeReturn.OK
        # Idle and selecting preserve nvinfer/nvtracker overlap. A pending seed
        # first drains ALL earlier YOLO frames, not just the most recent one.
        # Release/loss drains the last bypass frame before reopening YOLO.
        needs_barrier = (self.roi.barrier_requested.is_set() or self.roi.owns_selection
                         or self.barrier_engaged) and self.roi.abort_reason is None
        if needs_barrier:
            self.roi.counts["barrier_waits"] += 1
            if not self.done.wait(self.barrier_timeout_s):
                self.roi.counts["barrier_timeouts"] += 1
                self.roi.abort("ROI released: downstream frame delayed")
                # A delayed queue/probe must not hold up YOLO. An already
                # executing native call, however, cannot safely be interrupted
                # or overlapped with inference. Drain only that call; shutdown
                # can always wake this router. Teardown stays on its owner.
                while not self.roi.native_idle.wait(.05):
                    if self.stopping:
                        return self.gst.PadProbeReturn.OK
        if self.stopping:
            return self.gst.PadProbeReturn.OK
        skipped = (self.roi.begin_frame(allow_seed=needs_barrier)
                   if self.roi.abort_reason is None and self.transition in (None, "starting") else False)
        self.roi.yolo_skipped = skipped
        self.barrier_engaged = skipped
        with self.frames_lock:
            self.frames[int(buffer.pts)] = (time.perf_counter_ns() if skipped else None, skipped)
            self.roi.counts["max_inflight_frames"] = max(
                self.roi.counts["max_inflight_frames"], len(self.frames))
            self.done.clear()
        selector = getattr(self, "selector", None)
        self.roi.selector_pause_ms = 0.0
        if selector is not None and self.selector_paused != skipped:
            pause_started = time.perf_counter_ns()
            try:
                selector.pause_for_roi() if skipped else selector.resume_from_roi()
            except RuntimeError as exc:
                self.done.set()
                self.fail(str(exc))
                return self.gst.PadProbeReturn.DROP
            self.roi.selector_pause_ms = (time.perf_counter_ns()-pause_started)/1e6
            self.selector_paused = skipped
        if skipped:
            self.stage_clock.record_infer_input(time.monotonic())
            # Same NV12 caps and batch-size=1 on both paths. Calling the common
            # queue sink directly transfers this buffer around BOTH GPU elements;
            # no interval approximation and no fake detector metadata.
            sink = self.pipeline.get_by_name("roi_owner").get_static_pad("sink")
            flow = sink.chain(buffer)
            if flow != self.gst.FlowReturn.OK:
                self.done.set()
                self.fail(f"ROI bypass failed: {flow}")
            return self.gst.PadProbeReturn.DROP
        return self.gst.PadProbeReturn.OK

    def metadata_probe(self, pad, info, data):
        from jetson.deepstream.pipeline import _metadata_probe
        buffer = info.get_buffer()
        key = None if buffer is None else int(buffer.pts)
        with self.frames_lock:
            started, self.metadata_skipped = self.frames.get(key, (None, False))
        try:
            if buffer is not None and self.pyds.gst_buffer_get_nvds_batch_meta(hash(buffer)) is None:
                self.roi.process(None, buffer)
            return _metadata_probe(pad, info, data)
        except Exception as exc:
            self.fail(f"ROI metadata fault: {exc}")
            return self.gst.PadProbeReturn.DROP
        finally:
            if started is not None:
                self.roi.frame_pipeline_ms = (time.perf_counter_ns()-started)/1e6
                self.roi.pipeline_timings.append((self.roi.frame_pipeline_ms, self.roi.vpi_ms,
                                                 self.metadata_skipped))
            with self.frames_lock:
                self.frames.pop(key, None)
                if not self.frames:
                    self.done.set()

    def gpu_entry(self, pad, info, _data):
        if info.get_buffer() is not None:
            name = pad.get_parent_element().get_name()
            self.roi.counts[name + "_input_buffers"] += 1
            with self.frames_lock:
                skipped = self.frames.get(int(info.get_buffer().pts), (None, False))[1]
            if skipped:
                self.roi.counts["unexpected_gpu_input"] += 1
                self.fail(f"ROI bypass contract violated: {name} received a skipped frame")
                return self.gst.PadProbeReturn.DROP
        return self.gst.PadProbeReturn.OK

    def record_frame(self, meta, frame):
        if self.identity_live and self.roi.cache_enabled and frame.source_identity_verified:
            self.mapper.record_verified_frame(int(meta.frame_num), SourceIdentity(
                frame.tracker_session, frame.frame_id, frame.source_time_ns, frame.source_clock_domain))

    def encoder_input(self, _pad, info, _data):
        if not self.identity_live or not self.roi.cache_enabled:
            return self.gst.PadProbeReturn.OK
        mapper = self.mapper
        buffer = info.get_buffer()
        if buffer is not None and 0 <= int(buffer.pts) < (1 << 63)-1:
            batch = self.pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch is not None:
                from jetson.deepstream.pipeline import _iterate_meta
                metas = list(_iterate_meta(self.pyds, batch.frame_meta_list, self.pyds.NvDsFrameMeta.cast))
                if len(metas) == 1:
                    mapper.record_encoder_input(int(buffer.pts), int(metas[0].frame_num))
        return self.gst.PadProbeReturn.OK

    def return_rtp(self, _pad, info, _data):
        if not self.identity_live or not self.roi.cache_enabled:
            return self.gst.PadProbeReturn.OK
        mapper, generation = self.mapper, self.identity_generation
        for buffer in probe_buffers(info, buffer_flag=self.gst.PadProbeType.BUFFER,
                                    buffer_list_flag=self.gst.PadProbeType.BUFFER_LIST):
            if not self.identity_live or not self.roi.cache_enabled or generation != self.identity_generation:
                break
            ok, mapping = buffer.map(self.gst.MapFlags.READ)
            if not ok:
                continue
            try:
                packet = parse_rtp_identity(bytes(mapping.data[:12]))
                pts = int(buffer.pts)
                message = mapper.encoded_packet(pts if 0 <= pts < (1 << 63)-1 else None, packet)
                if message is not None:
                    with self.frames_lock:
                        if (not self.identity_live or not self.roi.cache_enabled
                                or generation != self.identity_generation):
                            continue
                        self.identity_frames.append(SourceIdentity(message["tracker_session"],
                            message["frame_id"], message["source_time_ns"], message["source_clock_domain"]))
                        try:
                            self.messages.put_nowait(message)
                        except Full:
                            self.roi.counts["identity_dropped"] += 1
            except ValueError:
                self.roi.counts["invalid_return_rtp"] += 1
            finally:
                buffer.unmap(mapping)
        return self.gst.PadProbeReturn.OK

    def poll(self):
        import zmq
        for _ in range(4):
            try:
                raw = self.pull.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                self.receive(json.loads(raw))
            except (ValueError, TypeError, AttributeError):
                self.roi.counts["invalid_commands"] += 1
        # Capability/freshness still works before the first selecting lease,
        # without invoking ROI code from ordinary frame/packet probes.
        stats = self.metadata_data[1]
        latest = stats.latest_source_frame
        if latest is not None:
            frame, seen_s = latest
            domain = "jetson_monotonic" if self.roi.source == "camera" else "pc_monotonic"
            if (frame.source_identity_verified and frame.tracker_session == self.roi.tracker_session
                    and frame.source_clock_domain == domain and (frame.width, frame.height) == (1280, 720)):
                self.roi.last_verified_s = seen_s
                self.roi.current_frame_id = frame.frame_id
        if stats.roi_idle_state.capable != self.roi.capable:
            stats.roi_idle_state = RoiStateV2(capable=self.roi.capable, owns_selection=False)
        try:
            self.lifecycle()
        except RuntimeError as exc:
            self.fail(str(exc))
            return False
        for queue in (self.roi.messages, self.messages):
            for _ in range(64):
                try:
                    message = queue.get_nowait()
                except Empty:
                    break
                try:
                    self.pub.send_json(message, flags=zmq.NOBLOCK)
                except zmq.Again:
                    self.roi.counts["sideband_dropped"] += 1
        now = time.monotonic()
        if (self.active and self.transition is None
                and not self.roi.cache_enabled and not self.roi.owns_selection
                and (self.roi.selecting is not None or self.roi.cache._frames)
                and not self.cache_expiry_sent):
            self.cache_expiry_sent = True
            if not self.send_owner_event("idcs-roi-expire-cache"):
                self.cache_expiry_sent = False
        if now-self.last_status_s >= .1:
            try:
                self.pub.send_json(self.roi.source_status(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            self.last_status_s = now
        return True

    def owner_event(self, _pad, info, _data):
        event = info.get_event()
        structure = event.get_structure() if event is not None else None
        if structure is not None and structure.get_name() == "idcs-roi-expire-cache":
            try:
                self.roi.expire_cache()
            finally:
                self.cache_expiry_sent = False
        if structure is not None and structure.get_name() == "idcs-roi-retire":
            try:
                self.roi.retire()
            except Exception as exc:
                self.close_error = str(exc)
                self.fail(f"ROI owner retirement failed: {exc}")
            finally:
                self.retired.set()
        if (event is not None and event.type == self.gst.EventType.EOS
                or structure is not None and structure.get_name() == "idcs-roi-close"):
            try:
                self.roi.close()
            except Exception as exc:
                self.close_error = str(exc)
            finally:
                self.done.set()
                self.closed.set()
        return self.gst.PadProbeReturn.OK

    def close(self):
        self.stopping = True
        self.roi.abort("runtime stopping")
        self.done.set()
        self.unblock()
        # A serialized event traverses the queue on its streaming task, after
        # its last buffer. Never close VPI from the GLib/main thread.
        if self.owner is None:
            self.roi.close()
            self.closed.set()
        sent = self.closed.is_set() or self.send_owner_event("idcs-roi-close")
        clean = sent and self.closed.wait(2.0)
        self.glib.source_remove(self.timer)
        self.pull.close(0)
        self.pub.close(0)
        self.context.term()
        for name, pad_name, _ in list(self.probes):
            self.detach(name, pad_name)
        self.active = self.identity_live = False
        self.transition = None
        self.identity_frames.clear()
        self.roi.probe_status.update(roi_probes_active=False, roi_identity_probes_active=False,
            roi_probe_transition=None, roi_identity_history_frames=0)
        if not clean:
            raise RuntimeError("ROI frame-owner teardown did not acknowledge close")
        if self.close_error is not None:
            raise RuntimeError(f"ROI frame-owner teardown failed: {self.close_error}")
