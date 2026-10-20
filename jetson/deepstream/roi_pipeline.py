"""GStreamer routing/sideband adapter. Imported only by the live pipeline."""
import json
import threading
import time
from queue import Empty, Full, Queue

from common.gst_buffer_probe import probe_buffers
from common.rtp_identity import parse_rtp_identity
from jetson.visual_lock.return_identity import ReturnRtpMapper, SourceIdentity


class RoiPipeline:
    def __init__(self, roi, pipeline, gst, glib, pyds, *, identity_bind, seed_bind, fail, stage_clock,
                 selector=None):
        import zmq
        self.roi, self.pipeline, self.gst, self.pyds = roi, pipeline, gst, pyds
        self.fail = fail
        self.stage_clock = stage_clock
        self.selector = selector
        self.done = threading.Event()
        self.done.set()
        self.closed = threading.Event()
        self.stopping = False
        self.close_error = None
        self.mapper = ReturnRtpMapper()
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
        self.attach("mux", "src", self.route)
        self.attach("primary", "sink", self.gpu_entry)
        if pipeline.get_by_name("tracker") is not None:
            self.attach("tracker", "sink", self.gpu_entry)
        self.attach("osd", "sink", self.owner_event, gst.PadProbeType.EVENT_DOWNSTREAM)
        self.attach("encoder", "sink", self.encoder_input)
        self.attach("return_wire", "src", self.return_rtp,
                    gst.PadProbeType.BUFFER | gst.PadProbeType.BUFFER_LIST)

    def attach(self, name, pad_name, callback, flags=None):
        element = self.pipeline.get_by_name(name)
        if element is None:
            raise RuntimeError(f"ROI pipeline element missing: {name}")
        element.get_static_pad(pad_name).add_probe(flags or self.gst.PadProbeType.BUFFER, callback, None)

    def route(self, _pad, info, _data):
        buffer = info.get_buffer()
        if buffer is None:
            return self.gst.PadProbeReturn.OK
        # nvtracker may output asynchronously. One outstanding buffer is the
        # barrier: no detector GPU work can race acquisition/updates downstream.
        if not self.done.wait(2.0):
            self.fail("ROI frame-owner barrier timed out")
            return self.gst.PadProbeReturn.DROP
        if self.stopping:
            return self.gst.PadProbeReturn.DROP
        self.done.clear()
        self.frame_started_ns = time.perf_counter_ns()
        self.stage_clock.record_infer_input(time.monotonic())
        skipped = self.roi.begin_frame()
        selector = getattr(self, "selector", None)
        self.roi.selector_pause_ms = 0.0
        if selector is not None:
            pause_started = time.perf_counter_ns()
            try:
                selector.pause_for_roi() if skipped else selector.resume_from_roi()
            except RuntimeError as exc:
                self.done.set()
                self.fail(str(exc))
                return self.gst.PadProbeReturn.DROP
            self.roi.selector_pause_ms = (time.perf_counter_ns()-pause_started)/1e6
        if skipped:
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
        try:
            buffer = info.get_buffer()
            if buffer is not None and self.pyds.gst_buffer_get_nvds_batch_meta(hash(buffer)) is None:
                self.roi.process(None, buffer)
            return _metadata_probe(pad, info, data)
        except Exception as exc:
            self.fail(f"ROI metadata fault: {exc}")
            return self.gst.PadProbeReturn.DROP
        finally:
            self.roi.frame_pipeline_ms = (time.perf_counter_ns()-self.frame_started_ns)/1e6
            self.roi.pipeline_timings.append((self.roi.frame_pipeline_ms, self.roi.vpi_ms,
                                             self.roi.yolo_skipped))
            self.done.set()

    def gpu_entry(self, pad, info, _data):
        if info.get_buffer() is not None:
            name = pad.get_parent_element().get_name()
            self.roi.counts[name + "_input_buffers"] += 1
            if self.roi.yolo_skipped:
                self.roi.counts["unexpected_gpu_input"] += 1
                self.fail(f"ROI bypass contract violated: {name} received a skipped frame")
                return self.gst.PadProbeReturn.DROP
        return self.gst.PadProbeReturn.OK

    def record_frame(self, meta, frame):
        if frame.source_identity_verified:
            self.mapper.record_verified_frame(int(meta.frame_num), SourceIdentity(
                frame.tracker_session, frame.frame_id, frame.source_time_ns, frame.source_clock_domain))

    def encoder_input(self, _pad, info, _data):
        buffer = info.get_buffer()
        if buffer is not None and 0 <= int(buffer.pts) < (1 << 63)-1:
            batch = self.pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch is not None:
                from jetson.deepstream.pipeline import _iterate_meta
                metas = list(_iterate_meta(self.pyds, batch.frame_meta_list, self.pyds.NvDsFrameMeta.cast))
                if len(metas) == 1:
                    self.mapper.record_encoder_input(int(buffer.pts), int(metas[0].frame_num))
        return self.gst.PadProbeReturn.OK

    def return_rtp(self, _pad, info, _data):
        for buffer in probe_buffers(info, buffer_flag=self.gst.PadProbeType.BUFFER,
                                    buffer_list_flag=self.gst.PadProbeType.BUFFER_LIST):
            ok, mapping = buffer.map(self.gst.MapFlags.READ)
            if not ok:
                continue
            try:
                packet = parse_rtp_identity(bytes(mapping.data[:12]))
                pts = int(buffer.pts)
                message = self.mapper.encoded_packet(pts if 0 <= pts < (1 << 63)-1 else None, packet)
                if message is not None:
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
                self.roi.enqueue(json.loads(raw))
            except (ValueError, TypeError):
                self.roi.counts["invalid_commands"] += 1
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
        if (event is not None and event.type == self.gst.EventType.EOS
                or structure is not None and structure.get_name() == "idcs-roi-close"):
            try:
                self.roi.close()
            except Exception as exc:
                self.close_error = str(exc)
            finally:
                self.closed.set()
        return self.gst.PadProbeReturn.OK

    def close(self):
        self.stopping = True
        # A serialized event traverses the queue on its streaming task, after
        # its last buffer. Never close VPI from the GLib/main thread.
        event = self.gst.Event.new_custom(self.gst.EventType.CUSTOM_DOWNSTREAM,
                                         self.gst.Structure.new_empty("idcs-roi-close"))
        sink = self.pipeline.get_by_name("roi_owner").get_static_pad("sink")
        sent = self.closed.is_set() or sink.send_event(event)
        clean = sent and self.closed.wait(2.0)
        self.glib.source_remove(self.timer)
        self.pull.close(0)
        self.pub.close(0)
        self.context.term()
        if not clean:
            raise RuntimeError("ROI frame-owner teardown did not acknowledge close")
        if self.close_error is not None:
            raise RuntimeError(f"ROI frame-owner teardown failed: {self.close_error}")
