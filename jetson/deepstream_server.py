"""DeepStream-powered pipeline for the Jetson server."""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import zmq
from common.ranging import (
    KnownSizeRangingConfig,
    iter_distance_estimates,
    iter_ranging_candidates,
    resolve_class_label,
)
from common.schemas import Box, CamState, DetectionMsg, detection_msg_to_json

try:  # pragma: no cover - used for typing only
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover
    TYPE_CHECKING = False

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import zmq
    from common.camera import CameraIntrinsics
    from jetson.controller import ControlLoop


_GST_INITIALIZED = False
Gst = None
GLib = None
pyds = None

_MAX_OSD_ELEMENTS = 16


@dataclass
class DeepStreamPipelineConfig:
    """Configuration for the DeepStream ingestion and inference pipeline."""

    udp_port: int
    width: int
    height: int
    fps: float
    infer_config: Path
    batch_size: int = 1
    payload_type: int = 96
    jitter_latency_ms: int = 200
    live_source: bool = True
    udp_buffer_size: Optional[int] = None
    batched_push_timeout_us: Optional[int] = None
    engine_path: Optional[Path] = None
    gpu_id: int = 0
    nvbuf_memory_type: Optional[int] = None
    return_host: Optional[str] = None
    return_port: Optional[int] = None
    return_payload_type: int = 97
    return_bitrate: Optional[int] = None
    return_vbv_size: Optional[int] = None
    return_iframe_interval: Optional[int] = None
    return_idr_interval: Optional[int] = None
    return_insert_sps_pps: bool = True
    record_path: Optional[Path] = None
    record_container: str = "mp4"


@dataclass
class DeepStreamRuntime:
    """Runtime dependencies needed to translate DeepStream output."""

    header_provider: Callable[[], Mapping[str, Any]]
    result_publisher: Optional[zmq.Socket]
    controller: Optional["ControlLoop"]
    camera_intrinsics: "CameraIntrinsics"
    ranging_cfg: KnownSizeRangingConfig
    class_labels: Mapping[str, str]
    cli_json_logs: bool = False
    control_tick_interval_s: float = 0.05
    ranging_log_interval_s: float = 0.5
    cam_state_provider: Optional[Callable[[], Optional["CamState"]]] = None


@dataclass
class _DetectionEvent:
    """Work item emitted from the GStreamer streaming thread."""

    boxes: List[Box]
    frame_width: int
    frame_height: int
    pts_ms: Optional[int]
    rx_ts_ms: int
    frame_num: int


@dataclass
class _RangingLogState:
    """Mutable state tracking ranging log cadence."""

    last_log_time: float = 0.0
    logged_once: bool = False
    last_target_idx: Optional[int] = None


@dataclass
class _OverlayFrame:
    """Overlay instructions keyed by DeepStream frame number."""

    frame_num: int
    message: DetectionMsg
    cam_state: Optional["CamState"]
    created_at: float


class DeepStreamServer:
    """Owns the GStreamer pipeline that ingests RTP and runs nvinfer."""

    def __init__(
        self,
        config: DeepStreamPipelineConfig,
        runtime: Optional[DeepStreamRuntime] = None,
    ) -> None:
        self._config = config
        self._pipeline = None
        self._loop = None
        self._runtime = runtime
        self._pending_events: "queue.Queue[_DetectionEvent]" = queue.Queue()
        self._idle_lock = threading.Lock()
        self._idle_scheduled = False
        self._control_tick_id: Optional[int] = None
        self._ranging_log = logging.getLogger("jetson.ranging")
        self._ranging_state = _RangingLogState()
        self._overlay_lock = threading.Lock()
        self._overlay_frames: Dict[int, _OverlayFrame] = {}

    def run(self) -> None:
        """Build the pipeline and enter the GLib main loop."""

        pipeline = self._ensure_pipeline()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logging.info(
            "Starting DeepStream pipeline on UDP port %d (batch=%d, infer=%s)",
            self._config.udp_port,
            self._config.batch_size,
            self._config.infer_config,
        )

        pipeline.set_state(Gst.State.PLAYING)
        try:
            self._loop = GLib.MainLoop()
            self._loop.run()
        except KeyboardInterrupt:
            logging.info("DeepStream pipeline interrupted by user")
        finally:
            logging.info("Stopping DeepStream pipeline")
            pipeline.set_state(Gst.State.NULL)
            bus.remove_signal_watch()
            if self._control_tick_id is not None:
                GLib.source_remove(self._control_tick_id)
                self._control_tick_id = None
            self._loop = None

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.quit()

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        _require_gstreamer()
        _require_pyds()
        cfg = self._config
        pipeline = Gst.Pipeline.new("deepstream-pipeline")
        if pipeline is None:
            raise RuntimeError("failed to allocate DeepStream pipeline")

        src = Gst.ElementFactory.make("udpsrc", "udp-source")
        jitter = Gst.ElementFactory.make("rtpjitterbuffer", "rtp-jitter")
        depay = Gst.ElementFactory.make("rtph264depay", "rtp-depay")
        parse = Gst.ElementFactory.make("h264parse", "h264-parse")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        queue = Gst.ElementFactory.make("queue", "decoder-queue")
        mux = Gst.ElementFactory.make("nvstreammux", "stream-mux")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-infer")
        post_queue = Gst.ElementFactory.make("queue", "post-infer-queue")
        osd = Gst.ElementFactory.make("nvdsosd", "on-screen-display")
        convert = Gst.ElementFactory.make("nvvideoconvert", "return-convert")
        capsfilter = Gst.ElementFactory.make("capsfilter", "return-capsfilter")
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "return-encoder")
        encode_parse = Gst.ElementFactory.make("h264parse", "return-parse")
        tee = Gst.ElementFactory.make("tee", "return-tee")

        elements = [
            src,
            jitter,
            depay,
            parse,
            decoder,
            queue,
            mux,
            pgie,
            post_queue,
            osd,
            convert,
            capsfilter,
            encoder,
            encode_parse,
            tee,
        ]
        if any(element is None for element in elements):
            missing = [
                name
                for name, element in (
                    ("udpsrc", src),
                    ("rtpjitterbuffer", jitter),
                    ("rtph264depay", depay),
                    ("h264parse", parse),
                    ("nvv4l2decoder", decoder),
                    ("queue", queue),
                    ("nvstreammux", mux),
                    ("nvinfer", pgie),
                    ("queue", post_queue),
                    ("nvdsosd", osd),
                    ("nvvideoconvert", convert),
                    ("capsfilter", capsfilter),
                    ("nvv4l2h264enc", encoder),
                    ("h264parse", encode_parse),
                    ("tee", tee),
                )
                if element is None
            ]
            raise RuntimeError(
                "missing DeepStream GStreamer plugins: " + ", ".join(missing)
            )

        pipeline.add(src)
        pipeline.add(jitter)
        pipeline.add(depay)
        pipeline.add(parse)
        pipeline.add(decoder)
        pipeline.add(queue)
        pipeline.add(mux)
        pipeline.add(pgie)
        pipeline.add(post_queue)
        pipeline.add(osd)
        pipeline.add(convert)
        pipeline.add(capsfilter)
        pipeline.add(encoder)
        pipeline.add(encode_parse)
        pipeline.add(tee)

        if not src.link(jitter):
            raise RuntimeError("failed to link udpsrc -> rtpjitterbuffer")
        if not jitter.link(depay):
            raise RuntimeError("failed to link rtpjitterbuffer -> rtph264depay")
        if not depay.link(parse):
            raise RuntimeError("failed to link rtph264depay -> h264parse")
        if not parse.link(decoder):
            raise RuntimeError("failed to link h264parse -> nvv4l2decoder")
        if not decoder.link(queue):
            raise RuntimeError("failed to link nvv4l2decoder -> queue")

        mux_sink_pad = mux.get_request_pad("sink_0")
        if mux_sink_pad is None:
            raise RuntimeError("failed to acquire nvstreammux sink pad")
        queue_src_pad = queue.get_static_pad("src")
        if queue_src_pad is None:
            raise RuntimeError("queue is missing src pad")
        if queue_src_pad.link(mux_sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("failed to link queue -> nvstreammux")

        if not mux.link(pgie):
            raise RuntimeError("failed to link nvstreammux -> nvinfer")
        if not pgie.link(post_queue):
            raise RuntimeError("failed to link nvinfer -> post queue")
        if not post_queue.link(osd):
            raise RuntimeError("failed to link post queue -> nvdsosd")
        if not osd.link(convert):
            raise RuntimeError("failed to link nvdsosd -> nvvideoconvert")
        if not convert.link(capsfilter):
            raise RuntimeError("failed to link nvvideoconvert -> capsfilter")
        if not capsfilter.link(encoder):
            raise RuntimeError("failed to link capsfilter -> nvv4l2h264enc")
        if not encoder.link(encode_parse):
            raise RuntimeError("failed to link nvv4l2h264enc -> h264parse")
        if not encode_parse.link(tee):
            raise RuntimeError("failed to link h264parse -> tee")

        src.set_property("port", int(cfg.udp_port))
        caps = Gst.Caps.from_string(
            "application/x-rtp,media=video,encoding-name=H264,payload=%d,clock-rate=90000"
            % int(cfg.payload_type)
        )
        src.set_property("caps", caps)
        if cfg.udp_buffer_size is not None:
            src.set_property("buffer-size", int(cfg.udp_buffer_size))

        jitter.set_property("latency", int(cfg.jitter_latency_ms))
        prop_info = None
        if getattr(jitter, "find_property", None) is not None:
            prop_info = jitter.find_property("drop-on-late")
        if prop_info is not None:
            jitter.set_property("drop-on-late", True)
        else:  # pragma: no cover - defensive logging for Jetson plugin variants
            logging.debug("rtpjitterbuffer drop-on-late property unavailable; skipping")
        jitter.set_property("mode", 4)

        mux.set_property("batch-size", int(cfg.batch_size))
        mux.set_property("width", int(cfg.width))
        mux.set_property("height", int(cfg.height))
        mux.set_property("live-source", 1 if cfg.live_source else 0)
        mux.set_property("gpu-id", int(cfg.gpu_id))
        if cfg.nvbuf_memory_type is not None:
            try:
                mux.set_property("nvbuf-memory-type", int(cfg.nvbuf_memory_type))
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.warning(
                    "failed to set nvstreammux nvbuf-memory-type to %s: %s",
                    cfg.nvbuf_memory_type,
                    exc,
                )
        timeout_us = cfg.batched_push_timeout_us
        if timeout_us is None:
            if cfg.fps > 0.0:
                timeout_us = int(max(1, 1_000_000 / cfg.fps))
            else:
                timeout_us = 33_000
        mux.set_property("batched-push-timeout", int(timeout_us))

        pgie.set_property("config-file-path", str(cfg.infer_config))
        if cfg.engine_path is not None:
            pgie.set_property("model-engine-file", str(cfg.engine_path))
        if cfg.batch_size > 0:
            pgie.set_property("batch-size", int(cfg.batch_size))
        pgie.set_property("gpu-id", int(cfg.gpu_id))

        osd.set_property("process-mode", 0)
        osd.set_property("display-clock", 0)
        osd.set_property("gpu-id", int(cfg.gpu_id))

        convert.set_property("gpu-id", int(cfg.gpu_id))
        if cfg.nvbuf_memory_type is not None:
            try:
                convert.set_property("nvbuf-memory-type", int(cfg.nvbuf_memory_type))
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.warning(
                    "failed to set nvvideoconvert nvbuf-memory-type to %s: %s",
                    cfg.nvbuf_memory_type,
                    exc,
                )

        caps = _build_return_caps(cfg.width, cfg.height, cfg.fps)
        capsfilter.set_property("caps", caps)

        bitrate = cfg.return_bitrate if cfg.return_bitrate else None
        if not bitrate or bitrate <= 0:
            bitrate = int(4_000_000)
        encoder.set_property("bitrate", int(bitrate))
        encoder.set_property("control-rate", 1)
        encoder.set_property("maxperf-enable", 1)
        encoder.set_property("preset-level", 1)
        encoder.set_property("device-id", int(cfg.gpu_id))
        iframe_interval = _default_gop(cfg.fps)
        if cfg.return_iframe_interval:
            iframe_interval = max(1, int(cfg.return_iframe_interval))
        idr_interval = iframe_interval
        if cfg.return_idr_interval:
            idr_interval = max(1, int(cfg.return_idr_interval))
        encoder.set_property("iframeinterval", int(iframe_interval))
        encoder.set_property("idrinterval", int(idr_interval))
        encoder.set_property("insert-sps-pps", 1 if cfg.return_insert_sps_pps else 0)
        encoder.set_property("EnableTwopassCBR", True)
        if cfg.return_vbv_size:
            try:
                encoder.set_property("vbv-size", int(cfg.return_vbv_size))
            except Exception:
                logging.warning("failed to set encoder vbv-size; continuing with default")
        encode_parse.set_property("config-interval", 1)
        encode_parse.set_property("disable-passthrough", True)

        branch_count = 0

        if cfg.return_host and cfg.return_port:
            udp_queue = Gst.ElementFactory.make("queue", "return-udp-queue")
            pay = Gst.ElementFactory.make("rtph264pay", "return-pay")
            udp_sink = Gst.ElementFactory.make("udpsink", "return-udp")
            if None in (udp_queue, pay, udp_sink):
                raise RuntimeError("failed to allocate DeepStream return video branch")
            pipeline.add(udp_queue)
            pipeline.add(pay)
            pipeline.add(udp_sink)
            tee_src_pad = tee.get_request_pad("src_%u")
            queue_sink_pad = udp_queue.get_static_pad("sink")
            if tee_src_pad is None or queue_sink_pad is None:
                raise RuntimeError("failed to acquire pads for return UDP branch")
            if tee_src_pad.link(queue_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("failed to link tee -> return UDP queue")
            if not udp_queue.link(pay):
                raise RuntimeError("failed to link return queue -> rtph264pay")
            if not pay.link(udp_sink):
                raise RuntimeError("failed to link rtph264pay -> udpsink")
            pay.set_property("pt", int(cfg.return_payload_type))
            pay.set_property("config-interval", 1)
            udp_sink.set_property("host", str(cfg.return_host))
            udp_sink.set_property("port", int(cfg.return_port))
            udp_sink.set_property("sync", False)
            udp_sink.set_property("async", False)
            udp_sink.set_property("qos", False)
            branch_count += 1

        if cfg.record_path is not None:
            try:
                cfg.record_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.warning(
                    "failed to create directory for DeepStream recording %s: %s",
                    cfg.record_path.parent,
                    exc,
                )
            record_queue = Gst.ElementFactory.make("queue", "return-record-queue")
            container = (cfg.record_container or "mp4").lower()
            mux_name = "return-mux"
            if container in {"mp4", "mov"}:
                mux_elem = Gst.ElementFactory.make("qtmux", mux_name)
            elif container in {"mkv", "matroska"}:
                mux_elem = Gst.ElementFactory.make("matroskamux", mux_name)
            else:
                logging.warning(
                    "unsupported DeepStream record_container %r; defaulting to mp4",
                    cfg.record_container,
                )
                container = "mp4"
                mux_elem = Gst.ElementFactory.make("qtmux", mux_name)
            file_sink = Gst.ElementFactory.make("filesink", "return-file")
            if None in (record_queue, mux_elem, file_sink):
                raise RuntimeError("failed to allocate DeepStream recording branch")
            pipeline.add(record_queue)
            pipeline.add(mux_elem)
            pipeline.add(file_sink)
            tee_src_pad = tee.get_request_pad("src_%u")
            queue_sink_pad = record_queue.get_static_pad("sink")
            if tee_src_pad is None or queue_sink_pad is None:
                raise RuntimeError("failed to acquire pads for recording branch")
            if tee_src_pad.link(queue_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("failed to link tee -> recording queue")
            record_queue_src = record_queue.get_static_pad("src")
            mux_sink_pad = mux_elem.get_request_pad("video_%u")
            if record_queue_src is None or mux_sink_pad is None:
                raise RuntimeError("failed to acquire mux pads for recording branch")
            if record_queue_src.link(mux_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("failed to link recording queue -> qtmux")
            if not mux_elem.link(file_sink):
                raise RuntimeError("failed to link qtmux -> filesink")
            file_sink.set_property("location", str(cfg.record_path))
            file_sink.set_property("sync", False)
            file_sink.set_property("async", False)
            if container in {"mp4", "mov"}:
                mux_elem.set_property("faststart", True)
            branch_count += 1

        if branch_count == 0:
            drop_sink = Gst.ElementFactory.make("fakesink", "return-fakesink")
            if drop_sink is None:
                raise RuntimeError("failed to allocate fallback return sink")
            pipeline.add(drop_sink)
            tee_src_pad = tee.get_request_pad("src_%u")
            drop_sink_pad = drop_sink.get_static_pad("sink")
            if tee_src_pad is None or drop_sink_pad is None:
                raise RuntimeError("failed to acquire pads for fallback sink")
            if tee_src_pad.link(drop_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("failed to link tee -> fallback sink")
            drop_sink.set_property("sync", False)
            drop_sink.set_property("async", False)

        pgie_src = pgie.get_static_pad("src")
        if pgie_src is None:
            raise RuntimeError("nvinfer element missing src pad")
        pgie_src.add_probe(Gst.PadProbeType.BUFFER, self._on_infer_buffer)

        osd_sink = osd.get_static_pad("sink")
        if osd_sink is None:
            raise RuntimeError("nvdsosd element missing sink pad")
        osd_sink.add_probe(Gst.PadProbeType.BUFFER, self._on_osd_buffer)

        if self._runtime and self._runtime.controller is not None:
            interval = max(0.001, float(self._runtime.control_tick_interval_s))
            interval_ms = int(max(1, round(interval * 1000.0)))
            self._control_tick_id = GLib.timeout_add(
                interval_ms,
                self._on_control_tick,
                priority=GLib.PRIORITY_DEFAULT,
            )

        self._pipeline = pipeline
        return pipeline

    def _on_bus_message(self, bus, message):
        msg_type = message.type
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logging.error("DeepStream pipeline error: %s", err)
            if debug:
                logging.debug("DeepStream debug: %s", debug)
            self.stop()
        elif msg_type == Gst.MessageType.EOS:
            logging.info("DeepStream pipeline reached EOS")
            self.stop()

    def _on_control_tick(self):
        runtime = self._runtime
        if runtime is None or runtime.controller is None:
            return False
        try:
            runtime.controller.tick(time.monotonic())
        except Exception:  # pragma: no cover - defensive logging
            logging.exception("control tick raised an exception")
        return True

    def _on_infer_buffer(self, pad, info):  # pragma: no cover - GPU callback
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except AttributeError:
                break
            if frame_meta is None:
                l_frame = l_frame.next
                continue

            frame_w = int(frame_meta.source_frame_width or self._config.width)
            frame_h = int(frame_meta.source_frame_height or self._config.height)
            pts_ns = getattr(frame_meta, "buf_pts", 0)
            pts_ms = int(pts_ns / 1_000_000) if pts_ns else None
            rx_ts_ms = int(time.monotonic_ns() / 1_000_000)
            frame_num = int(getattr(frame_meta, "frame_num", 0))

            boxes: List[Box] = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except AttributeError:
                    break
                if obj_meta is None:
                    l_obj = l_obj.next
                    continue

                rect = obj_meta.rect_params
                left = float(getattr(rect, "left", 0.0))
                top = float(getattr(rect, "top", 0.0))
                width = float(getattr(rect, "width", 0.0))
                height = float(getattr(rect, "height", 0.0))
                conf = float(getattr(obj_meta, "confidence", 0.0))
                class_id = str(getattr(obj_meta, "class_id", ""))

                if frame_w <= 0 or frame_h <= 0 or width <= 0.0 or height <= 0.0:
                    l_obj = l_obj.next
                    continue

                x = max(0.0, min(1.0, left / frame_w))
                y = max(0.0, min(1.0, top / frame_h))
                w = max(0.0, min(1.0 - x, width / frame_w))
                h = max(0.0, min(1.0 - y, height / frame_h))

                box = Box(x=x, y=y, w=w, h=h, cls=class_id, conf=conf)
                boxes.append(box)

                l_obj = l_obj.next

            if boxes:
                event = _DetectionEvent(
                    boxes=boxes,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    pts_ms=pts_ms,
                    rx_ts_ms=rx_ts_ms,
                    frame_num=frame_num,
                )
                self._queue_detection_event(event)

            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    def _on_osd_buffer(self, pad, info):  # pragma: no cover - GPU callback
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except AttributeError:
                break
            if frame_meta is None:
                l_frame = l_frame.next
                continue

            frame_num = int(getattr(frame_meta, "frame_num", 0))
            overlay = self._acquire_overlay_frame(frame_num)
            if overlay is not None:
                try:
                    self._apply_overlay(batch_meta, frame_meta, overlay)
                except Exception:  # pragma: no cover - defensive logging
                    logging.exception("failed to apply DeepStream overlay metadata")

            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    def _queue_detection_event(self, event: _DetectionEvent) -> None:
        if self._runtime is None:
            return
        self._pending_events.put(event)
        with self._idle_lock:
            if not self._idle_scheduled:
                self._idle_scheduled = True
                GLib.idle_add(self._drain_detection_queue, priority=GLib.PRIORITY_DEFAULT)

    def _drain_detection_queue(self):
        try:
            while True:
                event = self._pending_events.get_nowait()
                self._handle_detection_event(event)
        except queue.Empty:
            pass
        finally:
            with self._idle_lock:
                self._idle_scheduled = False
        return False

    def _handle_detection_event(self, event: _DetectionEvent) -> None:
        runtime = self._runtime
        if runtime is None:
            return

        header: Mapping[str, int]
        try:
            header = runtime.header_provider()
        except Exception:  # pragma: no cover - defensive
            logging.exception("failed to fetch header for DeepStream detection")
            header = {"frame_id": 0, "src_ts_ms": 0}

        frame_id = int(header.get("frame_id", 0))
        src_ts_ms = int(header.get("src_ts_ms", 0))

        class_labels = runtime.class_labels or {}
        for box in event.boxes:
            if class_labels:
                box.cls = resolve_class_label(box.cls, class_labels)

        log_rows = []
        if runtime.ranging_cfg.enabled:
            try:
                candidates = list(
                    iter_ranging_candidates(
                        event.boxes,
                        (event.frame_width, event.frame_height),
                        class_labels,
                        runtime.ranging_cfg,
                    )
                )
                estimates = list(
                    iter_distance_estimates(
                        candidates, runtime.camera_intrinsics, runtime.ranging_cfg
                    )
                )
                box_index_map = {id(box): idx for idx, box in enumerate(event.boxes)}
                for estimate in estimates:
                    target_box = estimate.candidate.box
                    target_box.distance_m = estimate.distance_m
                    target_box.distance_src = estimate.source
                    idx = box_index_map.get(id(target_box))
                    if idx is not None:
                        log_rows.append(
                            {
                                "idx": idx,
                                "label": target_box.cls,
                                "conf": target_box.conf,
                                "source": estimate.source,
                                "pixel_size_px": estimate.pixel_size_px,
                                "distance_m": estimate.distance_m,
                            }
                        )
            except Exception:  # pragma: no cover - defensive logging
                logging.exception("failed to compute ranging estimates from DeepStream metadata")

        infer_ts_ms = event.pts_ms if event.pts_ms is not None else event.rx_ts_ms

        msg = DetectionMsg(
            frame_id=frame_id,
            src_ts_ms=src_ts_ms,
            rx_ts_ms=event.rx_ts_ms,
            infer_ts_ms=infer_ts_ms,
            img_w=event.frame_width,
            img_h=event.frame_height,
            boxes=event.boxes,
        )

        controller = runtime.controller
        if controller is not None:
            try:
                controller.update_detection(msg)
            except Exception:  # pragma: no cover - defensive logging
                logging.exception("controller.update_detection raised an exception")

        if log_rows:
            self._maybe_log_ranging(msg, log_rows, runtime)

        publisher = runtime.result_publisher
        if publisher is not None:
            try:
                publisher.send_string(detection_msg_to_json(msg), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception:  # pragma: no cover - defensive logging
                logging.exception("failed to publish DeepStream detection message")

        if controller is not None:
            try:
                controller.tick(time.monotonic())
            except Exception:  # pragma: no cover - defensive logging
                logging.exception("controller.tick raised an exception")

        self._submit_overlay_frame(event.frame_num, msg)

    def _maybe_log_ranging(
        self,
        msg: DetectionMsg,
        rows: List[Mapping[str, object]],
        runtime: DeepStreamRuntime,
    ) -> None:
        now_log = time.monotonic()
        state = self._ranging_state
        log_interval = max(0.0, float(runtime.ranging_log_interval_s))

        should_log = False
        if not state.logged_once:
            should_log = True
        elif msg.target_idx != state.last_target_idx:
            should_log = True
        elif (now_log - state.last_log_time) >= log_interval:
            should_log = True

        if not should_log:
            return

        target_idx = msg.target_idx
        smoothed = msg.target_distance_smoothed_m
        annotated_rows = []
        for row in rows:
            entry = dict(row)
            idx = entry.get("idx")
            if target_idx is not None and idx == target_idx:
                entry["target"] = True
                if smoothed is not None:
                    entry["distance_smoothed_m"] = smoothed
            annotated_rows.append(entry)

        payload = {
            "frame_id": msg.frame_id,
            "src_ts_ms": msg.src_ts_ms,
            "rx_ts_ms": msg.rx_ts_ms,
            "infer_ts_ms": msg.infer_ts_ms,
            "ranging": [_round_for_log(row) for row in annotated_rows],
        }

        if runtime.cli_json_logs:
            self._ranging_log.info(json.dumps(_round_for_log(payload)))
        else:
            self._ranging_log.info(
                _format_ranging_log(
                    frame_id=payload["frame_id"],
                    src_ts_ms=payload["src_ts_ms"],
                    rx_ts_ms=payload["rx_ts_ms"],
                    infer_ts_ms=payload["infer_ts_ms"],
                    rows=annotated_rows,
                )
            )

        state.logged_once = True
        state.last_target_idx = msg.target_idx
        state.last_log_time = now_log

    def _submit_overlay_frame(self, frame_num: int, msg: DetectionMsg) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            msg_copy = msg.model_copy(deep=True)
        except AttributeError:  # pragma: no cover - pydantic compatibility
            msg_copy = DetectionMsg(**json.loads(detection_msg_to_json(msg)))

        cam_state: Optional[CamState] = None
        provider = runtime.cam_state_provider
        if provider is not None:
            try:
                cam_state = provider()
            except Exception:  # pragma: no cover - defensive logging
                logging.exception("failed to fetch CamState for DeepStream overlay")
                cam_state = None

        overlay = _OverlayFrame(
            frame_num=frame_num,
            message=msg_copy,
            cam_state=cam_state,
            created_at=time.monotonic(),
        )
        with self._overlay_lock:
            self._overlay_frames[frame_num] = overlay
            self._prune_overlay_cache_locked()

    def _acquire_overlay_frame(self, frame_num: int) -> Optional[_OverlayFrame]:
        with self._overlay_lock:
            overlay = self._overlay_frames.pop(frame_num, None)
            if overlay is None:
                self._prune_overlay_cache_locked()
            return overlay

    def _prune_overlay_cache_locked(self) -> None:
        now = time.monotonic()
        stale_keys = [
            key
            for key, value in list(self._overlay_frames.items())
            if (now - value.created_at) > 1.0
        ]
        for key in stale_keys:
            self._overlay_frames.pop(key, None)
        if len(self._overlay_frames) <= _MAX_OSD_ELEMENTS:
            return
        ordered = sorted(
            self._overlay_frames.items(), key=lambda item: item[1].created_at
        )
        for key, _ in ordered[:-_MAX_OSD_ELEMENTS]:
            self._overlay_frames.pop(key, None)

    def _apply_overlay(self, batch_meta, frame_meta, overlay: _OverlayFrame) -> None:
        msg = overlay.message
        runtime = self._runtime
        if runtime is None:
            return

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if display_meta is None:
            return

        frame_w = max(1, int(msg.img_w or self._config.width))
        frame_h = max(1, int(msg.img_h or self._config.height))

        rect_index = 0
        text_index = 0
        line_index = 0
        circle_index = 0

        for idx, box in enumerate(msg.boxes[:_MAX_OSD_ELEMENTS]):
            rect = display_meta.rect_params[rect_index]
            left = _clip_to_int(box.x * frame_w, frame_w)
            top = _clip_to_int(box.y * frame_h, frame_h)
            width = max(1, _clip_to_int(box.w * frame_w, frame_w))
            height = max(1, _clip_to_int(box.h * frame_h, frame_h))
            rect.left = left
            rect.top = top
            rect.width = width
            rect.height = height
            rect.border_width = 3
            rect.has_bg_color = 0
            is_target = msg.target_idx is not None and idx == msg.target_idx
            if is_target:
                rect.border_color.set(1.0, 0.84, 0.0, 1.0)
            else:
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)
            rect_index += 1

            if text_index < _MAX_OSD_ELEMENTS:
                label = _format_box_label(box, idx, is_target)
                text = display_meta.text_params[text_index]
                _configure_text(
                    text,
                    label,
                    left,
                    max(0, top - 18),
                    font_size=16,
                )
                text_index += 1

        laser_origin = _clip_point(msg.laser_origin_px, frame_w, frame_h)
        laser_dot = _clip_point(msg.laser_dot_px, frame_w, frame_h)
        if laser_origin and circle_index < _MAX_OSD_ELEMENTS:
            circle = display_meta.circle_params[circle_index]
            circle.center_x = float(laser_origin[0])
            circle.center_y = float(laser_origin[1])
            circle.radius = 6.0
            circle.circle_color.set(0.0, 0.8, 1.0, 1.0)
            circle.has_bg_color = 0
            circle_index += 1
        if laser_dot and circle_index < _MAX_OSD_ELEMENTS:
            circle = display_meta.circle_params[circle_index]
            circle.center_x = float(laser_dot[0])
            circle.center_y = float(laser_dot[1])
            circle.radius = 5.0
            if msg.laser_on_target:
                circle.circle_color.set(0.0, 1.0, 0.0, 1.0)
            else:
                circle.circle_color.set(1.0, 0.25, 0.25, 1.0)
            circle.has_bg_color = 0
            circle_index += 1
        if laser_origin and laser_dot and line_index < _MAX_OSD_ELEMENTS:
            line = display_meta.line_params[line_index]
            line.x1 = float(laser_origin[0])
            line.y1 = float(laser_origin[1])
            line.x2 = float(laser_dot[0])
            line.y2 = float(laser_dot[1])
            line.line_width = 3
            if msg.laser_on_target:
                line.line_color.set(0.0, 1.0, 0.0, 1.0)
            else:
                line.line_color.set(0.0, 0.7, 1.0, 1.0)
            line_index += 1

        if text_index < _MAX_OSD_ELEMENTS:
            range_text = _format_range_text(msg)
            if range_text:
                text = display_meta.text_params[text_index]
                _configure_text(text, range_text, 20, 40, font_size=18)
                text_index += 1

        if text_index < _MAX_OSD_ELEMENTS and overlay.cam_state is not None:
            cam_state = overlay.cam_state
            az = _degrees(cam_state.pan)
            el = _degrees(cam_state.tilt)
            fov = runtime.camera_intrinsics.fov_deg or (None, None)
            if fov and fov[0] is not None and fov[1] is not None:
                attitude = f"Az {az:.1f}° | El {el:.1f}° | FOV {fov[0]:.1f}/{fov[1]:.1f}°"
            else:
                attitude = f"Az {az:.1f}° | El {el:.1f}°"
            text = display_meta.text_params[text_index]
            _configure_text(text, attitude, 20, 70, font_size=18)
            text_index += 1

        display_meta.num_rects = rect_index
        display_meta.num_labels = text_index
        display_meta.num_lines = line_index
        display_meta.num_circles = circle_index

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def _configure_text(text_param, text: str, x: int, y: int, *, font_size: int = 18) -> None:
    text_param.display_text = text
    text_param.x_offset = int(x)
    text_param.y_offset = int(y)
    text_param.font_params.font_name = "Sans"
    text_param.font_params.font_size = int(font_size)
    text_param.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    text_param.set_bg_clr = 1
    text_param.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)


def _format_box_label(box: Box, idx: int, is_target: bool) -> str:
    parts = [str(box.cls or idx)]
    try:
        conf = float(box.conf)
    except (TypeError, ValueError):
        conf = None
    if conf is not None and math.isfinite(conf):
        parts.append(f"{conf:.2f}")
    distance = box.distance_m
    if distance is not None:
        try:
            distance_val = float(distance)
        except (TypeError, ValueError):
            distance_val = None
        if distance_val is not None and math.isfinite(distance_val):
            suffix = f"{distance_val:.1f}m"
            if box.distance_src:
                suffix += f" ({box.distance_src})"
            parts.append(suffix)
    if is_target:
        parts.append("TARGET")
    return " ".join(parts)


def _format_range_text(msg: DetectionMsg) -> Optional[str]:
    distance = msg.laser_range_m
    if distance is None:
        distance = msg.target_distance_smoothed_m
    if distance is None:
        return None
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    source = msg.laser_range_source or ""
    if source:
        return f"Range {value:.1f} m ({source})"
    return f"Range {value:.1f} m"


def _clip_point(
    point: Optional[Tuple[float, float]], width: int, height: int
) -> Optional[Tuple[int, int]]:
    if point is None:
        return None
    x, y = point
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (_clip_to_int(x, width), _clip_to_int(y, height))


def _clip_to_int(value: float, limit: int) -> int:
    if not math.isfinite(value):
        return 0
    return int(max(0, min(limit - 1, round(float(value)))))


def _degrees(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return math.degrees(value)


def _build_return_caps(width: int, height: int, fps: float):
    num, den = _fps_to_fraction(fps)
    caps_str = (
        "video/x-raw(memory:NVMM),format=NV12,width=%d,height=%d,framerate=%d/%d"
        % (int(width), int(height), num, den)
    )
    return Gst.Caps.from_string(caps_str)


def _default_gop(fps: float) -> int:
    if fps and fps > 0:
        return max(1, int(round(fps)))
    return 30


def _fps_to_fraction(fps: float) -> Tuple[int, int]:
    if not fps or fps <= 0.0:
        return (30, 1)
    num = fps
    den = 1
    if abs(round(fps) - fps) > 1e-3:
        den = 1000
        num = fps * den
    num_i = max(1, int(round(num)))
    den_i = max(1, int(round(den)))
    return (num_i, den_i)


def _require_gstreamer() -> None:
    global _GST_INITIALIZED, Gst, GLib
    if _GST_INITIALIZED:
        return
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import GLib as _GLib, Gst as _Gst

    _Gst.init(None)
    Gst = _Gst
    GLib = _GLib
    _GST_INITIALIZED = True


def _require_pyds() -> None:
    global pyds
    if pyds is not None:
        return
    import pyds as _pyds

    pyds = _pyds


def _round_for_log(value, precision: int = 4):
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, list):
        return [_round_for_log(v, precision) for v in value]
    if isinstance(value, tuple):
        return [_round_for_log(v, precision) for v in value]
    if isinstance(value, dict):
        return {k: _round_for_log(v, precision) for k, v in value.items()}
    return value


def _format_ranging_log(
    *,
    frame_id,
    src_ts_ms,
    rx_ts_ms,
    infer_ts_ms,
    rows,
    precision: int = 4,
) -> str:
    header = (
        f"frame={frame_id} src_ts={src_ts_ms} rx_ts={rx_ts_ms} infer_ts={infer_ts_ms}"
    )
    formatted_rows = []
    for row in rows:
        idx = row.get("idx")
        label = row.get("label") or "?"
        distance = row.get("distance_m")
        source = row.get("source")
        px_size = row.get("pixel_size_px")
        conf = row.get("conf")
        target = row.get("target")
        smoothed = row.get("distance_smoothed_m")

        idx_text = f"#{idx}" if idx is not None else "#?"
        label_text = f"{label}".strip() or "?"
        dist_text = "dist=?"
        if isinstance(distance, (int, float)):
            src_hint = {
                "height": "h",
                "width": "w",
                "average": "avg",
            }.get(str(source), str(source) if source else "")
            suffix = f" ({src_hint})" if src_hint else ""
            dist_text = f"dist={float(distance):.{precision}f}m{suffix}"

        px_text = "px=?"
        if isinstance(px_size, (int, float)):
            px_text = f"px={float(px_size):.{precision}f}"

        conf_text = "conf=?"
        if isinstance(conf, (int, float)):
            conf_text = f"conf={float(conf):.{precision}f}"

        extras = []
        if target:
            extras.append("target")
        if isinstance(smoothed, (int, float)):
            extras.append(f"ema={float(smoothed):.{precision}f}")
        extras_text = f" ({', '.join(extras)})" if extras else ""

        formatted_rows.append(
            f"{idx_text}:{label_text} {conf_text} {px_text} {dist_text}{extras_text}"
        )

    body = "; ".join(formatted_rows)
    return f"{header} | {body}"


__all__ = [
    "DeepStreamPipelineConfig",
    "DeepStreamServer",
    "DeepStreamRuntime",
]
