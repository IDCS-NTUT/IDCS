"""DeepStream-powered pipeline for the Jetson server."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional

import zmq
from common.ranging import (
    KnownSizeRangingConfig,
    iter_distance_estimates,
    iter_ranging_candidates,
    resolve_class_label,
)
from common.schemas import Box, DetectionMsg, detection_msg_to_json

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


@dataclass
class _DetectionEvent:
    """Work item emitted from the GStreamer streaming thread."""

    boxes: List[Box]
    frame_width: int
    frame_height: int
    pts_ms: Optional[int]
    rx_ts_ms: int


@dataclass
class _RangingLogState:
    """Mutable state tracking ranging log cadence."""

    last_log_time: float = 0.0
    logged_once: bool = False
    last_target_idx: Optional[int] = None


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
        sink = Gst.ElementFactory.make("fakesink", "fake-sink")

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
            sink,
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
                    ("fakesink", sink),
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
        pipeline.add(sink)

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
        if not post_queue.link(sink):
            raise RuntimeError("failed to link post queue -> sink")

        src.set_property("port", int(cfg.udp_port))
        caps = Gst.Caps.from_string(
            "application/x-rtp,media=video,encoding-name=H264,payload=%d,clock-rate=90000"
            % int(cfg.payload_type)
        )
        src.set_property("caps", caps)
        if cfg.udp_buffer_size is not None:
            src.set_property("buffer-size", int(cfg.udp_buffer_size))

        jitter.set_property("latency", int(cfg.jitter_latency_ms))
        jitter.set_property("drop-on-late", True)
        jitter.set_property("mode", 4)

        mux.set_property("batch-size", int(cfg.batch_size))
        mux.set_property("width", int(cfg.width))
        mux.set_property("height", int(cfg.height))
        mux.set_property("live-source", 1 if cfg.live_source else 0)
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

        sink.set_property("sync", False)
        sink.set_property("async", False)

        pgie_src = pgie.get_static_pad("src")
        if pgie_src is None:
            raise RuntimeError("nvinfer element missing src pad")
        pgie_src.add_probe(Gst.PadProbeType.BUFFER, self._on_infer_buffer)

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
                )
                self._queue_detection_event(event)

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
