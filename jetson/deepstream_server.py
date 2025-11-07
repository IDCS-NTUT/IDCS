"""DeepStream-powered pipeline for the Jetson server."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_GST_INITIALIZED = False
Gst = None
GLib = None


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


class DeepStreamServer:
    """Owns the GStreamer pipeline that ingests RTP and runs nvinfer."""

    def __init__(self, config: DeepStreamPipelineConfig) -> None:
        self._config = config
        self._pipeline = None
        self._loop = None

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
            self._loop = None

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.quit()

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        _require_gstreamer()
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


__all__ = [
    "DeepStreamPipelineConfig",
    "DeepStreamServer",
]
