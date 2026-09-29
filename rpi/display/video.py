"""Return-video pipeline for the panel screen.

    udpsrc -> rtpjitterbuffer -> depay -> parse -> v4l2h264dec (hardware) --\\
                                                                           input-selector
    videotestsrc black (shown while the stream is absent) -----------------/
        -> overlaycomposition (status bar, alerts, menu) -> sink

The overlay is blended only inside its own rectangles, so the Pi's CPU cost
is decode bookkeeping plus a few small blends (about 5% of one core at
720p30). The black fallback keeps the overlay visible, with its NO VIDEO
alert, when the Jetson is not sending.
"""

from __future__ import annotations

import threading
from typing import Callable, Sequence

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402

from rpi.display.render import OverlayImage  # noqa: E402

REQUIRED_ELEMENTS = (
    "udpsrc", "rtpjitterbuffer", "rtph264depay", "h264parse", "v4l2h264dec",
    "videotestsrc", "input-selector", "overlaycomposition",
)


def missing_elements(sink: str) -> list[str]:
    names = [*REQUIRED_ELEMENTS, sink.split()[0]]
    return [name for name in names if Gst.ElementFactory.find(name) is None]


def pipeline_description(*, port: int, jitter_ms: int, sink: str, width: int, height: int) -> str:
    caps = "application/x-rtp,media=video,encoding-name=H264,payload=97,clock-rate=90000"
    return (
        "input-selector name=select sync-streams=false ! "
        "overlaycomposition name=overlay ! "
        f"{sink} "
        f"udpsrc port={port} buffer-size=4000000 caps={caps} ! "
        f"rtpjitterbuffer latency={jitter_ms} drop-on-latency=true ! rtph264depay ! "
        "h264parse ! v4l2h264dec ! video/x-raw,format=I420 ! "
        "queue leaky=downstream max-size-buffers=1 ! identity name=decoded ! select.sink_0 "
        "videotestsrc is-live=true pattern=black ! "
        f"video/x-raw,format=I420,width={width},height={height},framerate=10/1 ! select.sink_1"
    )


class ReturnVideo:
    """Owns the pipeline; the overlay is swapped in whole from any thread."""

    def __init__(self, *, port: int, jitter_ms: int, sink: str, width: int = 1280,
                 height: int = 720, on_frame: Callable[[], None]) -> None:
        self.pipeline = Gst.parse_launch(
            pipeline_description(port=port, jitter_ms=jitter_ms, sink=sink, width=width, height=height))
        self._select = self.pipeline.get_by_name("select")
        self._live_pad = self._select.get_static_pad("sink_0")
        self._fallback_pad = self._select.get_static_pad("sink_1")
        self._composition: GstVideo.VideoOverlayComposition | None = None
        self._lock = threading.Lock()
        self.frame_size = (width, height)
        overlay = self.pipeline.get_by_name("overlay")
        overlay.connect("draw", self._on_draw)
        overlay.connect("caps-changed", self._on_caps)
        self.pipeline.get_by_name("decoded").get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, lambda _pad, _info: (on_frame(), Gst.PadProbeReturn.OK)[1])
        self._select.set_property("active-pad", self._fallback_pad)

    def start(self) -> None:
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("return video pipeline failed to start")

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)

    def show_live(self, live: bool) -> None:
        pad = self._live_pad if live else self._fallback_pad
        if self._select.get_property("active-pad") != pad:
            self._select.set_property("active-pad", pad)

    def set_overlay(self, images: Sequence[OverlayImage]) -> None:
        composition = None
        for image in images:
            buffer = Gst.Buffer.new_wrapped(image.data)
            GstVideo.buffer_add_video_meta_full(
                buffer, GstVideo.VideoFrameFlags.NONE, GstVideo.VideoFormat.BGRA,
                image.width, image.height, 1, [0, 0, 0, 0], [image.stride, 0, 0, 0])
            rectangle = GstVideo.VideoOverlayRectangle.new_raw(
                buffer, image.x, image.y, image.width, image.height,
                GstVideo.VideoOverlayFormatFlags.PREMULTIPLIED_ALPHA)
            if composition is None:
                composition = GstVideo.VideoOverlayComposition.new(rectangle)
            else:
                composition.add_rectangle(rectangle)
        with self._lock:
            self._composition = composition

    def _on_draw(self, _overlay, _sample):
        with self._lock:
            return self._composition

    def _on_caps(self, _overlay, caps, _window_width, _window_height) -> None:
        structure = caps.get_structure(0)
        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if ok_w and ok_h:
            self.frame_size = (width, height)
