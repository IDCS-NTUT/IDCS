"""Return-video pipeline for the panel screen.

    udpsrc -> rtpjitterbuffer -> depay -> parse -> v4l2h264dec (hardware) --\\
                                                                           input-selector
    videotestsrc black (shown while the stream is absent) -----------------/
        -> v4l2convert (hardware I420 -> BGRx; the desktop's Wayland only takes RGB)
        -> overlaycomposition (status bar, alerts, menu, HUD) -> sink

The overlay is blended only inside its own rectangles, so the Pi's CPU cost
is decode bookkeeping plus a few small blends (about 5% of one core at
720p30). The black fallback keeps the overlay visible, with its NO VIDEO
alert, when the Jetson is not sending.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Sequence

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402

from rpi.display.render import OverlayImage  # noqa: E402
from common.return_frame_identity import ReturnFrameIdentity  # noqa: E402
from common.rtp_identity import parse_rtp_identity  # noqa: E402
from common.return_rtp import receiver_head, ReceiverRecovery  # noqa: E402

REQUIRED_ELEMENTS = (
    "udpsrc", "rtpbin", "rtpjitterbuffer", "rtph264depay", "h264parse", "v4l2h264dec",
    "videotestsrc", "input-selector", "overlaycomposition", "v4l2convert",
)
RECORDING_ELEMENTS = ("tee", "queue", "identity", "splitmuxsink", "matroskamux")


def missing_elements(sink: str, *, recording: bool = False) -> list[str]:
    names = [*REQUIRED_ELEMENTS, sink.split()[0], *(RECORDING_ELEMENTS if recording else ())]
    return [name for name in names if Gst.ElementFactory.find(name) is None]


def pipeline_description(*, port: int, jitter_ms: int, sink: str, width: int, height: int,
                         feedback_host: str | None = None, feedback_port: int = 5003,
                         sender_report_port: int = 5004, record_segment_s: int | None = None) -> str:
    # Recording branches attach to this tee per recording (rpi.display.recording).
    tee = "tee name=return_tee allow-not-linked=true ! " if record_segment_s else ""
    return (
        "input-selector name=select sync-streams=false ! "
        "v4l2convert ! video/x-raw,format=BGRx ! "
        "overlaycomposition name=overlay ! "
        f"{sink} "
        + receiver_head(port, jitter_ms, feedback_host=feedback_host,
                        feedback_port=feedback_port, sender_report_port=sender_report_port) +
        f"h264parse ! {tee}v4l2h264dec ! video/x-raw,format=I420 ! "
        "queue leaky=downstream max-size-buffers=1 ! identity name=decoded ! select.sink_0 "
        "videotestsrc is-live=true pattern=black ! "
        f"video/x-raw,format=I420,width={width},height={height},framerate=10/1 ! select.sink_1"
    )


class ReturnVideo:
    """Owns the pipeline; the overlay is swapped in whole from any thread."""

    def __init__(self, *, port: int, jitter_ms: int, sink: str, width: int = 1280,
                 height: int = 720, on_frame: Callable[[], None], feedback_host: str | None = None,
                 feedback_port: int = 5003, sender_report_port: int = 5004,
                 recording: dict | None = None) -> None:
        self.pipeline = Gst.parse_launch(
            pipeline_description(port=port, jitter_ms=jitter_ms, sink=sink, width=width, height=height,
                                 feedback_host=feedback_host, feedback_port=feedback_port,
                                 sender_report_port=sender_report_port,
                                 record_segment_s=recording["segment_s"] if recording else None))
        self.recorder = None
        if recording:
            from rpi.display.recording import ReturnRecorder
            self.recorder = ReturnRecorder(
                self.pipeline, self.pipeline.get_by_name("return_tee"), recording["dir"],
                segment_s=recording["segment_s"], budget_bytes=int(recording["max_total_gb"] * 1e9), gst=Gst)
        self._select = self.pipeline.get_by_name("select")
        self._live_pad = self._select.get_static_pad("sink_0")
        self._fallback_pad = self._select.get_static_pad("sink_1")
        self._composition: GstVideo.VideoOverlayComposition | None = None
        self._lock = threading.Lock()
        self.frame_identity = ReturnFrameIdentity()
        self.frame_size = (width, height)
        overlay = self.pipeline.get_by_name("overlay")
        overlay.connect("draw", self._on_draw)
        overlay.connect("caps-changed", self._on_caps)
        self.pipeline.get_by_name("decoded").get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._on_decoded_frame)
        self._on_frame = on_frame
        self._last_rtp_sample = None
        self.recovery = ReceiverRecovery(self.pipeline, on_rtp_packet=self._on_return_rtp)
        overlay.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._on_output_frame)
        self._select.set_property("active-pad", self._fallback_pad)
        self._showing_live = False

    def start(self) -> None:
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("return video pipeline failed to start")

    def stop(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
        self.pipeline.set_state(Gst.State.NULL)

    def receiver_stats(self) -> dict[str, int]:
        return self.recovery.stats()

    def show_live(self, live: bool) -> None:
        # Switch only on a change: every switch makes the selector resend caps
        # downstream, and comparing pad wrappers each tick switched constantly.
        if live == self._showing_live:
            return
        self._showing_live = live
        self.frame_identity.clear()
        self._last_rtp_sample = None
        self._select.set_property("active-pad", self._live_pad if live else self._fallback_pad)

    def _on_return_rtp(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        try:
            # Most packets are redundant fragments of the same AU. Avoid
            # mapping payloads/allocating identities/taking the join lock for
            # each fragment, while retaining every distinct or conflicting PTS.
            header = buffer.extract_dup(0,12)
            if len(header)!=12 or header[0]>>6!=2:
                return Gst.PadProbeReturn.OK
            pts = int(buffer.pts)
            sample=(header[4:12],pts)
            marker=bool(header[1]&128)
            if not marker and sample==self._last_rtp_sample:
                return Gst.PadProbeReturn.OK
            self._last_rtp_sample=None if marker else sample
            packet = parse_rtp_identity(header)
            self.frame_identity.push_rtp_packet(
                pts if 0 <= pts < (1 << 63) - 1 else None, packet)
        except ValueError:
            pass
        return Gst.PadProbeReturn.OK

    def _on_decoded_frame(self, _pad, info):
        buffer = info.get_buffer()
        self._on_frame()
        if buffer is not None:
            pts = int(buffer.pts)
            if 0 <= pts < (1 << 63) - 1:
                self.frame_identity.push_decoded(pts)
        return Gst.PadProbeReturn.OK

    def _on_output_frame(self, _pad, info):
        buffer = info.get_buffer()
        if self._showing_live:
            if buffer is None or int(buffer.pts) == int(Gst.CLOCK_TIME_NONE):
                self.frame_identity.unidentified_output()
            else:
                self.frame_identity.displayed(int(buffer.pts), time.monotonic())
        return Gst.PadProbeReturn.OK

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
