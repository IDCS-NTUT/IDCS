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
import logging
from collections import deque
from typing import Callable, Sequence

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstVideo  # noqa: E402

from rpi.display.render import OverlayImage  # noqa: E402
from common.return_frame_identity import ReturnFrameIdentity  # noqa: E402
from common.rtp_identity import parse_rtp_identity  # noqa: E402
from common.return_rtp import receiver_head, ReceiverRecovery  # noqa: E402

REQUIRED_ELEMENTS = (
    "udpsrc", "rtpbin", "rtpjitterbuffer", "rtph264depay", "h264parse", "v4l2h264dec",
    "videotestsrc", "input-selector", "overlaycomposition", "v4l2convert",
)
RECORDING_ELEMENTS = ("tee", "queue", "h264parse", "matroskamux", "filesink")
log = logging.getLogger(__name__)


def missing_elements(sink: str, *, recording: bool = False) -> list[str]:
    names = [*REQUIRED_ELEMENTS, sink.split()[0], *(RECORDING_ELEMENTS if recording else ())]
    return [name for name in names if Gst.ElementFactory.find(name) is None]


def display_description(*, sink: str, width: int, height: int) -> str:
    return (
        "input-selector name=select sync-streams=false ! "
        "v4l2convert ! video/x-raw,format=BGRx ! "
        "overlaycomposition name=overlay ! "
        f"{sink} "
        "videotestsrc is-live=true pattern=black ! "
        f"video/x-raw,format=I420,width={width},height={height},framerate=30/1 ! "
        "identity name=fallback ! select."
    )


def receiver_description(*, port: int, jitter_ms: int, feedback_host: str | None = None,
                         feedback_port: int = 5003, sender_report_port: int = 5004,
                         record_segment_s: int | None = None) -> str:
    tee = "tee name=return_tee allow-not-linked=true ! " if record_segment_s else ""
    return (receiver_head(port, jitter_ms, feedback_host=feedback_host,
                          feedback_port=feedback_port, sender_report_port=sender_report_port)
            + f"h264parse ! {tee}v4l2h264dec ! video/x-raw,format=I420 ! "
            "queue leaky=downstream max-size-buffers=1 ! identity name=decoded")


class ReturnVideo:
    """Owns the pipeline; the overlay is swapped in whole from any thread."""

    def __init__(self, *, port: int, jitter_ms: int, sink: str, width: int = 1280,
                 height: int = 720, on_frame: Callable[[], None], feedback_host: str | None = None,
                 feedback_port: int = 5003, sender_report_port: int = 5004,
                 recording: dict | None = None,
                 on_receiver_lost: Callable[[str], None] = lambda _reason: None) -> None:
        self._receiver_text = receiver_description(
            port=port, jitter_ms=jitter_ms, feedback_host=feedback_host,
            feedback_port=feedback_port, sender_report_port=sender_report_port,
            record_segment_s=recording["segment_s"] if recording else None)
        self._recording_config = recording
        self._on_receiver_lost = on_receiver_lost
        self._generation = 0
        self._closed = False
        self._receiver_failed = False
        self._restart_at = None
        self._attempts = self._restarts = 0
        self._last_error = None
        self._state = "waiting"
        self._live_since = None
        self._retired = deque(maxlen=2)
        self._retiring_recorder = None
        self._stopping_since = None
        self.pipeline = Gst.parse_launch(display_description(sink=sink,width=width,height=height))
        self.recorder = None
        self._select = self.pipeline.get_by_name("select")
        self._fallback_pad = self.pipeline.get_by_name("fallback").get_static_pad("src").get_peer()
        self._live_pad = self._select.request_pad_simple("sink_%u")
        self._composition: GstVideo.VideoOverlayComposition | None = None
        self._lock = threading.Lock()
        self.frame_identity = ReturnFrameIdentity()
        self.frame_size = (width, height)
        overlay = self.pipeline.get_by_name("overlay")
        overlay.connect("draw", self._on_draw)
        overlay.connect("caps-changed", self._on_caps)
        self._on_frame = on_frame
        self._last_rtp_sample = None
        self.receiver = self._new_receiver()
        self.receiver.set_name("return_receiver_0")
        self.pipeline.add(self.receiver)
        self.receiver.set_locked_state(True)  # Start the fallback/output independently.
        if self.receiver.get_static_pad("src").link(self._live_pad) != Gst.PadLinkReturn.OK:
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("return receiver could not link to display")
        self._wire_receiver()
        overlay.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._on_output_frame)
        self._select.set_property("active-pad", self._fallback_pad)
        self._showing_live = False

    def _new_receiver(self):
        # RTP's dynamic source has not linked yet. Auto-ghosting every unlinked
        # pad would ghost its future depay/caps sink and prevent that link.
        receiver = Gst.parse_bin_from_description(self._receiver_text, False)
        output = receiver.get_by_name("decoded").get_static_pad("src")
        if not receiver.add_pad(Gst.GhostPad.new("src", output)):
            raise RuntimeError("receiver output ghost pad creation failed")
        return receiver

    def _wire_receiver(self) -> None:
        receiver, generation = self.receiver, self._generation
        def current():
            return not self._closed and generation == self._generation and not self._receiver_failed
        def decoded(pad, info):
            with self._lock:
                return self._on_decoded_frame(pad, info) if current() else Gst.PadProbeReturn.OK
        def packet(pad, info):
            with self._lock:
                return self._on_return_rtp(pad, info) if current() else Gst.PadProbeReturn.OK
        def event(_pad, info):
            if info.get_event().type == Gst.EventType.EOS:
                GLib.idle_add(self._receiver_fault, "return stream EOS", generation)
                return Gst.PadProbeReturn.DROP  # Never EOS the fallback/HUD/sink.
            return Gst.PadProbeReturn.OK
        receiver.get_by_name("decoded").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, decoded)
        receiver.get_static_pad("src").add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, event)
        self.recovery = ReceiverRecovery(receiver, on_rtp_packet=packet, bus=self.pipeline.get_bus())
        if self._recording_config:
            from rpi.display.recording import ReturnRecorder
            cfg = self._recording_config
            self.recorder = ReturnRecorder(receiver, receiver.get_by_name("return_tee"), cfg["dir"],
                segment_s=cfg["segment_s"], budget_bytes=int(cfg["max_total_gb"]*1e9), gst=Gst)

    def start(self) -> None:
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("return video pipeline failed to start")
        self.receiver.set_locked_state(False)
        if not self.receiver.sync_state_with_parent():
            self._receiver_fault("return receiver failed to start", self._generation)

    def stop(self) -> None:
        self._closed = True
        self._restart_at = None
        if self.recorder is not None:
            self.recorder.close()
        if self._retiring_recorder is not None:
            self._retiring_recorder.close()
        self.recovery.close()
        self.pipeline.set_state(Gst.State.NULL)

    def handle_message(self, message) -> bool:
        """True for receiver-local errors; output/sink failures remain fatal."""
        if message.type != Gst.MessageType.ERROR:
            return False
        for receiver in (self.receiver, *self._retired):
            if receiver is not None and (message.src == receiver or message.src.has_as_ancestor(receiver)):
                if receiver == self.receiver:
                    error, debug = message.parse_error()
                    self._receiver_fault(f"{error.message} ({debug})", self._generation)
                return True
        return False

    def _receiver_fault(self, reason: str, generation: int) -> bool:
        with self._lock:
            if self._closed or generation != self._generation or self._receiver_failed:
                return False
            self._receiver_failed = True
            self.frame_identity.clear()
        self._attempts += 1
        self._last_error = reason
        self._live_since = None
        self._state = "recovering" if self._attempts <= 5 else "failed"
        self._restart_at = time.monotonic() + min(.25*2**min(self._attempts-1, 4), 4.)
        self.show_live(False)
        self._last_rtp_sample = None
        self._on_receiver_lost(reason)
        if self.recorder is not None:
            self.recorder.begin_close()
            self._retiring_recorder, self.recorder = self.recorder, None
        log.error("return receiver %s (attempt %s): %s", self._state, self._attempts, reason)
        return False

    def retry_receiver(self) -> None:
        if not self._closed and self._state == "failed":
            self._attempts = 1
            self._state = "recovering"
            self._restart_at = time.monotonic()

    def service(self, now: float) -> None:
        """GLib/UI thread only; never rebuild from a streaming callback."""
        if self._closed or self._restart_at is None or now < self._restart_at:
            return
        if self._retiring_recorder is not None:
            if not self._retiring_recorder.finish_close(now):
                return
            self._retiring_recorder = None
        if self.receiver is not None:
            self.recovery.close()
            if self._stopping_since is None:
                self._stopping_since = now
                self.receiver.set_state(Gst.State.NULL)
            if self.receiver.get_state(0)[1] != Gst.State.NULL:
                if now-self._stopping_since > 2.:
                    self._state, self._restart_at = "failed", None
                    self._last_error = "receiver did not stop; no replacement decoder started"
                return
            pad = self.receiver.get_static_pad("src")
            if pad is not None and pad.is_linked():
                pad.unlink(self._live_pad)
            self.pipeline.remove(self.receiver)
            self._retired.append(self.receiver)
            self.receiver = None
            self._stopping_since = None
        if self._state == "failed":
            self._restart_at = None
            return
        self._generation += 1
        self.frame_identity.clear()
        self._last_rtp_sample = None
        self._receiver_failed = False
        try:
            self.receiver = self._new_receiver()
            self.receiver.set_name(f"return_receiver_{self._generation}")
            self.pipeline.add(self.receiver)
            if self.receiver.get_static_pad("src").link(self._live_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("replacement receiver could not link to display")
            self._wire_receiver()
            if not self.receiver.sync_state_with_parent():
                raise RuntimeError("replacement receiver could not start")
            self._restart_at = None
            self._state = "waiting"
            self._restarts += 1
        except Exception as exc:
            self._receiver_fault(str(exc), self._generation)

    def recovery_status(self) -> dict:
        return dict(state=self._state, attempts=self._attempts, restarts=self._restarts,
                    generation=self._generation, last_error=self._last_error)

    def receiver_stats(self) -> dict[str, int]:
        return self.recovery.stats()

    def show_live(self, live: bool) -> None:
        live = bool(live and not self._receiver_failed and not self._closed)
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
        now = time.monotonic()
        self._state = "live"
        if self._live_since is None:
            self._live_since = now
        elif now-self._live_since >= 10.:
            self._attempts = 0  # Only sustained healthy decoding resets a fault burst.
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
