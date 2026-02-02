import time
from typing import Optional

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo


class GstAppSrcWriter:
    def __init__(self, pipeline: str, fps: int, width: int, height: int) -> None:
        Gst.init(None)
        self._pipeline = Gst.parse_launch(pipeline)
        self._appsrc = self._pipeline.get_by_name("src")
        self._fps = max(1, int(fps))
        self._width = int(width)
        self._height = int(height)
        self._duration = Gst.SECOND // self._fps
        self._pts = 0
        self._opened = False
        if self._appsrc is not None:
            caps = Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={self._width},height={self._height},framerate={self._fps}/1"
            )
            self._appsrc.set_property("caps", caps)
            self._appsrc.set_property("is-live", True)
            self._appsrc.set_property("block", False)
            self._appsrc.set_property("format", Gst.Format.TIME)
            self._appsrc.set_property("do-timestamp", True)
            state_result = self._pipeline.set_state(Gst.State.PLAYING)
            self._opened = state_result != Gst.StateChangeReturn.FAILURE

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame: np.ndarray) -> bool:
        if not self._opened or self._appsrc is None:
            return False
        if frame is None:
            return False
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            raise ValueError(
                f"frame shape mismatch: got {frame.shape[1]}x{frame.shape[0]},"
                f" expected {self._width}x{self._height}"
            )
        if not frame.flags.c_contiguous:
            frame = frame.copy()
        data = frame.tobytes()
        buffer = Gst.Buffer.new_allocate(None, len(data), None)
        buffer.fill(0, data)
        buffer.pts = self._pts
        buffer.dts = self._pts
        buffer.duration = self._duration
        self._pts += self._duration
        result = self._appsrc.emit("push-buffer", buffer)
        return result == Gst.FlowReturn.OK

    def end_of_stream(self) -> None:
        if self._appsrc is not None:
            self._appsrc.end_of_stream()

    def close(self, send_eos: bool = False) -> None:
        if send_eos:
            self.end_of_stream()
            time.sleep(0.05)
        self._pipeline.set_state(Gst.State.NULL)
        self._opened = False

    def release(self) -> None:
        self.close(send_eos=False)


class GstAppSinkReader:
    def __init__(self, pipeline: str, stop_event: Optional[object] = None) -> None:
        Gst.init(None)
        self._pipeline = Gst.parse_launch(pipeline)
        self._appsink = self._pipeline.get_by_name("sink")
        self._bus = self._pipeline.get_bus()
        self._opened = False
        self._eos = False
        self._stop_event = stop_event
        if self._appsink is not None:
            state_result = self._pipeline.set_state(Gst.State.PLAYING)
            self._opened = state_result != Gst.StateChangeReturn.FAILURE

    def isOpened(self) -> bool:
        return self._opened

    @property
    def eos(self) -> bool:
        return self._eos

    def _signal_stop(self) -> None:
        if self._stop_event is not None and hasattr(self._stop_event, "set"):
            try:
                self._stop_event.set()
            except Exception:
                pass

    def _drain_bus(self) -> None:
        if self._bus is None:
            return
        while True:
            msg = self._bus.pop_filtered(
                Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
            if msg is None:
                break
            if msg.type == Gst.MessageType.EOS:
                self._eos = True
                self._signal_stop()
                break
            if msg.type == Gst.MessageType.ERROR:
                self._eos = True
                self._signal_stop()
                break

    def read(self, timeout_s: float = 0.02):
        if not self._opened or self._appsink is None:
            return False, None
        self._drain_bus()
        if self._eos:
            return False, None
        timeout_ns = int(timeout_s * Gst.SECOND)
        sample = self._appsink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            self._drain_bus()
            return False, None
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        if buffer is None or caps is None:
            return False, None
        info = GstVideo.VideoInfo()
        info.from_caps(caps)
        data = buffer.extract_dup(0, buffer.get_size())
        if not data:
            return False, None
        channels = info.finfo.n_components if info.finfo else 4
        frame = np.frombuffer(data, dtype=np.uint8)
        expected = info.width * info.height * channels
        if frame.size < expected:
            return False, None
        frame = frame[:expected].reshape((info.height, info.width, channels)).copy()
        return True, frame

    def release(self) -> None:
        self._pipeline.set_state(Gst.State.NULL)
        self._opened = False
