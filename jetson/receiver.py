import threading
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

class GRecv:
    """Receive H.264 RTP and deliver CPU BGR frames via appsink (HW decode only)."""
    def __init__(
        self,
        port: int,
        w: int,
        h: int,
        *,
        stop_event: Optional[threading.Event] = None,
    ):
        Gst.init(None)
        self.port, self.w, self.h = port, w, h
        self._pipeline = None
        self._appsink = None
        self._bus = None
        self._eos = False
        self._stop_event = stop_event
        self._open()

    def _pipeline(self) -> str:
        # HW decode → nvvidconv → videoconvert → CPU RGBA → appsink
        # (no memory:NVMM in the last caps so OpenCV gets system-memory buffers)
        return (
            f"udpsrc port={self.port} "
            "caps=application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! "
            "rtpjitterbuffer latency=200 mode=1 do-lost=true ! "
            "rtph264depay ! h264parse ! "
            "video/x-h264,stream-format=byte-stream,alignment=au ! "
            "nvv4l2decoder enable-max-performance=1 disable-dpb=true ! "
            "nvvidconv ! video/x-raw,format=RGBA ! "
            "videoconvert ! video/x-raw,format=RGBA ! "
            "queue leaky=downstream max-size-buffers=2 ! "
            "appsink name=sink drop=true sync=false max-buffers=1"
        )

    def _open(self):
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        pipe = self._pipeline()
        print("[GRecv] opening pipeline:\n", pipe)
        self._pipeline = Gst.parse_launch(pipe)
        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            raise RuntimeError("GRecv pipeline missing appsink named 'sink'")
        self._appsink.set_property("sync", False)
        self._appsink.set_property("max-buffers", 1)
        self._appsink.set_property("drop", True)
        self._bus = self._pipeline.get_bus()
        self._pipeline.set_state(Gst.State.PLAYING)
        self._eos = False

    @property
    def eos(self) -> bool:
        return self._eos

    def read(self):
        if self._eos:
            return False, None
        if self._stop_event is not None and self._stop_event.is_set():
            return False, None
        if self._bus is not None:
            msg = self._bus.timed_pop_filtered(
                0, Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
            if msg is not None:
                if msg.type == Gst.MessageType.EOS:
                    print("[GRecv] EOS received; stopping.")
                else:
                    err, dbg = msg.parse_error()
                    print(f"[GRecv] ERROR: {err} ({dbg})")
                self._eos = True
                if self._pipeline is not None:
                    try:
                        self._pipeline.set_state(Gst.State.NULL)
                    except Exception:
                        pass
                if self._stop_event is not None:
                    self._stop_event.set()
                return False, None
        if self._appsink is None:
            return False, None

        sample = self._appsink.emit("try-pull-sample", 20 * 1_000_000)
        if sample is None:
            return False, None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None else None
        width = structure.get_value("width") if structure is not None else self.w
        height = structure.get_value("height") if structure is not None else self.h
        fmt = structure.get_value("format") if structure is not None else "RGBA"

        success, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not success:
            return False, None
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
            frame = frame.reshape((int(height), int(width), -1))
            if fmt == "RGBA" and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] > 3:
                frame = frame[:, :, :3]
            frame = frame.copy()
        finally:
            buffer.unmap(mapinfo)
        return True, frame

    def release(self):
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._pipeline = None
        self._appsink = None
        self._bus = None


class FileVideoReader:
    """Read frames from a local video file for offline inference."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.path = str(Path(path))
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video file: {self.path}")
        self._src_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._src_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if fps > 0.0 else 0.0
        self._target_size: Optional[Tuple[int, int]] = target_size

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_width(self) -> int:
        if self._target_size is not None:
            return int(self._target_size[0])
        return int(self._src_width)

    @property
    def frame_height(self) -> int:
        if self._target_size is not None:
            return int(self._target_size[1])
        return int(self._src_height)

    def set_target_size(self, size: Optional[Tuple[int, int]]) -> None:
        if size is None:
            self._target_size = None
            return
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("target_size must be positive integers")
        self._target_size = (int(width), int(height))

    def read(self) -> Tuple[bool, Optional[Any]]:
        ok, frame = self.cap.read() if self.cap else (False, None)
        if not ok or frame is None:
            return False, None
        if self._target_size is not None:
            target_w, target_h = self._target_size
            if frame.shape[1] != target_w or frame.shape[0] != target_h:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return True, frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
