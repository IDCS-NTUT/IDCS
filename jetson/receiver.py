import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import cv2

class GRecv:
    """Receive H.264 RTP and deliver CPU BGR frames via appsink (HW decode only)."""
    def __init__(self, port: int, w: int, h: int):
        self.port, self.w, self.h = port, w, h
        self.cap = None
        self.fail_count = 0
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
            "appsink drop=true sync=false max-buffers=1"
        )

    def _open(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
        pipe = self._pipeline()
        print("[GRecv] opening pipeline:\n", pipe)
        self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        print("[GRecv] isOpened:", self.cap.isOpened())

    def read(self):
        ok, frame = self.cap.read() if self.cap else (False, None)
        if not ok or frame is None:
            self.fail_count += 1
            time.sleep(0.02)
            if self.fail_count >= 20:           # ~400 ms of misses → reopen
                print("[GRecv] reopening after consecutive failures...")
                self._open()
                self.fail_count = 0
            return False, None

        # Ensure 3-channel BGR for downstream
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        self.fail_count = 0
        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


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


class CsiVideoReader:
    """Capture frames from a Jetson CSI camera via nvarguscamerasrc."""

    def __init__(
        self,
        w: int,
        h: int,
        fps: int,
        *,
        sensor_id: int = 0,
        flip_method: int = 0,
    ) -> None:
        self.w = int(w)
        self.h = int(h)
        self.fps = int(fps)
        self.sensor_id = int(sensor_id)
        self.flip_method = int(flip_method)
        self.cap = None
        self.fail_count = 0
        self._open()

    def _pipeline(self) -> str:
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={self.w},height={self.h},framerate={self.fps}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            "video/x-raw,format=RGBA ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "queue leaky=downstream max-size-buffers=2 ! "
            "appsink drop=true sync=false max-buffers=1"
        )

    def _open(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        pipe = self._pipeline()
        print("[CsiVideoReader] opening pipeline:\n", pipe)
        self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        print("[CsiVideoReader] isOpened:", self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[Any]]:
        ok, frame = self.cap.read() if self.cap else (False, None)
        if not ok or frame is None:
            self.fail_count += 1
            time.sleep(0.02)
            if self.fail_count >= 20:
                print("[CsiVideoReader] reopening after consecutive failures...")
                self._open()
                self.fail_count = 0
            return False, None

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        self.fail_count = 0
        return True, frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
