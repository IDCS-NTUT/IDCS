import time
import cv2

class GRecv:
    def __init__(self, port: int, w: int, h: int):
        self.port, self.w, self.h = port, w, h
        self.cap = None
        self.fail_count = 0
        self._open()

    def _pipeline(self):
        # rtpjitterbuffer smooths bursts after (re)start
        # h264parse + explicit h264 caps help NVDEC renegotiate
        return (
    f"udpsrc port={self.port} "
    "caps=application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! "
    "rtpjitterbuffer latency=120 drop-on-late=true mode=1 ! "
    "rtph264depay ! h264parse ! "
    "video/x-h264,stream-format=byte-stream,alignment=au ! "
    "nvv4l2decoder enable-max-performance=1 ! nvvidconv ! "
    f"video/x-raw,format=BGRx,width={self.w},height={self.h} ! "
    "videoconvert ! appsink drop=true sync=false max-buffers=1"
	)

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self._pipeline(), cv2.CAP_GSTREAMER)

    def read(self):
        # try reading; if caps not ready, OpenCV returns (False,None) briefly
        ok, frame = self.cap.read() if self.cap else (False, None)
        if not ok or frame is None:
            self.fail_count += 1
            # give sender time to (re)start and SPS/PPS to arrive
            time.sleep(0.02)
            if self.fail_count >= 10:   # ~200ms of failures → reopen pipeline
                self._open()
                self.fail_count = 0
            return False, None
        self.fail_count = 0
        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

