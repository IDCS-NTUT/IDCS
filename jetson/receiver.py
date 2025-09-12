import time, cv2

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

