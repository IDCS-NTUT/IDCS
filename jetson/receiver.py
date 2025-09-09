import cv2

class GRecv:
    def __init__(self, port: int, w: int, h: int):
        self.pipeline = (
            f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
            f"video/x-raw,format=BGRx,width={w},height={h} ! videoconvert ! appsink drop=true sync=false"
        )
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Jetson receiver failed to open pipeline")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
