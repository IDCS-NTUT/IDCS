import argparse, time, cv2, yaml
from pc.sim_camera import SimCamera

PIPELINE = (
    "appsrc is-live=true block=true format=time "
    "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
    "videoconvert ! "
    "video/x-raw,format=I420,colorimetry=bt709,interlace-mode=progressive,chromasite=mpeg2 ! "
    "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bitrate={br} byte-stream=true ! "
    "h264parse ! "
    "rtph264pay pt=96 config-interval=1 ! "
    "udpsink host={host} port={port} sync=false async=false"
)


def open_source(spec: str, w: int, h: int, fps: int):
    if spec.startswith("webcam:"):
        idx = int(spec.split(":",1)[1])
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap
    elif spec.startswith("file:"):
        return cv2.VideoCapture(spec.split(":",1)[1])
    elif spec.startswith("sim"):
        # Wrap SimCamera into a VideoCapture-like object
        class _SimCap:
            def __init__(self, W, H, fps):
                self.gen = SimCamera(width=W, height=H)
                self.period = 1.0 / max(1, fps)
                self._t = time.monotonic()
            def isOpened(self): return True
            def read(self):
                # pace to approx fps
                now = time.monotonic()
                sleep = self.period - (now - self._t)
                if sleep > 0: time.sleep(sleep)
                self._t = time.monotonic()
                return self.gen.next_frame()
            def release(self): pass
        return _SimCap(w, h, fps)
    else:
        raise ValueError("Unknown source, use webcam:<idx> | file:<path> | sim")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    w,h,fps = cfg['video']['width'], cfg['video']['height'], cfg['video']['fps']
    br = cfg['video']['bitrate_kbps']
    host,port = cfg['net']['jetson_ip'], cfg['net']['rtp_port']

    cap = open_source(cfg.get('source','webcam:0'), w,h,fps)
    if not cap.isOpened():
        raise SystemExit("Failed to open source")

    gst = PIPELINE.format(w=w,h=h,fps=fps, br=br, host=host, port=port)
    out = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, float(fps), (w,h))
    if not out.isOpened():
        raise SystemExit("Failed to open GStreamer pipeline")

    frame_id = 0
    t0 = time.monotonic_ns()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        out.write(cv2.resize(frame, (w,h)))
        # simple FPS display
        if frame_id % (fps*2) == 0:
            dt = (time.monotonic_ns() - t0)/1e9
            print(f"Sent {frame_id} frames, avg FPS ~ {frame_id/dt:.1f}")

    cap.release(); out.release()

if __name__ == "__main__":
    main()
