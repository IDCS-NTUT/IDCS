import argparse, time, cv2, yaml
from pc.sim_camera import SimCamera
import zmq, time
from common.shutdown import install_signal_handlers


PIPELINE = (
    "appsrc is-live=true block=false do-timestamp=true format=time "  # <-- non-blocking, self timestamps
    "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
    "videoconvert ! "
    "video/x-raw,format=NV12,colorimetry=bt709,interlace-mode=progressive,chromasite=mpeg2 ! "
    "nvh264enc preset=low-latency-hq zerolatency=true rc-mode=cbr bframes=0 gop-size=30 bitrate={br} ! "
    "h264parse ! "
    "queue leaky=downstream max-size-buffers=120 max-size-bytes=0 max-size-time=0 ! "  # <-- drop if downstream slow
    "rtph264pay pt=96 config-interval=1 ! "
    "udpsink host={host} port={port} sync=false async=false"
)


'''
PIPELINE_X264 = (
    "appsrc is-live=true block=true format=time "
    "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
    "videoconvert ! "
    "video/x-raw,format=I420,colorimetry=bt709,interlace-mode=progressive,chromasite=mpeg2 ! "
    "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bitrate={br} byte-stream=true ! "
    "h264parse ! "
    "rtph264pay pt=96 config-interval=1 ! "
    "udpsink host={host} port={port} sync=false async=false"
)
'''

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

    # --- signals
    stop_event = install_signal_handlers()

    # --- ZMQ (local context so we can term())
    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.SNDHWM, 1)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(cfg['net']['header_push'])

    cap = open_source(cfg.get('source','webcam:0'), w,h,fps)
    if not cap.isOpened():
        raise SystemExit("Failed to open source")

    gst = PIPELINE.format(w=w,h=h,fps=fps, br=br, host=host, port=port)
    out = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, float(fps), (w,h))
    if not out.isOpened():
        raise SystemExit("Failed to open GStreamer pipeline")

    frame_id = 0
    t0 = time.monotonic_ns()
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame_id += 1
            src_ts_ms = int(time.monotonic_ns() / 1e6)
            # non-blocking header send
            try:
                push.send_json({"frame_id": frame_id, "src_ts_ms": src_ts_ms}, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            out.write(cv2.resize(frame, (w,h)))

            if frame_id % max(1,fps*2) == 0:
                dt = (time.monotonic_ns() - t0)/1e9
                print(f"[streamer] Sent {frame_id} frames, ~{frame_id/dt:.1f} FPS")
    except KeyboardInterrupt:
        pass
    finally:
        print("[streamer] shutting down...")
        try: cap.release()
        except: pass
        try: out.release()
        except: pass
        try: push.close(0)
        except: pass
        try: ctx.term()
        except: pass
        # give GStreamer a tick to flush
        time.sleep(0.05)

if __name__ == "__main__":
    main()