# pc/ui.py
import argparse, json, yaml, zmq, cv2, time
import numpy as np
from common.schemas import DetectionMsg

FONT = cv2.FONT_HERSHEY_SIMPLEX

def open_return_video(port, w, h):
    pipeline = (
    f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=97,clock-rate=90000 ! "
    "rtpjitterbuffer latency=120 ! rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! appsink drop=true sync=false max-buffers=1"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    # UTF-8 config read
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    w, h = cfg['video']['width'], cfg['video']['height']

    # Prepare a blank canvas and a named window immediately
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detections", w, h)
    ret = open_return_video(cfg['net']['rtp_return_port'], w, h)

    # ZMQ SUB with poll (non-blocking)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(cfg['net']['zmq_results'])           # e.g. tcp://<JETSON_IP>:5556
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    last_draw = time.time()
    fps_txt = "waiting for data..."

    last_frame_id = -1
    last_e2e_ms = 0
    last_draw = time.time()
    fps_est = 0.0

    while True:
        # keep the window alive even if no new ZMQ message
        okv, video = (ret.read() if ret and ret.isOpened() else (False, None))
        if okv and video is not None:
            frame = video
            draw_local = False    # Jetson already drew
        else:
            frame[:] = 0
            draw_local = True     # draw locally so you still see something

        # Poll ZMQ for up to 50 ms so UI stays responsive
        events = dict(poller.poll(timeout=50))
        if sub in events and events[sub] == zmq.POLLIN:
            s = sub.recv_string()
            d = json.loads(s)
            msg = DetectionMsg(**d)
            now_ms = int(time.monotonic_ns() / 1e6)
            last_frame_id = msg.frame_id
            last_e2e_ms = (now_ms - msg.src_ts_ms) if msg.src_ts_ms else 0

            # draw latest detections onto current frame
            if draw_local:
                for b in msg.boxes:
                    x1 = int(b.x * w); y1 = int(b.y * h)
                    x2 = int((b.x + b.w) * w); y2 = int((b.y + b.h) * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{b.cls}:{b.conf:.2f}", (x1, max(0, y1-5)), FONT, 0.6, (0,255,0), 2)

        # update FPS every loop (smoothed)
        now = time.time()
        inst_fps = 1.0 / max(1e-6, (now - last_draw))
        last_draw = now
        fps_est = 0.9 * fps_est + 0.1 * inst_fps if fps_est > 0 else inst_fps

        # compose ONE status line (don¡¦t overwrite)
        status = f"frame #{last_frame_id if last_frame_id>=0 else '-'}  e2e {int(last_e2e_ms)} ms  ~{fps_est:4.1f} fps"
        cv2.putText(frame, status, (10, 25), FONT, 0.7, (255,255,255), 2)

        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
