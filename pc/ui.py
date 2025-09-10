# pc/ui.py
import argparse, json, yaml, zmq, cv2, time
import numpy as np
from common.schemas import DetectionMsg

FONT = cv2.FONT_HERSHEY_SIMPLEX

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

    # ZMQ SUB with poll (non-blocking)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(cfg['net']['zmq_results'])           # e.g. tcp://<JETSON_IP>:5556
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    last_draw = time.time()
    fps_txt = "waiting for data..."

    while True:
        # Poll ZMQ for up to 50 ms so UI stays responsive
        events = dict(poller.poll(timeout=50))
        if sub in events and events[sub] == zmq.POLLIN:
            s = sub.recv_string()
            d = json.loads(s)
            msg = DetectionMsg(**d)

            # redraw canvas
            frame[:] = 0
            for b in msg.boxes:
                x1 = int(b.x * w); y1 = int(b.y * h)
                x2 = int((b.x + b.w) * w); y2 = int((b.y + b.h) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{b.cls}:{b.conf:.2f}", (x1, max(0, y1-5)), FONT, 0.6, (0,255,0), 2)

            now = time.time()
            fps = 1.0 / max(1e-6, (now - last_draw))
            last_draw = now
            fps_txt = f"frame #{msg.frame_id}  ~{fps:4.1f} fps"

        # Always paint something so the window appears
        cv2.putText(frame, fps_txt, (10, 25), FONT, 0.7, (255,255,255), 2)
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) == 27:   # ESC quits
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
