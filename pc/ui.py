import argparse, json, yaml, zmq, cv2, time
from common.schemas import DetectionMsg

FONT = cv2.FONT_HERSHEY_SIMPLEX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(cfg['net']['zmq_results'])
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    w,h = cfg['video']['width'], cfg['video']['height']
    canvas = None
    last = time.time()

    while True:
        s = sub.recv_string()
        d = json.loads(s)
        msg = DetectionMsg(**d)
        if canvas is None:
            canvas = 255 * (0 * (0,0,0))
            canvas = 255 * (cv2.UMat(h, w, cv2.CV_8UC3).get() * 0)
        frame = canvas.copy()
        # draw boxes (normalized)
        for b in msg.boxes:
          with open(args.config, "r", encoding="utf-8") as f:
            x1 = int((b.x) * w)
            y1 = int((b.y) * h)
            x2 = int((b.x + b.w) * w)
            y2 = int((b.y + b.h) * h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{b.cls}:{b.conf:.2f}", (x1, max(0,y1-5)), FONT, 0.5, (0,255,0), 1)
        cv2.putText(frame, f"frame #{msg.frame_id}", (10,25), FONT, 0.7, (255,255,255), 2)
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()
