# pc/ui.py
import argparse, yaml, zmq, cv2, time
import numpy as np
from common.control import LaserConfigError, LaserMountConfig
from common.schemas import DetectionMsg, detection_msg_from_json
from common.shutdown import install_signal_handlers
FONT = cv2.FONT_HERSHEY_SIMPLEX

def open_return_video(port, w, h):
    pipeline = (
        f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=97,clock-rate=90000 ! "
        "rtpjitterbuffer latency=120 ! rtph264depay ! h264parse ! avdec_h264 ! "
        "videoconvert ! queue leaky=downstream max-size-buffers=5 ! appsink drop=true sync=false max-buffers=1"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap


def _draw_status_overlay(frame, frame_id, e2e_ms, fps_est):
    h, w = frame.shape[:2]
    margin = 8
    scale = 0.5
    thickness = 1
    colour = (0, 255, 0)

    if frame_id is None or frame_id < 0:
        frame_bit = "frame:--"
    else:
        frame_bit = f"frame:{int(frame_id)}"

    if e2e_ms is None or e2e_ms <= 0:
        latency_bit = "e2e:--"
    elif e2e_ms >= 1000:
        latency_bit = f"e2e:{(e2e_ms/1000.0):.2f}s"
    else:
        latency_bit = f"e2e:{int(round(e2e_ms))}ms"

    if fps_est and fps_est > 0:
        fps_bit = f"fps:{fps_est:4.1f}"
    else:
        fps_bit = "fps:--"

    status_text = " | ".join((frame_bit, latency_bit, fps_bit))
    text_size, baseline = cv2.getTextSize(status_text, FONT, scale, thickness)
    text_w, text_h = text_size
    text_x = max(margin, w - margin - text_w)
    text_y = h - margin

    cv2.rectangle(
        frame,
        (text_x - 4, text_y - text_h - baseline - 4),
        (text_x + text_w + 4, text_y + 4),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )
    cv2.putText(frame, status_text, (text_x, text_y), FONT, scale, colour, thickness, cv2.LINE_AA)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        laser_cfg = LaserMountConfig.from_raw_config(cfg)
    except LaserConfigError as exc:
        raise SystemExit(f"invalid laser configuration: {exc}") from exc

    stop_event = install_signal_handlers()

    w,h = cfg['video']['width'], cfg['video']['height']
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detections", w, h)

    ret = open_return_video(cfg['net']['rtp_return_port'], w, h)

    # ZMQ
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(cfg['net']['zmq_results'])
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    last_frame_id = -1
    last_e2e_ms = None
    last_draw = time.time()
    fps_est = 0.0

    try:
        while not stop_event.is_set():
            okv, video = (ret.read() if ret and ret.isOpened() else (False, None))
            if okv and video is not None:
                frame = video
            else:
                frame[:] = 0

            events = dict(poller.poll(timeout=50))
            if sub in events and events[sub] == zmq.POLLIN:
                payload = sub.recv()
                msg = detection_msg_from_json(payload)
                now_ms = int(time.monotonic_ns() / 1e6)
                last_frame_id = msg.frame_id
                last_e2e_ms = (now_ms - msg.src_ts_ms) if msg.src_ts_ms else None
                # (Optional) you disabled local drawing; keep it off

            now = time.time()
            inst = 1.0 / max(1e-6, (now - last_draw))
            last_draw = now
            fps_est = inst if fps_est == 0.0 else (0.9*fps_est + 0.1*inst)
            if frame.size:
                _draw_status_overlay(frame, last_frame_id, last_e2e_ms, fps_est)
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("[ui] shutting down...")
        try:
            if ret: ret.release()
        except: pass
        try: sub.close(0)
        except: pass
        try: ctx.term()
        except: pass
        # make sure window goes away on all platforms
        for _ in range(3):
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        time.sleep(0.05)

if __name__ == "__main__":
    main()
