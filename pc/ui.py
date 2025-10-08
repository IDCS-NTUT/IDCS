# pc/ui.py
import argparse, yaml, zmq, cv2
import numpy as np
from common.clock import create_clock
from common.control import LaserConfigError, LaserMountConfig
from common.debug import DebugConfig, DebugConfigError
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



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        debug_cfg = DebugConfig.from_raw_config(cfg)
    except DebugConfigError as exc:
        raise SystemExit(f"invalid debug configuration: {exc}") from exc

    clock = create_clock(step_mode=debug_cfg.step_mode.enabled)

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
    last_e2e_ms = 0
    last_draw = clock.wall_time()
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
                now_ms = clock.now_ms()
                last_frame_id = msg.frame_id
                last_e2e_ms = (now_ms - msg.src_ts_ms) if msg.src_ts_ms else 0
                # (Optional) you disabled local drawing; keep it off

            now = clock.wall_time()
            inst = 1.0 / max(1e-6, (now - last_draw))
            last_draw = now
            fps_est = inst if fps_est == 0.0 else (0.9*fps_est + 0.1*inst)
            status = (
                f"frame #{last_frame_id if last_frame_id>=0 else '-'}  "
                f"e2e {int(last_e2e_ms)} ms  ~{fps_est:4.1f} fps"
            )

            font = FONT
            scale = 0.5
            thickness = 1
            margin = 8
            text_colour = (255, 255, 255)

            text_size, baseline = cv2.getTextSize(status, font, scale, thickness)
            text_w, text_h = text_size
            h, w = frame.shape[:2]

            origin_x = max(margin + 4, w - margin - text_w)
            origin_y = max(margin + text_h, h - margin - baseline)

            rect_tl = (int(origin_x - 4), int(max(0, origin_y - text_h - 4)))
            rect_br = (
                int(min(w - 1, origin_x + text_w + 4)),
                int(min(h - 1, origin_y + baseline + 4)),
            )

            cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), thickness=cv2.FILLED)
            cv2.putText(
                frame,
                status,
                (int(origin_x), int(origin_y)),
                font,
                scale,
                text_colour,
                thickness,
                cv2.LINE_AA,
            )
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
        clock.sleep_wall(0.05)

if __name__ == "__main__":
    main()
