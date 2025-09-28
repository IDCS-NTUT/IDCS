# pc/ui.py
import argparse, math, yaml, zmq, cv2, time
import numpy as np
from typing import Optional
from common.control import LaserConfigError, LaserMountConfig
from common.schemas import DetectionMsg, detection_msg_from_json
from common.shutdown import install_signal_handlers
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_marker(frame, uv, colour, size=10, thickness=2):
    if uv is None:
        return
    if not all(math.isfinite(coord) for coord in uv):
        return
    x = int(round(uv[0]))
    y = int(round(uv[1]))
    cv2.drawMarker(
        frame,
        (x, y),
        colour,
        markerType=cv2.MARKER_CROSS,
        markerSize=size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )


def _target_centroid(msg: DetectionMsg):
    if msg.target_idx is None:
        return None
    if not (0 <= msg.target_idx < len(msg.boxes)):
        return None
    box = msg.boxes[msg.target_idx]
    u = (box.x + (box.w / 2.0)) * msg.img_w
    v = (box.y + (box.h / 2.0)) * msg.img_h
    return (u, v), box


def _draw_tracker_overlay(frame, msg: DetectionMsg):
    info_lines = []
    measured_uv = None
    centroid = _target_centroid(msg)
    if centroid is not None:
        measured_uv = centroid[0]
        _draw_marker(frame, measured_uv, (0, 215, 255))

    predicted_uv = msg.tracker_uv_pred
    if predicted_uv is not None:
        _draw_marker(frame, predicted_uv, (0, 255, 0))

    err_mag = None
    if measured_uv is not None and predicted_uv is not None:
        du = predicted_uv[0] - measured_uv[0]
        dv = predicted_uv[1] - measured_uv[1]
        err_mag = math.hypot(du, dv)

    if msg.predict_horizon_ms is not None:
        rates = msg.cam_rates_radps or (0.0, 0.0)
        info = (
            f"horizon {msg.predict_horizon_ms:5.0f} ms  "
            f"cam ω [{rates[0]:+5.2f}, {rates[1]:+5.2f}] rad/s"
        )
        if err_mag is not None:
            info += f"  tracker Δ {err_mag:4.1f}px"
        info_lines.append(info)

    if msg.tracker_uv_vel is not None:
        vx, vy = msg.tracker_uv_vel
        info_lines.append(f"tracker vel [{vx:+5.1f}, {vy:+5.1f}] px/s")

    if msg.tracker_z_pred_m is not None:
        line = f"range {msg.tracker_z_pred_m:5.1f} m"
        if msg.tracker_z_source:
            line += f" ({msg.tracker_z_source})"
        if msg.tracker_z_vel_mps is not None:
            line += f"  vr {msg.tracker_z_vel_mps:+4.1f} m/s"
        info_lines.append(line)

    status_colour = (0, 255, 0) if msg.laser_on_target else (255, 255, 255)
    return info_lines, status_colour

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
    last_draw = time.time()
    fps_est = 0.0
    last_msg: Optional[DetectionMsg] = None

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
                last_e2e_ms = (now_ms - msg.src_ts_ms) if msg.src_ts_ms else 0
                last_msg = msg

            now = time.time()
            inst = 1.0 / max(1e-6, (now - last_draw))
            last_draw = now
            fps_est = inst if fps_est == 0.0 else (0.9*fps_est + 0.1*inst)
            overlay_lines = []
            status_colour = (255, 255, 255)
            if last_msg is not None:
                extra, colour = _draw_tracker_overlay(frame, last_msg)
                overlay_lines.extend(extra)
                status_colour = colour

            status = f"frame #{last_frame_id if last_frame_id>=0 else '-'}  e2e {int(last_e2e_ms)} ms  ~{fps_est:4.1f} fps"
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_colour, 2)

            for idx, line in enumerate(overlay_lines, start=1):
                y = 25 + idx * 22
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_colour, 2)
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
