# pc/ui.py
import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import zmq

from common.config_sync import (
    ConfigSyncError,
    load_sync_marker,
    parse_config_text,
    read_snapshot,
    resolve_active_video_profile,
    resolve_config_sync_endpoint,
    sync_as_client,
    write_sync_marker,
)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    ap.add_argument(
        "--config-sync-timeout",
        type=float,
        default=None,
        help=(
            "Maximum seconds to wait for Jetson config sync before continuing. "
            "Use 0 to skip the handshake."
        ),
    )
    ap.add_argument(
        "--config-sync-mode",
        choices=("auto", "force", "skip"),
        default="auto",
        help=(
            "auto: reuse the streamer sync marker when available; "
            "force: always perform the handshake; "
            "skip: never perform the handshake."
        ),
    )
    args = ap.parse_args()

    if args.config_sync_timeout is not None and args.config_sync_timeout < 0:
        raise SystemExit("--config-sync-timeout must be >= 0")

    config_path = Path(args.config)
    initial_snapshot = read_snapshot(config_path)
    preview_cfg = parse_config_text(initial_snapshot.text, str(config_path))
    sync_endpoint = resolve_config_sync_endpoint(preview_cfg)

    final_text = initial_snapshot.text
    final_meta = initial_snapshot.metadata

    marker_info = load_sync_marker(config_path)
    marker_meta = marker_info[0] if marker_info else None

    skip_reason: Optional[str] = None
    if args.config_sync_timeout == 0:
        skip_reason = "--config-sync-timeout=0"
    elif args.config_sync_mode == "skip":
        skip_reason = "--config-sync-mode=skip"
    elif args.config_sync_mode == "auto" and marker_meta is not None:
        if marker_meta.sha256 == initial_snapshot.metadata.sha256:
            skip_reason = "streamer marker matches local configuration"

    if skip_reason is not None:
        print(f"[ui] Config sync: skipping handshake ({skip_reason})")
    else:
        try:
            final_text, final_meta = sync_as_client(
                config_path,
                sync_endpoint,
                max_wait=args.config_sync_timeout,
            )
        except ConfigSyncError as exc:
            raise SystemExit(f"config synchronization failed: {exc}") from exc

        if final_meta.sha256 != initial_snapshot.metadata.sha256:
            print(
                "[ui] Config sync: updated local configuration "
                f"(sha256={final_meta.sha256})"
            )
        write_sync_marker(config_path, final_meta)

    cfg = parse_config_text(final_text, str(config_path))

    video_cfg, active_profile = resolve_active_video_profile(cfg)
    try:
        w = int(video_cfg["width"])
        h = int(video_cfg["height"])
    except KeyError as exc:
        raise SystemExit("config missing video.width/video.height") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.width/video.height must be integers") from exc

    try:
        laser_cfg = LaserMountConfig.from_raw_config(cfg)
    except LaserConfigError as exc:
        raise SystemExit(f"invalid laser configuration: {exc}") from exc

    try:
        return_port = int(cfg["net"]["rtp_return_port"])
    except KeyError as exc:
        raise SystemExit("config missing net.rtp_return_port") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit("net.rtp_return_port must be an integer") from exc

    stop_event = install_signal_handlers()

    if active_profile:
        print(
            "[ui] Using video profile %s (%dx%d)" % (active_profile, w, h)
        )

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detections", w, h)

    cap = None
    last_cap_open = 0.0

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

    try:
        while not stop_event.is_set():
            now = time.time()
            if (cap is None or not cap.isOpened()) and (now - last_cap_open) > 0.5:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                print(f"[ui] opening return video (port {return_port})")
                cap = open_return_video(return_port, w, h)
                last_cap_open = now

            okv, video = (cap.read() if cap and cap.isOpened() else (False, None))
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
                # (Optional) you disabled local drawing; keep it off

            now = time.time()
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
            if cap:
                cap.release()
        except Exception:
            pass
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
