import argparse, logging, time, yaml, zmq

from pydantic import ValidationError

from common.control import ControlConfig, ControlConfigError
from common.schemas import CamState, DetectionMsg
from jetson.receiver import GRecv
from jetson.controller import ControlLoop
from jetson.yolo_engine import YoloEngine
# Build a GStreamer encoder pipeline for return video
import threading
import cv2
from common.shutdown import install_signal_handlers

def make_return_writer(pc_ip, port, w, h, fps=30, bitrate=4000, vbv_size=None):
    br_bps = bitrate * 1000
    if vbv_size is None:
            vbv_size = int((br_bps / fps) * 2)
    pipeline = (
        # App source (CPU memory, BGR from OpenCV)
        f"appsrc is-live=true block=false do-timestamp=true format=time "
        f"caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
        # CPU colorspace to NV12
        "videoconvert ! video/x-raw,format=NV12 ! "
        # Move into NVMM for HW encoder
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12,width={w},height={h},framerate={fps}/1 ! "
        # Low-latency encoder (CBR, IDR every 1s)
        "nvv4l2h264enc maxperf-enable=1 control-rate=1 bitrate={bitrate} "
        f"vbv-size={vbv_size} EnableTwopassCBR=true "
        "iframeinterval={fps} idrinterval={fps} insert-sps-pps=true preset-level=1 ! "
        # Packetize
        "h264parse ! rtph264pay pt=97 config-interval=1 ! "
        # Send
        f"udpsink host={pc_ip} port={port} sync=false async=false"
    ).format(w=w, h=h, fps=fps, bitrate=bitrate*1000)
    vw = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, float(fps), (w, h))
    if not vw.isOpened():
        print("[server] WARN: failed to open return video pipeline")
    return vw

MS = 1_000_000


def _parse_tcp_port(endpoint: str, key: str) -> int:
    if not endpoint:
        raise SystemExit(f"config missing net.{key} endpoint")
    if not endpoint.startswith("tcp://"):
        raise SystemExit(f"net.{key} must be a tcp://HOST:PORT endpoint, got {endpoint!r}")
    host_port = endpoint[len("tcp://"):]
    if ":" not in host_port:
        raise SystemExit(f"net.{key} is missing a port: {endpoint!r}")
    host, port_str = host_port.rsplit(":", 1)
    if not host:
        raise SystemExit(f"net.{key} is missing a host: {endpoint!r}")
    try:
        port = int(port_str)
    except ValueError as exc:
        raise SystemExit(f"net.{key} has an invalid port: {endpoint!r}") from exc
    if not (0 < port < 65536):
        raise SystemExit(f"net.{key} port out of range: {port}")
    return port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    w,h = cfg['video']['width'], cfg['video']['height']
    try:
        control_cfg = ControlConfig.from_raw_config(cfg, (w, h))
    except ControlConfigError as exc:
        raise SystemExit(f"invalid control configuration: {exc}") from exc
    port = cfg['net']['rtp_port']

    stop_event = install_signal_handlers()

    recv = GRecv(port, w, h)

    yolo = YoloEngine(
  engine_path=cfg['yolo']['engine_path'],
  conf_thres=cfg['yolo']['conf_thres'],
  iou_thres=cfg['yolo']['iou_thres'],
  input_size=cfg['yolo']['input_size'],
  preprocess_mode=cfg['yolo'].get('preprocess_mode', 'bilinear'),
  direct_to_device=True
    )

    # --- ZMQ (local ctx)
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)

    ep = cfg['net'].get('zmq_results')  # e.g. tcp://<JETSON_IP>:5556
    results_port = _parse_tcp_port(ep, "zmq_results")
    pub.bind(f"tcp://0.0.0.0:{results_port}")

    ctrl_pub = ctx.socket(zmq.PUB)
    ctrl_pub.setsockopt(zmq.SNDHWM, 1)
    ctrl_pub.setsockopt(zmq.LINGER, 0)
    ctrl_ep = cfg['net'].get('zmq_control')
    ctrl_port = _parse_tcp_port(ctrl_ep, "zmq_control")
    ctrl_pub.bind(f"tcp://0.0.0.0:{ctrl_port}")

    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.RCVHWM, 10)
    pull.setsockopt(zmq.LINGER, 0)
    header_ep = cfg['net'].get('header_push')
    header_port = _parse_tcp_port(header_ep, "header_push")
    pull.bind(f"tcp://0.0.0.0:{header_port}")
    pull.RCVTIMEO = 0  # non-blocking

    ret_vw = make_return_writer(
        cfg['net']['pc_ip'], cfg['net']['rtp_return_port'], w, h,
        fps=cfg['video']['fps'], bitrate=cfg['video']['bitrate_kbps']
    )

    latest_header = {"frame_id": 0, "src_ts_ms": 0}
    controller = ControlLoop(control_cfg, ctrl_pub)

    try:
        while not stop_event.is_set():
            # receive frame
            ok, frame = recv.read()
            if not ok:
                controller.tick(time.monotonic())
                continue

            # headers (non-blocking drain)
            try:
                while True:
                    header_obj = pull.recv_json(flags=zmq.NOBLOCK)
                    if isinstance(header_obj, dict) and header_obj.get("type") == "CamState":
                        try:
                            cam_state = CamState(**header_obj)
                        except ValidationError as exc:
                            logging.warning("invalid CamState header: %s", exc)
                        else:
                            controller.update_cam_state(cam_state)
                            # CamState carries the originating frame metadata. Use it to
                            # refresh our latest header so DetectionMsg instances keep
                            # advancing even if the bare header message was dropped.
                            latest_header = header_obj
                    else:
                        latest_header = header_obj
            except zmq.Again:
                pass

            if frame.ndim == 3 and frame.shape[2] == 4:  # RGBA->BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            rx_ts_ms = int(time.monotonic_ns() / 1e6)
            boxes = yolo.infer(frame)
            infer_ts_ms = int(time.monotonic_ns() / 1e6)

            msg = DetectionMsg(
                frame_id=latest_header.get("frame_id", 0),
                src_ts_ms=latest_header.get("src_ts_ms", 0),
                rx_ts_ms=rx_ts_ms,
                infer_ts_ms=infer_ts_ms,
                img_w=w, img_h=h,
                boxes=boxes,
            )
            try:
                pub.send_string(msg.model_dump_json(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            controller.update_detection(msg)
            controller.tick(time.monotonic())

            # draw + return video
            ov = frame.copy()
            for b in boxes:
                x1 = int(b.x * w); y1 = int(b.y * h)
                x2 = int((b.x + b.w) * w); y2 = int((b.y + b.h) * h)
                colour = (0, 255, 0)
                cv2.rectangle(ov, (x1, y1), (x2, y2), colour, 2)
                if b.cls in ("person", "1"):
                    cv2.line(ov, (x1, y1), (x2, y2), colour, 2)
                    cv2.line(ov, (x1, y2), (x2, y1), colour, 2)
            if ret_vw and ret_vw.isOpened():
                ret_vw.write(ov)

    except KeyboardInterrupt:
        pass
    finally:
        print("[server] shutting down...")
        try: recv.release()
        except: pass
        try: 
            if ret_vw: ret_vw.release()
        except: pass
        for s in (pub, ctrl_pub, pull):
            try: s.close(0)
            except: pass
        try: ctx.term()
        except: pass
        time.sleep(0.05)

if __name__ == "__main__":
    main()
