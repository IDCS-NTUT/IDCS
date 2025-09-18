import argparse, time, yaml, zmq
from common.schemas import DetectionMsg
from jetson.receiver import GRecv
from jetson.yolo_engine import YoloEngine
# Build a GStreamer encoder pipeline for return video
import threading
import cv2
from common.shutdown import install_signal_handlers
from urllib.parse import urlparse
import numpy as np

def make_return_writer(pc_ip, port, rw, rh, fps=30, bitrate_kbps=6000, vbv_scale=2):
    br_bps = bitrate_kbps * 1000
    # vbv candidates: try bits; if your build wants bytes, halve by /8
    vbv_size = int((br_bps / fps) * vbv_scale)

    pipeline = (
        f"appsrc is-live=true block=false do-timestamp=true format=time "
        f"caps=video/x-raw,format=BGR,width={rw},height={rh},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=NV12 ! "
        "nvvidconv ! queue leaky=downstream max-size-buffers=30 ! "
        f"video/x-raw(memory:NVMM),format=NV12,width={rw},height={rh},framerate={fps}/1 ! "
        f"nvv4l2h264enc maxperf-enable=1 control-rate=1 bitrate={br_bps} "
        f"vbv-size={vbv_size} EnableTwopassCBR=true "
        f"iframeinterval={fps*3} idrinterval=0 insert-sps-pps=true preset-level=1 ! "
        "h264parse ! rtph264pay pt=97 config-interval=1 ! "
        f"udpsink host={pc_ip} port={port} sync=false async=false"
    )
    vw = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, float(fps), (rw, rh))
    if not vw.isOpened():
        print("[server] WARN: failed to open return video pipeline")
    else:
        print(f"[server] return enc opened: {rw}x{rh}@{fps} br={br_bps} vbv={vbv_size}")
    return vw

MS = 1_000_000


def letterbox_resize(img, dst_w, dst_h):
    """Resize ``img`` to fit within ``dst_w``×``dst_h`` with preserved aspect."""
    if img is None:
        return None

    src_h, src_w = img.shape[:2]
    if src_w == 0 or src_h == 0:
        return None

    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    if new_w != dst_w or new_h != dst_h:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        if resized.ndim == 3:
            letterboxed = np.zeros((dst_h, dst_w, resized.shape[2]), dtype=resized.dtype)
        else:
            letterboxed = np.zeros((dst_h, dst_w), dtype=resized.dtype)
        x_off = (dst_w - new_w) // 2
        y_off = (dst_h - new_h) // 2
        letterboxed[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return letterboxed

    if new_w != src_w or new_h != src_h:
        return cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    uw, uh = cfg['uplink']['width'], cfg['uplink']['height']
    port   = cfg['net']['rtp_port']

    stop_event = install_signal_handlers()

    recv = GRecv(port, uw, uh)

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

    ep = cfg['net']['zmq_results']  # e.g. tcp://<JETSON_IP>:5556
    hostport = ep[len("tcp://"):]
    results_port = hostport.split(":")[-1]
    pub.bind(f"tcp://0.0.0.0:{results_port}")

    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.RCVHWM, 10)
    pull.setsockopt(zmq.LINGER, 0)

    header_push_ep = cfg['net']['header_push'].strip()
    pull_bind_ep = header_push_ep
    try:
        parsed = urlparse(header_push_ep)
    except ValueError:
        parsed = None

    if parsed and parsed.scheme and parsed.port is not None:
        if parsed.hostname not in (None, "", "*", "0.0.0.0"):
            pull_bind_ep = f"{parsed.scheme}://0.0.0.0:{parsed.port}"

    pull.bind(pull_bind_ep)
    pull.RCVTIMEO = 0  # non-blocking

    rw, rh = cfg['return']['width'], cfg['return']['height']
    rfps   = cfg['return']['fps']
    rbr    = cfg['return']['bitrate_kbps']
    rscale = cfg['return'].get('vbv_scale', 2)
    ret_vw = make_return_writer(
        cfg['net']['pc_ip'], cfg['net']['rtp_return_port'],
        rw, rh, fps=rfps, bitrate_kbps=rbr, vbv_scale=rscale
    )


    latest_header = {"frame_id": 0, "src_ts_ms": 0}

    try:
        while not stop_event.is_set():
            # receive frame
            ok, frame = recv.read()
            if not ok:
                continue

            # headers (non-blocking drain)
            try:
                while True:
                    latest_header = pull.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            if frame.ndim == 3 and frame.shape[2] == 4:  # RGBA->BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            frame_h, frame_w = frame.shape[:2] if frame is not None else (uh, uw)

            rx_ts_ms = int(time.monotonic_ns() / 1e6)
            boxes = yolo.infer(frame)
            infer_ts_ms = int(time.monotonic_ns() / 1e6)

            msg = DetectionMsg(
                frame_id=latest_header.get("frame_id", 0),
                src_ts_ms=latest_header.get("src_ts_ms", 0),
                rx_ts_ms=rx_ts_ms,
                infer_ts_ms=infer_ts_ms,
                img_w=frame_w, img_h=frame_h,
                boxes=boxes,
            )
            try:
                pub.send_string(msg.model_dump_json(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            # draw + return video
            ov = frame.copy()
            for b in boxes:
                x1 = int(b.x * frame_w); y1 = int(b.y * frame_h)
                x2 = int((b.x + b.w) * frame_w); y2 = int((b.y + b.h) * frame_h)
                cv2.rectangle(ov, (x1,y1), (x2,y2), (0,255,0), 2)
            if ret_vw and ret_vw.isOpened():
                if ov.shape[1] != rw or ov.shape[0] != rh:
                    ov_resized = letterbox_resize(ov, rw, rh)
                else:
                    ov_resized = ov
                if ov_resized is not None:
                    ret_vw.write(ov_resized)

    except KeyboardInterrupt:
        pass
    finally:
        print("[server] shutting down...")
        try: recv.release()
        except: pass
        try: 
            if ret_vw: ret_vw.release()
        except: pass
        for s in (pub, pull):
            try: s.close(0)
            except: pass
        try: ctx.term()
        except: pass
        time.sleep(0.05)

if __name__ == "__main__":
    main()
