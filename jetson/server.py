import argparse, time, yaml, zmq
from common.schemas import DetectionMsg
from jetson.receiver import GRecv
from jetson.yolo_engine import YoloEngine
# Build a GStreamer encoder pipeline for return video
import threading
import cv2


def make_return_writer(pc_ip, port, w, h, fps=30, bitrate=4000):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    w,h = cfg['video']['width'], cfg['video']['height']
    port = cfg['net']['rtp_port']

    recv = GRecv(port, w, h)
    yolo = YoloEngine(
    engine_path=cfg['yolo']['engine_path'],
    conf_thres=cfg['yolo']['conf_thres'],
    iou_thres=cfg['yolo']['iou_thres'],
    input_size=cfg['yolo']['input_size'],
    )
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1); pub.setsockopt(zmq.LINGER, 0)
    ep = cfg['net']['zmq_results']  # e.g., "tcp://192.168.0.2:5556"
    assert ep.startswith("tcp://")
    hostport = ep[len("tcp://"):]   # "192.168.0.2:5556"
    port = hostport.split(":")[-1]  # "5556"
    pub.bind(f"tcp://0.0.0.0:{port}")
    pull = ctx.socket(zmq.PULL)
    pull.bind("tcp://0.0.0.0:5555")  # Bind; PC connects to cfg['net']['header_push']
    pull.RCVTIMEO = 0               # non-blocking
    latest_header = {"frame_id": 0, "src_ts_ms": 0}
    ret_vw = make_return_writer(cfg['net']['pc_ip'], cfg['net']['rtp_return_port'], w, h, fps=cfg['video']['fps'], bitrate=cfg['video']['bitrate_kbps'])
    if not ret_vw or not ret_vw.isOpened():
        print("[server] Falling back to software return encoder")
        ret_vw = make_return_writer_sw(...)
    frame_id = 0
    while True:
        ok, frame = recv.read()
        if not ok:
            continue
        try:
            while True:
                latest_header = pull.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            pass                  
        frame_id += 1
        rx_ts_ms = int(time.monotonic_ns() / 1e6)
        # ensure 3-ch BGR for YOLO
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        boxes = yolo.infer(frame)
        infer_ts_ms = int(time.monotonic_ns() / 1e6)
        msg = DetectionMsg(
            frame_id=latest_header["frame_id"],
            src_ts_ms=latest_header["src_ts_ms"],
            rx_ts_ms=rx_ts_ms,
            infer_ts_ms=infer_ts_ms,
            img_w=w, img_h=h,
            boxes=boxes,
        )
        pub.send_string(msg.model_dump_json())
        # draw simple overlays on a copy
        ov = frame.copy()
        for b in boxes:
            x1 = int(b.x * w); y1 = int(b.y * h)
            x2 = int((b.x + b.w) * w); y2 = int((b.y + b.h) * h)
            cv2.rectangle(ov, (x1,y1), (x2,y2), (0,255,0), 2)
        if ret_vw and ret_vw.isOpened():
            ret_vw.write(ov)



if __name__ == "__main__":
    main()
