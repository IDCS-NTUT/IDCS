import argparse, time, yaml, zmq
from common.schemas import DetectionMsg
from jetson.receiver import GRecv
from jetson.yolo_engine import YoloEngine

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
    yolo = YoloEngine(cfg['yolo']['conf_thres'], cfg['yolo']['iou_thres'])

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    ep = cfg['net']['zmq_results']  # e.g., "tcp://192.168.0.2:5556"
    assert ep.startswith("tcp://")
    hostport = ep[len("tcp://"):]   # "192.168.0.2:5556"
    port = hostport.split(":")[-1]  # "5556"
    pub.bind(f"tcp://0.0.0.0:{port}")

    frame_id = 0
    while True:
        ok, frame = recv.read()
        if not ok:
            break
        frame_id += 1
        rx_ts_ms = int(time.monotonic_ns() / 1e6)
        boxes = yolo.infer(frame)
        infer_ts_ms = int(time.monotonic_ns() / 1e6)
        msg = DetectionMsg(
            frame_id=frame_id,
            src_ts_ms=0,     # fill if you add header channel later
            rx_ts_ms=rx_ts_ms,
            infer_ts_ms=infer_ts_ms,
            img_w=w, img_h=h,
            boxes=boxes,
        )
        pub.send_string(msg.model_dump_json())

if __name__ == "__main__":
    main()
