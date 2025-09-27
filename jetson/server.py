import argparse, json, logging, time, yaml, zmq
from typing import Any, Dict, Mapping, Optional, Sequence

from pydantic import ValidationError

from common.camera import CameraIntrinsics, CameraIntrinsicsConfigError
from common.control import ControlConfig, ControlConfigError
from common.ranging import (
    KnownSizeRangingConfig,
    KnownSizeRangingConfigError,
    iter_distance_estimates,
    iter_ranging_candidates,
    resolve_class_label,
)
from common.schemas import CamState, DetectionMsg, detection_msg_to_json
from jetson.receiver import GRecv
from jetson.controller import ControlLoop
from jetson.yolo_engine import YoloEngine
# Build a GStreamer encoder pipeline for return video
import threading
import cv2
from common.shutdown import install_signal_handlers


_RANGING_LOG = logging.getLogger("jetson.ranging")
_RANGING_LOG_PRECISION = 4


def _round_for_log(value: Any, precision: int = _RANGING_LOG_PRECISION) -> Any:
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, list):
        return [_round_for_log(v, precision) for v in value]
    if isinstance(value, tuple):
        return [_round_for_log(v, precision) for v in value]
    if isinstance(value, dict):
        return {k: _round_for_log(v, precision) for k, v in value.items()}
    return value


def _format_ranging_log(
    *,
    frame_id: Any,
    src_ts_ms: Any,
    rx_ts_ms: Any,
    infer_ts_ms: Any,
    rows: Sequence[Mapping[str, Any]],
    precision: int = _RANGING_LOG_PRECISION,
) -> str:
    header = (
        f"frame={frame_id} src_ts={src_ts_ms} rx_ts={rx_ts_ms} infer_ts={infer_ts_ms}"
    )
    formatted_rows = []
    for row in rows:
        idx = row.get("idx")
        label = row.get("label") or "?"
        distance = row.get("distance_m")
        source = row.get("source")
        px_size = row.get("pixel_size_px")
        conf = row.get("conf")
        target = row.get("target")
        smoothed = row.get("distance_smoothed_m")

        idx_text = f"#{idx}" if idx is not None else "#?"
        label_text = f"{label}".strip() or "?"
        dist_text = "dist=?"
        if isinstance(distance, (int, float)):
            src_hint = {
                "height": "h",
                "width": "w",
                "average": "avg",
            }.get(str(source), str(source) if source else "")
            suffix = f" ({src_hint})" if src_hint else ""
            dist_text = f"dist={distance:.{precision}f}m{suffix}"
        px_text = (
            f"px={px_size:.{precision}f}"
            if isinstance(px_size, (int, float))
            else None
        )
        conf_text = (
            f"conf={conf:.{min(3, precision)}f}"
            if isinstance(conf, (int, float))
            else None
        )
        target_text = None
        if target:
            if isinstance(smoothed, (int, float)):
                target_text = f"target sm={smoothed:.{precision}f}m"
            else:
                target_text = "target"

        columns = [idx_text, label_text, dist_text]
        if px_text:
            columns.append(px_text)
        if conf_text:
            columns.append(conf_text)
        if target_text:
            columns.append(target_text)

        formatted_rows.append(" ".join(columns))

    return header + " | " + " | ".join(formatted_rows)

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


def _parse_class_labels(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """Return a mapping from YOLO class IDs to human-readable labels."""

    yolo_section = cfg.get("yolo", {})
    if not isinstance(yolo_section, Mapping):
        raise SystemExit("config missing 'yolo' section")

    raw_map = yolo_section.get("class_labels", {})
    if raw_map is None:
        return {}
    if not isinstance(raw_map, Mapping):
        raise SystemExit("yolo.class_labels must be a mapping when provided")

    parsed: Dict[str, str] = {}
    for raw_key, raw_value in raw_map.items():
        key = str(raw_key).strip()
        if not key:
            raise SystemExit("yolo.class_labels keys must be non-empty strings")
        if raw_value is None:
            continue
        label = str(raw_value).strip()
        if not label:
            raise SystemExit(
                f"yolo.class_labels[{raw_key!r}] must map to a non-empty string"
            )
        parsed[key] = label
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _RANGING_LOG.setLevel(logging.INFO)

    logging_cfg = cfg.get("logging", {}) or {}
    cli_json_logs = bool(logging_cfg.get("cli_json", False))

    w,h = cfg['video']['width'], cfg['video']['height']
    try:
        control_cfg = ControlConfig.from_raw_config(cfg, (w, h))
    except ControlConfigError as exc:
        raise SystemExit(f"invalid control configuration: {exc}") from exc

    try:
        camera_intrinsics = CameraIntrinsics.from_raw_config(cfg, (w, h))
    except CameraIntrinsicsConfigError as exc:
        raise SystemExit(f"invalid camera configuration: {exc}") from exc

    try:
        ranging_cfg = KnownSizeRangingConfig.from_raw_config(cfg)
    except KnownSizeRangingConfigError as exc:
        raise SystemExit(f"invalid known-size ranging configuration: {exc}") from exc
    class_labels = _parse_class_labels(cfg)
    port = cfg['net']['rtp_port']

    stop_event = install_signal_handlers()

    recv = GRecv(port, w, h)

    yolo = YoloEngine(
        engine_path=cfg['yolo']['engine_path'],
        conf_thres=cfg['yolo']['conf_thres'],
        iou_thres=cfg['yolo']['iou_thres'],
        input_size=cfg['yolo']['input_size'],
        preprocess_mode=cfg['yolo'].get('preprocess_mode', 'bilinear')
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
    distance_alpha = ranging_cfg.ema_alpha if ranging_cfg.enabled else None
    controller = ControlLoop(
        control_cfg,
        ctrl_pub,
        distance_alpha=distance_alpha,
        cli_json_logs=cli_json_logs,
    )
    ranging_log_interval_s = getattr(controller, "log_interval_s", 0.5)
    ranging_last_log_time = 0.0
    ranging_logged_once = False
    ranging_last_target_idx: Optional[int] = None

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
            box_index_map = {id(box): idx for idx, box in enumerate(boxes)}
            ranging_log_entries: Dict[int, Dict[str, Any]] = {}
            if class_labels:
                for box in boxes:
                    box.cls = resolve_class_label(box.cls, class_labels)
            if ranging_cfg.enabled:
                _ranging_candidates = list(
                    iter_ranging_candidates(boxes, (w, h), class_labels, ranging_cfg)
                )
                _distance_estimates = list(
                    iter_distance_estimates(_ranging_candidates, camera_intrinsics, ranging_cfg)
                )
                for estimate in _distance_estimates:
                    estimate.candidate.box.distance_m = estimate.distance_m
                    estimate.candidate.box.distance_src = estimate.source
                    idx = box_index_map.get(id(estimate.candidate.box))
                    entry = {
                        "idx": idx,
                        "label": estimate.candidate.box.cls,
                        "conf": estimate.candidate.box.conf,
                        "source": estimate.source,
                        "pixel_size_px": estimate.pixel_size_px,
                        "distance_m": estimate.distance_m,
                    }
                    if idx is not None:
                        ranging_log_entries[idx] = entry
            infer_ts_ms = int(time.monotonic_ns() / 1e6)

            msg = DetectionMsg(
                frame_id=latest_header.get("frame_id", 0),
                src_ts_ms=latest_header.get("src_ts_ms", 0),
                rx_ts_ms=rx_ts_ms,
                infer_ts_ms=infer_ts_ms,
                img_w=w, img_h=h,
                boxes=boxes,
            )

            controller.update_detection(msg)

            if ranging_cfg.enabled and ranging_log_entries:
                target_idx = msg.target_idx
                target_smoothed = msg.target_distance_smoothed_m
                log_rows = []
                for idx in sorted(ranging_log_entries):
                    base_entry = dict(ranging_log_entries[idx])
                    if target_idx is not None and idx == target_idx:
                        base_entry["target"] = True
                        base_entry["distance_smoothed_m"] = target_smoothed
                    log_rows.append(_round_for_log(base_entry))
                if log_rows:
                    now_log = time.monotonic()
                    should_log = False
                    if not ranging_logged_once:
                        should_log = True
                    elif target_idx != ranging_last_target_idx:
                        should_log = True
                    elif (now_log - ranging_last_log_time) >= ranging_log_interval_s:
                        should_log = True

                    if should_log:
                        log_payload = {
                            "frame_id": msg.frame_id,
                            "src_ts_ms": msg.src_ts_ms,
                            "rx_ts_ms": rx_ts_ms,
                            "infer_ts_ms": infer_ts_ms,
                            "ranging": log_rows,
                        }
                        if cli_json_logs:
                            _RANGING_LOG.info(json.dumps(_round_for_log(log_payload)))
                        else:
                            _RANGING_LOG.info(
                                _format_ranging_log(
                                    frame_id=log_payload["frame_id"],
                                    src_ts_ms=log_payload["src_ts_ms"],
                                    rx_ts_ms=log_payload["rx_ts_ms"],
                                    infer_ts_ms=log_payload["infer_ts_ms"],
                                    rows=log_rows,
                                    precision=_RANGING_LOG_PRECISION,
                                )
                            )
                        ranging_last_log_time = now_log
                        ranging_last_target_idx = target_idx
                        ranging_logged_once = True

            try:
                pub.send_string(detection_msg_to_json(msg), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            controller.tick(time.monotonic())

            # draw + return video (draw directly on the frame once inference is done)
            for b in boxes:
                x1 = int(b.x * w); y1 = int(b.y * h)
                x2 = int((b.x + b.w) * w); y2 = int((b.y + b.h) * h)
                colour = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                if b.cls in ("person", "1"):
                    cv2.line(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.line(frame, (x1, y2), (x2, y1), colour, 2)

                label_text = None
                if ranging_cfg.enabled and getattr(b, "distance_src", None):
                    indicator_thickness = 2
                    indicator_len = max(4, int(0.25 * max(y2 - y1, x2 - x1)))
                    mid_y = (y1 + y2) // 2
                    mid_x = (x1 + x2) // 2
                    if b.distance_src == "height":
                        start_pt = (x1, max(y1, mid_y - indicator_len // 2))
                        end_pt = (x1, min(y2, mid_y + indicator_len // 2))
                        cv2.line(frame, start_pt, end_pt, colour, indicator_thickness)
                    elif b.distance_src == "width":
                        start_pt = (max(x1, mid_x - indicator_len // 2), y1)
                        end_pt = (min(x2, mid_x + indicator_len // 2), y1)
                        cv2.line(frame, start_pt, end_pt, colour, indicator_thickness)
                    elif b.distance_src == "average":
                        vert_start = (x1, max(y1, mid_y - indicator_len // 2))
                        vert_end = (x1, min(y2, mid_y + indicator_len // 2))
                        horiz_start = (max(x1, mid_x - indicator_len // 2), y1)
                        horiz_end = (min(x2, mid_x + indicator_len // 2), y1)
                        cv2.line(frame, vert_start, vert_end, colour, indicator_thickness)
                        cv2.line(frame, horiz_start, horiz_end, colour, indicator_thickness)
                if ranging_cfg.enabled and b.distance_m is not None:
                    label_text = f"{b.distance_m:.2f} m"
                    cls_label = (b.cls or "").strip()
                    if cls_label:
                        label_text = f"{label_text} - {cls_label}"

                if label_text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    text_size, baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    text_w, text_h = text_size
                    # Try to place the label above the box; fall back to below if needed.
                    text_x = max(0, min(x1, w - text_w - 4))
                    text_y = y1 - 8
                    if text_y - text_h - baseline < 0:
                        text_y = min(h - 4, y2 + text_h + 8)
                    box_pt1 = (text_x - 2, text_y - text_h - baseline - 2)
                    box_pt2 = (text_x + text_w + 2, text_y + 2)
                    cv2.rectangle(frame, box_pt1, box_pt2, (0, 0, 0), thickness=cv2.FILLED)
                    cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, colour, thickness, cv2.LINE_AA)
            if ret_vw and ret_vw.isOpened():
                ret_vw.write(frame)

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
