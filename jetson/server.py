import argparse, json, logging, math, time, yaml, zmq
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from common.camera import CameraIntrinsics, CameraIntrinsicsConfigError
from common.control import (
    ControlConfig,
    ControlConfigError,
    LaserConfigError,
    LaserMountConfig,
)
from common.ranging import (
    KnownSizeRangingConfig,
    KnownSizeRangingConfigError,
    iter_distance_estimates,
    iter_ranging_candidates,
    resolve_class_label,
)
from common.schemas import CamState, DetectionMsg, detection_msg_to_json
from pc.renderers._geometry import clip_segment_to_rect
from jetson.receiver import GRecv
from jetson.controller import ControlLoop
from jetson.yolo_engine import YoloEngine
# Build a GStreamer encoder pipeline for return video
import threading
import cv2
import numpy as np
from common.shutdown import install_signal_handlers


_RANGING_LOG = logging.getLogger("jetson.ranging")
_RANGING_LOG_PRECISION = 4


def _is_finite_point(point: Tuple[float, float]) -> bool:
    return all(math.isfinite(coord) for coord in point)


def _draw_laser_overlay(
    frame,
    msg: DetectionMsg,
    laser_cfg: Optional[LaserMountConfig],
) -> None:
    """Overlay the projected laser origin/beam/dot on the return frame."""

    if laser_cfg is None:
        return

    origin = msg.laser_origin_px
    dot = msg.laser_dot_px
    on_target = msg.laser_on_target
    if origin is None and dot is None:
        return

    h, w = frame.shape[:2]

    def _within_image(pt: Tuple[float, float]) -> bool:
        if not _is_finite_point(pt):
            return False
        x, y = pt
        return -w <= x <= 2 * w and -h <= y <= 2 * h

    colour_base = tuple(int(c) for c in laser_cfg.render.colour_bgr)
    colour_on = (0, 255, 0)
    colour_warn = (0, 191, 255)
    if on_target is True:
        beam_colour = colour_on
        dot_colour = colour_on
    elif on_target is False:
        beam_colour = colour_warn
        dot_colour = colour_base
    else:
        beam_colour = colour_base
        dot_colour = colour_base

    thickness = max(1, int(laser_cfg.render.thickness_px))

    origin_pt = None
    if origin is not None and _within_image(origin):
        origin_pt = (int(round(origin[0])), int(round(origin[1])))
        cv2.circle(frame, origin_pt, max(2, thickness + 1), beam_colour, thickness)
        cv2.circle(frame, origin_pt, max(1, thickness - 1), (0, 0, 0), cv2.FILLED)

    dot_pt = None
    if dot is not None and _within_image(dot):
        dot_pt = (int(round(dot[0])), int(round(dot[1])))
        cv2.circle(frame, dot_pt, max(3, thickness + 2), dot_colour, thickness)
        cv2.circle(frame, dot_pt, max(1, thickness - 1), dot_colour, cv2.FILLED)

    beam_segment = None
    if origin is not None and dot is not None:
        clipped = clip_segment_to_rect(origin, dot, w, h)
        if clipped is not None:
            beam_segment = clipped

    if beam_segment is not None:
        (sx, sy), (ex, ey) = beam_segment
        start_pt = (int(round(sx)), int(round(sy)))
        end_pt = (int(round(ex)), int(round(ey)))
        if start_pt != end_pt:
            cv2.line(frame, start_pt, end_pt, beam_colour, thickness)

    status_bits = []
    if msg.parallax_compensation_active:
        status_bits.append("parallax:active")
    elif msg.parallax_compensation_active is False:
        status_bits.append("parallax:inactive")
    if on_target is True:
        status_bits.append("laser:on-target")
    elif on_target is False:
        status_bits.append("laser:adjusting")
    range_source = msg.laser_range_source
    if range_source:
        status_bits.append(f"range:{range_source}")
    assumed = msg.laser_range_m
    if assumed is not None and math.isfinite(assumed) and assumed > 0:
        status_bits.append(f"assume:{assumed:.1f}m")
    distance = msg.target_distance_smoothed_m
    if distance is not None and math.isfinite(distance) and distance > 0:
        status_bits.append(f"meas:{distance:.1f}m")

    if status_bits:
        status_text = " | ".join(status_bits)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness_text = 1
        margin = 8
        text_size, baseline = cv2.getTextSize(status_text, font, scale, thickness_text)
        text_w, text_h = text_size
        x0 = max(0, w - text_w - 2 * margin)
        y0 = margin + text_h
        cv2.rectangle(
            frame,
            (x0 - 4, y0 - text_h - baseline - 4),
            (x0 + text_w + 4, y0 + 4),
            (0, 0, 0),
            thickness=cv2.FILLED,
        )
        cv2.putText(frame, status_text, (x0, y0), font, scale, beam_colour, thickness_text, cv2.LINE_AA)


def _wrap_degrees(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    wrapped = (value + 180.0) % 360.0
    return wrapped - 180.0


def _draw_attitude_overlay(frame, cam_state: Optional[CamState]) -> None:
    """Render azimuth/elevation guides for the return video feed."""

    if cam_state is None:
        return

    yaw_rad = getattr(cam_state, "pan", None)
    pitch_rad = getattr(cam_state, "tilt", None)

    if yaw_rad is None and pitch_rad is None:
        return

    h, w = frame.shape[:2]
    margin = 12
    colour = (0, 255, 0)
    thickness = 2

    yaw_deg = None
    if yaw_rad is not None and math.isfinite(yaw_rad):
        yaw_deg = _wrap_degrees(math.degrees(float(yaw_rad)))

    pitch_deg = None
    if pitch_rad is not None and math.isfinite(pitch_rad):
        pitch_deg = _wrap_degrees(math.degrees(float(pitch_rad)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thickness = 2

    def _draw_degree_label(
        image,
        label_text: str,
        org: Tuple[int, int],
        *,
        align: str = "left",
    ) -> None:
        text_size, _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        text_w, text_h = text_size

        if align == "center":
            org_x = int(round(org[0] - text_w / 2))
            org_y = org[1]
        elif align == "right":
            org_x = int(round(org[0] - text_w))
            org_y = org[1]
        else:
            org_x, org_y = org

        if org_y < 0 or org_y >= h + text_h:
            return

        cv2.putText(
            image,
            label_text,
            (org_x, org_y),
            font,
            font_scale,
            colour,
            font_thickness,
            cv2.LINE_AA,
        )

        circle_radius = max(1, int(round(text_h * 0.25)))
        degree_centre_x = org_x + text_w + circle_radius
        degree_centre_y = org_y - text_h + circle_radius
        cv2.circle(
            image,
            (int(round(degree_centre_x)), int(round(degree_centre_y))),
            circle_radius,
            colour,
            font_thickness,
            cv2.LINE_AA,
        )

    if yaw_deg is not None:
        x0 = margin
        x1 = w - margin
        centre_x = (x0 + x1) * 0.5
        base_y = margin
        small_notch = 10
        large_notch = 18
        px_per_deg = max(2.0, (x1 - x0) / 180.0)
        span_deg = (x1 - x0) / px_per_deg
        half_span = span_deg * 0.5
        start_deg = yaw_deg - half_span - 5.0
        end_deg = yaw_deg + half_span + 5.0

        cv2.line(frame, (int(x0), int(base_y)), (int(x1), int(base_y)), colour, thickness)

        pointer_height = 10
        pointer_half = 12
        pointer_pts = [
            (int(round(centre_x)), int(round(base_y - pointer_height))),
            (int(round(centre_x - pointer_half)), int(round(base_y - thickness))),
            (int(round(centre_x + pointer_half)), int(round(base_y - thickness))),
        ]
        cv2.fillConvexPoly(frame, np.array(pointer_pts, dtype=np.int32), colour)

        tick_start = int(math.floor(start_deg / 5.0)) * 5
        tick_end = int(math.ceil(end_deg / 5.0)) * 5
        for deg in range(tick_start, tick_end + 1, 5):
            x = centre_x + (deg - yaw_deg) * px_per_deg
            if x < x0 - 1 or x > x1 + 1:
                continue
            notch_len = small_notch
            if deg % 10 == 0:
                notch_len = large_notch
            start_pt = (int(round(x)), int(round(base_y)))
            end_pt = (int(round(x)), int(round(base_y + notch_len)))
            cv2.line(frame, start_pt, end_pt, colour, thickness)
            if deg % 10 == 0:
                label = f"{int(_wrap_degrees(float(deg))):d}"
                text_y = int(round(base_y + notch_len + 12))
                if 0 <= text_y < h + 20:
                    _draw_degree_label(frame, label, (int(round(x)), text_y), align="center")

    if pitch_deg is not None:
        y0 = margin
        y1 = h - margin
        centre_y = (y0 + y1) * 0.5
        base_x = w - margin
        small_notch = 10
        large_notch = 18
        pitch_min = -45.0
        pitch_max = 45.0
        display_pitch_deg = max(pitch_min, min(pitch_max, pitch_deg))
        px_per_deg = max(2.0, (y1 - y0) / (pitch_max - pitch_min))

        cv2.line(frame, (int(base_x), int(y0)), (int(base_x), int(y1)), colour, thickness)

        pointer_width = 10
        pointer_height = 12
        pointer_centre_y = centre_y - display_pitch_deg * px_per_deg
        pointer_centre_y = max(y0, min(pointer_centre_y, y1))
        top_y = max(y0, pointer_centre_y - pointer_height)
        bottom_y = min(y1, pointer_centre_y + pointer_height)
        pointer_pts = [
            (int(round(base_x + thickness)), int(round(pointer_centre_y))),
            (int(round(base_x + pointer_width)), int(round(top_y))),
            (int(round(base_x + pointer_width)), int(round(bottom_y))),
        ]
        cv2.fillConvexPoly(frame, np.array(pointer_pts, dtype=np.int32), colour)

        tick_start = int(math.ceil(pitch_min / 5.0)) * 5
        tick_end = int(math.floor(pitch_max / 5.0)) * 5
        for deg in range(tick_start, tick_end + 1, 5):
            y = centre_y - deg * px_per_deg
            if y < y0 - 1 or y > y1 + 1:
                continue
            notch_len = small_notch
            if deg % 10 == 0:
                notch_len = large_notch
            start_pt = (int(round(base_x)), int(round(y)))
            end_pt = (int(round(base_x - notch_len)), int(round(y)))
            cv2.line(frame, start_pt, end_pt, colour, thickness)
            if deg % 10 == 0:
                label = f"{int(_wrap_degrees(float(deg))):d}"
                text_y = int(round(y + 6))
                if -20 < text_y < h:
                    _draw_degree_label(
                        frame,
                        label,
                        (int(round(base_x - notch_len - 8)), text_y),
                        align="right",
                    )


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
        # Convert to lightweight format before NVMM upload
        "videoconvert ! video/x-raw,format=BGRx,width={w},height={h},framerate={fps}/1 ! "
        # Upload + convert to NV12 in NVMM for HW encoder
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


def _derive_return_file_path(source_spec: str) -> Path:
    raw = (source_spec or "").strip()
    if raw.startswith("file:"):
        raw = raw.split(":", 1)[1]
    raw = raw.strip()
    if raw:
        candidate = Path(raw)
        stem = candidate.stem or "return"
    else:
        stem = "return"
    return Path.cwd() / f"{stem}_annotated.mp4"


def make_file_return_writer(path: Path, w: int, h: int, fps: int = 30, codec: str = "mp4v"):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[server] WARN: failed to create directory for return video: {exc}")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if writer.isOpened():
        print(f"[server] writing return video to {path}")
    else:
        print(f"[server] WARN: failed to open return video file at {path}")
    return writer

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

    source_spec = str(cfg.get("source", "") or "")
    file_source = source_spec.strip().startswith("file:")
    if file_source:
        logging.info("source is file; disabling control publisher and recording return video")

    w,h = cfg['video']['width'], cfg['video']['height']
    try:
        control_cfg = ControlConfig.from_raw_config(cfg, (w, h))
    except ControlConfigError as exc:
        raise SystemExit(f"invalid control configuration: {exc}") from exc

    try:
        laser_cfg = LaserMountConfig.from_raw_config(cfg)
    except LaserConfigError as exc:
        raise SystemExit(f"invalid laser configuration: {exc}") from exc

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

    ctrl_pub: Optional[zmq.Socket] = None
    ctrl_ep = cfg['net'].get('zmq_control')
    if not file_source and ctrl_ep:
        ctrl_pub = ctx.socket(zmq.PUB)
        ctrl_pub.setsockopt(zmq.SNDHWM, 1)
        ctrl_pub.setsockopt(zmq.LINGER, 0)
        ctrl_port = _parse_tcp_port(ctrl_ep, "zmq_control")
        ctrl_pub.bind(f"tcp://0.0.0.0:{ctrl_port}")
    elif not file_source:
        raise SystemExit("config missing net.zmq_control endpoint")

    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.RCVHWM, 10)
    pull.setsockopt(zmq.LINGER, 0)
    header_ep = cfg['net'].get('header_push')
    header_port = _parse_tcp_port(header_ep, "header_push")
    pull.bind(f"tcp://0.0.0.0:{header_port}")
    pull.RCVTIMEO = 0  # non-blocking

    if file_source:
        return_file_path = _derive_return_file_path(source_spec)
        ret_vw = make_file_return_writer(
            return_file_path,
            w,
            h,
            fps=cfg['video']['fps'],
        )
    else:
        ret_vw = make_return_writer(
            cfg['net']['pc_ip'],
            cfg['net']['rtp_return_port'],
            w,
            h,
            fps=cfg['video']['fps'],
            bitrate=cfg['video']['bitrate_kbps'],
        )

    latest_header = {"frame_id": 0, "src_ts_ms": 0}
    latest_cam_state: Optional[CamState] = None
    controller: Optional[ControlLoop] = None
    if not file_source:
        distance_alpha = ranging_cfg.ema_alpha if ranging_cfg.enabled else None
        if ctrl_pub is None:
            raise RuntimeError("control publisher is not initialized")
        controller = ControlLoop(
            control_cfg,
            ctrl_pub,
            laser_mount=laser_cfg,
            distance_alpha=distance_alpha,
            cli_json_logs=cli_json_logs,
        )
        ranging_log_interval_s = getattr(controller, "log_interval_s", 0.5)
    else:
        ranging_log_interval_s = 0.5
    ranging_last_log_time = 0.0
    ranging_logged_once = False
    ranging_last_target_idx: Optional[int] = None

    try:
        while not stop_event.is_set():
            # receive frame
            ok, frame = recv.read()
            if not ok:
                if controller is not None:
                    controller.tick(time.monotonic())
                continue

            # headers (non-blocking drain)
            try:
                while True:
                    header_obj = pull.recv_json(flags=zmq.NOBLOCK)
                    if (
                        controller is not None
                        and isinstance(header_obj, dict)
                        and header_obj.get("type") == "CamState"
                    ):
                        try:
                            cam_state = CamState(**header_obj)
                        except ValidationError as exc:
                            logging.warning("invalid CamState header: %s", exc)
                        else:
                            controller.update_cam_state(cam_state)
                            latest_cam_state = cam_state
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

            if controller is not None:
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
            if controller is not None:
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

            _draw_attitude_overlay(frame, latest_cam_state)
            _draw_laser_overlay(frame, msg, laser_cfg)
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
        for s in (pub, pull):
            try: s.close(0)
            except: pass
        if ctrl_pub is not None:
            try: ctrl_pub.close(0)
            except: pass
        try: ctx.term()
        except: pass
        time.sleep(0.05)

if __name__ == "__main__":
    main()
