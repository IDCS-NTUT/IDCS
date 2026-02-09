import argparse, json, logging, math, time, zmq
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from common.camera import CameraIntrinsics, CameraIntrinsicsConfigError
from common.control import (
    ControlConfig,
    ControlConfigError,
    LaserConfigError,
    LaserMountConfig,
)
from common.config_sync import (
    ConfigSyncError,
    merge_config_maps,
    parse_config_text,
    read_snapshot,
    resolve_active_video_profile,
    resolve_config_sync_endpoint,
    sync_as_server,
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
from jetson.receiver import CsiVideoReader, FileVideoReader, GRecv
from jetson.controller import ControlLoop
from jetson.yolo_engine import YoloEngine
# Build a GStreamer encoder pipeline for return video
import threading
import cv2
import gi
from common.shutdown import install_signal_handlers

gi.require_version("Gst", "1.0")
from gi.repository import Gst

_YOLO_ENGINE_DIR = Path("/home/idcs/Desktop/project/repo/IDCS/assets")
_YOLO_ENGINE_SIZES = {"nano", "small"}
_YOLO_RES_SUFFIX_BY_PROFILE = {"720p": "1280", "1080p": "1920"}
_YOLO_RES_SUFFIX_BY_WIDTH = {1280: "1280", 1920: "1920"}


_RANGING_LOG = logging.getLogger("jetson.ranging")
_RANGING_LOG_PRECISION = 4

_FILE_SOURCE_SYNC_TIMEOUT_S = 5.0


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
        x0 = margin + 4
        y0 = max(margin + text_h, h - margin - baseline)
        rect_tl = (
            int(max(0, x0 - 4)),
            int(max(0, y0 - text_h - baseline - 4)),
        )
        rect_br = (
            int(min(w - 1, x0 + text_w + 4)),
            int(min(h - 1, y0 + 4)),
        )
        cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), thickness=cv2.FILLED)
        cv2.putText(
            frame,
            status_text,
            (int(x0), int(y0)),
            font,
            scale,
            beam_colour,
            thickness_text,
            cv2.LINE_AA,
        )


def _draw_lead_overlay(frame: Any, msg: DetectionMsg) -> None:
    target_idx = msg.target_idx
    lead_uv = msg.target_lead_uv
    velocity = msg.target_velocity_px_s

    if target_idx is None or lead_uv is None or velocity is None:
        return

    if target_idx < 0 or target_idx >= len(msg.boxes):
        return

    box = msg.boxes[target_idx]
    centre_u = (box.x + box.w / 2.0) * msg.img_w
    centre_v = (box.y + box.h / 2.0) * msg.img_h

    if not _is_finite_point((centre_u, centre_v)) or not _is_finite_point(lead_uv):
        return

    h, w = frame.shape[:2]

    start_pt = (centre_u, centre_v)
    end_pt = (lead_uv[0], lead_uv[1])

    clipped = clip_segment_to_rect(start_pt, end_pt, w, h)
    if clipped is not None:
        start_pt, end_pt = clipped

    start_px = (int(round(start_pt[0])), int(round(start_pt[1])))
    end_px = (int(round(end_pt[0])), int(round(end_pt[1])))

    colour = (32, 160, 255)
    tip_length = 0.2

    if start_px != end_px:
        cv2.arrowedLine(frame, start_px, end_px, colour, 2, cv2.LINE_AA, tipLength=tip_length)

    lead_circle = (int(round(lead_uv[0])), int(round(lead_uv[1])))
    cv2.circle(frame, lead_circle, 4, colour, thickness=2, lineType=cv2.LINE_AA)

    speed = math.hypot(velocity[0], velocity[1])
    lead_time = msg.target_lead_time_s or 0.0
    label = f"lead {speed:.1f}px/s"
    if lead_time > 0.0:
        label = f"{label} @ {lead_time * 1000.0:.0f}ms"

    text_origin = (
        int(round(min(max(0.0, lead_uv[0] + 6.0), w - 1))),
        int(round(min(max(0.0, lead_uv[1] - 6.0), h - 1))),
    )
    _draw_text_box(
        frame,
        label,
        text_origin,
        colour,
        font_scale=0.5,
        thickness=1,
        padding=4,
    )


def _draw_predictive_overlay(frame: Any, msg: DetectionMsg) -> None:
    if not getattr(msg, "predictive_active", False):
        return

    box = msg.predictive_box_px
    if box is None:
        return

    if not all(math.isfinite(value) for value in box):
        return

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return

    def _clip_coord(value: float, upper: int) -> int:
        return int(round(max(0.0, min(value, float(max(upper - 1, 0))))))

    x1_i = _clip_coord(x1, w)
    y1_i = _clip_coord(y1, h)
    x2_i = _clip_coord(x2, w)
    y2_i = _clip_coord(y2, h)

    if x2_i <= x1_i or y2_i <= y1_i:
        return

    # Match the standard detection styling (green, 2px lines) while blinking by
    # skipping every other frame worth of draw calls.
    phase = int(time.monotonic() * 4.0) & 1
    if phase == 1:
        return

    colour = (0, 255, 0)
    thickness = 2
    cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), colour, thickness, lineType=cv2.LINE_AA)


def _draw_text_box(
    frame: Any,
    text: str,
    origin: Tuple[int, int],
    colour: Tuple[int, int, int],
    *,
    font_scale: float,
    thickness: int,
    padding: int = 4,
    box_colour: Tuple[int, int, int] = (0, 32, 0),
) -> None:
    metrics = _measure_text(text, font_scale, thickness)
    text_w = metrics["width"]
    text_h = metrics["height"]
    baseline = metrics["baseline"]
    x, y = origin
    h, w = frame.shape[:2]
    top_left = (
        max(0, x - padding),
        max(0, y - text_h - baseline - padding),
    )
    bottom_right = (
        min(w - 1, x + text_w + padding),
        min(h - 1, y + padding),
    )
    cv2.rectangle(frame, top_left, bottom_right, box_colour, cv2.FILLED)
    _draw_text_with_metrics(
        frame,
        origin,
        colour,
        metrics,
        outline_colour=(0, 0, 0),
    )


def _measure_text(text: str, font_scale: float, thickness: int) -> Mapping[str, Any]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    has_degree = text.endswith("°")
    base_text = text[:-1] if has_degree else text
    measure_text = base_text if base_text else "0"
    text_size, baseline = cv2.getTextSize(measure_text, font, font_scale, thickness)
    raw_width, text_height = text_size
    if not base_text:
        raw_width = 0
    if has_degree:
        circle_radius = max(1, int(round(max(text_height, thickness * 2) * 0.25)))
        circle_spacing = max(1, int(round(circle_radius * 0.6)))
        total_width = raw_width + circle_spacing + circle_radius * 2
        total_height = max(text_height, circle_radius * 2)
    else:
        circle_radius = 0
        circle_spacing = 0
        total_width = raw_width
        total_height = text_height
    return {
        "text": text,
        "font": font,
        "font_scale": font_scale,
        "thickness": thickness,
        "base_text": base_text,
        "has_degree": has_degree,
        "raw_width": raw_width,
        "width": total_width,
        "height": total_height,
        "baseline": baseline,
        "circle_radius": circle_radius,
        "circle_spacing": circle_spacing,
    }


def _draw_text_with_metrics(
    frame: Any,
    origin: Tuple[int, int],
    colour: Tuple[int, int, int],
    metrics: Mapping[str, Any],
    *,
    outline_colour: Optional[Tuple[int, int, int]] = None,
) -> None:
    font = metrics["font"]
    font_scale = metrics["font_scale"]
    thickness = metrics["thickness"]
    base_text = metrics["base_text"]
    has_degree = metrics["has_degree"]
    raw_width = metrics["raw_width"]
    circle_radius = metrics["circle_radius"]
    circle_spacing = metrics["circle_spacing"]
    x, y = origin
    top_y = y - metrics["height"]
    circle_center = None
    if has_degree and circle_radius > 0:
        circle_center = (
            int(round(x + raw_width + circle_spacing + circle_radius)),
            int(round(top_y + circle_radius)),
        )
    if outline_colour is not None and (base_text or has_degree):
        if base_text:
            cv2.putText(
                frame,
                base_text,
                (int(x), int(y)),
                font,
                font_scale,
                outline_colour,
                thickness + 2,
                cv2.LINE_AA,
            )
        if circle_center is not None:
            outline_radius = circle_radius + 1
            cv2.circle(
                frame,
                circle_center,
                outline_radius,
                outline_colour,
                cv2.FILLED,
                lineType=cv2.LINE_AA,
            )
    if base_text:
        cv2.putText(
            frame,
            base_text,
            (int(x), int(y)),
            font,
            font_scale,
            colour,
            thickness,
            cv2.LINE_AA,
        )
    if circle_center is not None:
        cv2.circle(
            frame,
            circle_center,
            circle_radius,
            colour,
            cv2.FILLED,
            lineType=cv2.LINE_AA,
        )


def _safe_degrees(value: Optional[float]) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return math.degrees(value)


def _wrap_degrees_360(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    wrapped = value % 360.0
    return wrapped if wrapped >= 0.0 else wrapped + 360.0


def _draw_attitude_overlay(
    frame: Any,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    hfov_deg: float,
    vfov_deg: float,
) -> None:
    if not math.isfinite(hfov_deg) or not math.isfinite(vfov_deg):
        return
    if hfov_deg <= 0.0 or vfov_deg <= 0.0:
        return

    h, w = frame.shape[:2]
    top_band_height = max(36, min(64, h // 16 or 1))
    right_band_width = max(36, min(80, w // 16 or 1))
    top_band_height = min(top_band_height, max(1, h - 40))
    band_margin = 6

    vert_top = top_band_height + band_margin
    vert_bottom = h - band_margin
    if vert_bottom <= vert_top:
        vert_top = band_margin
        vert_bottom = h - band_margin

    base_colour = (0, 255, 0)
    half_hfov = hfov_deg / 2.0
    half_vfov = vfov_deg / 2.0
    if half_hfov <= 0.0 or half_vfov <= 0.0:
        return

    usable_width = max(1, w - 2 * band_margin)
    center_x = band_margin + usable_width / 2.0
    base_y = band_margin + top_band_height
    long_len = max(12, int(top_band_height * 0.65))
    short_len = max(8, int(top_band_height * 0.45))
    step = 5.0

    min_az = azimuth_deg - half_hfov
    max_az = azimuth_deg + half_hfov
    start_idx = int(math.floor(min_az / step))
    end_idx = int(math.ceil(max_az / step))

    major_tick_thickness = 3
    minor_tick_thickness = 2
    center_tick_thickness = major_tick_thickness + 1

    def _round_deg_signed(value: float) -> int:
        return int(round(value)) if math.isfinite(value) else 0

    def _round_deg_unsigned(value: float) -> int:
        if not math.isfinite(value):
            return 0
        rounded = int(round(_wrap_degrees_360(value)))
        if rounded == 360:
            return 0
        return rounded

    for idx in range(start_idx, end_idx + 1):
        tick_value = idx * step
        if not (min_az - 1e-3 <= tick_value <= max_az + 1e-3):
            continue
        offset = tick_value - azimuth_deg
        norm = (offset / half_hfov) if half_hfov else 0.0
        x = int(round(center_x + norm * (usable_width / 2.0)))
        x = max(0, min(w - 1, x))
        colour = base_colour
        is_major = abs(idx) % 2 == 0
        length = long_len if is_major else short_len
        is_center = abs(offset) < 1e-6
        if is_center:
            thickness = center_tick_thickness
        else:
            thickness = major_tick_thickness if is_major else minor_tick_thickness
        start_y = max(0, base_y - length)
        cv2.line(frame, (x, base_y), (x, start_y), colour, thickness)
        if is_major and (min_az - 1e-3) <= tick_value <= (max_az + 1e-3):
            label_value = _round_deg_unsigned(tick_value)
            label = f"{label_value:d}°"
            metrics = _measure_text(label, 0.45, 1)
            text_w = metrics["width"]
            text_h = metrics["height"]
            baseline = metrics["baseline"]
            text_x = int(round(x - text_w / 2))
            text_y = int(
                round(max(text_h + 2, start_y - baseline - 2))
            )
            text_x = max(2, min(w - text_w - 2, text_x))
            text_y = max(text_h + 2, min(base_y - 2, text_y))
            _draw_text_with_metrics(
                frame,
                (text_x, text_y),
                base_colour,
                metrics,
            )

    center_tick_len = long_len + 4
    center_start_y = max(0, base_y - center_tick_len)
    cv2.line(
        frame,
        (int(round(center_x)), base_y),
        (int(round(center_x)), center_start_y),
        base_colour,
        center_tick_thickness,
    )

    az_text = f"{_round_deg_unsigned(azimuth_deg):d}°"
    az_metrics = _measure_text(az_text, 0.55, 1)
    text_w = az_metrics["width"]
    text_h = az_metrics["height"]
    baseline = az_metrics["baseline"]
    text_x = int(round(center_x - text_w / 2))
    text_y = max(text_h + 4, center_start_y - baseline - 2)
    text_x = max(2, min(w - text_w - 2, text_x))
    _draw_text_box(
        frame,
        az_text,
        (text_x, text_y),
        base_colour,
        font_scale=0.55,
        thickness=1,
    )

    usable_height = max(1, vert_bottom - vert_top)
    center_y = vert_top + usable_height / 2.0
    long_len_v = max(14, int(right_band_width * 0.65))
    short_len_v = max(8, int(right_band_width * 0.45))

    span_limit = min(half_vfov, 20.0)
    min_el = elevation_deg - span_limit
    max_el = elevation_deg + span_limit
    start_idx_v = int(math.floor(min_el / step))
    end_idx_v = int(math.ceil(max_el / step))
    tick_right = w - band_margin - 2

    center_label_metrics: Optional[Tuple[int, int, int, int]] = None

    for idx in range(start_idx_v, end_idx_v + 1):
        tick_value = idx * step
        if not (min_el - 1e-3 <= tick_value <= max_el + 1e-3):
            continue
        offset = tick_value - elevation_deg
        norm = (offset / half_vfov) if half_vfov else 0.0
        y = int(round(center_y - norm * (usable_height / 2.0)))
        y = max(vert_top, min(vert_bottom - 1, y))
        colour = base_colour
        is_major = abs(idx) % 2 == 0
        length = long_len_v if is_major else short_len_v
        start_x = max(0, tick_right - length)
        is_center = abs(offset) < 1e-6
        if is_center:
            thickness = center_tick_thickness
        else:
            thickness = major_tick_thickness if is_major else minor_tick_thickness
        cv2.line(frame, (tick_right, y), (start_x, y), colour, thickness)
        if is_major and (min_el - 1e-3) <= tick_value <= (max_el + 1e-3):
            label = f"{int(round(tick_value))}°"
            metrics = _measure_text(label, 0.45, 1)
            text_w = metrics["width"]
            text_h = metrics["height"]
            baseline = metrics["baseline"]
            text_x = max(2, start_x - text_w - 4)
            text_y = min(
                vert_bottom - 4,
                max(vert_top + text_h, y + text_h // 2),
            )
            if is_center:
                center_label_metrics = (
                    text_x + text_w,
                    text_y,
                    text_h,
                    baseline,
                )
                continue
            _draw_text_with_metrics(
                frame,
                (int(text_x), int(text_y)),
                base_colour,
                metrics,
            )

    center_tick_len_v = long_len_v + 4
    center_start_x = max(0, tick_right - center_tick_len_v)
    cv2.line(
        frame,
        (tick_right, int(round(center_y))),
        (center_start_x, int(round(center_y))),
        base_colour,
        center_tick_thickness,
    )

    el_text = f"{_round_deg_signed(elevation_deg):+d}°"
    el_metrics = _measure_text(el_text, 0.55, 1)
    text_w = el_metrics["width"]
    text_h = el_metrics["height"]
    baseline = el_metrics["baseline"]
    if center_label_metrics is None:
        text_x = max(2, tick_right - text_w)
        text_y = int(round(center_y))
        text_y = min(vert_bottom - baseline - 2, max(vert_top + text_h, text_y))
        center_label_origin = (text_x, text_y)
    else:
        label_right, label_y, label_h, label_baseline = center_label_metrics
        text_x = int(round(label_right - text_w))
        text_x = max(2, min(w - text_w - 2, text_x))
        label_top = label_y - label_h
        label_bottom = label_y + label_baseline
        label_mid = (label_top + label_bottom) / 2.0
        target_y = label_mid + (text_h - baseline) / 2.0
        text_y = int(round(target_y))
        text_y = min(vert_bottom - baseline - 2, max(vert_top + text_h, text_y))
        center_label_origin = (text_x, text_y)
    _draw_text_box(
        frame,
        el_text,
        center_label_origin,
        base_colour,
        font_scale=0.55,
        thickness=1,
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


class GstVideoWriter:
    def __init__(self, pipeline: str, *, fps: int) -> None:
        self._pipeline = Gst.parse_launch(pipeline)
        self._appsrc = self._pipeline.get_by_name("src")
        if self._appsrc is None:
            raise RuntimeError("GStreamer pipeline missing appsrc named 'src'")
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._frame_count = 0
        self._frame_duration_ns = int(1e9 / fps) if fps > 0 else None
        self._pipeline.set_state(Gst.State.PLAYING)
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame) -> bool:
        if not self._opened:
            return False
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        if self._frame_duration_ns is not None:
            buf.duration = self._frame_duration_ns
            buf.pts = self._frame_count * self._frame_duration_ns
            buf.dts = buf.pts
        self._frame_count += 1
        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            self._opened = False
            return False
        return True

    def end_of_stream(self) -> None:
        if self._appsrc is not None:
            self._appsrc.end_of_stream()

    def release(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self._opened = False


def make_return_writer(pc_ip, port, w, h, fps=30, bitrate=4000, vbv_size=None):
    br_bps = bitrate * 1000
    if vbv_size is None:
        vbv_size = int((br_bps / fps) * 2)
    pipeline = (
        # App source (CPU memory, BGR from OpenCV)
        f"appsrc name=src is-live=true block=false do-timestamp=true format=time "
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
    vw = GstVideoWriter(pipeline, fps=fps)
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


def _prepare_config_sync_endpoint(cfg: Mapping[str, Any]) -> Tuple[str, str]:
    endpoint = resolve_config_sync_endpoint(cfg)
    port = _parse_tcp_port(endpoint, "config_sync")
    bind_endpoint = f"tcp://0.0.0.0:{port}"
    return endpoint, bind_endpoint


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
    Gst.init(None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    ap.add_argument(
        "--config-extra",
        default="configs/dev_extra.yaml",
        help="Optional second YAML config merged over --config.",
    )
    ap.add_argument(
        "--config-sync-timeout",
        type=float,
        default=None,
        help=(
            "Maximum seconds to wait for PC config sync when source=file:. "
            "Default 5s; use 0 to continue immediately."
        ),
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    extra_path = Path(args.config_extra) if args.config_extra else None
    config_paths = [config_path] + ([extra_path] if extra_path else [])

    initial_snapshots = {path: read_snapshot(path) for path in config_paths}
    cfg = merge_config_maps(
        *(
            parse_config_text(snapshot.text, str(path))
            for path, snapshot in initial_snapshots.items()
        )
    )

    _, bind_endpoint = _prepare_config_sync_endpoint(cfg)
    initial_source = str(cfg.get("source", "") or "")
    initial_file_source = initial_source.strip().startswith("file:")

    if args.config_sync_timeout is not None and args.config_sync_timeout < 0:
        raise SystemExit("--config-sync-timeout must be >= 0")

    if initial_file_source:
        wait_timeout: Optional[float] = (
            args.config_sync_timeout
            if args.config_sync_timeout is not None
            else _FILE_SOURCE_SYNC_TIMEOUT_S
        )
    else:
        wait_timeout = None

    config_sync_logs: List[Tuple[int, str]] = []
    final_texts: Dict[Path, str] = {}
    for path in config_paths:
        snapshot = initial_snapshots[path]
        try:
            final_text, final_meta = sync_as_server(
                path,
                bind_endpoint,
                config_id=path.name,
                wait_timeout=wait_timeout,
            )
        except ConfigSyncError as exc:
            if wait_timeout is not None:
                final_text = snapshot.text
                final_meta = snapshot.metadata
                config_sync_logs.append(
                    (
                        logging.WARNING,
                        (
                            f"Config sync timed out for {path} after {wait_timeout:.1f}s; "
                            f"continuing with local configuration ({exc})"
                        ),
                    )
                )
            else:
                raise SystemExit(f"config synchronization failed: {exc}") from exc
        else:
            if final_meta.sha256 != snapshot.metadata.sha256:
                config_sync_logs.append(
                    (
                        logging.INFO,
                        "Config sync: accepted client configuration "
                        f"for {path} (sha256={final_meta.sha256})",
                    )
                )
            else:
                config_sync_logs.append(
                    (
                        logging.INFO,
                        "Config sync: using local configuration "
                        f"for {path} (sha256={final_meta.sha256})",
                    )
                )
        final_texts[path] = final_text

    cfg = merge_config_maps(
        *(
            parse_config_text(final_texts[path], str(path))
            for path in config_paths
        )
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _RANGING_LOG.setLevel(logging.INFO)

    for level, message in config_sync_logs:
        logging.log(level, message)

    logging_cfg = cfg.get("logging", {}) or {}
    cli_json_logs = bool(logging_cfg.get("cli_json", False))

    source_spec = str(cfg.get("source", "") or "")
    source_clean = source_spec.strip()
    source_lower = source_clean.lower()
    file_source = source_lower.startswith("file:")
    csi_source = (
        source_lower in {"csi", "webcam"}
        or source_lower.startswith("csi:")
        or source_lower.startswith("webcam:")
    )
    file_source_path: Optional[Path] = None
    if file_source:
        logging.info(
            "source is file; disabling control publisher and recording return video"
        )
        path_spec = source_clean.split(":", 1)[1] if ":" in source_clean else ""
        path_spec = path_spec.strip()
        if not path_spec:
            raise SystemExit("file source requires a path, e.g. file:/path/to/video")
        file_source_path = Path(path_spec).expanduser()

    video_cfg, active_profile = resolve_active_video_profile(cfg)

    def _coerce_dimension(name: str, raw: Any) -> Optional[int]:
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"video.{name} must be an integer, got {raw!r}") from exc
        if value <= 0:
            raise SystemExit(f"video.{name} must be positive, got {value}")
        return value

    def _coerce_fps(raw: Any) -> Optional[float]:
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"video.fps must be numeric, got {raw!r}") from exc
        if value <= 0.0:
            raise SystemExit(f"video.fps must be positive, got {value}")
        return value

    video_w = _coerce_dimension("width", video_cfg.get("width"))
    video_h = _coerce_dimension("height", video_cfg.get("height"))
    cfg_fps = _coerce_fps(video_cfg.get("fps"))
    try:
        bitrate_kbps = int(video_cfg.get("bitrate_kbps", 4000) or 4000)
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.bitrate_kbps must be an integer") from exc

    source_fps = cfg_fps if cfg_fps is not None else 0.0

    recv: Optional[Any] = None
    if file_source:
        if file_source_path is None:
            raise SystemExit("file source requires a valid path")
        try:
            file_reader = FileVideoReader(file_source_path)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if video_w is None:
            video_w = file_reader.frame_width
        if video_h is None:
            video_h = file_reader.frame_height
        if video_w is None or video_h is None or video_w <= 0 or video_h <= 0:
            raise SystemExit(
                "could not determine video dimensions from config or source"
            )
        file_reader.set_target_size((video_w, video_h))
        if not source_fps or source_fps <= 0.0:
            source_fps = file_reader.fps
        if not source_fps or source_fps <= 0.0:
            source_fps = 30.0
            logging.info(
                "video.fps not provided; defaulting to 30 FPS for file playback"
            )
        recv = file_reader
    else:
        if video_w is None or video_h is None:
            raise SystemExit("config missing video.width/video.height")
        if source_fps <= 0.0:
            raise SystemExit("config missing positive video.fps")

    video_w = int(video_w)
    video_h = int(video_h)
    source_fps = float(source_fps)

    yolo_cfg = cfg.get("yolo")
    if not isinstance(yolo_cfg, Mapping):
        raise SystemExit("config missing 'yolo' section")

    engine_size_raw = yolo_cfg.get("engine_size")
    if not isinstance(engine_size_raw, str) or not engine_size_raw.strip():
        raise SystemExit("config missing yolo.engine_size (expected 'nano' or 'small')")
    engine_size = engine_size_raw.strip().lower()
    if engine_size not in _YOLO_ENGINE_SIZES:
        supported = ", ".join(sorted(_YOLO_ENGINE_SIZES))
        raise SystemExit(
            f"unsupported yolo.engine_size {engine_size_raw!r}; expected one of: {supported}"
        )

    suffix = _YOLO_RES_SUFFIX_BY_PROFILE.get(active_profile)
    if suffix is None:
        suffix = _YOLO_RES_SUFFIX_BY_WIDTH.get(video_w)
    if suffix is None:
        raise SystemExit(
            f"no YOLO engine available for video width {video_w} (profile {active_profile!r})"
        )

    engine_path = _YOLO_ENGINE_DIR / f"{engine_size}_{suffix}.engine"
    if not engine_path.exists():
        raise SystemExit(f"YOLO engine not found at {engine_path}")

    try:
        derived_input_size = int(suffix)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"invalid YOLO engine suffix {suffix!r} for engine {engine_path.name}"
        ) from exc

    configured_input_size = yolo_cfg.get("input_size")
    yolo_input_size = derived_input_size
    if configured_input_size is None:
        logging.info(
            "yolo.input_size not provided; using derived size %d for %s",
            yolo_input_size,
            engine_path.name,
        )
    else:
        try:
            configured_input_size = int(configured_input_size)
        except (TypeError, ValueError) as exc:
            raise SystemExit("yolo.input_size must be an integer") from exc
        if configured_input_size != derived_input_size:
            logging.warning(
                "yolo.input_size (%d) does not match engine resolution %d; overriding",
                configured_input_size,
                derived_input_size,
            )
        else:
            yolo_input_size = configured_input_size

    try:
        control_cfg = ControlConfig.from_raw_config(cfg, (video_w, video_h))
    except ControlConfigError as exc:
        raise SystemExit(f"invalid control configuration: {exc}") from exc

    try:
        laser_cfg = LaserMountConfig.from_raw_config(cfg)
    except LaserConfigError as exc:
        raise SystemExit(f"invalid laser configuration: {exc}") from exc

    try:
        camera_intrinsics = CameraIntrinsics.from_raw_config(cfg, (video_w, video_h))
    except CameraIntrinsicsConfigError as exc:
        raise SystemExit(f"invalid camera configuration: {exc}") from exc

    try:
        ranging_cfg = KnownSizeRangingConfig.from_raw_config(cfg)
    except KnownSizeRangingConfigError as exc:
        raise SystemExit(f"invalid known-size ranging configuration: {exc}") from exc
    class_labels = _parse_class_labels(cfg)

    net_cfg = cfg.get("net") or {}

    stop_event = install_signal_handlers()

    if csi_source:
        device_path: Optional[str] = None
        pipeline_override: Optional[str] = None
        if ":" in source_clean:
            csi_arg = source_clean.split(":", 1)[1].strip()
            if csi_arg:
                if csi_arg.startswith("/"):
                    device_path = csi_arg
                elif csi_arg.isdigit():
                    device_path = f"/dev/video{int(csi_arg)}"
                else:
                    pipeline_override = csi_arg
        recv = CsiVideoReader(
            device=device_path,
            width=video_w,
            height=video_h,
            fps=source_fps,
            pipeline=pipeline_override,
            stop_event=stop_event,
        )
    elif not file_source:
        try:
            port_value = net_cfg.get("rtp_port")
        except AttributeError as exc:
            raise SystemExit("config missing net section") from exc
        if port_value is None:
            raise SystemExit("config missing net.rtp_port endpoint")
        try:
            port = int(port_value)
        except (TypeError, ValueError) as exc:
            raise SystemExit("net.rtp_port must be an integer") from exc
        recv = GRecv(port, video_w, video_h, stop_event=stop_event)

    if recv is None:
        raise SystemExit("failed to initialize video source")

    yolo = YoloEngine(
        engine_path=str(engine_path),
        conf_thres=yolo_cfg['conf_thres'],
        iou_thres=yolo_cfg['iou_thres'],
        input_size=yolo_input_size,
        preprocess_mode=yolo_cfg.get('preprocess_mode', 'bilinear')
    )

    logging.info(
        "processing video at %dx%d @ %.2f FPS", video_w, video_h, source_fps
    )

    # --- ZMQ (local ctx)
    ctx = zmq.Context()
    pub: Optional[zmq.Socket] = None
    ep = net_cfg.get('zmq_results') if isinstance(net_cfg, Mapping) else None
    if ep:
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 1)
        pub.setsockopt(zmq.LINGER, 0)
        results_port = _parse_tcp_port(ep, "zmq_results")
        pub.bind(f"tcp://0.0.0.0:{results_port}")
    elif not file_source:
        raise SystemExit("config missing net.zmq_results endpoint")

    ctrl_pub: Optional[zmq.Socket] = None
    ctrl_ep = net_cfg.get('zmq_control') if isinstance(net_cfg, Mapping) else None
    if not file_source and ctrl_ep:
        ctrl_pub = ctx.socket(zmq.PUB)
        ctrl_pub.setsockopt(zmq.SNDHWM, 1)
        ctrl_pub.setsockopt(zmq.LINGER, 0)
        ctrl_port = _parse_tcp_port(ctrl_ep, "zmq_control")
        ctrl_pub.bind(f"tcp://0.0.0.0:{ctrl_port}")
    elif not file_source:
        raise SystemExit("config missing net.zmq_control endpoint")

    pull: Optional[zmq.Socket] = None
    header_ep = net_cfg.get('header_push') if isinstance(net_cfg, Mapping) else None
    if header_ep:
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.RCVHWM, 10)
        pull.setsockopt(zmq.LINGER, 0)
        header_port = _parse_tcp_port(header_ep, "header_push")
        pull.bind(f"tcp://0.0.0.0:{header_port}")
        pull.RCVTIMEO = 0  # non-blocking
    elif not file_source:
        raise SystemExit("config missing net.header_push endpoint")

    writer_fps = source_fps if source_fps > 0.0 else (cfg_fps or 30.0)
    profile_fps = cfg_fps if cfg_fps and cfg_fps > 0.0 else writer_fps
    if active_profile:
        logging.info(
            "video profile %s resolved to %dx%d @ %.2f FPS, %d kbps",
            active_profile,
            video_w,
            video_h,
            profile_fps,
            bitrate_kbps,
        )
    if file_source:
        return_file_path = _derive_return_file_path(source_spec)
        ret_vw = make_file_return_writer(
            return_file_path,
            video_w,
            video_h,
            fps=writer_fps,
        )
    else:
        pc_ip = net_cfg.get('pc_ip') if isinstance(net_cfg, Mapping) else None
        if not pc_ip:
            raise SystemExit("config missing net.pc_ip")
        return_port_value = net_cfg.get('rtp_return_port') if isinstance(net_cfg, Mapping) else None
        if return_port_value is None:
            raise SystemExit("config missing net.rtp_return_port")
        try:
            return_port = int(return_port_value)
        except (TypeError, ValueError) as exc:
            raise SystemExit("net.rtp_return_port must be an integer") from exc
        ret_vw = make_return_writer(
            pc_ip,
            return_port,
            video_w,
            video_h,
            fps=max(1, int(round(writer_fps))),
            bitrate=bitrate_kbps,
        )

    file_frame_idx = -1
    file_frame_interval_ms = (
        1000.0 / writer_fps if file_source and writer_fps > 0.0 else None
    )
    file_src_start_ns = 0

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
            if not ok or frame is None:
                if getattr(recv, "eos", False):
                    stop_event.set()
                    break
                if stop_event.is_set():
                    break
                if file_source:
                    logging.info("end of video file reached")
                    break
                if controller is not None:
                    controller.tick(time.monotonic())
                continue

            frame_h, frame_w = frame.shape[:2]

            if file_source:
                file_frame_idx += 1
                if file_frame_interval_ms is not None:
                    src_ts_ms = int(round(file_frame_idx * file_frame_interval_ms))
                else:
                    if file_src_start_ns == 0:
                        file_src_start_ns = time.monotonic_ns()
                    src_ts_ms = int((time.monotonic_ns() - file_src_start_ns) / 1e6)
                latest_header = {"frame_id": file_frame_idx, "src_ts_ms": src_ts_ms}

            # headers (non-blocking drain)
            if pull is not None:
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
                    iter_ranging_candidates(
                        boxes, (frame_w, frame_h), class_labels, ranging_cfg
                    )
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
                img_w=frame_w,
                img_h=frame_h,
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

            if pub is not None:
                try:
                    pub.send_string(detection_msg_to_json(msg), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
            if controller is not None:
                controller.tick(time.monotonic())

            # draw + return video (draw directly on the frame once inference is done)
            for b in boxes:
                x1 = int(b.x * frame_w)
                y1 = int(b.y * frame_h)
                x2 = int((b.x + b.w) * frame_w)
                y2 = int((b.y + b.h) * frame_h)
                colour = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                if b.cls in ("person", "1"):
                    cv2.line(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.line(frame, (x1, y2), (x2, y1), colour, 2)

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
                label_parts: List[str] = []
                cls_label_raw = getattr(b, "cls", "")
                cls_label = str(cls_label_raw).strip()
                if cls_label:
                    label_parts.append(cls_label)
                conf_val = getattr(b, "conf", None)
                if isinstance(conf_val, (int, float)) and math.isfinite(float(conf_val)):
                    label_parts.append(f"{float(conf_val):.2f}")
                distance_val = getattr(b, "distance_m", None)
                if isinstance(distance_val, (int, float)) and math.isfinite(float(distance_val)):
                    label_parts.append(f"{float(distance_val):.2f} m")
                label_text = " | ".join(label_parts) if label_parts else None
                if label_text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    text_size, baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    text_w, text_h = text_size
                    # Try to place the label above the box; fall back to below if needed.
                    text_x = max(0, min(x1, frame_w - text_w - 4))
                    text_y = y1 - 8
                    if text_y - text_h - baseline < 0:
                        text_y = min(frame_h - 4, y2 + text_h + 8)
                    box_pt1 = (text_x - 2, text_y - text_h - baseline - 2)
                    box_pt2 = (text_x + text_w + 2, text_y + 2)
                    cv2.rectangle(frame, box_pt1, box_pt2, (0, 0, 0), thickness=cv2.FILLED)
                    cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, colour, thickness, cv2.LINE_AA)

            _draw_lead_overlay(frame, msg)
            _draw_predictive_overlay(frame, msg)

            if (
                not file_source
                and camera_intrinsics.fov_deg
                and len(camera_intrinsics.fov_deg) == 2
            ):
                hfov, vfov = camera_intrinsics.fov_deg
                azimuth_deg = _safe_degrees(
                    latest_cam_state.pan if latest_cam_state is not None else None
                )
                elevation_deg = _safe_degrees(
                    latest_cam_state.tilt if latest_cam_state is not None else None
                )
                _draw_attitude_overlay(
                    frame,
                    azimuth_deg=azimuth_deg,
                    elevation_deg=elevation_deg,
                    hfov_deg=hfov,
                    vfov_deg=vfov,
                )

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
            if ret_vw:
                try:
                    ret_vw.end_of_stream()
                except Exception:
                    pass
                ret_vw.release()
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
