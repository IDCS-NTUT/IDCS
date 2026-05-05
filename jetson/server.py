import argparse, json, logging, math, os, time, zmq
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from common.camera import CameraIntrinsics, CameraIntrinsicsConfigError
from common.control import (
    ControlConfig,
    ControlConfigError,
    LaserConfigError,
    LaserMountConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.config_sync import (
    ConfigSyncError,
    expand_config_paths,
    merge_config_maps,
    parse_config_text,
    read_snapshot,
    resolve_active_return_video_profile,
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
from common.schemas import (
    CamState,
    ControlCmd,
    DetectionMsg,
    ManualControlState,
    detection_msg_to_json,
    manual_control_state_from_json,
)
from pc.renderers._geometry import clip_segment_to_rect
from jetson.receiver import CsiVideoReader, FileVideoReader, GRecv
from jetson.controller import ControlLoop
from jetson.yolo_engine import YoloEngine
# Build a GStreamer encoder pipeline for return video
import threading
import cv2
import gi
import numpy as np
import concurrent.futures
from common.shutdown import install_signal_handlers
from jetson.multi_target_tracker import (
    BotSortSearchTracker,
    assign_track_ids_to_boxes,
    boxes_to_tracker_arrays,
)

gi.require_version("Gst", "1.0")
from gi.repository import Gst

_YOLO_ENGINE_DIR = Path(__file__).resolve().parents[1] / "assets" / "models" / "yolo"
_YOLO_ENGINE_SIZES = {"nano", "small", "medium"}
_YOLO_RES_SUFFIX_BY_PROFILE = {"720p": "1280", "1080p": "1920"}
_YOLO_RES_SUFFIX_BY_WIDTH = {1280: "1280", 1920: "1920"}


_RANGING_LOG = logging.getLogger("jetson.ranging")
_RANGING_LOG_PRECISION = 4

def _expected_header_origins(*, rpi_source: bool, file_source: bool) -> Tuple[str, ...]:
    if file_source:
        return ()
    if rpi_source:
        return ("rpi2",)
    return ("pc",)


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

    span_limit = half_vfov * 3
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


def _draw_control_authority_overlay(
    frame: Any,
    *,
    authority: str,
    reason: str,
    negotiation_enabled: bool,
    negotiation_mode: str,
    manual_state_age_s: Optional[float],
) -> None:
    h, _w = frame.shape[:2]

    authority_clean = str(authority or "auto").strip().lower()
    reason_clean = str(reason or "-").strip()
    mode_clean = str(negotiation_mode or "-").strip()

    if authority_clean == "manual":
        primary_colour = (0, 140, 255)
        box_colour = (20, 20, 80)
    else:
        primary_colour = (64, 224, 64)
        box_colour = (20, 60, 20)

    age_text = "n/a"
    if isinstance(manual_state_age_s, (int, float)) and math.isfinite(float(manual_state_age_s)):
        age_text = f"{float(manual_state_age_s):.2f}s"

    line = (
        f"ctrl:{authority_clean} | nego:{'on' if negotiation_enabled else 'off'}"
        f"({mode_clean}) | state_age:{age_text} | {reason_clean}"
    )
    _draw_text_box(
        frame,
        line,
        (12, max(24, h - 14)),
        primary_colour,
        font_scale=0.5,
        thickness=1,
        padding=4,
        box_colour=box_colour,
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
        f"vbv-size={vbv_size} "
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


def _box_center_px(box: Any, img_w: int, img_h: int) -> Tuple[float, float]:
    return (
        (float(box.x) + (float(box.w) / 2.0)) * float(img_w),
        (float(box.y) + (float(box.h) / 2.0)) * float(img_h),
    )


def _crop_rect_around_point(
    center_uv: Tuple[float, float],
    frame_w: int,
    frame_h: int,
    crop_w: int,
    crop_h: int,
) -> Tuple[int, int, int, int]:
    cx = float(center_uv[0])
    cy = float(center_uv[1])
    half_w = crop_w // 2
    half_h = crop_h // 2

    x1 = int(round(cx)) - half_w
    y1 = int(round(cy)) - half_h
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_w:
        shift = x2 - frame_w
        x1 -= shift
        x2 = frame_w
    if y2 > frame_h:
        shift = y2 - frame_h
        y1 -= shift
        y2 = frame_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)
    return x1, y1, x2, y2


def _project_boxes_from_crop(
    boxes: Sequence[Any],
    crop_rect: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> List[Any]:
    x1, y1, x2, y2 = crop_rect
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    projected: List[Any] = []
    for box in boxes:
        gx = (x1 + (float(box.x) * crop_w)) / float(frame_w)
        gy = (y1 + (float(box.y) * crop_h)) / float(frame_h)
        gw = (float(box.w) * crop_w) / float(frame_w)
        gh = (float(box.h) * crop_h) / float(frame_h)
        box.x = float(max(0.0, min(gx, 1.0)))
        box.y = float(max(0.0, min(gy, 1.0)))
        box.w = float(max(0.0, min(gw, 1.0)))
        box.h = float(max(0.0, min(gh, 1.0)))
        projected.append(box)
    return projected


def _target_uv_from_msg(msg: DetectionMsg) -> Optional[Tuple[float, float]]:
    target_idx = msg.target_idx
    if target_idx is None:
        return None
    if target_idx < 0 or target_idx >= len(msg.boxes):
        return None
    box = msg.boxes[target_idx]
    return (
        (float(box.x) + (float(box.w) / 2.0)) * float(msg.img_w),
        (float(box.y) + (float(box.h) / 2.0)) * float(msg.img_h),
    )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _camstate_warp_from_delta(
    *,
    prev_state: CamState,
    curr_state: CamState,
    fx_px: float,
    fy_px: float,
) -> Optional[np.ndarray]:
    """Estimate previous->current image affine warp from camera pan/tilt deltas."""

    try:
        delta_pan = _wrap_angle(float(curr_state.pan) - float(prev_state.pan))
        delta_tilt = _wrap_angle(float(curr_state.tilt) - float(prev_state.tilt))
    except (TypeError, ValueError):
        return None

    tx = -float(fx_px) * math.tan(delta_pan)
    ty = float(fy_px) * math.tan(delta_tilt)
    if not (math.isfinite(tx) and math.isfinite(ty)):
        return None

    return np.asarray(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
        ],
        dtype=np.float32,
    )


def _send_transition_cmd(
    pub: zmq.Socket,
    *,
    msg: DetectionMsg,
    target_uv: Tuple[float, float],
    control_cfg: ControlConfig,
    speed_rad_s: float,
) -> None:
    px_err = pixel_delta(
        float(target_uv[0]),
        float(target_uv[1]),
        control_cfg.cx_px,
        control_cfg.cy_px,
        control_cfg,
        apply_deadband=False,
    )
    ang_err = angular_error_from_pixel_delta(px_err, control_cfg, linearize=False)

    yaw_sign = 0.0 if abs(float(ang_err.yaw)) < 1e-6 else math.copysign(1.0, float(ang_err.yaw))
    pitch_sign = 0.0 if abs(float(ang_err.pitch)) < 1e-6 else math.copysign(1.0, float(ang_err.pitch))
    yaw_rate = yaw_sign * min(float(speed_rad_s), float(control_cfg.rate_limits.yaw))
    pitch_rate = pitch_sign * min(float(speed_rad_s), float(control_cfg.rate_limits.pitch))

    cmd = ControlCmd(
        frame_id=int(msg.frame_id),
        src_ts_ms=int(msg.src_ts_ms),
        cmd_ts_ms=int(time.monotonic_ns() / 1e6),
        target_ok=True,
        target_uv=(float(target_uv[0]), float(target_uv[1])),
        err_uv=(float(px_err.yaw), float(px_err.pitch)),
        err_rad=(float(ang_err.yaw), float(ang_err.pitch)),
        pan_rate_cmd=float(yaw_rate),
        tilt_rate_cmd=float(pitch_rate),
        controller_mode=str(control_cfg.controller),
    )
    try:
        pub.send_string(cmd.model_dump_json(), flags=zmq.NOBLOCK)
    except zmq.Again:
        logging.warning("transition_control_pub_backpressure")


def _publish_hold_control_cmd(
    pub: zmq.Socket,
    *,
    frame_id: int,
    src_ts_ms: int,
    controller_mode: str,
) -> None:
    cmd = ControlCmd(
        frame_id=int(frame_id),
        src_ts_ms=int(src_ts_ms),
        cmd_ts_ms=int(time.monotonic_ns() / 1e6),
        target_ok=False,
        target_uv=(0.0, 0.0),
        err_uv=(0.0, 0.0),
        err_rad=(0.0, 0.0),
        pan_rate_cmd=0.0,
        tilt_rate_cmd=0.0,
        controller_mode=controller_mode,
    )
    try:
        pub.send_string(cmd.model_dump_json(exclude_none=True), flags=zmq.NOBLOCK)
    except zmq.Again:
        logging.warning("hold_control_pub_backpressure")


def _publish_manual_passthrough_control_cmd(
    pub: zmq.Socket,
    *,
    frame_id: int,
    src_ts_ms: int,
    controller_mode: str,
    manual_state: ManualControlState,
    max_yaw_rate: float,
    max_pitch_rate: float,
) -> None:
    active_motion = bool(manual_state.active) and not bool(manual_state.emergency)
    yaw_cmd = float(manual_state.joystick_rate_cmd[0]) if active_motion else 0.0
    pitch_cmd = float(manual_state.joystick_rate_cmd[1]) if active_motion else 0.0
    yaw_cmd = max(-float(max_yaw_rate), min(float(max_yaw_rate), yaw_cmd))
    pitch_cmd = max(-float(max_pitch_rate), min(float(max_pitch_rate), pitch_cmd))

    cmd = ControlCmd(
        frame_id=int(frame_id),
        src_ts_ms=int(src_ts_ms),
        cmd_ts_ms=int(time.monotonic_ns() / 1e6),
        target_ok=active_motion,
        target_uv=(0.0, 0.0),
        err_uv=(0.0, 0.0),
        err_rad=(0.0, 0.0),
        pan_rate_cmd=float(yaw_cmd),
        tilt_rate_cmd=float(pitch_cmd),
        controller_mode=controller_mode,
    )
    try:
        pub.send_string(cmd.model_dump_json(exclude_none=True), flags=zmq.NOBLOCK)
    except zmq.Again:
        logging.warning("manual_control_pub_backpressure")


def _parse_engine_spec(
    yolo_cfg: Mapping[str, Any],
    key: str,
    *,
    default_size: Optional[str] = None,
    default_input_size: Optional[int] = None,
    required: bool = False,
) -> Optional[Tuple[str, int]]:
    section = yolo_cfg.get(key)
    if section is None:
        if required:
            raise SystemExit(f"config missing yolo.{key} section")
        if default_size is None or default_input_size is None:
            return None
        return default_size, default_input_size

    if not isinstance(section, Mapping):
        raise SystemExit(f"yolo.{key} must be a mapping when provided")

    size_raw = section.get("size", default_size)
    if not isinstance(size_raw, str) or not size_raw.strip():
        raise SystemExit(f"yolo.{key}.size must be a non-empty string")
    size = size_raw.strip().lower()
    if size not in _YOLO_ENGINE_SIZES:
        supported = ", ".join(sorted(_YOLO_ENGINE_SIZES))
        raise SystemExit(
            f"unsupported yolo.{key}.size {size_raw!r}; expected one of: {supported}"
        )

    input_raw = section.get("input_size", default_input_size)
    try:
        input_size = int(input_raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"yolo.{key}.input_size must be an integer") from exc
    if input_size <= 0:
        raise SystemExit(f"yolo.{key}.input_size must be > 0")

    return size, input_size


def _parse_dual_tracker_cfg(yolo_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    dual_cfg = yolo_cfg.get("dual_tracker")
    if dual_cfg is None:
        return {
            "enabled": False,
            "search": {},
            "track": {},
        }
    if not isinstance(dual_cfg, Mapping):
        raise SystemExit("yolo.dual_tracker must be a mapping when provided")

    enabled = bool(dual_cfg.get("enabled", False))

    search_raw = dual_cfg.get("search")
    track_raw = dual_cfg.get("track")
    has_nested = search_raw is not None or track_raw is not None

    if search_raw is None:
        search_raw = {}
    if track_raw is None:
        track_raw = {}
    if not isinstance(search_raw, Mapping):
        raise SystemExit("yolo.dual_tracker.search must be a mapping when provided")
    if not isinstance(track_raw, Mapping):
        raise SystemExit("yolo.dual_tracker.track must be a mapping when provided")

    legacy_keys = {
        "enter_track_hits",
        "track_takeover_hits",
        "exit_track_misses",
        "recover_timeout_ms",
        "heartbeat_interval_frames",
        "transition_speed_rad_s",
        "arrival_tolerance_px",
        "transition_timeout_ms",
    }
    if not has_nested and any(key in dual_cfg for key in legacy_keys):
        logging.warning(
            "yolo.dual_tracker flat keys are deprecated; migrate to dual_tracker.search and dual_tracker.track"
        )

    def _as_pos_int(raw: Any, key_path: str, default: int) -> int:
        value_raw = default if raw is None else raw
        try:
            value = int(value_raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{key_path} must be an integer") from exc
        if value <= 0:
            raise SystemExit(f"{key_path} must be > 0")
        return value

    def _as_nonneg_int(raw: Any, key_path: str, default: int) -> int:
        value_raw = default if raw is None else raw
        try:
            value = int(value_raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{key_path} must be an integer") from exc
        if value < 0:
            raise SystemExit(f"{key_path} must be >= 0")
        return value

    def _as_pos_float(raw: Any, key_path: str, default: float) -> float:
        value_raw = default if raw is None else raw
        try:
            value = float(value_raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{key_path} must be numeric") from exc
        if value <= 0.0:
            raise SystemExit(f"{key_path} must be > 0")
        return value

    def _as_unit_float(raw: Any, key_path: str, default: float) -> float:
        value_raw = default if raw is None else raw
        try:
            value = float(value_raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{key_path} must be numeric") from exc
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"{key_path} must be within [0, 1]")
        return value

    def _as_bool(raw: Any, key_path: str, default: bool) -> bool:
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise SystemExit(f"{key_path} must be boolean")

    def _pick(section: Mapping[str, Any], key: str, legacy_key: Optional[str], default: Any) -> Any:
        if key in section:
            return section.get(key)
        if legacy_key is not None and legacy_key in dual_cfg:
            return dual_cfg.get(legacy_key)
        return default

    search_tracker_raw = _pick(search_raw, "tracker", None, "botsort")
    if not isinstance(search_tracker_raw, str) or not search_tracker_raw.strip():
        raise SystemExit("yolo.dual_tracker.search.tracker must be a non-empty string")
    search_tracker = search_tracker_raw.strip().lower()
    if search_tracker != "botsort":
        raise SystemExit("yolo.dual_tracker.search.tracker currently supports only 'botsort'")

    botsort_raw = search_raw.get("botsort")
    if botsort_raw is None:
        botsort_raw = {}
    if not isinstance(botsort_raw, Mapping):
        raise SystemExit("yolo.dual_tracker.search.botsort must be a mapping when provided")

    search_cfg = {
        "tracker": search_tracker,
        "enter_track_hits": _as_pos_int(
            _pick(search_raw, "enter_track_hits", "enter_track_hits", 3),
            "yolo.dual_tracker.search.enter_track_hits",
            3,
        ),
        "heartbeat_interval_frames": _as_nonneg_int(
            _pick(search_raw, "heartbeat_interval_frames", "heartbeat_interval_frames", 12),
            "yolo.dual_tracker.search.heartbeat_interval_frames",
            12,
        ),
        "assign_iou_thresh": _as_unit_float(
            _pick(search_raw, "assign_iou_thresh", None, 0.2),
            "yolo.dual_tracker.search.assign_iou_thresh",
            0.2,
        ),
        "camstate_gmc_enabled": _as_bool(
            _pick(search_raw, "camstate_gmc_enabled", None, True),
            "yolo.dual_tracker.search.camstate_gmc_enabled",
            True,
        ),
        "camstate_timeout_ms": _as_pos_int(
            _pick(search_raw, "camstate_timeout_ms", None, 300),
            "yolo.dual_tracker.search.camstate_timeout_ms",
            300,
        ),
        "botsort": {
            "track_high_thresh": _as_unit_float(
                botsort_raw.get("track_high_thresh", 0.25),
                "yolo.dual_tracker.search.botsort.track_high_thresh",
                0.25,
            ),
            "track_low_thresh": _as_unit_float(
                botsort_raw.get("track_low_thresh", 0.1),
                "yolo.dual_tracker.search.botsort.track_low_thresh",
                0.1,
            ),
            "new_track_thresh": _as_unit_float(
                botsort_raw.get("new_track_thresh", 0.25),
                "yolo.dual_tracker.search.botsort.new_track_thresh",
                0.25,
            ),
            "track_buffer": _as_pos_int(
                botsort_raw.get("track_buffer", 30),
                "yolo.dual_tracker.search.botsort.track_buffer",
                30,
            ),
            "match_thresh": _as_unit_float(
                botsort_raw.get("match_thresh", 0.8),
                "yolo.dual_tracker.search.botsort.match_thresh",
                0.8,
            ),
            "fuse_score": _as_bool(
                botsort_raw.get("fuse_score", True),
                "yolo.dual_tracker.search.botsort.fuse_score",
                True,
            ),
            "gmc_method": (
                str(botsort_raw.get("gmc_method", "sparseOptFlow") or "sparseOptFlow").strip()
                or "sparseOptFlow"
            ),
            "proximity_thresh": _as_unit_float(
                botsort_raw.get("proximity_thresh", 0.5),
                "yolo.dual_tracker.search.botsort.proximity_thresh",
                0.5,
            ),
            "appearance_thresh": _as_unit_float(
                botsort_raw.get("appearance_thresh", 0.8),
                "yolo.dual_tracker.search.botsort.appearance_thresh",
                0.8,
            ),
            "with_reid": _as_bool(
                botsort_raw.get("with_reid", False),
                "yolo.dual_tracker.search.botsort.with_reid",
                False,
            ),
            "reid_model": str(botsort_raw.get("model", "auto") or "auto").strip() or "auto",
        },
    }

    track_cfg = {
        "takeover_hits": _as_pos_int(
            _pick(track_raw, "takeover_hits", "track_takeover_hits", 3),
            "yolo.dual_tracker.track.takeover_hits",
            3,
        ),
        "exit_misses": _as_pos_int(
            _pick(track_raw, "exit_misses", "exit_track_misses", 4),
            "yolo.dual_tracker.track.exit_misses",
            4,
        ),
        "recover_timeout_ms": _as_pos_int(
            _pick(track_raw, "recover_timeout_ms", "recover_timeout_ms", 450),
            "yolo.dual_tracker.track.recover_timeout_ms",
            450,
        ),
        "transition_speed_rad_s": _as_pos_float(
            _pick(track_raw, "transition_speed_rad_s", "transition_speed_rad_s", 1.0),
            "yolo.dual_tracker.track.transition_speed_rad_s",
            1.0,
        ),
        "arrival_tolerance_px": _as_pos_float(
            _pick(track_raw, "arrival_tolerance_px", "arrival_tolerance_px", 24.0),
            "yolo.dual_tracker.track.arrival_tolerance_px",
            24.0,
        ),
        "transition_timeout_ms": _as_pos_int(
            _pick(track_raw, "transition_timeout_ms", "transition_timeout_ms", 1200),
            "yolo.dual_tracker.track.transition_timeout_ms",
            1200,
        ),
    }

    return {
        "enabled": enabled,
        "search": search_cfg,
        "track": track_cfg,
    }


def main():
    Gst.init(None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/network.yaml")
    ap.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
        help="Comma-separated YAML configs merged over --config.",
    )
    ap.add_argument(
        "--config-sync-timeout",
        type=float,
        default=3.0,
        help=(
            "Maximum seconds to wait per config file for each sync peer. "
            "Use 0 to continue immediately when peers are unavailable."
        ),
    )
    ap.add_argument(
        "--config-sync-timeout-action",
        choices=("continue", "exit"),
        default="exit",
        help=(
            "Action when a required sync peer times out: "
            "continue with local config or exit immediately."
        ),
    )
    ap.add_argument(
        "--source-override",
        default=None,
        help=(
            "Override config source on Jetson only for this process "
            "(for example: sim, webcam, rpi, file:/path/to/video)."
        ),
    )
    ap.add_argument(
        "--required",
        default="",
        help=(
            "Comma-separated peer IDs to require during config sync startup. "
            "Overrides source-based defaults."
        ),
    )
    args = ap.parse_args()

    config_paths = expand_config_paths(args.config, args.config_extra)

    initial_snapshots = {path: read_snapshot(path) for path in config_paths}
    cfg = merge_config_maps(
        *(
            parse_config_text(snapshot.text, str(path))
            for path, snapshot in initial_snapshots.items()
        )
    )

    source_override = str(args.source_override).strip() if args.source_override is not None else ""
    source_override_active = bool(source_override)
    if source_override_active:
        cfg = dict(cfg)
        cfg["source"] = source_override

    _, bind_endpoint = _prepare_config_sync_endpoint(cfg)
    initial_source = str(cfg.get("source", "") or "")
    initial_source_lower = initial_source.strip().lower()
    initial_sim_source = initial_source_lower.startswith("sim")

    def _peer_list(raw: object) -> List[str]:
        if isinstance(raw, str):
            parts = raw.split(",")
        elif isinstance(raw, (list, tuple)):
            parts = raw
        else:
            return []
        peers: List[str] = []
        for peer in parts:
            peer_id = str(peer).strip()
            if peer_id and peer_id not in peers:
                peers.append(peer_id)
        return peers

    default_known_peers = ["pc", "rpi", "rpi2"]
    cli_required_peers = _peer_list(args.required)

    if cli_required_peers:
        required_sync_peers = list(cli_required_peers)
    else:
        required_sync_peers = ["pc"] if initial_sim_source else []

    optional_sync_peers = [
        peer for peer in default_known_peers if peer not in required_sync_peers
    ]

    optional_sync_peers = [peer for peer in optional_sync_peers if peer not in required_sync_peers]

    if args.config_sync_timeout is not None and args.config_sync_timeout < 0:
        raise SystemExit("--config-sync-timeout must be >= 0")

    wait_timeout: Optional[float] = args.config_sync_timeout
    timeout_action = "continue" if initial_sim_source else args.config_sync_timeout_action
    sim_peer_timeouts = {"rpi": 5.0, "pc": 3.0} if initial_sim_source else {}

    config_sync_logs: List[Tuple[int, str]] = []
    if initial_sim_source:
        config_sync_logs.append(
            (
                logging.INFO,
                "Config sync: source=sim requires peers " + ", ".join(required_sync_peers),
            )
        )
    else:
        config_sync_logs.append(
            (
                logging.INFO,
                "Config sync: source!=sim requires peers " + ", ".join(required_sync_peers),
            )
        )
    if required_sync_peers:
        config_sync_logs.append(
            (
                logging.INFO,
                "Config sync: timeout policy for required peers is "
                + timeout_action,
            )
        )
    if initial_sim_source:
        config_sync_logs.append(
            (
                logging.INFO,
                "Config sync: source=sim peer timeout policy rpi=5.0s, pc=3.0s",
            )
        )
    if optional_sync_peers:
        config_sync_logs.append(
            (
                logging.INFO,
                "Config sync: optional peers " + ", ".join(optional_sync_peers),
            )
        )
    if cli_required_peers:
        config_sync_logs.append(
            (
                logging.INFO,
                "Config sync: required peer override active (--required) => "
                + ", ".join(required_sync_peers),
            )
        )
    else:
        config_sync_logs.append(
            (
                logging.INFO,
                "Config sync: default policy active (sim=>require pc; otherwise all optional)",
            )
        )
    if source_override_active:
        config_sync_logs.append(
            (
                logging.INFO,
                f"Config sync: source override active ({initial_source})",
            )
        )

    startup_state = {
        "effective_source": initial_source,
        "source_override_active": source_override_active,
        "required_sync_peers": list(required_sync_peers),
    }
    final_texts: Dict[Path, str] = {}
    final_texts.update({path: snapshot.text for path, snapshot in initial_snapshots.items()})
    successful_sync_peers: set[str] = set()

    if required_sync_peers:
        restart_required_on_fail = True
        sync_round = 0
        while True:
            sync_round += 1
            if sync_round > 1:
                config_sync_logs.append(
                    (
                        logging.WARNING,
                        f"Config sync: restarting required sync round {sync_round}",
                    )
                )

            required_failed = False
            successful_sync_peers.difference_update(required_sync_peers)

            for peer_id in required_sync_peers:
                peer_wait_timeout = sim_peer_timeouts.get(peer_id, 3.0) if initial_sim_source else wait_timeout
                config_sync_logs.append(
                    (
                        logging.INFO,
                        f"Config sync: waiting for peer {peer_id} to sync all config files (timeout={peer_wait_timeout})",
                    )
                )
                peer_failed = False
                for path in config_paths:
                    snapshot = read_snapshot(path)
                    if peer_failed:
                        final_texts[path] = snapshot.text
                        continue
                    try:
                        final_text, final_meta = sync_as_server(
                            path,
                            bind_endpoint,
                            config_id=path.name,
                            required_peer_ids=[peer_id],
                            enforce_peer_match=True,
                            wait_timeout=peer_wait_timeout,
                            server_state=startup_state,
                        )
                    except ConfigSyncError as exc:
                        if peer_wait_timeout is not None:
                            final_text = snapshot.text
                            final_meta = snapshot.metadata
                            peer_failed = True
                            config_sync_logs.append(
                                (
                                    logging.WARNING,
                                    (
                                        f"Config sync timed out for {path} after {peer_wait_timeout:.1f}s "
                                        f"while waiting for required peer {peer_id} ({exc})"
                                    ),
                                )
                            )
                            if timeout_action == "exit" and not restart_required_on_fail:
                                raise SystemExit(
                                    f"config synchronization failed: required peer {peer_id} unavailable"
                                ) from exc
                            if restart_required_on_fail:
                                config_sync_logs.append(
                                    (
                                        logging.WARNING,
                                        "Config sync: required peer sync failed; restarting sync server round",
                                    )
                                )
                                required_failed = True
                                break
                            config_sync_logs.append(
                                (
                                    logging.WARNING,
                                    f"Config sync: continuing without required peer {peer_id}",
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
                                    f"for {path} from peer {peer_id} (sha256={final_meta.sha256})",
                                )
                            )
                        else:
                            config_sync_logs.append(
                                (
                                    logging.INFO,
                                    "Config sync: using local configuration "
                                    f"for {path} after peer {peer_id} check (sha256={final_meta.sha256})",
                                )
                            )
                    final_texts[path] = final_text
                if required_failed:
                    break
                if not peer_failed:
                    successful_sync_peers.add(peer_id)

            if required_failed:
                time.sleep(0.2)
                continue
            break

    if optional_sync_peers:
        for peer_id in optional_sync_peers:
            config_sync_logs.append(
                (
                    logging.INFO,
                    f"Config sync: waiting for optional peer {peer_id} to sync all config files",
                )
            )
            peer_failed = False
            for path in config_paths:
                snapshot = read_snapshot(path)
                if peer_failed:
                    final_texts[path] = snapshot.text
                    continue
                try:
                    final_text, final_meta = sync_as_server(
                        path,
                        bind_endpoint,
                        config_id=path.name,
                        required_peer_ids=[peer_id],
                        enforce_peer_match=True,
                        wait_timeout=wait_timeout,
                        server_state=startup_state,
                    )
                except ConfigSyncError as exc:
                    if wait_timeout is not None:
                        final_text = snapshot.text
                        final_meta = snapshot.metadata
                        peer_failed = True
                        config_sync_logs.append(
                            (
                                logging.INFO,
                                (
                                    f"Config sync timed out for optional peer {peer_id} on {path} "
                                    f"after {wait_timeout:.1f}s; continuing ({exc})"
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
                                f"for {path} from optional peer {peer_id} (sha256={final_meta.sha256})",
                            )
                        )
                    else:
                        config_sync_logs.append(
                            (
                                logging.INFO,
                                "Config sync: using local configuration "
                                f"for {path} after optional peer {peer_id} check (sha256={final_meta.sha256})",
                            )
                        )
                final_texts[path] = final_text
            if not peer_failed:
                successful_sync_peers.add(peer_id)

    if not required_sync_peers and not optional_sync_peers:
        for path in config_paths:
            snapshot = initial_snapshots[path]
            try:
                final_text, final_meta = sync_as_server(
                    path,
                    bind_endpoint,
                    config_id=path.name,
                    required_peer_ids=None,
                    enforce_peer_match=False,
                    wait_timeout=wait_timeout,
                    server_state=startup_state,
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
    if source_override_active:
        cfg = dict(cfg)
        cfg["source"] = initial_source

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _RANGING_LOG.setLevel(logging.INFO)

    for level, message in config_sync_logs:
        logging.log(level, message)

    logging_cfg = cfg.get("logging", {}) or {}
    cli_json_logs = bool(logging_cfg.get("cli_json", False))

    source_spec = str(cfg.get("source", "") or "")
    source_clean = source_spec.strip()
    source_lower = source_clean.lower()
    sim_source = source_lower.startswith("sim")
    file_source = source_lower.startswith("file:")
    csi_source = (
        source_lower in {"csi", "webcam"}
        or source_lower.startswith("csi:")
        or source_lower.startswith("webcam:")
    )
    rpi_source = source_lower.startswith("rpi") or source_lower.startswith("rpi:")
    if rpi_source:
        # `rpi` is an alias indicating the camera will be streamed from a Pi
        # over the network to this Jetson. The Jetson will still receive RTP
        # on `net.rtp_port` — ensure the Pi streamer targets that address.
        logging.info("source is rpi; expecting external Pi streamer to send RTP to this Jetson")
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
    return_video_cfg, active_return_profile = resolve_active_return_video_profile(cfg)

    def _coerce_dimension(name: str, raw: Any) -> Optional[int]:
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc
        if value <= 0:
            raise SystemExit(f"{name} must be positive, got {value}")
        return value

    def _coerce_fps(name: str, raw: Any) -> Optional[float]:
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{name} must be numeric, got {raw!r}") from exc
        if value <= 0.0:
            raise SystemExit(f"{name} must be positive, got {value}")
        return value

    video_w = _coerce_dimension("video.width", video_cfg.get("width"))
    video_h = _coerce_dimension("video.height", video_cfg.get("height"))
    cfg_fps = _coerce_fps("video.fps", video_cfg.get("fps"))
    try:
        bitrate_kbps = int(video_cfg.get("bitrate_kbps", 4000) or 4000)
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.bitrate_kbps must be an integer") from exc

    return_w = _coerce_dimension("video.return.width", return_video_cfg.get("width"))
    return_h = _coerce_dimension("video.return.height", return_video_cfg.get("height"))
    return_cfg_fps = _coerce_fps("video.return.fps", return_video_cfg.get("fps"))
    try:
        return_bitrate_kbps = int(return_video_cfg.get("bitrate_kbps", bitrate_kbps) or bitrate_kbps)
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.return.bitrate_kbps must be an integer") from exc

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
    if return_w is None:
        return_w = video_w
    if return_h is None:
        return_h = video_h
    return_w = int(return_w)
    return_h = int(return_h)

    yolo_cfg = cfg.get("yolo")
    if not isinstance(yolo_cfg, Mapping):
        raise SystemExit("config missing 'yolo' section")

    suffix = _YOLO_RES_SUFFIX_BY_PROFILE.get(active_profile) if active_profile is not None else None
    if suffix is None:
        suffix = _YOLO_RES_SUFFIX_BY_WIDTH.get(video_w)
    if suffix is None:
        raise SystemExit(
            f"no YOLO engine available for video width {video_w} (profile {active_profile!r})"
        )

    try:
        derived_input_size = int(suffix)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid YOLO engine suffix {suffix!r}") from exc

    legacy_engine_size_raw = yolo_cfg.get("engine_size")
    legacy_engine_size: Optional[str]
    if isinstance(legacy_engine_size_raw, str) and legacy_engine_size_raw.strip():
        legacy_engine_size = legacy_engine_size_raw.strip().lower()
    else:
        legacy_engine_size = None
    if legacy_engine_size is not None and legacy_engine_size not in _YOLO_ENGINE_SIZES:
        supported = ", ".join(sorted(_YOLO_ENGINE_SIZES))
        raise SystemExit(
            f"unsupported yolo.engine_size {legacy_engine_size_raw!r}; expected one of: {supported}"
        )

    legacy_input_raw = yolo_cfg.get("input_size")
    legacy_input_size: Optional[int]
    if legacy_input_raw is None:
        legacy_input_size = None
    else:
        try:
            legacy_input_size = int(legacy_input_raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit("yolo.input_size must be an integer") from exc

    search_spec = _parse_engine_spec(
        yolo_cfg,
        "search_engine",
        default_size=legacy_engine_size or "small",
        default_input_size=legacy_input_size or derived_input_size,
    )
    if search_spec is None:
        raise SystemExit("failed to resolve yolo.search_engine")
    engine_size, yolo_input_size = search_spec

    engine_path = _YOLO_ENGINE_DIR / f"{engine_size}_{yolo_input_size}.engine"
    if not engine_path.exists():
        raise SystemExit(f"YOLO search engine not found at {engine_path}")

    try:
        control_cfg = ControlConfig.from_raw_config(cfg, (video_w, video_h))
    except ControlConfigError as exc:
        raise SystemExit(f"invalid control configuration: {exc}") from exc

    control_section = cfg.get("control") if isinstance(cfg, Mapping) else None
    if not isinstance(control_section, Mapping):
        control_section = {}
    negotiation_raw = control_section.get("negotiation")
    if not isinstance(negotiation_raw, Mapping):
        negotiation_raw = {}

    negotiation_enabled = bool(negotiation_raw.get("enabled", rpi_source and not file_source))
    negotiation_mode = str(negotiation_raw.get("mode", "rpi_priority")).strip().lower()
    if negotiation_mode not in {"auto_only", "manual_only", "rpi_priority"}:
        raise SystemExit(
            "control.negotiation.mode must be one of: auto_only, manual_only, rpi_priority"
        )
    try:
        negotiation_state_timeout_s = float(negotiation_raw.get("rpi_state_timeout_s", 0.75))
    except (TypeError, ValueError) as exc:
        raise SystemExit("control.negotiation.rpi_state_timeout_s must be numeric") from exc
    if negotiation_state_timeout_s <= 0.0:
        raise SystemExit("control.negotiation.rpi_state_timeout_s must be > 0")

    negotiation_manual_when_no_state = bool(
        negotiation_raw.get("manual_when_no_state", rpi_source and not file_source)
    )
    negotiation_manual_on_emergency = bool(negotiation_raw.get("manual_on_emergency", True))
    negotiation_manual_on_active = bool(negotiation_raw.get("manual_on_active", True))
    negotiation_command_mode = str(
        negotiation_raw.get("command_mode", "always")
    ).strip().lower()
    if negotiation_command_mode not in {"off", "toggle", "always"}:
        raise SystemExit(
            "control.negotiation.command_mode must be one of: off, toggle, always"
        )

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
        argus_sensor_id: Optional[int] = None
        argus_sensor_mode: Optional[int] = None
        argus_capture_width: Optional[int] = None
        argus_capture_height: Optional[int] = None
        argus_capture_fps: Optional[float] = None
        argus_exposuretimerange: Optional[str] = None
        argus_gainrange: Optional[str] = None
        argus_aeantibanding: Optional[int] = None
        argus_tnr_strength: Optional[float] = None
        argus_ee_strength: Optional[float] = None

        camera_cfg = cfg.get("camera") if isinstance(cfg, Mapping) else None
        if isinstance(camera_cfg, Mapping):
            argus_cfg = camera_cfg.get("argus")
            if isinstance(argus_cfg, Mapping):
                raw_capture_width = argus_cfg.get("capture_width")
                if raw_capture_width is not None:
                    try:
                        argus_capture_width = int(raw_capture_width)
                    except (TypeError, ValueError) as exc:
                        raise SystemExit(
                            f"camera.argus.capture_width must be an integer, got {raw_capture_width!r}"
                        ) from exc
                    if argus_capture_width <= 0:
                        raise SystemExit("camera.argus.capture_width must be > 0")

                raw_capture_height = argus_cfg.get("capture_height")
                if raw_capture_height is not None:
                    try:
                        argus_capture_height = int(raw_capture_height)
                    except (TypeError, ValueError) as exc:
                        raise SystemExit(
                            f"camera.argus.capture_height must be an integer, got {raw_capture_height!r}"
                        ) from exc
                    if argus_capture_height <= 0:
                        raise SystemExit("camera.argus.capture_height must be > 0")

                raw_capture_fps = argus_cfg.get("capture_fps")
                if raw_capture_fps is not None:
                    try:
                        argus_capture_fps = float(raw_capture_fps)
                    except (TypeError, ValueError) as exc:
                        raise SystemExit(
                            f"camera.argus.capture_fps must be numeric, got {raw_capture_fps!r}"
                        ) from exc
                    if argus_capture_fps <= 0.0:
                        raise SystemExit("camera.argus.capture_fps must be > 0")

                raw_exposure_range = argus_cfg.get("exposuretimerange")
                if raw_exposure_range is not None:
                    argus_exposuretimerange = str(raw_exposure_range).strip()
                    if not argus_exposuretimerange:
                        raise SystemExit("camera.argus.exposuretimerange must not be empty")

                raw_gain_range = argus_cfg.get("gainrange")
                if raw_gain_range is not None:
                    argus_gainrange = str(raw_gain_range).strip()
                    if not argus_gainrange:
                        raise SystemExit("camera.argus.gainrange must not be empty")

                raw_aeantibanding = argus_cfg.get("aeantibanding")
                if raw_aeantibanding is not None:
                    try:
                        argus_aeantibanding = int(raw_aeantibanding)
                    except (TypeError, ValueError) as exc:
                        raise SystemExit(
                            f"camera.argus.aeantibanding must be an integer, got {raw_aeantibanding!r}"
                        ) from exc

                raw_tnr_strength = argus_cfg.get("tnr_strength")
                if raw_tnr_strength is not None:
                    try:
                        argus_tnr_strength = float(raw_tnr_strength)
                    except (TypeError, ValueError) as exc:
                        raise SystemExit(
                            f"camera.argus.tnr_strength must be numeric, got {raw_tnr_strength!r}"
                        ) from exc

                raw_ee_strength = argus_cfg.get("ee_strength")
                if raw_ee_strength is not None:
                    try:
                        argus_ee_strength = float(raw_ee_strength)
                    except (TypeError, ValueError) as exc:
                        raise SystemExit(
                            f"camera.argus.ee_strength must be numeric, got {raw_ee_strength!r}"
                        ) from exc

            libcamera_cfg = camera_cfg.get("libcamera")
            if isinstance(libcamera_cfg, Mapping):
                raw_sensor_mode = libcamera_cfg.get("sensor_mode")
                if raw_sensor_mode is not None:
                    try:
                        argus_sensor_mode = int(raw_sensor_mode)
                    except (TypeError, ValueError) as exc:
                        raise SystemExit(
                            f"camera.libcamera.sensor_mode must be an integer, got {raw_sensor_mode!r}"
                        ) from exc
                    if argus_sensor_mode < 0:
                        raise SystemExit("camera.libcamera.sensor_mode must be >= 0")
        if ":" in source_clean:
            csi_arg = source_clean.split(":", 1)[1].strip()
            if csi_arg:
                if csi_arg.startswith("/"):
                    device_path = csi_arg
                elif csi_arg.isdigit():
                    argus_sensor_id = int(csi_arg)
                else:
                    pipeline_override = csi_arg
        recv = CsiVideoReader(
            device=device_path,
            width=video_w,
            height=video_h,
            fps=source_fps,
            pipeline=pipeline_override,
            argus_sensor_id=argus_sensor_id,
            argus_sensor_mode=argus_sensor_mode,
            argus_capture_width=argus_capture_width,
            argus_capture_height=argus_capture_height,
            argus_capture_fps=argus_capture_fps,
            argus_exposuretimerange=argus_exposuretimerange,
            argus_gainrange=argus_gainrange,
            argus_aeantibanding=argus_aeantibanding,
            argus_tnr_strength=argus_tnr_strength,
            argus_ee_strength=argus_ee_strength,
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

    dual_tracker_cfg = _parse_dual_tracker_cfg(yolo_cfg)
    dual_tracker_enabled = bool(dual_tracker_cfg.get("enabled", False))
    dual_search_cfg = dual_tracker_cfg.get("search", {})
    if not isinstance(dual_search_cfg, Mapping):
        dual_search_cfg = {}
    dual_track_cfg = dual_tracker_cfg.get("track", {})
    if not isinstance(dual_track_cfg, Mapping):
        dual_track_cfg = {}

    search_tracker: Optional[BotSortSearchTracker] = None
    search_assign_iou_thresh = 0.2
    search_camstate_gmc_enabled = True
    search_camstate_timeout_s = 0.3
    if dual_tracker_enabled:
        search_tracker_name = str(dual_search_cfg.get("tracker", "botsort")).strip().lower()
        search_assign_iou_thresh = float(dual_search_cfg.get("assign_iou_thresh", 0.2) or 0.2)
        search_camstate_gmc_enabled = bool(dual_search_cfg.get("camstate_gmc_enabled", True))
        search_camstate_timeout_s = (
            float(dual_search_cfg.get("camstate_timeout_ms", 300) or 300) / 1000.0
        )
        if search_tracker_name == "botsort":
            botsort_cfg = dual_search_cfg.get("botsort")
            if not isinstance(botsort_cfg, Mapping):
                raise SystemExit("yolo.dual_tracker.search.botsort must be a mapping")
            try:
                search_tracker = BotSortSearchTracker(
                    frame_rate=source_fps,
                    config=dict(botsort_cfg),
                )
            except Exception as exc:
                raise SystemExit(f"failed to initialize BoT-SORT search tracker: {exc}") from exc
        else:
            raise SystemExit(
                f"unsupported yolo.dual_tracker.search.tracker {search_tracker_name!r}; expected 'botsort'"
            )

    track_yolo: Optional[YoloEngine] = None
    track_crop_w: Optional[int] = None
    track_crop_h: Optional[int] = None
    if dual_tracker_enabled:
        track_spec = _parse_engine_spec(
            yolo_cfg,
            "track_engine",
            default_size=None,
            default_input_size=None,
            required=True,
        )
        if track_spec is None:
            raise SystemExit("failed to resolve yolo.track_engine")
        track_engine_size, track_input_size = track_spec
        track_engine_path = _YOLO_ENGINE_DIR / f"{track_engine_size}_{track_input_size}.engine"
        if not track_engine_path.exists():
            raise SystemExit(f"YOLO dual-tracker engine not found at {track_engine_path}")
        track_yolo = YoloEngine(
            engine_path=str(track_engine_path),
            conf_thres=yolo_cfg['conf_thres'],
            iou_thres=yolo_cfg['iou_thres'],
            input_size=track_input_size,
            preprocess_mode=yolo_cfg.get('preprocess_mode', 'bilinear'),
        )

        track_crop_w = min(video_w, max(1, int(track_input_size)))
        track_crop_h = max(1, int(round(track_crop_w * (float(video_h) / float(video_w)))))
        if track_crop_h > video_h:
            track_crop_h = video_h
            track_crop_w = max(1, int(round(track_crop_h * (float(video_w) / float(video_h)))))

        logging.info(
            "dual tracker enabled: search=%s track=%s crop=%dx%d tracker=%s",
            engine_path.name,
            track_engine_path.name,
            track_crop_w,
            track_crop_h,
            str(dual_search_cfg.get("tracker", "botsort")),
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
    expected_header_origins = set(
        _expected_header_origins(rpi_source=rpi_source, file_source=file_source)
    )
    last_header_origin_drop_log = 0.0
    header_ep = net_cfg.get('header_push') if isinstance(net_cfg, Mapping) else None
    if header_ep and not file_source:
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.RCVHWM, 10)
        pull.setsockopt(zmq.LINGER, 0)
        header_port = _parse_tcp_port(header_ep, "header_push")
        pull.bind(f"tcp://0.0.0.0:{header_port}")
        pull.RCVTIMEO = 0  # non-blocking
    elif not file_source:
        raise SystemExit("config missing net.header_push endpoint")

    manual_pull: Optional[zmq.Socket] = None
    manual_state_ep = net_cfg.get('zmq_manual_state') if isinstance(net_cfg, Mapping) else None
    if manual_state_ep and not file_source:
        manual_pull = ctx.socket(zmq.PULL)
        manual_pull.setsockopt(zmq.RCVHWM, 10)
        manual_pull.setsockopt(zmq.LINGER, 0)
        manual_state_port = _parse_tcp_port(manual_state_ep, "zmq_manual_state")
        manual_pull.bind(f"tcp://0.0.0.0:{manual_state_port}")
        manual_pull.RCVTIMEO = 0
    elif rpi_source and not file_source:
        logging.warning("source=rpi but net.zmq_manual_state is not configured")

    gimbal_sub: Optional[zmq.Socket] = None
    gimbal_state_ep = net_cfg.get('zmq_gimbal_state') if isinstance(net_cfg, Mapping) else None
    if gimbal_state_ep and not file_source:
        gimbal_sub = ctx.socket(zmq.SUB)
        gimbal_sub.setsockopt(zmq.CONFLATE, 1)
        gimbal_sub.setsockopt(zmq.RCVHWM, 1)
        gimbal_sub.setsockopt(zmq.LINGER, 0)
        gimbal_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        gimbal_sub.connect(str(gimbal_state_ep))
        gimbal_sub.RCVTIMEO = 0

    writer_fps = return_cfg_fps if return_cfg_fps and return_cfg_fps > 0.0 else source_fps
    if writer_fps <= 0.0:
        writer_fps = cfg_fps or 30.0
    profile_fps = return_cfg_fps if return_cfg_fps and return_cfg_fps > 0.0 else writer_fps
    if active_profile:
        logging.info(
            "video profile %s resolved to %dx%d @ %.2f FPS, %d kbps",
            active_profile,
            video_w,
            video_h,
            cfg_fps if cfg_fps and cfg_fps > 0.0 else source_fps,
            bitrate_kbps,
        )
    if active_return_profile:
        logging.info(
            "return video profile %s resolved to %dx%d @ %.2f FPS, %d kbps",
            active_return_profile,
            return_w,
            return_h,
            profile_fps,
            return_bitrate_kbps,
        )
    if file_source:
        return_file_path = _derive_return_file_path(source_spec)
        ret_vw = make_file_return_writer(
            return_file_path,
            return_w,
            return_h,
            fps=writer_fps,
        )
    else:
        has_rpi_peer = "rpi" in successful_sync_peers
        rpi_ip = net_cfg.get('rpi_ip') if isinstance(net_cfg, Mapping) else None
        if has_rpi_peer and rpi_ip is not None and str(rpi_ip).strip():
            return_ip_key = 'rpi_ip'
            return_ip = rpi_ip
            logging.info("return feed destination: net.rpi_ip (rpi sync successful)")
        else:
            return_ip_key = 'pc_ip'
            return_ip = net_cfg.get(return_ip_key) if isinstance(net_cfg, Mapping) else None
            if has_rpi_peer and (rpi_ip is None or not str(rpi_ip).strip()):
                logging.warning(
                    "rpi sync succeeded but net.rpi_ip is missing/empty; falling back to net.pc_ip"
                )
            else:
                logging.info("return feed destination: net.pc_ip (rpi sync unavailable)")

        return_ip_override = net_cfg.get('return_ip') if isinstance(net_cfg, Mapping) else None
        if return_ip_override is not None and str(return_ip_override).strip():
            override_ip = str(return_ip_override).strip()
            if return_ip_key == 'rpi_ip' or rpi_source or csi_source:
                return_ip_key = 'return_ip'
                return_ip = override_ip
            else:
                logging.info(
                    "ignoring net.return_ip override for source=%s; using net.%s",
                    source_lower,
                    return_ip_key,
                )
        if not return_ip:
            raise SystemExit(f"config missing net.{return_ip_key}")
        return_port_value = net_cfg.get('rtp_return_port') if isinstance(net_cfg, Mapping) else None
        if return_port_value is None:
            raise SystemExit("config missing net.rtp_return_port")
        try:
            return_port = int(return_port_value)
        except (TypeError, ValueError) as exc:
            raise SystemExit("net.rtp_return_port must be an integer") from exc
        ret_vw = make_return_writer(
            str(return_ip),
            return_port,
            return_w,
            return_h,
            fps=max(1, int(round(writer_fps))),
            bitrate=return_bitrate_kbps,
        )

    file_frame_idx = -1
    file_frame_interval_ms = (
        1000.0 / writer_fps if file_source and writer_fps > 0.0 else None
    )
    file_src_start_ns = 0

    latest_header: Optional[Dict[str, int]] = None
    latest_manual_state: Optional[ManualControlState] = None
    latest_manual_state_rx_mono: Optional[float] = None
    last_manual_state_log_ts = 0.0
    last_hold_cmd_mono = 0.0
    last_authority_log_mono = 0.0
    current_control_authority = "auto"
    current_control_authority_reason = "default"
    last_control_command_log_mono = 0.0
    current_control_commands_enabled = True
    current_control_commands_reason = "default"
    local_frame_id = 0
    waiting_for_header_logged = False
    latest_cam_state: Optional[CamState] = None
    latest_cam_state_mono: Optional[float] = None
    latest_gimbal_cam_state_mono: Optional[float] = None
    cam_state_prefer_gimbal_window_s = 0.30
    tracker_prev_cam_state: Optional[CamState] = None
    tracker_prev_cam_state_mono: Optional[float] = None
    controller_search: Optional[ControlLoop] = None
    controller_track: Optional[ControlLoop] = None
    transition_control_cfg = control_cfg
    if not file_source:
        distance_alpha = ranging_cfg.ema_alpha if ranging_cfg.enabled else None
        if ctrl_pub is None:
            raise RuntimeError("control publisher is not initialized")
        if dual_tracker_enabled:
            search_cfg = replace(control_cfg, controller="pid")
            track_controller = str(control_cfg.controller)
            track_cfg = replace(control_cfg, controller=track_controller)
            controller_search = ControlLoop(
                search_cfg,
                ctrl_pub,
                laser_mount=laser_cfg,
                distance_alpha=distance_alpha,
                cli_json_logs=cli_json_logs,
            )
            controller_track = ControlLoop(
                track_cfg,
                ctrl_pub,
                laser_mount=laser_cfg,
                distance_alpha=distance_alpha,
                cli_json_logs=cli_json_logs,
            )
            transition_control_cfg = search_cfg
            ranging_log_interval_s = max(
                getattr(controller_search, "log_interval_s", 0.5),
                getattr(controller_track, "log_interval_s", 0.5),
            )
            logging.info(
                "dual_tracker control split: search=pid track=%s",
                track_controller,
            )
        else:
            controller_search = ControlLoop(
                control_cfg,
                ctrl_pub,
                laser_mount=laser_cfg,
                distance_alpha=distance_alpha,
                cli_json_logs=cli_json_logs,
            )
            ranging_log_interval_s = getattr(controller_search, "log_interval_s", 0.5)
    else:
        ranging_log_interval_s = 0.5
    ranging_last_log_time = 0.0
    ranging_logged_once = False
    ranging_last_target_idx: Optional[int] = None
    tracker_mode = "search"
    tracker_hits = 0
    tracker_misses = 0
    tracker_recover_until = 0.0
    tracker_last_target_uv: Optional[Tuple[float, float]] = None
    tracker_frame_counter = 0
    tracker_slew_sent = False
    tracker_slew_started_at = 0.0
    tracker_slew_target_uv: Optional[Tuple[float, float]] = None
    tracker_slew_track_hits = 0
    tracker_active_track_id: Optional[int] = None

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
                active_controller = (
                    controller_track
                    if (dual_tracker_enabled and tracker_mode == "track" and controller_track is not None)
                    else controller_search
                )
                no_frame_now = time.monotonic()
                has_fresh_manual_for_no_frame = (
                    latest_manual_state is not None
                    and latest_manual_state_rx_mono is not None
                    and (no_frame_now - latest_manual_state_rx_mono) <= negotiation_state_timeout_s
                )
                if (
                    active_controller is not None
                    and current_control_authority == "auto"
                    and current_control_commands_enabled
                ):
                    active_controller.tick(time.monotonic())
                elif ctrl_pub is not None and (no_frame_now - last_hold_cmd_mono) >= 0.5:
                    if (
                        current_control_commands_enabled
                        and current_control_authority != "auto"
                        and has_fresh_manual_for_no_frame
                        and latest_manual_state is not None
                    ):
                        _publish_manual_passthrough_control_cmd(
                            ctrl_pub,
                            frame_id=-1,
                            src_ts_ms=int(latest_manual_state.src_ts_ms),
                            controller_mode=str(control_cfg.controller),
                            manual_state=latest_manual_state,
                            max_yaw_rate=float(control_cfg.rate_limits.yaw),
                            max_pitch_rate=float(control_cfg.rate_limits.pitch),
                        )
                    else:
                        _publish_hold_control_cmd(
                            ctrl_pub,
                            frame_id=-1,
                            src_ts_ms=0,
                            controller_mode=str(control_cfg.controller),
                        )
                    last_hold_cmd_mono = no_frame_now
                continue

            if csi_source or rpi_source:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

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
                waiting_for_header_logged = False

            # headers (non-blocking drain)
            if pull is not None:
                try:
                    while True:
                        header_obj = pull.recv_json(flags=zmq.NOBLOCK)
                        if not isinstance(header_obj, dict):
                            continue

                        origin_raw = header_obj.get("origin")
                        origin = str(origin_raw).strip().lower() if origin_raw is not None else ""
                        if expected_header_origins and origin not in expected_header_origins:
                            now = time.monotonic()
                            if (now - last_header_origin_drop_log) >= 2.0:
                                logging.info(
                                    "ignoring header from origin=%r (expected one of %s)",
                                    origin,
                                    sorted(expected_header_origins),
                                )
                                last_header_origin_drop_log = now
                            continue

                        if (
                            controller_search is not None
                            and header_obj.get("type") == "CamState"
                        ):
                            try:
                                cam_state = CamState(**header_obj)
                            except ValidationError as exc:
                                logging.warning("invalid CamState header: %s", exc)
                            else:
                                use_header_cam_state = True
                                if (
                                    gimbal_sub is not None
                                    and latest_gimbal_cam_state_mono is not None
                                ):
                                    gimbal_state_age_s = max(
                                        0.0,
                                        time.monotonic() - latest_gimbal_cam_state_mono,
                                    )
                                    if gimbal_state_age_s <= cam_state_prefer_gimbal_window_s:
                                        use_header_cam_state = False

                                if use_header_cam_state:
                                    if controller_search is not None:
                                        controller_search.update_cam_state(cam_state)
                                    if controller_track is not None:
                                        controller_track.update_cam_state(cam_state)
                                    latest_cam_state = cam_state
                                    latest_cam_state_mono = time.monotonic()
                                # CamState carries the originating frame metadata. Use it to
                                # refresh our latest header so DetectionMsg instances keep
                                # advancing even if the bare header message was dropped.
                                latest_header = header_obj
                        elif isinstance(header_obj, dict):
                            frame_id_raw = header_obj.get("frame_id")
                            src_ts_ms_raw = header_obj.get("src_ts_ms")
                            if frame_id_raw is None or src_ts_ms_raw is None:
                                continue
                            try:
                                latest_header = {
                                    "frame_id": int(frame_id_raw),
                                    "src_ts_ms": int(src_ts_ms_raw),
                                }
                            except (TypeError, ValueError):
                                continue
                except zmq.Again:
                    pass

            if manual_pull is not None:
                try:
                    while True:
                        raw_manual = manual_pull.recv(flags=zmq.NOBLOCK)
                        try:
                            manual_state = manual_control_state_from_json(raw_manual)
                        except Exception as exc:  # noqa: BLE001
                            logging.warning("invalid ManualControlState payload: %s", exc)
                            continue

                        latest_manual_state = manual_state
                        now_manual = time.monotonic()
                        latest_manual_state_rx_mono = now_manual
                        should_log_manual = (
                            manual_state.emergency
                            or manual_state.active_changed
                            or manual_state.emergency_entered
                            or manual_state.emergency_exited
                            or manual_state.control_cmd_changed
                            or (now_manual - last_manual_state_log_ts) >= 2.0
                        )
                        if should_log_manual:
                            logging.info(
                                "manual state source=%s active=%s emergency=%s cmd_enabled=%s joy=(%d,%d) rate=(%.3f,%.3f) serial_local=%s",
                                manual_state.source,
                                manual_state.active,
                                manual_state.emergency,
                                manual_state.control_cmd_enabled,
                                manual_state.joystick_raw[0],
                                manual_state.joystick_raw[1],
                                manual_state.joystick_rate_cmd[0],
                                manual_state.joystick_rate_cmd[1],
                                manual_state.serial_local_mode,
                            )
                            last_manual_state_log_ts = now_manual
                except zmq.Again:
                    pass

            if gimbal_sub is not None:
                try:
                    while True:
                        raw_cam_state = gimbal_sub.recv_json(flags=zmq.NOBLOCK)
                        try:
                            cam_state = CamState(**raw_cam_state)
                        except ValidationError as exc:
                            logging.warning("invalid CamState payload on zmq_gimbal_state: %s", exc)
                            continue

                        if controller_search is not None:
                            controller_search.update_cam_state(cam_state)
                        if controller_track is not None:
                            controller_track.update_cam_state(cam_state)
                        latest_cam_state = cam_state
                        latest_cam_state_mono = time.monotonic()
                        latest_gimbal_cam_state_mono = time.monotonic()
                except zmq.Again:
                    pass

            auto_control_allowed = not file_source
            control_authority_reason = "default"
            manual_state_age_s: Optional[float] = None
            now_auth = time.monotonic()
            if latest_manual_state_rx_mono is not None:
                manual_state_age_s = max(0.0, now_auth - latest_manual_state_rx_mono)
            has_fresh_manual_state = (
                latest_manual_state is not None
                and latest_manual_state_rx_mono is not None
                and (now_auth - latest_manual_state_rx_mono) <= negotiation_state_timeout_s
            )

            if not file_source and negotiation_enabled:

                if negotiation_mode == "manual_only":
                    auto_control_allowed = False
                    control_authority_reason = "mode=manual_only"
                elif negotiation_mode == "auto_only":
                    auto_control_allowed = True
                    control_authority_reason = "mode=auto_only"
                else:
                    if has_fresh_manual_state and latest_manual_state is not None:
                        if negotiation_manual_on_emergency and latest_manual_state.emergency:
                            auto_control_allowed = False
                            control_authority_reason = "rpi emergency active"
                        elif negotiation_manual_on_active and latest_manual_state.active:
                            auto_control_allowed = False
                            control_authority_reason = "rpi manual active"
                        else:
                            auto_control_allowed = True
                            control_authority_reason = "rpi state allows auto"
                    else:
                        require_manual_on_missing_state = (
                            rpi_source and negotiation_manual_when_no_state and not sim_source
                        )
                        auto_control_allowed = not require_manual_on_missing_state
                        control_authority_reason = (
                            "no fresh rpi state -> manual"
                            if require_manual_on_missing_state
                            else "no fresh rpi state -> auto"
                        )

            control_commands_enabled = not file_source
            control_commands_reason = "mode=always"
            if not file_source:
                if negotiation_command_mode == "off":
                    control_commands_enabled = False
                    control_commands_reason = "mode=off"
                elif negotiation_command_mode == "toggle":
                    if has_fresh_manual_state and latest_manual_state is not None:
                        control_commands_enabled = bool(latest_manual_state.control_cmd_enabled)
                        control_commands_reason = (
                            "rpi command toggle enabled"
                            if control_commands_enabled
                            else "rpi command toggle disabled"
                        )
                    else:
                        control_commands_enabled = False
                        control_commands_reason = "no fresh rpi command toggle -> off"

            desired_control_command_state = (
                "enabled" if control_commands_enabled else "disabled"
            )
            if (
                control_commands_enabled != current_control_commands_enabled
                or (now_auth - last_control_command_log_mono) >= 3.0
            ):
                logging.info(
                    "control commands=%s reason=%s mode=%s",
                    desired_control_command_state,
                    control_commands_reason,
                    negotiation_command_mode,
                )
                current_control_commands_enabled = control_commands_enabled
                current_control_commands_reason = control_commands_reason
                last_control_command_log_mono = now_auth
            else:
                current_control_commands_reason = control_commands_reason

            desired_authority = "auto" if auto_control_allowed else "manual"
            now_authority_log = time.monotonic()
            if (
                desired_authority != current_control_authority
                or (now_authority_log - last_authority_log_mono) >= 3.0
            ):
                logging.info(
                    "control authority=%s reason=%s negotiation_enabled=%s mode=%s",
                    desired_authority,
                    control_authority_reason,
                    negotiation_enabled,
                    negotiation_mode,
                )
                current_control_authority = desired_authority
                current_control_authority_reason = control_authority_reason
                last_authority_log_mono = now_authority_log
            else:
                current_control_authority_reason = control_authority_reason

            if csi_source:
                local_frame_id += 1
                latest_header = {
                    "frame_id": local_frame_id,
                    "src_ts_ms": int(time.monotonic_ns() / 1e6),
                }
                waiting_for_header_logged = False

            if not file_source and latest_header is None:
                if not waiting_for_header_logged:
                    logging.info("waiting for first external frame header before publishing detections")
                    waiting_for_header_logged = True
                active_controller = (
                    controller_track
                    if (dual_tracker_enabled and tracker_mode == "track" and controller_track is not None)
                    else controller_search
                )
                if (
                    active_controller is not None
                    and auto_control_allowed
                    and control_commands_enabled
                ):
                    active_controller.tick(time.monotonic())
                elif ctrl_pub is not None and (time.monotonic() - last_hold_cmd_mono) >= 0.5:
                    if (
                        control_commands_enabled
                        and not auto_control_allowed
                        and has_fresh_manual_state
                        and latest_manual_state is not None
                    ):
                        _publish_manual_passthrough_control_cmd(
                            ctrl_pub,
                            frame_id=-1,
                            src_ts_ms=int(latest_manual_state.src_ts_ms),
                            controller_mode=str(control_cfg.controller),
                            manual_state=latest_manual_state,
                            max_yaw_rate=float(control_cfg.rate_limits.yaw),
                            max_pitch_rate=float(control_cfg.rate_limits.pitch),
                        )
                    else:
                        _publish_hold_control_cmd(
                            ctrl_pub,
                            frame_id=-1,
                            src_ts_ms=0,
                            controller_mode=str(control_cfg.controller),
                        )
                    last_hold_cmd_mono = time.monotonic()
                continue

            if frame.ndim == 3 and frame.shape[2] == 4:  # RGBA->BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            tracker_frame_counter += 1
            rx_ts_ms = int(time.monotonic_ns() / 1e6)
            infer_source = "search"
            now_mono = time.monotonic()

            should_use_track = (
                dual_tracker_enabled
                and track_yolo is not None
                and tracker_mode == "track"
                and tracker_last_target_uv is not None
                and track_crop_w is not None
                and track_crop_h is not None
            )
            heartbeat_interval = int(dual_search_cfg.get("heartbeat_interval_frames", 0) or 0)
            heartbeat_due = (
                should_use_track
                and heartbeat_interval > 0
                and (tracker_frame_counter % heartbeat_interval == 0)
            )

            # --- asynchronous inference to avoid blocking the main/server thread ---
            # Uses a single-worker ThreadPoolExecutor to run the heavy GPU/TensorRT
            # inference off-thread. We keep the last known boxes if a new
            # inference result isn't ready yet so the video pipeline can continue.
            if "_inference_executor" not in globals():
                _inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                _inference_future = None
                _last_infer_result = None

            # prepare crop_rect for possible track inference (no heavy work here)
            crop_rect = None
            if (
                should_use_track
                and not heartbeat_due
                and track_yolo is not None
                and track_crop_w is not None
                and track_crop_h is not None
                and tracker_last_target_uv is not None
            ):
                crop_rect = _crop_rect_around_point(
                    tracker_last_target_uv,
                    frame_w,
                    frame_h,
                    int(track_crop_w),
                    int(track_crop_h),
                )

            def _infer_task(frame_copy, use_track, crop_rect_local):
                try:
                    if use_track and crop_rect_local is not None:
                        x1, y1, x2, y2 = crop_rect_local
                        crop_local = frame_copy[y1:y2, x1:x2]
                        if crop_local.size > 0:
                            track_boxes_local = track_yolo.infer(crop_local)
                            boxes_local = _project_boxes_from_crop(
                                track_boxes_local, crop_rect_local, frame_w, frame_h
                            )
                            src = "track"
                        else:
                            boxes_local = yolo.infer(frame_copy)
                            src = "search"
                    else:
                        boxes_local = yolo.infer(frame_copy)
                        src = "search"
                    return {"boxes": boxes_local, "infer_ts_ms": int(time.monotonic_ns() / 1e6), "infer_source": src}
                except Exception:
                    return {"boxes": [], "infer_ts_ms": int(time.monotonic_ns() / 1e6), "infer_source": "search"}

            # if no active future, submit one; otherwise, if it's done take the result
            try:
                if _inference_future is None:
                    frame_for_infer = frame.copy()
                    _inference_future = _inference_executor.submit(
                        _infer_task, frame_for_infer, should_use_track and not heartbeat_due, crop_rect
                    )
                    boxes = _last_infer_result["boxes"] if _last_infer_result is not None else []
                    infer_source = _last_infer_result["infer_source"] if _last_infer_result is not None else "search"
                    infer_ts_ms = int(time.monotonic_ns() / 1e6)
                elif _inference_future.done():
                    try:
                        res = _inference_future.result()
                    except Exception:
                        res = {"boxes": [], "infer_ts_ms": int(time.monotonic_ns() / 1e6), "infer_source": "search"}
                    _last_infer_result = res
                    boxes = res["boxes"]
                    infer_source = res["infer_source"]
                    infer_ts_ms = res["infer_ts_ms"]
                    # submit next task for the current frame
                    frame_for_infer = frame.copy()
                    _inference_future = _inference_executor.submit(
                        _infer_task, frame_for_infer, should_use_track and not heartbeat_due, crop_rect
                    )
                else:
                    # inference still running; use last known boxes to avoid blocking
                    boxes = _last_infer_result["boxes"] if _last_infer_result is not None else []
                    infer_source = _last_infer_result["infer_source"] if _last_infer_result is not None else "search"
                    infer_ts_ms = int(time.monotonic_ns() / 1e6)
            except Exception:
                boxes = []
                infer_source = "search"
                infer_ts_ms = int(time.monotonic_ns() / 1e6)

            if (
                dual_tracker_enabled
                and search_tracker is not None
                and tracker_mode in {"search", "slew", "recover"}
            ):
                tracker_warp: Optional[np.ndarray] = None
                if (
                    search_camstate_gmc_enabled
                    and latest_cam_state is not None
                    and latest_cam_state_mono is not None
                    and (now_mono - latest_cam_state_mono) <= search_camstate_timeout_s
                    and tracker_prev_cam_state is not None
                ):
                    tracker_warp = _camstate_warp_from_delta(
                        prev_state=tracker_prev_cam_state,
                        curr_state=latest_cam_state,
                        fx_px=float(camera_intrinsics.fx_px),
                        fy_px=float(camera_intrinsics.fy_px),
                    )

                track_xyxy, track_conf, track_cls = boxes_to_tracker_arrays(
                    boxes,
                    img_w=frame_w,
                    img_h=frame_h,
                )
                tracked_observations = search_tracker.update(
                    xyxy=track_xyxy,
                    conf=track_conf,
                    cls=track_cls,
                    frame=frame,
                    warp_override=tracker_warp,
                )

                assigned_track_ids = assign_track_ids_to_boxes(
                    boxes,
                    tracked_observations,
                    img_w=frame_w,
                    img_h=frame_h,
                    min_iou=search_assign_iou_thresh,
                )
                for box, track_id in zip(boxes, assigned_track_ids):
                    box.track_id = track_id

                if (
                    latest_cam_state is not None
                    and latest_cam_state_mono is not None
                    and (now_mono - latest_cam_state_mono) <= search_camstate_timeout_s
                ):
                    tracker_prev_cam_state = latest_cam_state
                    tracker_prev_cam_state_mono = latest_cam_state_mono

            box_index_map = {id(box): idx for idx, box in enumerate(boxes)}
            ranging_log_entries: Dict[int, Dict[str, Any]] = {}
            if class_labels:
                for box in boxes:
                    box.cls = resolve_class_label(box.cls, class_labels)

            slew_probe_hit = False
            if (
                dual_tracker_enabled
                and tracker_mode == "slew"
                and track_yolo is not None
                and track_crop_w is not None
                and track_crop_h is not None
                and tracker_last_target_uv is not None
            ):
                slew_crop_rect = _crop_rect_around_point(
                    tracker_last_target_uv,
                    frame_w,
                    frame_h,
                    int(track_crop_w),
                    int(track_crop_h),
                )
                sx1, sy1, sx2, sy2 = slew_crop_rect
                slew_crop = frame[sy1:sy2, sx1:sx2]
                if slew_crop.size > 0:
                    slew_probe_boxes = track_yolo.infer(slew_crop)
                    slew_probe_boxes = _project_boxes_from_crop(
                        slew_probe_boxes,
                        slew_crop_rect,
                        frame_w,
                        frame_h,
                    )
                    if class_labels:
                        for box in slew_probe_boxes:
                            box.cls = resolve_class_label(box.cls, class_labels)
                    slew_probe_hit = bool(slew_probe_boxes)

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
                frame_id=latest_header["frame_id"],
                src_ts_ms=latest_header["src_ts_ms"],
                rx_ts_ms=rx_ts_ms,
                infer_ts_ms=infer_ts_ms,
                img_w=frame_w,
                img_h=frame_h,
                boxes=boxes,
                infer_source=infer_source,
                tracker_mode=tracker_mode,
            )

            active_controller = (
                controller_track
                if (dual_tracker_enabled and tracker_mode == "track" and controller_track is not None)
                else controller_search
            )
            if active_controller is not None:
                active_controller.update_detection(msg)

            if dual_tracker_enabled:
                prev_tracker_mode = tracker_mode
                target_uv_now = _target_uv_from_msg(msg)
                has_target = target_uv_now is not None
                target_track_id_now = (
                    int(msg.target_track_id) if msg.target_track_id is not None else None
                )

                search_enter_track_hits = int(dual_search_cfg.get("enter_track_hits", 3) or 3)
                track_takeover_hits = int(dual_track_cfg.get("takeover_hits", 3) or 3)
                track_exit_misses = int(dual_track_cfg.get("exit_misses", 4) or 4)
                track_recover_timeout_s = float(dual_track_cfg.get("recover_timeout_ms", 450) or 450) / 1000.0
                track_transition_speed = float(dual_track_cfg.get("transition_speed_rad_s", 1.0) or 1.0)
                track_arrival_tolerance_px = float(dual_track_cfg.get("arrival_tolerance_px", 24.0) or 24.0)
                track_transition_timeout_s = float(dual_track_cfg.get("transition_timeout_ms", 1200) or 1200) / 1000.0

                if has_target and target_uv_now is not None:
                    tracker_last_target_uv = target_uv_now

                if tracker_mode == "slew":
                    if not tracker_slew_sent:
                        tracker_mode = "search"
                        tracker_hits = 0
                        tracker_slew_track_hits = 0
                        tracker_active_track_id = None
                    elif has_target and target_uv_now is not None:
                        track_id_matches = (
                            tracker_active_track_id is None
                            or target_track_id_now is None
                            or target_track_id_now == tracker_active_track_id
                        )

                        if track_id_matches and slew_probe_hit:
                            tracker_slew_track_hits += 1
                        else:
                            tracker_slew_track_hits = 0

                        if (
                            track_id_matches
                            and tracker_slew_track_hits >= track_takeover_hits
                        ):
                            tracker_mode = "track"
                            tracker_hits = 0
                            tracker_misses = 0
                            tracker_slew_sent = False
                            tracker_slew_track_hits = 0
                            if target_track_id_now is not None:
                                tracker_active_track_id = target_track_id_now
                            logging.info(
                                "dual_tracker slew takeover met (frame=%s)",
                                msg.frame_id,
                            )
                        else:
                            arrived = False
                            if control_cfg.aim_mode == "laser_point":
                                if msg.laser_on_target is True:
                                    arrived = True
                                elif msg.laser_dot_px is not None:
                                    dot_u, dot_v = msg.laser_dot_px
                                    err_u = float(dot_u) - float(target_uv_now[0])
                                    err_v = float(dot_v) - float(target_uv_now[1])
                                    arrived = math.hypot(err_u, err_v) <= track_arrival_tolerance_px
                            else:
                                px_err_now = pixel_delta(
                                    float(target_uv_now[0]),
                                    float(target_uv_now[1]),
                                    control_cfg.cx_px,
                                    control_cfg.cy_px,
                                    control_cfg,
                                    apply_deadband=False,
                                )
                                err_mag_px = math.hypot(float(px_err_now.yaw), float(px_err_now.pitch))
                                arrived = err_mag_px <= track_arrival_tolerance_px

                            if arrived and track_id_matches:
                                tracker_mode = "track"
                                tracker_hits = 0
                                tracker_misses = 0
                                tracker_slew_sent = False
                                tracker_slew_track_hits = 0
                                if target_track_id_now is not None:
                                    tracker_active_track_id = target_track_id_now
                                logging.info(
                                    "dual_tracker slew arrival met (frame=%s)",
                                    msg.frame_id,
                                )
                            elif now_mono >= (tracker_slew_started_at + track_transition_timeout_s):
                                tracker_mode = "search"
                                tracker_hits = 0
                                tracker_slew_sent = False
                                tracker_slew_track_hits = 0
                                tracker_active_track_id = None
                                logging.info(
                                    "dual_tracker slew timeout -> search (frame=%s)",
                                    msg.frame_id,
                                )
                    elif now_mono >= (tracker_slew_started_at + track_transition_timeout_s):
                        tracker_mode = "search"
                        tracker_hits = 0
                        tracker_slew_sent = False
                        tracker_slew_track_hits = 0
                        tracker_active_track_id = None
                        logging.info(
                            "dual_tracker slew lost target -> search (frame=%s)",
                            msg.frame_id,
                        )
                elif tracker_mode == "track":
                    track_ok = False
                    if has_target:
                        if tracker_active_track_id is None or target_track_id_now is None:
                            track_ok = True
                            if tracker_active_track_id is None and target_track_id_now is not None:
                                tracker_active_track_id = target_track_id_now
                        else:
                            track_ok = target_track_id_now == tracker_active_track_id

                    if track_ok:
                        tracker_misses = 0
                    else:
                        tracker_misses += 1
                        if tracker_misses >= track_exit_misses:
                            tracker_mode = "recover"
                            tracker_recover_until = now_mono + track_recover_timeout_s
                else:
                    if has_target:
                        if (
                            tracker_active_track_id is None
                            or target_track_id_now is None
                            or target_track_id_now == tracker_active_track_id
                        ):
                            tracker_hits += 1
                        else:
                            tracker_hits = 1
                        tracker_active_track_id = target_track_id_now
                        tracker_misses = 0
                        should_enter_track = tracker_hits >= search_enter_track_hits
                        if should_enter_track:
                            if (
                                ctrl_pub is not None
                                and controller_search is not None
                                and target_uv_now is not None
                                and auto_control_allowed
                                and control_commands_enabled
                            ):
                                _send_transition_cmd(
                                    ctrl_pub,
                                    msg=msg,
                                    target_uv=target_uv_now,
                                    control_cfg=transition_control_cfg,
                                    speed_rad_s=track_transition_speed,
                                )
                                tracker_mode = "slew"
                                tracker_slew_sent = True
                                tracker_slew_started_at = now_mono
                                tracker_slew_target_uv = target_uv_now
                                tracker_hits = 0
                                tracker_slew_track_hits = 0
                            else:
                                tracker_mode = "search"
                                tracker_hits = 0
                    else:
                        tracker_hits = 0
                        tracker_misses += 1
                        if tracker_mode == "recover" and now_mono >= tracker_recover_until:
                            tracker_mode = "search"
                            tracker_active_track_id = None

                msg.tracker_mode = tracker_mode
                if tracker_mode != prev_tracker_mode:
                    logging.info(
                        "dual_tracker mode %s -> %s (frame=%s source=%s)",
                        prev_tracker_mode,
                        tracker_mode,
                        msg.frame_id,
                        infer_source,
                    )

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
            active_controller = (
                controller_track
                if (dual_tracker_enabled and tracker_mode == "track" and controller_track is not None)
                else controller_search
            )
            if (
                active_controller is not None
                and auto_control_allowed
                and control_commands_enabled
            ):
                active_controller.tick(time.monotonic())
            elif ctrl_pub is not None and (time.monotonic() - last_hold_cmd_mono) >= 0.2:
                if (
                    control_commands_enabled
                    and not auto_control_allowed
                    and has_fresh_manual_state
                    and latest_manual_state is not None
                ):
                    _publish_manual_passthrough_control_cmd(
                        ctrl_pub,
                        frame_id=int(msg.frame_id),
                        src_ts_ms=int(msg.src_ts_ms),
                        controller_mode=str(control_cfg.controller),
                        manual_state=latest_manual_state,
                        max_yaw_rate=float(control_cfg.rate_limits.yaw),
                        max_pitch_rate=float(control_cfg.rate_limits.pitch),
                    )
                else:
                    _publish_hold_control_cmd(
                        ctrl_pub,
                        frame_id=int(msg.frame_id),
                        src_ts_ms=int(msg.src_ts_ms),
                        controller_mode=str(control_cfg.controller),
                    )
                last_hold_cmd_mono = time.monotonic()

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
                track_id_val = getattr(b, "track_id", None)
                if isinstance(track_id_val, (int, float)) and math.isfinite(float(track_id_val)):
                    label_parts.append(f"id:{int(track_id_val)}")
                threat_level = getattr(b, "threat_level", None)
                if isinstance(threat_level, str) and threat_level:
                    label_parts.append(threat_level)
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
                rank_val = getattr(b, "engagement_rank", None)
                if isinstance(rank_val, int) and rank_val > 0:
                    rank_text = f"{rank_val}"
                    rank_font = cv2.FONT_HERSHEY_SIMPLEX
                    rank_scale = 0.55
                    rank_thickness = 2
                    rank_size, rank_baseline = cv2.getTextSize(
                        rank_text, rank_font, rank_scale, rank_thickness
                    )
                    rank_w, rank_h = rank_size
                    rank_x = max(0, x2 - rank_w - 6)
                    rank_y = min(
                        frame_h - 4,
                        max(rank_h + rank_baseline + 4, y2 - 6),
                    )
                    rank_bg_tl = (
                        max(0, rank_x - 3),
                        max(0, rank_y - rank_h - rank_baseline - 3),
                    )
                    rank_bg_br = (
                        min(frame_w - 1, rank_x + rank_w + 3),
                        min(frame_h - 1, rank_y + 3),
                    )
                    cv2.rectangle(frame, rank_bg_tl, rank_bg_br, (0, 0, 0), thickness=cv2.FILLED)
                    cv2.putText(
                        frame,
                        rank_text,
                        (rank_x, rank_y),
                        rank_font,
                        rank_scale,
                        colour,
                        rank_thickness,
                        cv2.LINE_AA,
                    )

            if msg.tracker_mode == "track":
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

            if not file_source:
                _draw_control_authority_overlay(
                    frame,
                    authority=current_control_authority,
                    reason=current_control_authority_reason,
                    negotiation_enabled=negotiation_enabled,
                    negotiation_mode=negotiation_mode,
                    manual_state_age_s=manual_state_age_s,
                )

            _draw_laser_overlay(frame, msg, laser_cfg)
            if ret_vw and ret_vw.isOpened():
                frame_to_write = frame
                if frame_to_write.shape[0] != return_h or frame_to_write.shape[1] != return_w:
                    frame_to_write = cv2.resize(frame_to_write, (return_w, return_h))
                if not frame_to_write.flags.c_contiguous:
                    frame_to_write = frame_to_write.copy()
                if frame_to_write.shape[0] != return_h or frame_to_write.shape[1] != return_w:
                    raise RuntimeError(
                        f"return frame shape mismatch: got {frame_to_write.shape[1]}x{frame_to_write.shape[0]}, expected {return_w}x{return_h}"
                    )
                ret_vw.write(frame_to_write)

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
        for s in (pub, pull, manual_pull, gimbal_sub):
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
