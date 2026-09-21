"""Controller-independent V2 HUD composition for PC return video.

The DeepStream branch owns detector/tracker boxes and labels.  This module
adds the operator-facing cues that depend on V2 perception, CamState, and the
compact ControlCmd wire without coupling the UI to detector transport internals.
MPC cost-term diagnostics are intentionally outside this renderer.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping, Optional

import cv2
import numpy as np

from common.perception import PerceptionSnapshotV2, PerceptionTrackV2, TrackAssessmentV2
from common.schemas import CamState, ControlCmd


GREEN = (0, 255, 0)
AMBER = (0, 191, 255)
RED = (0, 64, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SUBPIXEL_SHIFT = 8
SUBPIXEL_SCALE = 1 << SUBPIXEL_SHIFT


@dataclass(frozen=True)
class HudRenderReport:
    """Exact overlay inventory drawn into one frame, used by visual QA."""

    elements: tuple[str, ...]


@dataclass(frozen=True)
class TapeTick:
    """One world-anchored tape tick projected to a display coordinate."""

    value_deg: float
    position_px: float
    major: bool


def heading_tape_ticks(
    yaw_deg: float,
    hfov_deg: float,
    width: int,
    *,
    margin_px: int = 8,
    step_deg: float = 5.0,
) -> tuple[TapeTick, ...]:
    """Return sliding heading ticks anchored to fixed world bearings."""

    half_fov = max(float(hfov_deg) * 0.5, 1e-6)
    usable = max(1.0, float(width - 2 * margin_px))
    first = math.floor((float(yaw_deg) - half_fov) / step_deg)
    last = math.ceil((float(yaw_deg) + half_fov) / step_deg)
    ticks: list[TapeTick] = []
    for index in range(first, last + 1):
        value = index * step_deg
        offset = value - float(yaw_deg)
        if abs(offset) > half_fov + 1e-9:
            continue
        position = margin_px + usable * (0.5 + offset / (2.0 * half_fov))
        ticks.append(TapeTick(value, position, index % 2 == 0))
    return tuple(ticks)


def elevation_tape_ticks(
    pitch_deg: float,
    vfov_deg: float,
    top_px: int,
    bottom_px: int,
    *,
    step_deg: float = 5.0,
) -> tuple[TapeTick, ...]:
    """Return sliding elevation ticks anchored to fixed world elevations."""

    half_fov = max(float(vfov_deg) * 0.5, 1e-6)
    usable = max(1.0, float(bottom_px - top_px))
    first = math.floor((float(pitch_deg) - half_fov) / step_deg)
    last = math.ceil((float(pitch_deg) + half_fov) / step_deg)
    ticks: list[TapeTick] = []
    for index in range(first, last + 1):
        value = index * step_deg
        offset = value - float(pitch_deg)
        if abs(offset) > half_fov + 1e-9:
            continue
        position = top_px + usable * (0.5 - offset / (2.0 * half_fov))
        ticks.append(TapeTick(value, position, index % 2 == 0))
    return tuple(ticks)


def resolve_hud_fov(
    config: Mapping[str, Any],
    frame_size: tuple[int, int],
) -> tuple[float, float]:
    """Resolve the renderer-matched FOV without depending on controller state."""

    width, height = frame_size
    sim = config.get("sim", {})
    if isinstance(sim, Mapping):
        camera = sim.get("camera", {})
        if isinstance(camera, Mapping) and _finite(camera.get("fov_y_deg")):
            vfov = float(camera["fov_y_deg"])
            fy = height / (2.0 * math.tan(math.radians(vfov) * 0.5))
            hfov = math.degrees(2.0 * math.atan(width / (2.0 * fy)))
            return hfov, vfov

    camera = config.get("camera", {})
    if isinstance(camera, Mapping):
        intrinsics = camera.get("intrinsics", {})
        if isinstance(intrinsics, Mapping):
            fov = intrinsics.get("fov_deg", {})
            if isinstance(fov, Mapping) and _finite(fov.get("h")) and _finite(fov.get("v")):
                return float(fov["h"]), float(fov["v"])

    control = config.get("control", {})
    if isinstance(control, Mapping):
        fov = control.get("fov_deg", {})
        if isinstance(fov, Mapping) and _finite(fov.get("h")) and _finite(fov.get("v")):
            return float(fov["h"]), float(fov["v"])
    return 90.0, 60.0


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _clamp_px(value: float, maximum: int) -> int:
    return int(round(max(0.0, min(float(max(maximum - 1, 0)), value))))


def _subpixel_line(
    frame: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw an anti-aliased line without quantizing its position to a pixel."""

    start_fixed = tuple(int(round(value * SUBPIXEL_SCALE)) for value in start)
    end_fixed = tuple(int(round(value * SUBPIXEL_SCALE)) for value in end)
    cv2.line(
        frame,
        start_fixed,
        end_fixed,
        colour,
        thickness,
        cv2.LINE_AA,
        shift=SUBPIXEL_SHIFT,
    )


def _text_box(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int] = GREEN,
    *,
    scale: float = 0.46,
    thickness: int = 1,
    background: tuple[int, int, int] = BLACK,
    min_width_px: int = 0,
) -> None:
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    height, width = frame.shape[:2]
    right_padding = 12
    panel_width = max(text_w + right_padding, int(min_width_px))
    x = max(4, min(width - panel_width - 1, x))
    y = max(text_h + 5, min(height - baseline - 5, y))
    cv2.rectangle(
        frame,
        (x - 4, y - text_h - baseline - 3),
        (x + panel_width, y + baseline + 3),
        background,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), FONT, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def _draw_center_reticle(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    gap = max(5, min(width, height) // 100)
    arm = max(12, min(width, height) // 40)
    for start, end in (
        ((cx - arm, cy), (cx - gap, cy)),
        ((cx + gap, cy), (cx + arm, cy)),
        ((cx, cy - arm), (cx, cy - gap)),
        ((cx, cy + gap), (cx, cy + arm)),
    ):
        cv2.line(frame, start, end, GREEN, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, GREEN, cv2.FILLED, cv2.LINE_AA)


def _draw_attitude(frame: np.ndarray, cam_state: CamState, hfov_deg: float, vfov_deg: float) -> None:
    height, width = frame.shape[:2]
    yaw_deg = math.degrees(float(cam_state.pan)) % 360.0
    pitch_deg = math.degrees(float(cam_state.tilt))
    hfov = max(1.0, float(hfov_deg))
    vfov = max(1.0, float(vfov_deg))

    # The encoded frame already contains DeepStream's health banner. Keep the
    # host-composed heading tape below that immutable row so neither layer
    # obscures the other.
    top_y = max(110, min(132, height // 8))
    margin = 8
    for tick in heading_tape_ticks(yaw_deg, hfov, width, margin_px=margin):
        x = tick.position_px
        length = 22 if tick.major else 13
        _subpixel_line(frame, (x, top_y), (x, top_y - length), GREEN, 2 if tick.major else 1)
        if tick.major:
            label = str(int(round(tick.value_deg)) % 360)
            (tw, _), _base = cv2.getTextSize(label, FONT, 0.4, 1)
            label_x = int(round(x - tw / 2.0))
            cv2.putText(frame, label, (label_x, max(12, top_y - length - 4)), FONT, 0.4, GREEN, 1, cv2.LINE_AA)
    _text_box(
        frame,
        f"{yaw_deg:05.1f}",
        (width // 2 - 14, top_y - 30),
        GREEN,
        scale=0.48,
    )

    right_x = width - 8
    tape_bottom = height - 20
    for tick in elevation_tape_ticks(pitch_deg, vfov, top_y, tape_bottom):
        y = tick.position_px
        length = 30 if tick.major else 18
        _subpixel_line(frame, (right_x, y), (right_x - length, y), GREEN, 2 if tick.major else 1)
        if tick.major:
            label = f"{int(round(tick.value_deg)):+d}"
            (tw, th), _base = cv2.getTextSize(label, FONT, 0.4, 1)
            label_y = int(round(y + th / 2.0))
            cv2.putText(frame, label, (right_x - length - tw - 5, label_y), FONT, 0.4, GREEN, 1, cv2.LINE_AA)
    _text_box(frame, f"{pitch_deg:+.1f}", (width - 72, height // 2), GREEN, scale=0.46)


def _track_center(track: PerceptionTrackV2, width: int, height: int) -> tuple[int, int]:
    return (
        _clamp_px((track.box.x + track.box.w * 0.5) * width, width),
        _clamp_px((track.box.y + track.box.h * 0.5) * height, height),
    )


def _track_rect(track: PerceptionTrackV2, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        _clamp_px(track.box.x * width, width),
        _clamp_px(track.box.y * height, height),
        _clamp_px((track.box.x + track.box.w) * width, width),
        _clamp_px((track.box.y + track.box.h) * height, height),
    )


def _draw_range_indicator(
    frame: np.ndarray,
    track: PerceptionTrackV2,
    assessment: TrackAssessmentV2,
    colour: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _track_rect(track, width, height)
    length = max(5, int(0.24 * max(x2 - x1, y2 - y1)))
    if assessment.distance_src in {"height", "average"}:
        mid = (y1 + y2) // 2
        cv2.line(frame, (x1, mid - length // 2), (x1, mid + length // 2), colour, 2, cv2.LINE_AA)
    if assessment.distance_src in {"width", "average"}:
        mid = (x1 + x2) // 2
        cv2.line(frame, (mid - length // 2, y1), (mid + length // 2, y1), colour, 2, cv2.LINE_AA)


class V2HudRenderer:
    """Stateful non-MPC HUD renderer for the V2 host display."""

    def __init__(
        self,
        *,
        hfov_deg: float,
        vfov_deg: float,
        authority_label: str = "passive",
        trail_length: int = 24,
    ) -> None:
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        authority = str(authority_label or "passive").strip()
        self.authority_label = "SIM-ONLY" if authority.lower() == "sim" else authority
        self._trails: dict[int, Deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=max(2, int(trail_length)))
        )

    def render(
        self,
        frame: np.ndarray,
        *,
        snapshot: Optional[PerceptionSnapshotV2],
        cam_state: Optional[CamState],
        control_cmd: Optional[ControlCmd],
        cam_state_age_s: Optional[float] = None,
        control_age_s: Optional[float] = None,
    ) -> HudRenderReport:
        elements: list[str] = []
        height, width = frame.shape[:2]

        _draw_center_reticle(frame)
        elements.append("center_reticle")

        if cam_state is not None:
            _draw_attitude(frame, cam_state, self.hfov_deg, self.vfov_deg)
            elements.append("attitude")

        selected_track: Optional[PerceptionTrackV2] = None
        selected_assessment: Optional[TrackAssessmentV2] = None
        if snapshot is not None:
            assessment_by_id = {item.track_id: item for item in snapshot.assessments}
            selected_id = None if snapshot.selection is None else snapshot.selection.track_id
            live_ids: set[int] = set()
            for track in snapshot.tracks:
                live_ids.add(track.track_id)
                center = _track_center(track, width, height)
                self._trails[track.track_id].append(center)
                points = np.asarray(self._trails[track.track_id], dtype=np.int32)
                if len(points) >= 2:
                    cv2.polylines(frame, [points], False, (0, 170, 0), 1, cv2.LINE_AA)
                if track.class_id.strip().lower() == "person":
                    x1, y1, x2, y2 = _track_rect(track, width, height)
                    cv2.line(frame, (x1, y1), (x2, y2), GREEN, 1, cv2.LINE_AA)
                    cv2.line(frame, (x1, y2), (x2, y1), GREEN, 1, cv2.LINE_AA)
                assessment = assessment_by_id.get(track.track_id)
                if selected_id == track.track_id:
                    selected_track = track
                    selected_assessment = assessment
            for track_id in tuple(self._trails):
                if track_id not in live_ids:
                    del self._trails[track_id]
            if snapshot.tracks:
                elements.extend(("tracks", "track_history"))

        if selected_track is not None:
            target = _track_center(selected_track, width, height)
            centre = (width // 2, height // 2)
            cue_colour = RED
            if selected_assessment is not None:
                if selected_assessment.threat_level == "benign":
                    cue_colour = GREEN
                elif selected_assessment.threat_level == "suspicious":
                    cue_colour = AMBER
            cv2.line(frame, centre, target, AMBER, 1, cv2.LINE_AA)
            cv2.circle(frame, target, 8, cue_colour, 2, cv2.LINE_AA)
            cv2.circle(frame, target, 2, cue_colour, cv2.FILLED, cv2.LINE_AA)
            elements.append("selection")
            if selected_assessment is not None:
                _draw_range_indicator(frame, selected_track, selected_assessment, cue_colour)
                if _finite(selected_assessment.distance_m):
                    elements.append("range")
                if selected_assessment.threat_level:
                    elements.append("threat")
                if selected_assessment.engagement_rank is not None:
                    elements.append("rank")

        if control_cmd is not None:
            target_colour = GREEN if control_cmd.target_ok else AMBER
            mode = str(control_cmd.controller_mode or "control").upper()
            state = "TRACK" if control_cmd.target_ok else "HOLD"
            if control_cmd.laser_on_target is True:
                parallax_state = "parallax ALIGNED"
            elif control_cmd.laser_on_target is False:
                parallax_state = "parallax OFFSET"
            else:
                parallax_state = "parallax n/a"
            line = (
                f"{self.authority_label} | {state} | {mode} | "
                f"yaw {control_cmd.pan_rate_cmd:+.3f} | pitch {control_cmd.tilt_rate_cmd:+.3f} | "
                f"{parallax_state}"
            )
            _text_box(
                frame,
                line,
                (12, height - 14),
                target_colour,
                scale=0.44,
                min_width_px=480,
            )
            elements.extend(("control_status", "parallax_status"))

            origin = control_cmd.laser_origin_px
            dot = control_cmd.laser_dot_px
            if origin is not None or dot is not None:
                parallax_colour = GREEN if control_cmd.laser_on_target else AMBER
                origin_px = None
                dot_px = None
                if origin is not None and all(_finite(value) for value in origin):
                    origin_px = (_clamp_px(origin[0], width), _clamp_px(origin[1], height))
                    cv2.circle(frame, origin_px, 5, parallax_colour, 2, cv2.LINE_AA)
                if dot is not None and all(_finite(value) for value in dot):
                    dot_px = (_clamp_px(dot[0], width), _clamp_px(dot[1], height))
                    cv2.drawMarker(
                        frame,
                        dot_px,
                        parallax_colour,
                        cv2.MARKER_DIAMOND,
                        12,
                        2,
                        cv2.LINE_AA,
                    )
                if origin_px is not None and dot_px is not None:
                    cv2.arrowedLine(
                        frame,
                        origin_px,
                        dot_px,
                        parallax_colour,
                        1,
                        cv2.LINE_AA,
                        tipLength=0.15,
                    )
                elements.append("parallax_cue")

        freshness: list[str] = []
        if cam_state_age_s is not None:
            freshness.append(f"cam {cam_state_age_s * 1000.0:.0f}ms")
        if control_age_s is not None:
            freshness.append(f"cmd {control_age_s * 1000.0:.0f}ms")
        if freshness:
            _text_box(frame, " | ".join(freshness), (12, height - 42), WHITE, scale=0.38)
            elements.append("freshness")

        # Keep this explicit: MPC term bars are a separate, opt-in diagnostic
        # renderer and must never appear as part of the operational V2 HUD.
        return HudRenderReport(elements=tuple(dict.fromkeys(elements)))
