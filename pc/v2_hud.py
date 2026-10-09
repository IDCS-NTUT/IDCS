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
from common.schemas import CamState, ControlCmd, ControlDiagnostics
from common.hud_geometry import TapeTick, heading_tape_ticks, elevation_tape_ticks, feedforward_state
from common.hud_geometry import ReticleStyle, tracking_reticle_style
from common import ui_style as theme
from common.ui_style import HudLayout
from pc.ui_text import text as ui_text, panel as ui_panel, put_text, get_text_size


GREEN = theme.bgr(theme.GREEN)
AMBER = theme.bgr(theme.AMBER)
RED = theme.bgr(theme.RED)
WHITE = theme.bgr(theme.WHITE)
CYAN = theme.bgr(theme.CYAN)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SUBPIXEL_SHIFT = 8
SUBPIXEL_SCALE = 1 << SUBPIXEL_SHIFT


@dataclass(frozen=True)
class HudRenderReport:
    """Exact overlay inventory drawn into one frame, used by visual QA."""

    elements: tuple[str, ...]
    feedforward_state: str = "unavailable"
    feedforward_yaw_rad_s: Optional[float] = None
    feedforward_pitch_rad_s: Optional[float] = None
    reticle_mode: str = 'TRACK'
    reticle_state: str = 'idle'


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
            if _finite(camera.get("fov_x_deg")):
                # Independent axes (the calibrated wide-angle camera is not a
                # pinhole aspect-ratio derivation); the simulator uses both.
                return float(camera["fov_x_deg"]), vfov
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
    height, width = frame.shape[:2]
    size = theme.pixel_size(scale*28, height)
    (text_w, text_h), baseline = get_text_size(text,FONT,scale,thickness,frame_height=height)
    pad = theme.pixel_size(theme.PADDING_PX,height)
    panel_width = min(width-2*pad,max(text_w,int(min_width_px)))
    x = max(pad, min(width-panel_width-pad, x))
    y = max(text_h+pad, min(height-baseline-pad, y))
    ui_text(frame,text,(x,y),colour,size=size,background=True,
            min_width=panel_width,max_width=width-x-pad)


def _draw_center_reticle(frame: np.ndarray, style: ReticleStyle) -> None:
    height, width = frame.shape[:2]
    layout = HudLayout(width,height)
    s = layout.scale
    cx, cy = width // 2, height // 2
    gap = max(1,round(6*s))
    arm = max(3,round(22*s))
    colour = style.rgb[::-1]  # OpenCV uses BGR; shared semantics are RGB.
    for start, end in (
        ((cx - arm, cy), (cx - gap, cy)),
        ((cx + gap, cy), (cx + arm, cy)),
        ((cx, cy - arm), (cx, cy - gap)),
        ((cx, cy + gap), (cx, cy + arm)),
    ):
        cv2.line(frame, start, end, colour, theme.pixel_size(theme.STROKE_PX,height), cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), theme.pixel_size(2,height), colour, cv2.FILLED, cv2.LINE_AA)
    badge = layout.mode_badge
    ui_panel(frame,(badge.x,badge.y,badge.width,badge.height))
    ui_text(frame,style.label,(round(badge.x+4*s),round(badge.y+18*s)),colour,
            size=theme.pixel_size(theme.TEXT_PX,height),max_width=badge.width-8*s)


def _draw_feedforward_indicator(
    frame: np.ndarray,
    diagnostics: Optional[ControlDiagnostics],
    diagnostics_age_s: Optional[float],
) -> tuple[str, Optional[float], Optional[float]]:
    """Draw compact signed feedforward attribution without implying authority."""

    state, yaw, pitch = feedforward_state(
        None if diagnostics is None else diagnostics.model_dump(mode='json'),
        0.0 if diagnostics_age_s is None else diagnostics_age_s)
    colour = WHITE if state == 'unavailable' else CYAN if state == 'active' else AMBER

    yaw_text = "--" if yaw is None else f"{yaw:+.3f}"
    pitch_text = "--" if pitch is None else f"{pitch:+.3f}"
    label = f"FF {state.upper()} | Y {yaw_text} P {pitch_text} rad/s"

    height, width = frame.shape[:2]
    layout = HudLayout(width,height)
    s = layout.scale
    region = layout.feedforward
    x0, y0 = round(region.x),round(region.y+15*s)
    ui_panel(frame,(region.x-3*s,region.y,region.width,region.height))
    ui_text(frame,label,(x0,y0),colour,size=theme.pixel_size(theme.TEXT_PX,height),max_width=region.width-8*s)

    # Two centered bars make sign and relative contribution visible at a
    # glance. Values beyond the nominal 0.5 rad/s display scale are clipped;
    # the exact unclipped values remain in the label.
    bar_x0, bar_x1 = x0, round(x0 + 330*s)
    centre_x = (bar_x0 + bar_x1) // 2
    cv2.line(frame, (bar_x0, round(y0+14*s)), (bar_x1, round(y0+14*s)), (77,77,77), 1, cv2.LINE_AA)
    cv2.line(frame, (bar_x0, round(y0+28*s)), (bar_x1, round(y0+28*s)), (77,77,77), 1, cv2.LINE_AA)
    cv2.line(frame, (centre_x, round(y0+10*s)), (centre_x, round(y0+32*s)), WHITE, 1, cv2.LINE_AA)
    for value, y in ((yaw, round(y0+14*s)), (pitch, round(y0+28*s))):
        if value is None:
            continue
        end_x = int(round(centre_x + max(-1.0, min(1.0, value / 0.5)) * (bar_x1 - centre_x)))
        cv2.line(frame, (centre_x, y), (end_x, y), colour, max(1,round(3*s)), cv2.LINE_AA)
    return state, yaw, pitch


def _draw_attitude(frame: np.ndarray, cam_state: CamState, hfov_deg: float, vfov_deg: float) -> None:
    height, width = frame.shape[:2]
    # Relative to the mount's home, as the camera view is: in HIL the bridge
    # reports absolute encoder angles (bench home ~ -2.4 / 1.125 rad), which
    # put the tapes 137 / 64 deg off the rendered view.
    yaw_deg = math.degrees(float(cam_state.pan) - float(cam_state.home_pan or 0.0)) % 360.0
    pitch_deg = math.degrees(float(cam_state.tilt) - float(cam_state.home_tilt or 0.0))
    hfov = max(1.0, float(hfov_deg))
    vfov = max(1.0, float(vfov_deg))

    # Heading tape at the top edge; DeepStream's status line is burned into
    # the encoded frame just below it (y ~58 px), so neither layer covers the
    # other.
    layout = HudLayout(width,height)
    s = layout.scale
    top_y = layout.heading_y
    margin = round(8*s)
    for tick in heading_tape_ticks(yaw_deg, hfov, width, margin_px=margin):
        x = tick.position_px
        length = (18 if tick.major else 9)*s
        _subpixel_line(frame, (x, top_y), (x, top_y - length), GREEN, theme.pixel_size(theme.STROKE_PX,height))
        if tick.major:
            label = str(int(round(tick.value_deg)) % 360)
            (tw, _), _base = get_text_size(label, FONT, 0.4, 1,frame_height=height)
            label_x = int(round(x - tw / 2.0))
            put_text(frame, label, (label_x, round(123*s)), FONT, 0.4, GREEN)
    _text_box(
        frame,
        f"Y {yaw_deg:05.1f}",
        (round(width/2-28*s), round(124*s)),
        GREEN,
        scale=0.43,
    )

    right_x = width - 8*s
    for tick in elevation_tape_ticks(pitch_deg, vfov, layout.elevation_top, layout.elevation_bottom):
        y = tick.position_px
        length = (30 if tick.major else 18)*s
        _subpixel_line(frame, (right_x, y), (right_x - length, y), GREEN, theme.pixel_size(theme.STROKE_PX,height))
        if tick.major:
            label = f"{int(round(tick.value_deg)):+d}"
            (tw, th), _base = get_text_size(label, FONT, 0.4, 1,frame_height=height)
            label_y = int(round(y + th / 2.0))
            put_text(frame, label, (round(width-89*s), label_y), FONT, 0.4, GREEN)
    _text_box(frame, f"P {pitch_deg:+.1f}", (round(width-89*s), height//2), GREEN, scale=0.43)


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


def _draw_diagnostics_status(
    frame: np.ndarray,
    diagnostics: ControlDiagnostics,
    diagnostics_age_s: Optional[float],
    authority_label: str,
    elements: list[str],
    *,
    allow_aim: bool = True,
) -> None:
    """Status line and aim cue from controller diagnostics (no ControlCmd).

    The video controller publishes ``ControlIntent`` to the bridge and
    ``ControlDiagnostics`` for displays; this draws the same operator cues the
    ControlCmd path draws: tracking state, commanded rates, and where the
    controller is aiming relative to the selected target.
    """
    height, width = frame.shape[:2]
    stale = diagnostics_age_s is not None and diagnostics_age_s > 0.25
    tracking = diagnostics.reason == "tracking" and not stale
    colour = theme.bgr(theme.control_rgb(diagnostics.reason,stale=stale))
    mode = "PID+FF" if diagnostics.yaw.estimator_enabled or diagnostics.pitch.estimator_enabled else "PID"
    aim = target = None
    if allow_aim and tracking and diagnostics.aim_reference_norm is not None and diagnostics.target_center_norm is not None:
        aim = (diagnostics.aim_reference_norm[0] * width, diagnostics.aim_reference_norm[1] * height)
        target = (diagnostics.target_center_norm[0] * width, diagnostics.target_center_norm[1] * height)
    offset_px = None
    if aim is not None and target is not None and all(_finite(v) for v in (*aim, *target)):
        offset_px = math.hypot(target[0] - aim[0], target[1] - aim[1])
    line=theme.control_summary(authority_label,diagnostics.reason,mode,
        diagnostics.yaw.final_rate_rad_s,diagnostics.pitch.final_rate_rad_s,stale=stale)
    layout = HudLayout(width,height)
    _text_box(frame, line, (round(layout.status.x),round(layout.status.y+35*layout.scale)), colour, scale=0.46)
    _text_box(frame,theme.parallax_summary(offset_px),
              (round(layout.status.x),round(layout.status.y+55*layout.scale)),WHITE)
    elements.extend(("control_status", "parallax_status"))
    if offset_px is not None and not stale:
        aim_px = (_clamp_px(aim[0], width), _clamp_px(aim[1], height))
        target_px = (_clamp_px(target[0], width), _clamp_px(target[1], height))
        cv2.drawMarker(frame, aim_px, colour, cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
        cv2.arrowedLine(frame, aim_px, target_px, colour, 1, cv2.LINE_AA, tipLength=0.15)
        elements.append("parallax_cue")


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
        control_diagnostics: Optional[ControlDiagnostics] = None,
        cam_state_age_s: Optional[float] = None,
        control_age_s: Optional[float] = None,
        diagnostics_age_s: Optional[float] = None,
        perception_age_s: Optional[float] = None,
    ) -> HudRenderReport:
        elements: list[str] = []
        height, width = frame.shape[:2]

        fresh = snapshot is not None and (perception_age_s is None or 0 <= perception_age_s <= .25)
        style = tracking_reticle_style(
            None if snapshot is None else snapshot.model_dump(mode='json'), fresh=fresh,
            mode_hint='sim' if self.authority_label == 'SIM-ONLY' else None)
        if not fresh:
            snapshot = None
            self._trails.clear()
        allow_aim = style.state == 'tracking' and style.mode != 'ROI'
        elements.extend(("center_reticle", "tracking_mode"))

        feedforward_state, feedforward_yaw, feedforward_pitch = _draw_feedforward_indicator(
            frame,
            control_diagnostics,
            diagnostics_age_s,
        )
        elements.append("feedforward_indicator")

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
                if selected_id == track.track_id and style.state == 'tracking':
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
            s=height/720
            cue_colour = RED
            if selected_assessment is not None:
                if selected_assessment.threat_level == "benign":
                    cue_colour = GREEN
                elif selected_assessment.threat_level == "suspicious":
                    cue_colour = AMBER
            cv2.line(frame, centre, target, AMBER, theme.pixel_size(1,height), cv2.LINE_AA)
            cv2.circle(frame, target, theme.pixel_size(8,height), cue_colour, theme.pixel_size(2,height), cv2.LINE_AA)
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
                (round(12*height/720), round(height-84*height/720)),
                target_colour,
                scale=0.44,
            )
            elements.extend(("control_status", "parallax_status"))

            origin = control_cmd.laser_origin_px
            dot = control_cmd.laser_dot_px
            if allow_aim and control_cmd.target_ok and (control_age_s is None or 0 <= control_age_s <= .25) and (origin is not None or dot is not None):
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

        elif control_diagnostics is not None:
            _draw_diagnostics_status(
                frame, control_diagnostics, diagnostics_age_s, self.authority_label, elements,
                allow_aim=allow_aim,
            )

        freshness: list[str] = []
        if cam_state_age_s is not None:
            freshness.append(f"cam {cam_state_age_s * 1000.0:.0f}ms")
        if control_age_s is not None:
            freshness.append(f"cmd {control_age_s * 1000.0:.0f}ms")
        elif control_diagnostics is not None and diagnostics_age_s is not None:
            freshness.append(f"ctl {diagnostics_age_s * 1000.0:.0f}ms")
        if freshness:
            _text_box(frame, "rx age; transit unknown: " + " | ".join(freshness),
                      (round(12*height/720), round(height-44*height/720)), WHITE, scale=0.39)
            elements.append("freshness")

        s = height/720
        info=theme.source_summary(self.authority_label,cam_state is not None,
            None if selected_track is None else selected_track.track_id,
            None if selected_assessment is None else selected_assessment.model_dump(mode='json'))
        _text_box(frame, info, (round(12*s),round(height-104*s)), WHITE)
        elements.append('source_status')
        if control_cmd is not None:
            _text_box(frame, 'Parallax: geometric aim cue only', (round(12*s),round(height-64*s)),WHITE,scale=.46)
        _text_box(frame,'HUD latest telemetry; video frame join unavailable',
                  (round(12*s),round(height-24*s)),WHITE,scale=.39)
        elements.append('freshness')

        # Keep this explicit: MPC term bars are a separate, opt-in diagnostic
        # renderer and must never appear as part of the operational V2 HUD.
        # Draw last: selection rays must not overwrite the mode-coloured dot.
        _draw_center_reticle(frame, style)
        return HudRenderReport(
            elements=tuple(dict.fromkeys(elements)),
            feedforward_state=feedforward_state,
            feedforward_yaw_rad_s=feedforward_yaw,
            feedforward_pitch_rad_s=feedforward_pitch,
            reticle_mode=style.mode,
            reticle_state=style.state,
        )
