"""Display-only geometry/semantics shared by host and panel, no GPU imports."""
import math
from dataclasses import dataclass
from typing import Mapping
from common.ui_style import GREEN, CYAN, AMBER, GRAY


@dataclass(frozen=True)
class TapeTick:
    value_deg: float
    position_px: float
    major: bool


@dataclass(frozen=True)
class ReticleStyle:
    mode: str
    state: str
    rgb: tuple[int, int, int]

    @property
    def label(self):
        return f'{self.mode} {self.state.upper()}'


def tracking_reticle_style(perception, *, fresh=True, mode_hint=None, state_hint=None):
    """Display identity, not motor authority or a claim of target depth.

    V2 has no lifecycle field: empty snapshots cannot identify an ROI mode.
    Explicit hints are only for owners that know that mode/state (ROI trial).
    Compact panel and full V2 payloads share these selection semantics.
    """
    if mode_hint not in (None, 'roi', 'sim', 'detector'):
        raise ValueError('unknown reticle mode')
    if state_hint not in (None, 'idle', 'acquiring', 'tracking', 'uncertain', 'lost'):
        raise ValueError('unknown reticle state')
    data = perception or {}
    tracks = data.get('tracks') or ()
    selected_id = data.get('selected_id', (data.get('selection') or {}).get('track_id'))
    selected = next((t for t in tracks if t.get('track_id') == selected_id), None)
    visual = selected or next((t for t in tracks if t.get('class_id') == 'visual'), {})
    mode = ('ROI' if visual.get('class_id') == 'visual' or mode_hint == 'roi' else
            'SIM' if mode_hint == 'sim' else
            'YOLO' if selected is not None or mode_hint == 'detector' else 'TRACK')
    observed = selected is not None and selected.get('missed_frames', 0) == 0
    if not fresh:
        state = 'stale'
    elif state_hint is not None:
        state = 'uncertain' if state_hint == 'tracking' and not observed else state_hint
    else:
        state = 'tracking' if observed else 'uncertain' if selected is not None else 'idle'
    rgb = (CYAN if mode == 'ROI' else GREEN) if state == 'tracking' else (
        AMBER if state in ('acquiring', 'uncertain') else GRAY)
    return ReticleStyle(mode, state, rgb)


def perception_progresses(previous, current):
    """Duplicate/backward observations cannot renew a display receive watchdog."""
    if previous is None:
        return True
    a, b = previous.frame, current.frame
    if (a.source_clock_domain, a.tracker_session) != (b.source_clock_domain, b.tracker_session):
        return True
    return current.sequence > previous.sequence and b.frame_id > a.frame_id


def _tape_focal(fov_deg, size, angle_deg, step_deg):
    if not all(math.isfinite(v) for v in (fov_deg, size, angle_deg, step_deg)):
        raise ValueError('finite tape geometry required')
    if not 0 < fov_deg < 180 or size <= 0 or step_deg <= 0:
        raise ValueError('positive size/step and FOV between 0 and 180 required')
    return size / (2 * math.tan(math.radians(fov_deg) / 2))


def heading_tape_ticks(yaw_deg, hfov_deg, width, *, margin_px=8, right_px=None, step_deg=5.0):
    """Project angular offsets onto the full image, then clip the edge strip.

    Trimming a strip must never shrink the FOV or shift the principal point.
    Heading is a camera-relative angular scale, not a world-horizon overlay.
    """
    focal = _tape_focal(hfov_deg, width, yaw_deg, step_deg)
    right = width-margin_px if right_px is None else min(right_px, width-margin_px)
    half = hfov_deg / 2
    ticks = []
    for i in range(math.ceil((yaw_deg-half)/step_deg), math.floor((yaw_deg+half)/step_deg)+1):
        x = width/2 + focal*math.tan(math.radians(i*step_deg-yaw_deg))
        if margin_px <= x <= right:
            ticks.append(TapeTick(i*step_deg, x, i % 2 == 0))
    return tuple(ticks)


def elevation_tape_ticks(pitch_deg, vfov_deg, top_px, bottom_px, *, frame_height, step_deg=5.0):
    """Positive elevation is up; zero aligns with the central-plane horizon.

    Use the full frame's principal point/focal length, not the strip midpoint.
    This is rectilinear, zero-roll geometry; it does not calibrate lens distortion.
    """
    focal = _tape_focal(vfov_deg, frame_height, pitch_deg, step_deg)
    half = vfov_deg / 2
    ticks = []
    for i in range(math.ceil((pitch_deg-half)/step_deg), math.floor((pitch_deg+half)/step_deg)+1):
        y = frame_height/2 - focal*math.tan(math.radians(i*step_deg-pitch_deg))
        if top_px <= y <= bottom_px:
            ticks.append(TapeTick(i*step_deg, y, i % 2 == 0))
    return tuple(ticks)


def feedforward_state(diagnostics: Mapping | None, age_s: float | None):
    if diagnostics is None:
        return 'unavailable', None, None
    axes = [diagnostics.get(name) or {} for name in ('yaw', 'pitch')]
    yaw, pitch = [axis.get('feedforward_term_rad_s') for axis in axes]
    if age_s is None or age_s > .25:
        state = 'stale'
    elif diagnostics.get('reason') != 'tracking' or not any(a.get('estimator_enabled') for a in axes) or yaw is None or pitch is None:
        state = 'inhibited'
    else:
        limited = any(a.get('desired_rate_pre_limit_rad_s') is not None and a.get('desired_rate_post_limit_rad_s') is not None
            and abs(a['desired_rate_pre_limit_rad_s'] - a['desired_rate_post_limit_rad_s']) > 1e-6 for a in axes)
        state = 'limited' if limited else 'active'
    return state, yaw, pitch


def resolve_hud_fov(config, frame_size):
    """Same configuration precedence as the PC; no OpenCV dependency."""
    width, height = frame_size
    camera = (config.get('sim') or {}).get('camera') or {}
    if camera.get('fov_y_deg') is not None:
        vfov = float(camera['fov_y_deg'])
        hfov = camera.get('fov_x_deg')
        if hfov is None:
            hfov = math.degrees(2 * math.atan(width / height * math.tan(math.radians(vfov) / 2)))
        return float(hfov), vfov
    for values in (((config.get('camera') or {}).get('intrinsics') or {}).get('fov_deg'),
                   (config.get('control') or {}).get('fov_deg')):
        if values and values.get('h') is not None and values.get('v') is not None:
            return float(values['h']), float(values['v'])
    return 90., 60.
