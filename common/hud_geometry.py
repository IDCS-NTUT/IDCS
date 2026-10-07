"""Display-only geometry/semantics shared by host and panel, no GPU imports."""
import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TapeTick:
    value_deg: float
    position_px: float
    major: bool


def heading_tape_ticks(yaw_deg, hfov_deg, width, *, margin_px=8, step_deg=5.0):
    half = max(float(hfov_deg) * .5, 1e-6)
    usable = max(1., float(width - 2 * margin_px))
    return tuple(TapeTick(i * step_deg, margin_px + usable * (.5 + (i * step_deg - yaw_deg) / (2 * half)), i % 2 == 0)
        for i in range(math.floor((yaw_deg - half) / step_deg), math.ceil((yaw_deg + half) / step_deg) + 1)
        if abs(i * step_deg - yaw_deg) <= half + 1e-9)


def elevation_tape_ticks(pitch_deg, vfov_deg, top_px, bottom_px, *, step_deg=5.0):
    half = max(float(vfov_deg) * .5, 1e-6)
    usable = max(1., float(bottom_px - top_px))
    return tuple(TapeTick(i * step_deg, top_px + usable * (.5 - (i * step_deg - pitch_deg) / (2 * half)), i % 2 == 0)
        for i in range(math.floor((pitch_deg - half) / step_deg), math.ceil((pitch_deg + half) / step_deg) + 1)
        if abs(i * step_deg - pitch_deg) <= half + 1e-9)


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
