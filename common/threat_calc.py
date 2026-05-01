"""Threat evaluation calculation utilities.

Provides helpers for computing distance-to-asset, zone membership,
distance rates, velocity, bounding box metrics, and zone dwell time.

All calculations use XY horizontal plane (ignoring Z/height for v1).

Key metrics computed:
- center_x, center_y: Bounding box center in image/world space
- bbox_width, bbox_height: Bounding box dimensions
- velocity_x, velocity_y: Target velocity in meters/second or pixels/second
- confidence: Detection confidence score [0, 1]
- distance_to_asset: Euclidean distance from target to defended asset (meters)
- distance_rate_to_asset: Rate of distance change (m/s, negative=approaching)
- zone_id: Current zone membership (critical/restricted/warning/normal)
- time_inside_zone: Cumulative time spent in current zone (seconds)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "compute_distance_to_asset",
    "compute_distance_rate",
    "compute_radial_closing_speed",
    "compute_breakthrough_time",
    "compute_zone_feature_vector",
    "estimate_axis_slew_time",
    "estimate_slew_time",
    "estimate_track_lock_time",
    "estimate_uncertainty_time",
    "estimate_time_to_engage",
    "compute_velocity",
    "compute_bbox_center",
    "validate_bbox",
    "validate_confidence",
    "get_zone_membership",
    "get_zone_id_for_distance",
    "TargetThreatState",
    "parse_zone_config",
    "validate_asset_position",
]


def compute_distance_to_asset(
    target_xy: Tuple[float, float],
    asset_xy: Tuple[float, float],
) -> float:
    """Compute Euclidean distance from target to asset in XY plane.

    Args:
        target_xy: (x, y) target position in world meters
        asset_xy: (x, y) asset position in world meters

    Returns:
        Distance in meters (always >= 0)

    Raises:
        ValueError: if coordinates contain NaN or infinite values
    """
    tx, ty = float(target_xy[0]), float(target_xy[1])
    ax, ay = float(asset_xy[0]), float(asset_xy[1])

    if not all(math.isfinite(v) for v in (tx, ty, ax, ay)):
        raise ValueError("All coordinates must be finite")

    dx = tx - ax
    dy = ty - ay
    distance = math.sqrt(dx * dx + dy * dy)
    return distance


def compute_distance_rate(
    current_distance_m: float,
    previous_distance_m: float,
    dt_s: float,
) -> float:
    """Compute rate of distance change (velocity towards/away from asset).

    Args:
        current_distance_m: Current distance to asset (meters)
        previous_distance_m: Previous frame's distance (meters)
        dt_s: Time elapsed since previous frame (seconds)

    Returns:
        Distance rate in meters/second
        - Negative: target approaching asset
        - Positive: target moving away from asset
        - Zero: no distance change

    Raises:
        ValueError: if inputs are non-finite or dt_s <= 0
    """
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    if not all(math.isfinite(v) for v in (current_distance_m, previous_distance_m)):
        raise ValueError("Distances must be finite")

    delta_distance = current_distance_m - previous_distance_m
    rate = delta_distance / dt_s
    return rate


def compute_radial_closing_speed(
    target_xy: Tuple[float, float],
    velocity_xy: Tuple[float, float],
    asset_xy: Tuple[float, float],
) -> float:
    """Compute positive closing speed toward the defended asset.

    Args:
        target_xy: Target position in meters.
        velocity_xy: Target planar velocity in meters/second.
        asset_xy: Asset position in meters.

    Returns:
        Positive scalar closing speed in meters/second. Receding, stationary,
        and purely lateral motion return 0.
    """
    tx, ty = float(target_xy[0]), float(target_xy[1])
    vx, vy = float(velocity_xy[0]), float(velocity_xy[1])
    ax, ay = float(asset_xy[0]), float(asset_xy[1])

    if not all(math.isfinite(v) for v in (tx, ty, vx, vy, ax, ay)):
        raise ValueError("All coordinates and velocities must be finite")

    rel_x = ax - tx
    rel_y = ay - ty
    distance = math.hypot(rel_x, rel_y)
    if distance <= 1e-9:
        return 0.0

    unit_x = rel_x / distance
    unit_y = rel_y / distance
    closing = vx * unit_x + vy * unit_y
    return max(0.0, closing)


def compute_breakthrough_time(
    distance_to_asset_m: float,
    radial_closing_speed_m_s: float,
    *,
    min_closing_speed_m_s: float = 1e-3,
) -> float:
    """Estimate time until breakthrough.

    A non-positive closing speed is treated as not currently threatening the
    asset, which yields ``math.inf`` for the breakthrough horizon.
    """
    distance = float(distance_to_asset_m)
    closing = float(radial_closing_speed_m_s)

    if math.isnan(distance) or distance < 0.0:
        raise ValueError("distance_to_asset_m must be non-negative and not NaN")
    if not math.isfinite(closing):
        raise ValueError("radial_closing_speed_m_s must be finite")
    if math.isinf(distance):
        return math.inf

    if closing <= min_closing_speed_m_s:
        return math.inf
    return distance / closing


def compute_zone_feature_vector(
    distance_m: float,
    zone_radii: Optional[Dict[str, float]],
) -> Tuple[float, float, float, float]:
    """Encode threat-evaluation zone context into fixed structured features."""
    if zone_radii is None:
        return (0.0, 0.0, 0.0, 0.0)

    distance = float(distance_m)
    if not math.isfinite(distance) or distance < 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    warning_radius = float(zone_radii.get("warning", 0.0) or 0.0)
    restricted_radius = float(zone_radii.get("restricted", 0.0) or 0.0)
    critical_radius = float(zone_radii.get("critical", 0.0) or 0.0)

    in_warning = 1.0 if warning_radius > 0.0 and distance <= warning_radius else 0.0
    in_restricted = 1.0 if restricted_radius > 0.0 and distance <= restricted_radius else 0.0
    in_critical = 1.0 if critical_radius > 0.0 and distance <= critical_radius else 0.0

    outer_radius = warning_radius
    if outer_radius <= 0.0:
        valid_radii = [float(radius) for radius in zone_radii.values() if float(radius) > 0.0]
        outer_radius = max(valid_radii, default=0.0)
    zone_progress = (
        float(np.clip(1.0 - (distance / outer_radius), 0.0, 1.0))
        if outer_radius > 0.0
        else 0.0
    )
    return (in_warning, in_restricted, in_critical, zone_progress)


def estimate_axis_slew_time(
    angle_error_rad: float,
    rate_limit_rad_s: float,
    accel_limit_rad_s2: Optional[float] = None,
    *,
    current_rate_rad_s: float = 0.0,
) -> float:
    """Estimate time to slew one axis under rate and optional accel limits."""
    error = abs(float(angle_error_rad))
    rate_limit = abs(float(rate_limit_rad_s))
    current_rate = abs(float(current_rate_rad_s))

    if not math.isfinite(error) or not math.isfinite(rate_limit) or not math.isfinite(current_rate):
        raise ValueError("Slew inputs must be finite")
    if rate_limit <= 0.0:
        return math.inf if error > 0.0 else 0.0
    if error <= 1e-9:
        return 0.0

    accel = None if accel_limit_rad_s2 is None else abs(float(accel_limit_rad_s2))
    if accel is None or accel <= 1e-6:
        return error / rate_limit

    current_rate = min(current_rate, rate_limit)
    accel_time = max(0.0, (rate_limit - current_rate) / accel)
    accel_distance = current_rate * accel_time + 0.5 * accel * accel_time * accel_time
    decel_distance = (rate_limit * rate_limit) / (2.0 * accel)

    if error >= accel_distance + decel_distance:
        cruise_distance = error - accel_distance - decel_distance
        cruise_time = cruise_distance / rate_limit
        return accel_time + cruise_time + (rate_limit / accel)

    # Triangular profile: accelerate from current_rate, then decelerate.
    peak_rate_sq = max(0.0, accel * error + 0.5 * current_rate * current_rate)
    peak_rate = min(rate_limit, math.sqrt(peak_rate_sq))
    accel_up_time = max(0.0, (peak_rate - current_rate) / accel)
    decel_time = peak_rate / accel
    return accel_up_time + decel_time


def estimate_slew_time(
    yaw_error_rad: float,
    pitch_error_rad: float,
    yaw_rate_limit_rad_s: float,
    pitch_rate_limit_rad_s: float,
    yaw_accel_limit_rad_s2: Optional[float] = None,
    pitch_accel_limit_rad_s2: Optional[float] = None,
    *,
    current_yaw_rate_rad_s: float = 0.0,
    current_pitch_rate_rad_s: float = 0.0,
    settle_margin_s: float = 0.0,
) -> float:
    """Estimate two-axis slew time using the slower axis plus settle margin."""
    yaw_time = estimate_axis_slew_time(
        yaw_error_rad,
        yaw_rate_limit_rad_s,
        yaw_accel_limit_rad_s2,
        current_rate_rad_s=current_yaw_rate_rad_s,
    )
    pitch_time = estimate_axis_slew_time(
        pitch_error_rad,
        pitch_rate_limit_rad_s,
        pitch_accel_limit_rad_s2,
        current_rate_rad_s=current_pitch_rate_rad_s,
    )
    base = max(yaw_time, pitch_time)
    if not math.isfinite(base):
        return base
    return max(0.0, base + float(settle_margin_s))


def estimate_track_lock_time(
    *,
    tracker_mode: Optional[str],
    confidence: float,
    track_observations: int,
    base_track_lock_s: float,
    search_track_lock_s: float,
    recover_track_lock_s: float,
    low_conf_threshold: float,
    low_conf_penalty_s: float,
    min_track_observations: int,
    low_continuity_penalty_s: float,
) -> float:
    """Estimate time spent reacquiring and stabilizing the track."""
    conf = validate_confidence(confidence)
    observations = max(0, int(track_observations))

    mode = (tracker_mode or "track").strip().lower()
    if mode == "recover":
        lock_time = float(recover_track_lock_s)
    elif mode in {"search", "slew"}:
        lock_time = float(search_track_lock_s)
    else:
        lock_time = float(base_track_lock_s)

    if conf < float(low_conf_threshold):
        ratio = (float(low_conf_threshold) - conf) / max(float(low_conf_threshold), 1e-6)
        lock_time += max(0.0, ratio) * float(low_conf_penalty_s)

    if observations < int(min_track_observations):
        deficit = int(min_track_observations) - observations
        lock_time += deficit * float(low_continuity_penalty_s)

    return max(0.0, lock_time)


def estimate_uncertainty_time(
    *,
    range_source: Optional[str],
    predictive_only: bool,
    missing_range_penalty_s: float,
    predictive_penalty_s: float,
) -> float:
    """Estimate extra time reserved for uncertain targeting conditions."""
    penalty = 0.0
    if predictive_only:
        penalty += float(predictive_penalty_s)
    if range_source not in {"known_size", "height", "width", "average", "default"}:
        penalty += float(missing_range_penalty_s)
    return max(0.0, penalty)


def estimate_time_to_engage(
    *,
    yaw_error_rad: float,
    pitch_error_rad: float,
    yaw_rate_limit_rad_s: float,
    pitch_rate_limit_rad_s: float,
    yaw_accel_limit_rad_s2: Optional[float],
    pitch_accel_limit_rad_s2: Optional[float],
    current_yaw_rate_rad_s: float,
    current_pitch_rate_rad_s: float,
    tracker_mode: Optional[str],
    confidence: float,
    track_observations: int,
    range_source: Optional[str],
    predictive_only: bool,
    base_track_lock_s: float,
    search_track_lock_s: float,
    recover_track_lock_s: float,
    low_conf_threshold: float,
    low_conf_penalty_s: float,
    min_track_observations: int,
    low_continuity_penalty_s: float,
    missing_range_penalty_s: float,
    predictive_penalty_s: float,
    effect_time_s: float,
    confirm_time_s: float,
    settle_margin_s: float = 0.0,
) -> float:
    """Compose deterministic timing estimates into one engagement duration."""
    slew_time = estimate_slew_time(
        yaw_error_rad,
        pitch_error_rad,
        yaw_rate_limit_rad_s,
        pitch_rate_limit_rad_s,
        yaw_accel_limit_rad_s2,
        pitch_accel_limit_rad_s2,
        current_yaw_rate_rad_s=current_yaw_rate_rad_s,
        current_pitch_rate_rad_s=current_pitch_rate_rad_s,
        settle_margin_s=settle_margin_s,
    )
    lock_time = estimate_track_lock_time(
        tracker_mode=tracker_mode,
        confidence=confidence,
        track_observations=track_observations,
        base_track_lock_s=base_track_lock_s,
        search_track_lock_s=search_track_lock_s,
        recover_track_lock_s=recover_track_lock_s,
        low_conf_threshold=low_conf_threshold,
        low_conf_penalty_s=low_conf_penalty_s,
        min_track_observations=min_track_observations,
        low_continuity_penalty_s=low_continuity_penalty_s,
    )
    uncertainty_time = estimate_uncertainty_time(
        range_source=range_source,
        predictive_only=predictive_only,
        missing_range_penalty_s=missing_range_penalty_s,
        predictive_penalty_s=predictive_penalty_s,
    )
    total = slew_time + lock_time + float(effect_time_s) + float(confirm_time_s) + uncertainty_time
    return max(0.0, total)


def get_zone_id_for_distance(
    distance_m: float,
    zone_radii: Dict[str, float],
) -> str:
    """Determine zone membership based on distance and configured radii.

    Zone assignment is based on distance tiers. A target belongs to the
    smallest zone whose radius is >= distance. If distance exceeds all zone
    radii, the zone is "normal" (outside all zones).

    Args:
        distance_m: Distance to asset in meters (>= 0)
        zone_radii: Dict mapping zone_id -> radius_m
            Example: {"critical": 5.0, "restricted": 10.0, "warning": 20.0}

    Returns:
        Zone ID string. If distance falls in multiple zones, the innermost
        (smallest radius) zone is returned. If no zones match, returns "normal".

    Raises:
        ValueError: if distance_m is negative or non-finite
    """
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")

    if not math.isfinite(distance_m):
        raise ValueError("distance_m must be finite")

    # Sort zones by radius (ascending) to assign innermost zone
    sorted_zones = sorted(zone_radii.items(), key=lambda x: x[1])

    for zone_id, radius_m in sorted_zones:
        if distance_m <= radius_m:
            return zone_id

    return "normal"


def get_zone_membership(
    distance_m: float,
    zone_radii: Dict[str, float],
) -> Dict[str, bool]:
    """Compute boolean zone membership for all configured zones.

    Args:
        distance_m: Distance to asset in meters
        zone_radii: Dict mapping zone_id -> radius_m

    Returns:
        Dict mapping zone_id -> is_inside (bool)
        Example output: {"critical": True, "restricted": True, "warning": True}

    Raises:
        ValueError: if distance_m is invalid
    """
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")

    if not math.isfinite(distance_m):
        raise ValueError("distance_m must be finite")

    membership = {}
    for zone_id, radius_m in zone_radii.items():
        membership[zone_id] = distance_m <= radius_m

    return membership


def parse_zone_config(
    zone_config: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Extract zone radii from config dict.

    Args:
        zone_config: Dict like {"critical": {"type": "circle", "radius_m": 5.0}, ...}

    Returns:
        Dict mapping zone_id -> radius_m
        Example: {"critical": 5.0, "restricted": 10.0, "warning": 20.0}

    Raises:
        ValueError: if required fields are missing or invalid
    """
    radii = {}

    for zone_id, zone_spec in zone_config.items():
        if not isinstance(zone_spec, dict):
            raise ValueError(f"Zone '{zone_id}' spec must be a dict")

        if "radius_m" not in zone_spec:
            raise ValueError(f"Zone '{zone_id}' missing required 'radius_m'")

        try:
            radius = float(zone_spec["radius_m"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Zone '{zone_id}' radius_m must be numeric: {e}")

        if radius < 0:
            raise ValueError(f"Zone '{zone_id}' radius_m must be non-negative")

        radii[zone_id] = radius

    return radii


def validate_asset_position(asset_pos: Tuple[float, float]) -> None:
    """Validate asset position tuple.

    Args:
        asset_pos: (x, y) position in world meters

    Raises:
        ValueError: if position is invalid (non-tuple, non-numeric, non-finite)
    """
    if not isinstance(asset_pos, (tuple, list)):
        raise ValueError("Asset position must be tuple or list")

    if len(asset_pos) < 2:
        raise ValueError("Asset position must have at least 2 elements (x, y)")

    try:
        x, y = float(asset_pos[0]), float(asset_pos[1])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Asset position coordinates must be numeric: {e}")

    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Asset position coordinates must be finite")


def compute_bbox_center(
    x: float,
    y: float,
    width: float,
    height: float,
) -> Tuple[float, float]:
    """Compute bounding box center from position and dimensions.

    Args:
        x: Left edge coordinate (image or world space)
        y: Top edge coordinate (image or world space)
        width: Bounding box width
        height: Bounding box height

    Returns:
        (center_x, center_y) tuple

    Raises:
        ValueError: if inputs are non-finite
    """
    x_val = float(x)
    y_val = float(y)
    w_val = float(width)
    h_val = float(height)

    if not all(math.isfinite(v) for v in (x_val, y_val, w_val, h_val)):
        raise ValueError("All bbox parameters must be finite")

    center_x = x_val + w_val / 2.0
    center_y = y_val + h_val / 2.0
    return center_x, center_y


def validate_bbox(
    x: float,
    y: float,
    width: float,
    height: float,
) -> Tuple[float, float, float, float]:
    """Validate and normalize bounding box parameters.

    Args:
        x: Left edge
        y: Top edge
        width: Width (must be positive)
        height: Height (must be positive)

    Returns:
        Tuple of (x, y, width, height) as floats

    Raises:
        ValueError: if any parameter is invalid
    """
    try:
        x_val = float(x)
        y_val = float(y)
        w_val = float(width)
        h_val = float(height)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Bbox parameters must be numeric: {e}")

    if not all(math.isfinite(v) for v in (x_val, y_val, w_val, h_val)):
        raise ValueError("All bbox parameters must be finite")

    if w_val <= 0:
        raise ValueError(f"Bbox width must be positive, got {w_val}")

    if h_val <= 0:
        raise ValueError(f"Bbox height must be positive, got {h_val}")

    return x_val, y_val, w_val, h_val


def validate_confidence(confidence: float) -> float:
    """Validate and normalize confidence score.

    Args:
        confidence: Confidence score

    Returns:
        Confidence as float clamped to [0, 1]

    Raises:
        ValueError: if confidence is non-finite
    """
    conf_val = float(confidence)

    if not math.isfinite(conf_val):
        raise ValueError("Confidence must be finite")

    # Clamp to [0, 1]
    return max(0.0, min(1.0, conf_val))


def compute_velocity(
    current_xy: Tuple[float, float],
    previous_xy: Tuple[float, float],
    dt_s: float,
) -> Tuple[float, float]:
    """Compute 2D velocity from position change.

    Args:
        current_xy: Current (x, y) position
        previous_xy: Previous (x, y) position
        dt_s: Time elapsed since previous measurement (seconds)

    Returns:
        (velocity_x, velocity_y) in same units as position per second

    Raises:
        ValueError: if inputs are invalid or dt_s <= 0
    """
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    curr_x = float(current_xy[0])
    curr_y = float(current_xy[1])
    prev_x = float(previous_xy[0])
    prev_y = float(previous_xy[1])

    if not all(math.isfinite(v) for v in (curr_x, curr_y, prev_x, prev_y)):
        raise ValueError("All position coordinates must be finite")

    vx = (curr_x - prev_x) / dt_s
    vy = (curr_y - prev_y) / dt_s

    return vx, vy


@dataclass
class TargetThreatState:
    """Tracks threat-related state for a single target across frames.

    Maintains history needed to compute time_inside_zone, velocity, and
    other derived metrics. Designed to be updated once per frame with
    new position and bounding box data.

    Attributes:
        target_id: Unique target identifier
        zone_radii: Dict mapping zone_id -> radius_m for zone computation
        asset_xy: (x, y) position of defended asset in meters
    """

    target_id: int
    zone_radii: Dict[str, float]
    asset_xy: Tuple[float, float]

    # Position history (in meters or pixels, depending on caller)
    _prev_position: Optional[Tuple[float, float]] = field(default=None, init=False)
    _prev_distance: Optional[float] = field(default=None, init=False)
    _prev_zone_id: Optional[str] = field(default=None, init=False)

    # Time tracking
    _zone_entry_time: Optional[float] = field(default=None, init=False)
    _total_zone_time: Dict[str, float] = field(default_factory=lambda: {}, init=False)
    _last_update_time: Optional[float] = field(default=None, init=False)

    def update(
        self,
        current_xy: Tuple[float, float],
        current_time: float,
        confidence: float = 1.0,
        bbox_x: float = 0.0,
        bbox_y: float = 0.0,
        bbox_width: float = 0.0,
        bbox_height: float = 0.0,
    ) -> Dict[str, float]:
        """Update target state and compute threat metrics for current frame.

        Args:
            current_xy: Current (x, y) position in meters
            current_time: Current timestamp in seconds
            confidence: Detection confidence [0, 1] (default 1.0)
            bbox_x: Bounding box left edge
            bbox_y: Bounding box top edge
            bbox_width: Bounding box width
            bbox_height: Bounding box height

        Returns:
            Dict containing computed metrics:
            - center_x, center_y: Bbox center
            - bbox_width, bbox_height: Bbox dimensions
            - velocity_x, velocity_y: 2D velocity (units/second)
            - confidence: Confidence score
            - distance_to_asset: Distance in meters
            - distance_rate_to_asset: Distance rate (m/s)
            - zone_id: Current zone
            - time_inside_zone: Time in current zone (seconds)

        Raises:
            ValueError: if inputs are invalid
            RuntimeError: if update called without prior initialization
        """
        # Validate inputs
        curr_x, curr_y = float(current_xy[0]), float(current_xy[1])
        curr_time = float(current_time)
        conf = validate_confidence(confidence)
        bbox_x_v, bbox_y_v, bbox_w_v, bbox_h_v = validate_bbox(
            bbox_x, bbox_y, bbox_width, bbox_height
        )

        if not math.isfinite(curr_time):
            raise ValueError("current_time must be finite")

        # Compute bbox center
        center_x, center_y = compute_bbox_center(
            bbox_x_v, bbox_y_v, bbox_w_v, bbox_h_v
        )

        # Compute distance to asset
        distance = compute_distance_to_asset(current_xy, self.asset_xy)

        # Compute zone
        zone_id = get_zone_id_for_distance(distance, self.zone_radii)

        # Compute velocity
        vx, vy = 0.0, 0.0
        if self._prev_position is not None and self._last_update_time is not None:
            dt = curr_time - self._last_update_time
            if dt > 0:
                vx, vy = compute_velocity(current_xy, self._prev_position, dt)

        # Compute distance rate
        distance_rate = 0.0
        if self._prev_distance is not None and self._last_update_time is not None:
            dt = curr_time - self._last_update_time
            if dt > 0:
                distance_rate = compute_distance_rate(distance, self._prev_distance, dt)

        # Track zone dwell time
        time_in_zone = 0.0
        if zone_id != self._prev_zone_id:
            # Zone transition
            self._zone_entry_time = curr_time
            self._prev_zone_id = zone_id
            time_in_zone = 0.0
        else:
            # Same zone - accumulate time
            if self._zone_entry_time is not None:
                time_in_zone = curr_time - self._zone_entry_time
                if zone_id in self._total_zone_time:
                    self._total_zone_time[zone_id] += time_in_zone
                else:
                    self._total_zone_time[zone_id] = time_in_zone

        # Update history
        self._prev_position = (curr_x, curr_y)
        self._prev_distance = distance
        self._last_update_time = curr_time

        return {
            "center_x": center_x,
            "center_y": center_y,
            "bbox_width": bbox_w_v,
            "bbox_height": bbox_h_v,
            "velocity_x": vx,
            "velocity_y": vy,
            "confidence": conf,
            "distance_to_asset": distance,
            "distance_rate_to_asset": distance_rate,
            "zone_id": zone_id,
            "time_inside_zone": time_in_zone,
        }

    def get_total_zone_time(self, zone_id: str) -> float:
        """Get cumulative time spent in a zone.

        Args:
            zone_id: Zone identifier

        Returns:
            Total time in zone (seconds), 0 if never visited
        """
        return self._total_zone_time.get(zone_id, 0.0)
