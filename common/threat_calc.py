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
