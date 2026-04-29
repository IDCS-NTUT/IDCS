"""Threat evaluation calculation utilities.

Provides helpers for computing distance-to-asset, zone membership,
distance rates, and other metrics needed for threat evaluation.

All calculations use XY horizontal plane (ignoring Z/height for v1).
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import numpy as np

__all__ = [
    "compute_distance_to_asset",
    "compute_distance_rate",
    "get_zone_membership",
    "get_zone_id_for_distance",
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
