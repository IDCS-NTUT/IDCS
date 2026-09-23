"""Shared V2 target geometry and parallax aiming calculations.

This module is deliberately pure: it accepts one immutable perception
snapshot and returns one immutable aiming solution.  Simulator and hardware
controller runtimes therefore use identical image geometry without sharing
controller tuning or motion models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from common.control import (
    ControlConfig,
    LaserMountConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.geometry import laser_ray_to_pixel
from common.perception import PerceptionSnapshotV2


@dataclass(frozen=True)
class AimingSolution:
    """Complete selected-target geometry for a single perception snapshot."""

    track_id: int
    class_id: str
    confidence: float
    target_px: tuple[float, float]
    aim_px: tuple[float, float]
    pixel_error: tuple[float, float]
    bearing_error_rad: tuple[float, float]
    distance_m: Optional[float]
    distance_source: Optional[str]
    parallax_active: bool
    on_target: bool


def _selected_assessment_distance(
    snapshot: PerceptionSnapshotV2,
    track_id: int,
) -> tuple[Optional[float], Optional[str]]:
    assessment = next(
        (item for item in snapshot.assessments if item.track_id == track_id),
        None,
    )
    if assessment is None or assessment.distance_m is None:
        return None, None
    distance = float(assessment.distance_m)
    if not math.isfinite(distance) or distance <= 0.0:
        return None, None
    suffix = assessment.distance_src or "unspecified"
    return distance, f"known_size:{suffix}"


def _resolve_parallax_range(
    snapshot: PerceptionSnapshotV2,
    track_id: int,
    control: ControlConfig,
) -> tuple[Optional[float], Optional[str]]:
    measured, measured_source = _selected_assessment_distance(snapshot, track_id)
    policy = str(control.laser.use_range or "").strip().lower()
    if policy in {"known_size", "auto", "ground_plane"} and measured is not None:
        return measured, measured_source
    fallback = float(control.laser.default_distance_m)
    if not math.isfinite(fallback) or fallback <= 0.0:
        return None, None
    if policy == "infinite":
        return fallback, "configured_infinite"
    return fallback, "config_default"


def solve_snapshot_aiming(
    snapshot: PerceptionSnapshotV2,
    control: ControlConfig,
    laser_mount: Optional[LaserMountConfig],
) -> Optional[AimingSolution]:
    """Resolve the selected V2 target and its configured image-plane aim point."""

    frame_control = control.for_frame_size(
        (snapshot.frame.width, snapshot.frame.height)
    )

    selection = snapshot.selection
    if selection is None:
        return None
    track = next(
        (item for item in snapshot.tracks if item.track_id == selection.track_id),
        None,
    )
    if track is None:
        return None

    target_px = (
        (track.box.x + 0.5 * track.box.w) * snapshot.frame.width,
        (track.box.y + 0.5 * track.box.h) * snapshot.frame.height,
    )
    aim_px = (float(frame_control.cx_px), float(frame_control.cy_px))
    distance_m: Optional[float] = None
    distance_source: Optional[str] = None
    parallax_active = False

    if frame_control.aim_mode == "laser_point" and laser_mount is not None:
        distance_m, distance_source = _resolve_parallax_range(
            snapshot, track.track_id, frame_control
        )
        if distance_m is not None:
            try:
                projected = laser_ray_to_pixel(
                    laser_mount.offset_m.as_tuple(),
                    laser_mount.dir_cam.as_tuple(),
                    fx_px=frame_control.fx_px,
                    fy_px=frame_control.fy_px,
                    cx_px=frame_control.cx_px,
                    cy_px=frame_control.cy_px,
                    depth_m=distance_m,
                )
            except ValueError:
                projected = None
            if projected is not None and all(math.isfinite(float(v)) for v in projected):
                aim_px = (float(projected[0]), float(projected[1]))
                parallax_active = True

    raw_error = (target_px[0] - aim_px[0], target_px[1] - aim_px[1])
    signed = pixel_delta(
        target_px[0],
        target_px[1],
        aim_px[0],
        aim_px[1],
        frame_control,
        apply_deadband=False,
    )
    angular = angular_error_from_pixel_delta(signed, frame_control)
    return AimingSolution(
        track_id=track.track_id,
        class_id=track.class_id,
        confidence=track.confidence,
        target_px=(float(target_px[0]), float(target_px[1])),
        aim_px=aim_px,
        pixel_error=(float(raw_error[0]), float(raw_error[1])),
        bearing_error_rad=angular.as_tuple(),
        distance_m=distance_m,
        distance_source=distance_source,
        parallax_active=parallax_active,
        on_target=math.hypot(*raw_error) <= float(frame_control.laser.tolerance_px),
    )
