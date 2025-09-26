"""Shared utilities available to both Jetson and PC processes."""

from .camera import (
    CameraIntrinsics,
    CameraIntrinsicsConfigError,
    focal_lengths_from_fov,
)
from .control import (
    AxisPair,
    ControlConfig,
    ControlConfigError,
    angular_error_from_pixels,
    pixel_error,
)
from .schemas import Box, CamState, ControlCmd, DetectionMsg

__all__ = [
    "CameraIntrinsics",
    "CameraIntrinsicsConfigError",
    "AxisPair",
    "ControlConfig",
    "ControlConfigError",
    "angular_error_from_pixels",
    "focal_lengths_from_fov",
    "pixel_error",
    "Box",
    "CamState",
    "ControlCmd",
    "DetectionMsg",
]
