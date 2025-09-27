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
from .ranging import (
    KnownSizeRangingConfig,
    KnownSizeRangingConfigError,
    RangingCandidate,
    iter_ranging_candidates,
    normalized_box_dimensions,
    resolve_class_label,
)
from .schemas import (
    Box,
    CamState,
    ControlCmd,
    DetectionMsg,
    detection_msg_from_json,
    detection_msg_to_json,
)

__all__ = [
    "CameraIntrinsics",
    "CameraIntrinsicsConfigError",
    "AxisPair",
    "ControlConfig",
    "ControlConfigError",
    "angular_error_from_pixels",
    "focal_lengths_from_fov",
    "KnownSizeRangingConfig",
    "KnownSizeRangingConfigError",
    "RangingCandidate",
    "iter_ranging_candidates",
    "normalized_box_dimensions",
    "resolve_class_label",
    "pixel_error",
    "Box",
    "CamState",
    "ControlCmd",
    "DetectionMsg",
    "detection_msg_from_json",
    "detection_msg_to_json",
]
