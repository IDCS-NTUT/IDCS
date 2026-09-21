"""Helpers for the known-size distance estimation pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Literal,
)

from common.camera import CameraIntrinsics


class KnownSizeRangingConfigError(ValueError):
    """Raised when the known-size ranging configuration is invalid."""


_VALID_DIMENSIONS = {"height", "width", "average"}


@dataclass(frozen=True)
class KnownSizeRangingConfig:
    """Typed view over the ``camera.known_size_ranging`` configuration block."""

    enabled: bool
    dimension: str
    class_sizes_m: Mapping[str, float]
    min_pixels: float
    ema_alpha: float
    class_aspect_ratio_limits: Mapping[str, Tuple[Optional[float], Optional[float]]]

    @classmethod
    def from_raw_config(cls, cfg: Mapping[str, Any]) -> "KnownSizeRangingConfig":
        camera_section: MutableMapping[str, Any] = dict(cfg.get("camera", {}))
        ranging_section = camera_section.get("known_size_ranging")

        if ranging_section is None:
            return cls(
                enabled=False,
                dimension="height",
                class_sizes_m={},
                min_pixels=0.0,
                ema_alpha=0.5,
                class_aspect_ratio_limits={},
            )

        if not isinstance(ranging_section, Mapping):
            raise KnownSizeRangingConfigError(
                "camera.known_size_ranging must be a mapping when provided"
            )

        enabled = bool(ranging_section.get("enabled", False))

        dimension_raw = str(ranging_section.get("dimension", "height")).strip().lower()
        if dimension_raw == "avg":
            dimension_raw = "average"
        if dimension_raw not in _VALID_DIMENSIONS:
            valid = ", ".join(sorted(_VALID_DIMENSIONS))
            raise KnownSizeRangingConfigError(
                f"camera.known_size_ranging.dimension must be one of: {valid}"
            )

        class_sizes = ranging_section.get("class_sizes_m", {})
        if not isinstance(class_sizes, Mapping):
            raise KnownSizeRangingConfigError(
                "camera.known_size_ranging.class_sizes_m must be a mapping"
            )

        sizes: Dict[str, float] = {}
        for cls_label, raw_size in class_sizes.items():
            if not isinstance(cls_label, str) or not cls_label:
                raise KnownSizeRangingConfigError(
                    "class_sizes_m keys must be non-empty strings"
                )
            try:
                size_m = float(raw_size)
            except (TypeError, ValueError) as exc:
                raise KnownSizeRangingConfigError(
                    "class_sizes_m values must be numeric distances in meters"
                ) from exc
            if size_m <= 0.0:
                raise KnownSizeRangingConfigError(
                    "class_sizes_m values must be positive"
                )
            sizes[cls_label] = size_m

        raw_aspect_limits = ranging_section.get("class_aspect_ratio_limits", {})
        if raw_aspect_limits is None:
            raw_aspect_limits = {}
        if not isinstance(raw_aspect_limits, Mapping):
            raise KnownSizeRangingConfigError(
                "camera.known_size_ranging.class_aspect_ratio_limits must be a mapping"
            )

        aspect_limits: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for cls_label, raw_limits in raw_aspect_limits.items():
            if not isinstance(cls_label, str) or not cls_label:
                raise KnownSizeRangingConfigError(
                    "class_aspect_ratio_limits keys must be non-empty strings"
                )
            if raw_limits is None:
                continue

            min_ratio: Optional[float] = None
            max_ratio: Optional[float] = None

            if isinstance(raw_limits, Mapping):
                if "min" in raw_limits:
                    try:
                        min_ratio = float(raw_limits["min"])
                    except (TypeError, ValueError) as exc:
                        raise KnownSizeRangingConfigError(
                            "class_aspect_ratio_limits min values must be numeric"
                        ) from exc
                if "max" in raw_limits:
                    try:
                        max_ratio = float(raw_limits["max"])
                    except (TypeError, ValueError) as exc:
                        raise KnownSizeRangingConfigError(
                            "class_aspect_ratio_limits max values must be numeric"
                        ) from exc
            elif isinstance(raw_limits, Sequence) and not isinstance(raw_limits, (str, bytes)):
                if len(raw_limits) != 2:
                    raise KnownSizeRangingConfigError(
                        "class_aspect_ratio_limits sequences must contain exactly two values"
                    )
                raw_min, raw_max = raw_limits
                if raw_min is not None:
                    try:
                        min_ratio = float(raw_min)
                    except (TypeError, ValueError) as exc:
                        raise KnownSizeRangingConfigError(
                            "class_aspect_ratio_limits min values must be numeric"
                        ) from exc
                if raw_max is not None:
                    try:
                        max_ratio = float(raw_max)
                    except (TypeError, ValueError) as exc:
                        raise KnownSizeRangingConfigError(
                            "class_aspect_ratio_limits max values must be numeric"
                        ) from exc
            else:
                raise KnownSizeRangingConfigError(
                    "class_aspect_ratio_limits values must be a mapping or [min, max] sequence"
                )

            if min_ratio is not None and min_ratio <= 0.0:
                raise KnownSizeRangingConfigError(
                    "class_aspect_ratio_limits min must be positive when provided"
                )
            if max_ratio is not None and max_ratio <= 0.0:
                raise KnownSizeRangingConfigError(
                    "class_aspect_ratio_limits max must be positive when provided"
                )
            if (
                min_ratio is not None
                and max_ratio is not None
                and min_ratio > max_ratio
            ):
                raise KnownSizeRangingConfigError(
                    "class_aspect_ratio_limits min cannot exceed max"
                )

            aspect_limits[cls_label] = (min_ratio, max_ratio)

        try:
            min_pixels = float(ranging_section.get("min_pixels", 0.0))
        except (TypeError, ValueError) as exc:
            raise KnownSizeRangingConfigError(
                "camera.known_size_ranging.min_pixels must be numeric"
            ) from exc
        if min_pixels < 0.0:
            raise KnownSizeRangingConfigError(
                "camera.known_size_ranging.min_pixels cannot be negative"
            )

        try:
            ema_alpha = float(ranging_section.get("ema_alpha", 0.5))
        except (TypeError, ValueError) as exc:
            raise KnownSizeRangingConfigError(
                "camera.known_size_ranging.ema_alpha must be numeric"
            ) from exc
        if not 0.0 <= ema_alpha <= 1.0:
            raise KnownSizeRangingConfigError(
                "camera.known_size_ranging.ema_alpha must be between 0 and 1"
            )

        return cls(
            enabled=enabled,
            dimension=dimension_raw,
            class_sizes_m=sizes,
            min_pixels=min_pixels,
            ema_alpha=ema_alpha,
            class_aspect_ratio_limits=aspect_limits,
        )


@dataclass(frozen=True)
class NormalizedDistanceEstimate:
    """Immutable range result for a normalized object without a legacy box."""

    class_label: str
    distance_m: float
    source: Literal["height", "width", "average"]
    pixel_size_px: float


def resolve_class_label(class_id: str, label_map: Mapping[str, str]) -> str:
    """Map a detector-provided class identifier to a human-friendly label."""

    key = str(class_id).strip()
    if key in label_map:
        return label_map[key]
    return key


def _distance_from_dimension(
    *,
    size_m: float,
    pixel_size_px: float,
    focal_length_px: float,
    min_pixels: float,
) -> Optional[float]:
    """Return the estimated distance for a single pixel dimension."""

    if pixel_size_px <= 0.0 or pixel_size_px < min_pixels:
        return None
    if focal_length_px <= 0.0:
        return None

    distance_m = (size_m * focal_length_px) / pixel_size_px
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        return None
    return distance_m


def _distance_from_dimensions(
    *,
    size_m: float,
    width_px: float,
    height_px: float,
    intrinsics: CameraIntrinsics,
    config: KnownSizeRangingConfig,
) -> Optional[Tuple[float, Literal["height", "width", "average"], float]]:
    min_pixels = max(0.0, config.min_pixels)

    height_distance = None
    if config.dimension in {"height", "average"}:
        height_distance = _distance_from_dimension(
            size_m=size_m,
            pixel_size_px=height_px,
            focal_length_px=intrinsics.fy_px,
            min_pixels=min_pixels,
        )
        if config.dimension == "height":
            if height_distance is None:
                return None
            return height_distance, "height", height_px

    width_distance = None
    if config.dimension in {"width", "average"}:
        width_distance = _distance_from_dimension(
            size_m=size_m,
            pixel_size_px=width_px,
            focal_length_px=intrinsics.fx_px,
            min_pixels=min_pixels,
        )
        if config.dimension == "width":
            if width_distance is None:
                return None
            return width_distance, "width", width_px

    if config.dimension != "average":
        return None

    components = []
    if height_distance is not None:
        components.append(("height", height_distance, height_px))
    if width_distance is not None:
        components.append(("width", width_distance, width_px))
    if not components:
        return None
    if len(components) == 1:
        source, distance_m, pixel_size_px = components[0]
        return distance_m, source, pixel_size_px
    return (
        sum(component[1] for component in components) / len(components),
        "average",
        sum(component[2] for component in components) / len(components),
    )


def estimate_normalized_distance(
    *,
    class_id: str,
    width_norm: float,
    height_norm: float,
    frame_size: Tuple[int, int],
    label_map: Mapping[str, str],
    intrinsics: CameraIntrinsics,
    config: KnownSizeRangingConfig,
) -> Optional[NormalizedDistanceEstimate]:
    """Estimate range directly from an immutable normalized-object contract."""

    frame_w, frame_h = frame_size
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame dimensions must be positive")
    width_px = max(0.0, float(width_norm) * frame_w)
    height_px = max(0.0, float(height_norm) * frame_h)
    if width_px <= 0.0 or height_px <= 0.0:
        return None
    class_label = resolve_class_label(class_id, label_map)
    size_m = config.class_sizes_m.get(class_label)
    if size_m is None:
        return None
    aspect_bounds = config.class_aspect_ratio_limits.get(class_label)
    if aspect_bounds is not None:
        min_ratio, max_ratio = aspect_bounds
        aspect_ratio = height_px / width_px
        if (
            (min_ratio is not None and aspect_ratio < min_ratio)
            or (max_ratio is not None and aspect_ratio > max_ratio)
        ):
            return None
    measurement = _distance_from_dimensions(
        size_m=size_m,
        width_px=width_px,
        height_px=height_px,
        intrinsics=intrinsics,
        config=config,
    )
    if measurement is None:
        return None
    distance_m, source, pixel_size_px = measurement
    return NormalizedDistanceEstimate(
        class_label=class_label,
        distance_m=distance_m,
        source=source,
        pixel_size_px=pixel_size_px,
    )
