"""Helpers for the known-size distance estimation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Sequence, Tuple

from common.schemas import Box


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
        )


@dataclass(frozen=True)
class RangingCandidate:
    """Pre-computed attributes for distance estimation of a single detection."""

    box: Box
    class_label: str
    width_px: float
    height_px: float
    size_m: float


def normalized_box_dimensions(box: Box, frame_size: Tuple[int, int]) -> Tuple[float, float]:
    """Convert a normalized :class:`Box` into pixel dimensions."""

    frame_w, frame_h = frame_size
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame dimensions must be positive")
    width_px = max(0.0, box.w * frame_w)
    height_px = max(0.0, box.h * frame_h)
    return width_px, height_px


def resolve_class_label(class_id: str, label_map: Mapping[str, str]) -> str:
    """Map a detector-provided class identifier to a human-friendly label."""

    key = str(class_id).strip()
    if key in label_map:
        return label_map[key]
    return key


def iter_ranging_candidates(
    boxes: Sequence[Box],
    frame_size: Tuple[int, int],
    label_map: Mapping[str, str],
    config: KnownSizeRangingConfig,
) -> Iterator[RangingCandidate]:
    """Yield ranging candidates with pixel geometry and canonical size.

    Only boxes whose resolved class label appears in ``config.class_sizes_m`` are
    returned so downstream steps can focus on detections that have a defined
    real-world size.
    """

    frame_w, frame_h = frame_size
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame dimensions must be positive")

    for box in boxes:
        width_px, height_px = normalized_box_dimensions(box, (frame_w, frame_h))
        if width_px <= 0.0 or height_px <= 0.0:
            continue
        class_label = resolve_class_label(box.cls, label_map)
        size_m = config.class_sizes_m.get(class_label)
        if size_m is None:
            continue
        yield RangingCandidate(
            box=box,
            class_label=class_label,
            width_px=width_px,
            height_px=height_px,
            size_m=size_m,
        )
