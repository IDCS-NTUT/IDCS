"""Helpers for resolving video configuration values."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


class VideoConfigError(ValueError):
    """Raised when video configuration cannot be resolved."""


@dataclass(frozen=True)
class AutoResolutionResult:
    width: Optional[int]
    height: Optional[int]
    engine_size: Optional[int]
    applied: bool
    used_fallback: bool


def _coerce_positive_int(name: str, raw: Any) -> Optional[int]:
    """Return ``raw`` as a positive integer or ``None`` when unset."""

    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - validated at runtime
        raise VideoConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise VideoConfigError(f"{name} must be positive, got {value}")
    return value


def parse_aspect_ratio(raw: Any, *, default: Tuple[int, int] = (4, 3)) -> Tuple[int, int]:
    """Parse an aspect ratio specification into a ``(width, height)`` pair."""

    if raw is None:
        return default
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return default
        if ":" in text:
            parts = text.split(":", 1)
        else:
            parts = [text, "1"]
        if len(parts) != 2:
            raise VideoConfigError(f"invalid aspect ratio {raw!r}")
        try:
            a = int(parts[0])
            b = int(parts[1])
        except ValueError as exc:
            raise VideoConfigError(f"invalid aspect ratio {raw!r}") from exc
    elif isinstance(raw, (tuple, list)):
        if len(raw) != 2:
            raise VideoConfigError(f"aspect ratio must have two entries, got {raw!r}")
        try:
            a = int(raw[0])
            b = int(raw[1])
        except (TypeError, ValueError) as exc:
            raise VideoConfigError(f"invalid aspect ratio {raw!r}") from exc
    else:
        raise VideoConfigError(f"invalid aspect ratio {raw!r}")

    if a <= 0 or b <= 0:
        raise VideoConfigError(f"aspect ratio values must be positive, got {raw!r}")
    return (a, b)


def detect_engine_input_size(
    engine_path: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[int]:
    """Inspect ``engine_path`` and return the square TensorRT input size."""

    if not engine_path:
        return None
    try:  # pragma: no cover - TensorRT unavailable in CI
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover - best effort on non-Jetson hosts
        if logger:
            logger.debug("TensorRT unavailable for engine inspection: %s", exc)
        return None

    try:
        with open(engine_path, "rb") as handle:
            engine_data = handle.read()
    except OSError as exc:
        if logger:
            logger.warning("Failed to read TensorRT engine %s: %s", engine_path, exc)
        return None

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        if logger:
            logger.warning(
                "TensorRT failed to deserialize engine at %s", engine_path
            )
        return None

    try:
        binding_index = None
        for idx in range(engine.num_bindings):
            if engine.binding_is_input(idx):
                binding_index = idx
                break
        if binding_index is None:
            if logger:
                logger.warning("TensorRT engine %s has no input bindings", engine_path)
            return None

        shape = engine.get_binding_shape(binding_index)
        if any(dim == -1 for dim in shape):
            if engine.num_optimization_profiles > 0:
                _, opt_shape, _ = engine.get_profile_shape(0, binding_index)
                shape = opt_shape
            else:
                if logger:
                    logger.warning(
                        "TensorRT engine %s exposes dynamic shape for binding %s",
                        engine_path,
                        engine.get_binding_name(binding_index),
                    )
                return None

        dims = tuple(int(dim) for dim in shape)
        if engine.has_implicit_batch_dimension:
            if len(dims) < 3:
                if logger:
                    logger.warning(
                        "TensorRT engine %s binding shape %s missing spatial dims",
                        engine_path,
                        dims,
                    )
                return None
            spatial = dims[-2:]
        else:
            if len(dims) < 4:
                if logger:
                    logger.warning(
                        "TensorRT engine %s binding shape %s missing spatial dims",
                        engine_path,
                        dims,
                    )
                return None
            spatial = dims[-2:]
        height, width = spatial
        if width != height:
            if logger:
                logger.warning(
                    "TensorRT engine %s input is not square: %s", engine_path, spatial
                )
            return None
        return int(width)
    finally:
        del engine


def apply_auto_video_resolution(
    video_cfg: Mapping[str, Any],
    yolo_cfg: Mapping[str, Any],
    width: Optional[int],
    height: Optional[int],
    *,
    logger: Optional[logging.Logger] = None,
) -> AutoResolutionResult:
    """Return resolved dimensions when ``auto_from_engine`` is enabled."""

    auto_enabled = bool(video_cfg.get("auto_from_engine"))
    if not auto_enabled:
        return AutoResolutionResult(width, height, None, False, False)

    aspect = parse_aspect_ratio(video_cfg.get("aspect_ratio"))
    ratio_w, ratio_h = aspect

    engine_path = str((yolo_cfg.get("engine_path") or "").strip())
    fallback_size = _coerce_positive_int("yolo.input_size", yolo_cfg.get("input_size"))

    engine_size = detect_engine_input_size(engine_path, logger=logger)
    used_fallback = False
    if engine_size is None:
        engine_size = fallback_size
        used_fallback = engine_size is not None

    if engine_size is None:
        raise VideoConfigError(
            "video.auto_from_engine enabled but TensorRT input size could not be determined; "
            "set yolo.input_size or disable auto detection"
        )

    width = int(engine_size)
    height = int(round(width * ratio_h / ratio_w))
    if height <= 0:
        raise VideoConfigError(
            f"computed video height {height} is invalid for aspect ratio {aspect}"
        )
    if height % 2:
        height += 1
    if width % 2:
        width += 1

    if logger:
        if used_fallback:
            logger.info(
                "video.auto_from_engine: using configured YOLO input size %d → %dx%d",
                engine_size,
                width,
                height,
            )
        else:
            logger.info(
                "video.auto_from_engine: detected engine size %d → %dx%d",
                engine_size,
                width,
                height,
            )

    return AutoResolutionResult(width, height, engine_size, True, used_fallback)


def coerce_configured_dimensions(
    video_cfg: Mapping[str, Any],
) -> Tuple[Optional[int], Optional[int]]:
    """Return configured width/height allowing ``None``/``auto`` placeholders."""

    width = _coerce_positive_int("video.width", video_cfg.get("width"))
    height = _coerce_positive_int("video.height", video_cfg.get("height"))
    return width, height
