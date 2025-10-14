"""Helpers for resolving YOLO TensorRT engine configuration from YAML."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


_LOG = logging.getLogger(__name__)


class EngineConfigError(RuntimeError):
    """Raised when the YOLO engine configuration is invalid."""


@dataclass(frozen=True)
class EngineSelection:
    """Represents a chosen engine file from the configuration."""

    path: Path
    name: Optional[str] = None
    input_size: Optional[int] = None


@dataclass(frozen=True)
class ResolvedEngineConfig:
    """Result of resolving the effective YOLO engine configuration."""

    selection: EngineSelection
    explicit_input_size: Optional[int]
    detected_input_size: Optional[int]

    @property
    def variant_input_size(self) -> Optional[int]:
        return self.selection.input_size

    @property
    def effective_input_size(self) -> Optional[int]:
        for value in (
            self.explicit_input_size,
            self.selection.input_size,
            self.detected_input_size,
        ):
            if value is not None:
                return value
        return None

    @property
    def video_size_hint(self) -> Optional[int]:
        for value in (
            self.selection.input_size,
            self.detected_input_size,
            self.explicit_input_size,
        ):
            if value is not None:
                return value
        return None

    @property
    def input_size_source(self) -> str:
        if self.explicit_input_size is not None:
            return "yolo.input_size"
        if self.selection.input_size is not None:
            if self.selection.name:
                return f"yolo.engine.variants[{self.selection.name}].input_size"
            return "yolo.engine.input_size"
        if self.detected_input_size is not None:
            return "engine file"
        return "unknown"

    @property
    def video_size_source(self) -> str:
        if self.selection.input_size is not None:
            if self.selection.name:
                return f"yolo.engine.variants[{self.selection.name}].input_size"
            return "yolo.engine.input_size"
        if self.detected_input_size is not None:
            return "engine file"
        if self.explicit_input_size is not None:
            return "yolo.input_size"
        return "unknown"


def resolve_engine_config(
    cfg: Mapping[str, Any], *, config_dir: Optional[Path] = None
) -> EngineSelection:
    """Return the selected engine description from ``cfg``.

    Supports two schemas:

    1. Legacy: ``yolo.engine_path`` and optional ``yolo.input_size``.
    2. Structured: ``yolo.engine`` with ``selected`` variant and ``variants`` map.
    """

    yolo_section = cfg.get("yolo")
    if not isinstance(yolo_section, Mapping):
        raise EngineConfigError("config missing 'yolo' section")

    engine_section = yolo_section.get("engine")
    if engine_section is None:
        raw_path = yolo_section.get("engine_path")
        if raw_path is None:
            raise EngineConfigError("config missing yolo.engine or yolo.engine_path")
        path = _resolve_path(raw_path, config_dir)
        input_size = _coerce_positive_int(
            yolo_section.get("input_size"), field_name="yolo.input_size"
        )
        return EngineSelection(path=path, input_size=input_size)

    if isinstance(engine_section, Mapping):
        base_dir = config_dir
        raw_base = engine_section.get("base_dir")
        if raw_base is not None:
            base_dir = _resolve_path(raw_base, config_dir)

        raw_selected = engine_section.get("selected")
        selected = str(raw_selected).strip() if raw_selected is not None else ""
        variants = engine_section.get("variants")
        if not isinstance(variants, Mapping) or not variants:
            raise EngineConfigError("yolo.engine.variants must be a non-empty mapping")
        if not selected:
            if len(variants) == 1:
                selected = next(iter(variants))
            else:
                raise EngineConfigError("yolo.engine.selected must specify a variant name")
        variant = variants.get(selected)
        if variant is None:
            raise EngineConfigError(
                f"yolo.engine.selected references unknown variant {selected!r}"
            )
        if isinstance(variant, Mapping):
            raw_path = variant.get("path") or variant.get("file")
            if raw_path is None:
                raise EngineConfigError(
                    f"yolo.engine.variants[{selected!r}] is missing a 'path' entry"
                )
            path = _resolve_path(raw_path, base_dir)
            input_size = _coerce_positive_int(
                variant.get("input_size"),
                field_name=f"yolo.engine.variants[{selected}].input_size",
            )
        elif isinstance(variant, str):
            path = _resolve_path(variant, base_dir)
            input_size = None
        else:
            raise EngineConfigError(
                f"yolo.engine.variants[{selected!r}] must be a mapping or string path"
            )
        return EngineSelection(path=path, name=selected, input_size=input_size)

    if isinstance(engine_section, str):
        path = _resolve_path(engine_section, config_dir)
        return EngineSelection(path=path)

    raise EngineConfigError("yolo.engine must be a mapping or string when provided")


def resolve_yolo_runtime_config(
    cfg: Mapping[str, Any],
    *,
    config_dir: Optional[Path] = None,
    detect_engine_size: bool = True,
) -> ResolvedEngineConfig:
    """Resolve the engine selection and determine input size hints."""

    selection = resolve_engine_config(cfg, config_dir=config_dir)
    yolo_section = cfg.get("yolo")
    explicit_input = _coerce_positive_int(
        yolo_section.get("input_size") if isinstance(yolo_section, Mapping) else None,
        field_name="yolo.input_size",
    )

    detected_input: Optional[int] = None
    if detect_engine_size:
        detected_input = detect_engine_input_size(selection.path)

    return ResolvedEngineConfig(
        selection=selection,
        explicit_input_size=explicit_input,
        detected_input_size=detected_input,
    )


ASPECT_WIDTH = 4
ASPECT_HEIGHT = 3


def ensure_video_dimensions(
    cfg: MutableMapping[str, Any],
    resolved: ResolvedEngineConfig,
) -> tuple[Optional[int], Optional[int]]:
    """Ensure ``cfg['video']`` contains integer width/height.

    If either dimension is missing and the engine provides a size hint, the value
    is inserted into the configuration mapping.
    """

    video_section = cfg.get("video")
    if not isinstance(video_section, MutableMapping):
        raise EngineConfigError("config missing 'video' section")

    width = _coerce_positive_int(video_section.get("width"), field_name="video.width")
    height = _coerce_positive_int(
        video_section.get("height"), field_name="video.height"
    )

    hint = resolved.video_size_hint

    if width is None and height is None and hint is not None:
        width = hint
        height = _derive_height_from_width(width)
        video_section["width"] = width
        video_section["height"] = height
        return width, height

    if width is None:
        if height is not None:
            width = _derive_width_from_height(height)
        elif hint is not None:
            width = hint
        if width is not None:
            video_section["width"] = width

    if height is None:
        if width is not None:
            height = _derive_height_from_width(width)
        elif hint is not None:
            height = _derive_height_from_width(hint)
        if height is not None:
            video_section["height"] = height

    return width, height


def _derive_height_from_width(width: int) -> int:
    if width <= 0:
        raise EngineConfigError("video.width must be positive to derive video.height")
    derived = int(round(width * ASPECT_HEIGHT / ASPECT_WIDTH))
    derived = max(1, derived)
    if derived % 2 != 0:
        derived += 1
    return derived


def _derive_width_from_height(height: int) -> int:
    if height <= 0:
        raise EngineConfigError("video.height must be positive to derive video.width")
    derived = int(round(height * ASPECT_WIDTH / ASPECT_HEIGHT))
    derived = max(1, derived)
    if derived % 2 != 0:
        derived += 1
    return derived


def detect_engine_input_size(engine_path: Path) -> Optional[int]:
    """Return the square input size encoded in ``engine_path``, if available."""

    try:
        import tensorrt as trt  # type: ignore
    except Exception:  # pragma: no cover - TensorRT unavailable on non-Jetson hosts
        return None

    try:
        data = engine_path.read_bytes()
    except OSError as exc:  # pragma: no cover - runtime error path
        _LOG.debug("Failed to read TensorRT engine at %s: %s", engine_path, exc)
        return None

    try:
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(data)
    except Exception as exc:  # pragma: no cover - deserialize failure
        _LOG.debug("Failed to deserialize TensorRT engine at %s: %s", engine_path, exc)
        return None

    try:
        for binding_index in range(engine.num_bindings):
            if not engine.binding_is_input(binding_index):
                continue
            shape = engine.get_binding_shape(binding_index)
            if any(dim < 0 for dim in shape):
                try:
                    _, opt_shape, _ = engine.get_profile_shape(0, binding_index)
                    shape = opt_shape
                except Exception:  # pragma: no cover - dynamic shape fallback
                    continue
            positive_dims = [int(dim) for dim in shape if dim > 0]
            if not positive_dims:
                continue
            if len(positive_dims) >= 2 and positive_dims[-1] == positive_dims[-2]:
                return positive_dims[-1]
            return positive_dims[-1]
    finally:
        try:
            del engine  # type: ignore  # pragma: no cover - cleanup
        except Exception:
            pass

    return None


def _resolve_path(raw: Any, base_dir: Optional[Path]) -> Path:
    if raw is None:
        raise EngineConfigError("engine path cannot be null")
    text = str(raw).strip()
    if not text:
        raise EngineConfigError("engine path cannot be empty")
    path = Path(text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _coerce_positive_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EngineConfigError(f"{field_name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise EngineConfigError(f"{field_name} must be positive, got {parsed}")
    return parsed

