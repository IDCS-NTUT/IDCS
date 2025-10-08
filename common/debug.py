"""Typed helpers for the debug/step-mode configuration tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class DebugConfigError(ValueError):
    """Raised when the debug configuration is invalid."""


@dataclass(frozen=True)
class BreakpointConfig:
    """Toggleable breakpoint with an optional numeric threshold."""

    enabled: bool
    threshold: Optional[float] = None


@dataclass(frozen=True)
class StepBreakpointsConfig:
    """Collection of breakpoint settings used by step mode."""

    target_switch: BreakpointConfig
    latency_jump_ms: BreakpointConfig
    distance_source_change: BreakpointConfig
    distance_jump_m: BreakpointConfig
    saturation: BreakpointConfig
    innovation_px: BreakpointConfig


@dataclass(frozen=True)
class ReplayConfig:
    """Optional replay sources for frames and detections."""

    frames_path: Optional[Path]
    detections_path: Optional[Path]


@dataclass(frozen=True)
class StepModeConfig:
    """Top-level configuration for deterministic step mode."""

    enabled: bool
    replay: ReplayConfig
    breakpoints: StepBreakpointsConfig


@dataclass(frozen=True)
class SnapshotExportConfig:
    """Controls how snapshots are exported to disk."""

    path: Optional[Path]
    cadence: str


@dataclass(frozen=True)
class RecomputeDefaultsConfig:
    """Baseline parameters for what-if recomputation."""

    params: Mapping[str, Any]


@dataclass(frozen=True)
class DebugConfig:
    """Aggregate debug configuration used across processes."""

    step_mode: StepModeConfig
    snapshot_export: SnapshotExportConfig
    recompute_defaults: RecomputeDefaultsConfig

    @classmethod
    def from_raw_config(cls, cfg: Mapping[str, Any]) -> "DebugConfig":
        debug_section = _get_mapping(cfg, "debug")

        step_section = _get_mapping(debug_section, "step_mode")
        enabled = bool(step_section.get("enabled", False))
        replay = _parse_replay(step_section.get("replay"))
        breakpoints = _parse_breakpoints(step_section.get("breakpoints"))

        snapshot_section = _get_mapping(debug_section, "snapshot")
        export_cfg = _parse_snapshot_export(snapshot_section.get("export"))

        recompute_section = _get_mapping(debug_section, "recompute")
        defaults_raw = recompute_section.get("defaults", {}) or {}
        if not isinstance(defaults_raw, Mapping):
            raise DebugConfigError("debug.recompute.defaults must be a mapping when provided")
        defaults: Dict[str, Any] = dict(defaults_raw)

        return cls(
            step_mode=StepModeConfig(
                enabled=enabled,
                replay=replay,
                breakpoints=breakpoints,
            ),
            snapshot_export=export_cfg,
            recompute_defaults=RecomputeDefaultsConfig(params=defaults),
        )


def _get_mapping(section: Optional[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    if not section:
        return {}
    value = section.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DebugConfigError(f"debug.{key} must be a mapping when provided")
    return value


def _parse_optional_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if value is False:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return None
        lowered = cleaned.lower()
        if lowered in {"disable", "none", "off"}:
            return None
        return Path(cleaned).expanduser()
    raise DebugConfigError("expected string path or disable flag")


def _parse_replay(section: Any) -> ReplayConfig:
    if section is None:
        return ReplayConfig(frames_path=None, detections_path=None)
    if not isinstance(section, Mapping):
        raise DebugConfigError("debug.step_mode.replay must be a mapping when provided")

    frames = _parse_optional_path(section.get("frames"))
    detections = _parse_optional_path(section.get("detections"))
    return ReplayConfig(frames_path=frames, detections_path=detections)


def _parse_breakpoints(section: Any) -> StepBreakpointsConfig:
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise DebugConfigError("debug.step_mode.breakpoints must be a mapping when provided")

    return StepBreakpointsConfig(
        target_switch=_parse_breakpoint(section.get("target_switch"), default_enabled=True),
        latency_jump_ms=_parse_breakpoint(
            section.get("latency_jump_ms"),
            default_enabled=False,
            default_threshold=30.0,
        ),
        distance_source_change=_parse_breakpoint(
            section.get("distance_source_change"),
            default_enabled=False,
        ),
        distance_jump_m=_parse_breakpoint(
            section.get("distance_jump_m"),
            default_enabled=False,
            default_threshold=5.0,
        ),
        saturation=_parse_breakpoint(
            section.get("saturation"),
            default_enabled=True,
        ),
        innovation_px=_parse_breakpoint(
            section.get("innovation_px"),
            default_enabled=False,
            default_threshold=20.0,
        ),
    )


def _parse_breakpoint(
    raw: Any,
    *,
    default_enabled: bool,
    default_threshold: Optional[float] = None,
) -> BreakpointConfig:
    if raw is None:
        return BreakpointConfig(enabled=default_enabled, threshold=default_threshold)
    if isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", default_enabled))
        threshold_value = raw.get("threshold", default_threshold)
        threshold = None
        if threshold_value is not None:
            try:
                threshold = float(threshold_value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise DebugConfigError(
                    "breakpoint threshold must be numeric when provided"
                ) from exc
        return BreakpointConfig(enabled=enabled, threshold=threshold)
    if isinstance(raw, bool):
        return BreakpointConfig(enabled=raw, threshold=default_threshold)
    if isinstance(raw, (int, float)):
        return BreakpointConfig(enabled=True, threshold=float(raw))
    raise DebugConfigError("invalid breakpoint specification")


def _parse_snapshot_export(section: Any) -> SnapshotExportConfig:
    if section is None:
        return SnapshotExportConfig(path=None, cadence="off")
    if not isinstance(section, Mapping):
        raise DebugConfigError("debug.snapshot.export must be a mapping when provided")

    path = _parse_optional_path(section.get("path"))
    cadence_raw = section.get("cadence", "off")
    if isinstance(cadence_raw, str):
        cadence = cadence_raw.strip().lower()
    else:
        raise DebugConfigError("snapshot export cadence must be a string")

    valid_cadence = {"off", "every_step", "on_breakpoints"}
    if cadence not in valid_cadence:
        raise DebugConfigError(
            f"snapshot export cadence must be one of {sorted(valid_cadence)}"
        )

    return SnapshotExportConfig(path=path, cadence=cadence)

