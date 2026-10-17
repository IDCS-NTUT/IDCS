"""Explicit section contract and update tiers; no device or network access.

The checked-in field inventory is source, not inferred from files at runtime.
Existing typed parsers remain authoritative for cross-field constraints.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from common.config import ConfigError

FIELDS = json.loads(Path(__file__).with_name("config_fields.json").read_text(encoding="utf-8"))


def tier(key: str) -> str:
    try:
        return FIELDS[key]["tier"]
    except KeyError as exc:
        raise ConfigError(f"unknown configuration key: {key}") from exc


def validate_fields(config: Mapping[str, Any], *, overlay: bool = False) -> None:
    def visit(value: Any, path: str) -> None:
        spec = FIELDS.get(path)
        if spec is None:
            raise ConfigError(f"unknown configuration key: {path}")
        if spec["tier"] == "fixed":
            raise ConfigError(f"{path} is fixed in common.config_constants")
        kind = ("null" if value is None else "bool" if isinstance(value, bool) else
                "number" if isinstance(value, (float, int)) else "str" if isinstance(value, str) else
                "map" if isinstance(value, Mapping) else "list" if isinstance(value, (list, tuple)) else "invalid")
        allowed = spec["types"]
        if kind not in allowed or (spec.get("integer") and kind == "number" and type(value) is not int):
            raise ConfigError(f"{path} must have type {allowed}")
        if kind == "number":
            if not math.isfinite(value):
                raise ConfigError(f"{path} must be finite")
            if "min" in spec and value < spec["min"] or "max" in spec and value > spec["max"]:
                raise ConfigError(f"{path} outside [{spec.get('min', '-inf')}, {spec.get('max', 'inf')}]; "
                                  + spec.get("range_note", "see the section contract"))
        if "enum" in spec and value not in spec["enum"]:
            raise ConfigError(f"{path} must be one of {spec['enum']}")
        if kind == "map":
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif kind == "list":
            for item in value:
                visit(item, f"{path}[]")
    for key, value in config.items():
        visit(value, key)


def validate_config(config: Mapping[str, Any], *, bridge_limits=None) -> None:
    """Validate authored values and retain all existing reader checks."""
    validate_fields(config)
    if "controller" in config and "net" in config:
        from jetson.control.runtime_config import ControlRuntimeConfig
        cfg = ControlRuntimeConfig.from_config(config)
        gimbal = config.get("gimbal", {})
        for axis in ("yaw", "pitch"):
            cap = gimbal.get(f"{axis}_rate_limit_rad_s")
            if cap is not None and max(cfg.rate_limit_rad_s, cfg.manual_rate_limit_rad_s) > cap:
                raise ConfigError(f"controller.rate_limit_rad_s exceeds gimbal.{axis}_rate_limit_rad_s")
            low, high = gimbal.get(f"{axis}_min_rad"), gimbal.get(f"{axis}_max_rad")
            if (cfg.local_clock != "pc" and low is not None and high is not None
                    and cfg.max_travel_rad * 2 > high - low):
                raise ConfigError(f"controller.max_travel_rad cannot fit within gimbal.{axis}_min_rad/max_rad")
        if bridge_limits is not None:
            for axis in ("yaw", "pitch"):
                cap = bridge_limits.get(f"{axis}_rate_limit_rad_s")
                if cap is not None and max(cfg.rate_limit_rad_s, cfg.manual_rate_limit_rad_s) > cap:
                    raise ConfigError(f"controller.rate_limit_rad_s exceeds release bridge {axis} cap {cap}")
        # max_travel is a symmetric envelope about the measured start pose,
        # whereas bridge min/max are absolute home coordinates. They cannot
        # safely be compared without the origin; bridge retains final authority.
    if "camera" in config:
        from common.ranging import KnownSizeRangingConfig
        KnownSizeRangingConfig.from_raw_config(config)
    if "laser" in config:
        from common.control import LaserMountConfig
        LaserMountConfig.from_raw_config(config)
    if "control" in config and "video" in config:
        from common.config import resolve_active_video_profile
        from common.control import ControlConfig
        video, _ = resolve_active_video_profile(config)
        ControlConfig.from_raw_config(config, (int(video["width"]), int(video["height"])))
    if "deepstream" in config and "net" in config:
        from jetson.deepstream.runtime import load_settings
        load_settings(config, base_dir=Path.cwd())
    if "operator" in config and "net" in config:
        from jetson.operator_agent import AgentConfig
        AgentConfig.from_config(config)
