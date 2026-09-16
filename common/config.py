"""Immutable, local-only configuration loading and provenance.

This module is the replacement boundary for peer-to-peer configuration sync.
It reads an ordered set of YAML layers once, validates the resolved value tree,
and returns a recursively immutable bundle with deterministic content hashes.
It never writes configuration, marker, lock, or runtime-state files.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - import guard for lightweight environments
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


class ConfigError(ValueError):
    """Raised when local configuration cannot be loaded deterministically."""


@dataclass(frozen=True)
class ConfigSource:
    """Provenance for one input file in layer order."""

    path: Path
    size: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {"path": str(self.path), "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class ConfigBundle:
    """Resolved immutable configuration plus reproducibility metadata."""

    data: Mapping[str, Any]
    digest: str
    sources: tuple[ConfigSource, ...]

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(source.path for source in self.sources)

    def mutable_copy(self) -> dict[str, Any]:
        """Return a detached mutable copy for APIs that require plain values."""

        return _thaw_mapping(self.data)

    def require_section(self, name: str) -> Mapping[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"configuration requires a {name!r} mapping")
        return value

    def provenance(self) -> dict[str, Any]:
        return {
            "config_digest": self.digest,
            "config_sources": [source.to_dict() for source in self.sources],
        }


def resolve_config_paths(
    primary: Path | str,
    extras: str | Iterable[Path | str] | None = None,
) -> tuple[Path, ...]:
    """Resolve the CLI primary/extra convention without reading any files."""

    paths = [Path(primary)]
    candidates = extras.split(",") if isinstance(extras, str) else extras or ()
    for candidate in candidates:
        text = str(candidate).strip()
        if text:
            paths.append(Path(text))
    return tuple(paths)


def resolve_active_video_profile(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Resolve the selected video profile from an immutable config bundle."""

    video = config.get("video")
    if not isinstance(video, Mapping):
        raise ConfigError("configuration requires a 'video' mapping")
    profiles = video.get("profiles")
    if not profiles:
        return _thaw_mapping(video), None
    if not isinstance(profiles, Mapping):
        raise ConfigError("video.profiles must be a mapping")
    active = video.get("active_profile")
    if not isinstance(active, str) or not active:
        raise ConfigError("video.active_profile must name a configured profile")
    selected = profiles.get(active)
    if not isinstance(selected, Mapping):
        raise ConfigError(f"video.active_profile {active!r} was not found")
    resolved = {
        key: _thaw_value(value)
        for key, value in video.items()
        if key not in {"profiles", "active_profile"}
    }
    resolved.update(_thaw_mapping(selected))
    return resolved, active


def resolve_active_return_video_profile(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Resolve the independently selected, display-only return profile."""

    video = config.get("video")
    if not isinstance(video, Mapping):
        raise ConfigError("configuration requires a 'video' mapping")
    profiles = video.get("profiles")
    if not profiles:
        return _thaw_mapping(video), None
    if not isinstance(profiles, Mapping):
        raise ConfigError("video.profiles must be a mapping")
    active = video.get("active_return_profile", video.get("active_profile"))
    if not isinstance(active, str) or not active:
        raise ConfigError(
            "video.active_return_profile must name a configured profile"
        )
    selected = profiles.get(active)
    if not isinstance(selected, Mapping):
        raise ConfigError(
            f"video.active_return_profile {active!r} was not found"
        )
    resolved = {
        key: _thaw_value(value)
        for key, value in video.items()
        if key not in {"profiles", "active_profile", "active_return_profile"}
    }
    resolved.update(_thaw_mapping(selected))
    return resolved, active


def merge_config_layers(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge layers; later scalars and sequences replace earlier ones."""

    merged: dict[str, Any] = {}
    for index, layer in enumerate(layers):
        normalized = _normalize_mapping(layer, f"layer[{index}]")
        _merge_into(merged, normalized)
    return merged


def load_config_bundle(
    paths: Sequence[Path | str],
    *,
    required_sections: Iterable[str] = (),
) -> ConfigBundle:
    """Read, validate, recursively merge, freeze, and hash YAML layers.

    File mtimes do not participate in resolution or hashing. Missing files,
    duplicate keys, non-mapping roots, and nonportable YAML values fail closed.
    """

    if yaml is None:
        raise ConfigError("PyYAML is required to load configuration")
    if not paths:
        raise ConfigError("at least one configuration path is required")

    layers: list[dict[str, Any]] = []
    sources: list[ConfigSource] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise ConfigError(f"configuration file does not exist: {path}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot read configuration file {path}: {exc}") from exc
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError(f"configuration file is not UTF-8: {path}") from exc
        try:
            parsed = yaml.load(text, Loader=_unique_key_loader())
        except ConfigError:
            raise
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, Mapping):
            raise ConfigError(f"{path} must contain a mapping at the top level")
        layers.append(_normalize_mapping(parsed, str(path)))
        sources.append(
            ConfigSource(
                path=path,
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )

    resolved = merge_config_layers(*layers)
    for section in required_sections:
        if not isinstance(section, str) or not section:
            raise ConfigError("required section names must be non-empty strings")
        if not isinstance(resolved.get(section), Mapping):
            raise ConfigError(f"configuration requires a {section!r} mapping")

    canonical = json.dumps(
        resolved,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ConfigBundle(
        data=_freeze_mapping(resolved),
        digest=hashlib.sha256(canonical).hexdigest(),
        sources=tuple(sources),
    )


def _unique_key_loader() -> type[Any]:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise ConfigError("configuration mapping keys must be scalar strings") from exc
            if duplicate:
                raise ConfigError(f"duplicate configuration key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return UniqueKeyLoader


def _normalize_mapping(value: Mapping[Any, Any], origin: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{origin} contains a non-string mapping key: {key!r}")
        result[key] = _normalize_value(item, f"{origin}.{key}")
    return result


def _normalize_value(value: Any, origin: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{origin} must be a finite number")
        return value
    if isinstance(value, Mapping):
        return _normalize_mapping(value, origin)
    if isinstance(value, (list, tuple)):
        return [
            _normalize_value(item, f"{origin}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ConfigError(
        f"{origin} has unsupported YAML value type {type(value).__name__}; "
        "use strings, finite numbers, booleans, null, lists, or mappings"
    )


def _merge_into(target: dict[str, Any], layer: Mapping[str, Any]) -> None:
    for key, value in layer.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge_into(current, value)
        else:
            target[key] = _copy_value(value)


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_value(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _thaw_value(item) for key, item in value.items()}


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _thaw_mapping(value)
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value
