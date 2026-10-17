"""Local configuration path, snapshot, merge and video-profile helpers.

The historical module name remains for existing imports. There is no network
synchronization, handshake, marker, locking, or configuration-writing path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:  # pragma: no cover - import guard for lightweight test envs
    import yaml
except ImportError:  # pragma: no cover - defer failure to actual use sites
    yaml = None  # type: ignore[assignment]



@dataclass(frozen=True)
class ConfigMetadata:
    """Metadata describing a configuration file snapshot."""

    mtime_ns: int
    size: int
    sha256: str


@dataclass(frozen=True)
class ConfigSnapshot:
    """File contents plus metadata captured at a moment in time."""

    text: str
    metadata: ConfigMetadata


def read_snapshot(path: Path) -> ConfigSnapshot:
    """Read ``path`` and return its contents + metadata.

    Missing files are treated as empty strings with zeroed metadata for legacy
    offline callers. Managed runtimes use the strict common.config loader.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        data = b""
        metadata = ConfigMetadata(mtime_ns=0, size=0, sha256=_sha256(data))
        return ConfigSnapshot(text="", metadata=metadata)

    stat = path.stat()
    data = text.encode("utf-8")
    metadata = ConfigMetadata(
        mtime_ns=getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)),
        size=len(data),
        sha256=_sha256(data),
    )
    return ConfigSnapshot(text=text, metadata=metadata)


def parse_config_text(text: str, origin: str) -> Mapping[str, Any]:
    """Parse ``text`` as YAML and ensure the root is a mapping."""

    if yaml is None:
        raise SystemExit("pyyaml is required to parse configuration files")
    data = yaml.safe_load(text) if text else {}
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise SystemExit(f"{origin} must contain a mapping at the top level")
    return data


def merge_config_maps(*configs: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge config mappings in order.

    Nested mappings merge key by key; later scalars and lists replace earlier
    ones. This matches ``common.config.load_config_bundle``, so an overlay
    means the same thing to every process that loads it.
    """

    def merge_into(target: Dict[str, Any], layer: Mapping[str, Any]) -> None:
        for key, value in layer.items():
            current = target.get(key)
            if isinstance(current, dict) and isinstance(value, Mapping):
                merge_into(current, value)
            elif isinstance(value, Mapping):
                target[key] = {}
                merge_into(target[key], value)
            else:
                target[key] = value

    merged: Dict[str, Any] = {}
    for cfg in configs:
        merge_into(merged, cfg)
    return merged


def expand_config_paths(
    config_path: Path | str,
    extra_paths: Optional[str] = None,
) -> list[Path]:
    """Return ``config_path`` plus any comma-separated extra config paths.

    A directory (``configs/base``) stands for its ``*.yaml`` files in name order.
    """

    raw = [Path(config_path)]
    if extra_paths:
        for raw_path in extra_paths.split(","):
            path_text = raw_path.strip()
            if path_text:
                raw.append(Path(path_text))
    paths: list[Path] = []
    for path in raw:
        if path.is_dir():
            files = sorted(path.glob("*.yaml"))
            if not files:
                raise ValueError(f"configuration directory has no .yaml files: {path}")
            paths.extend(files)
        else:
            paths.append(path)
    return paths


def load_merged_config(paths: Iterable[Path | str]) -> Dict[str, Any]:
    """Read and merge a sequence of YAML config files."""

    path_list = [Path(path) for path in paths]
    snapshots = [read_snapshot(path) for path in path_list]
    return merge_config_maps(
        *(
            parse_config_text(snapshot.text, str(path))
            for path, snapshot in zip(path_list, snapshots, strict=True)
        )
    )


def resolve_active_video_profile(
    cfg: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Return the effective video configuration and active profile name.

    The configuration supports two layouts:

    1. Legacy ``video`` section with width/height/fps directly under ``video``.
    2. ``video.profiles`` mapping with ``video.active_profile`` selecting a named profile.

    When profiles are present, the selected profile is merged on top of any
    additional keys defined alongside ``profiles`` (e.g. shared encoder
    settings). The returned mapping is a shallow copy so callers can mutate it
    without affecting the original configuration tree.
    """

    video_section = cfg.get("video")
    if not isinstance(video_section, Mapping):
        raise SystemExit("config missing 'video' section")

    profiles = video_section.get("profiles")
    if not profiles:
        # Either no profiles defined or explicitly empty/falsey; fall back to
        # the legacy structure. ``dict(...)`` ensures we always hand out a copy.
        return dict(video_section), None

    if not isinstance(profiles, Mapping):
        raise SystemExit("video.profiles must be a mapping of name -> settings")

    active_name = video_section.get("active_profile")
    if not isinstance(active_name, str) or not active_name:
        raise SystemExit("video.active_profile must be a non-empty string when profiles are defined")

    try:
        profile_values = profiles[active_name]
    except KeyError as exc:
        raise SystemExit(f"video.active_profile {active_name!r} not found in video.profiles") from exc

    if not isinstance(profile_values, Mapping):
        raise SystemExit(
            f"video.profiles[{active_name!r}] must be a mapping of settings"
        )

    base_cfg = {
        key: value
        for key, value in video_section.items()
        if key not in {"profiles", "active_profile"}
    }
    merged_cfg: Dict[str, Any] = dict(base_cfg)
    merged_cfg.update(profile_values)
    return merged_cfg, active_name


def resolve_active_return_video_profile(
    cfg: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Return the effective return-video configuration and profile name.

    When ``video.active_return_profile`` is provided, this selects a profile
    from ``video.profiles`` specifically for Jetson return-feed encoding.
    Otherwise this falls back to :func:`resolve_active_video_profile`.
    """

    video_cfg, active_name = resolve_active_video_profile(cfg)

    video_section = cfg.get("video")
    if not isinstance(video_section, Mapping):
        return video_cfg, active_name

    active_return_name_raw = video_section.get("active_return_profile")
    if active_return_name_raw is None:
        return video_cfg, active_name

    if not isinstance(active_return_name_raw, str) or not active_return_name_raw:
        raise SystemExit(
            "video.active_return_profile must be a non-empty string when provided"
        )

    profiles = video_section.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise SystemExit(
            "video.active_return_profile requires video.profiles to be defined"
        )

    try:
        return_profile_values = profiles[active_return_name_raw]
    except KeyError as exc:
        raise SystemExit(
            f"video.active_return_profile {active_return_name_raw!r} "
            "not found in video.profiles"
        ) from exc

    if not isinstance(return_profile_values, Mapping):
        raise SystemExit(
            f"video.profiles[{active_return_name_raw!r}] must be a mapping of settings"
        )

    base_cfg = {
        key: value
        for key, value in video_section.items()
        if key not in {"profiles", "active_profile", "active_return_profile"}
    }
    merged_cfg: Dict[str, Any] = dict(base_cfg)
    merged_cfg.update(return_profile_values)
    return merged_cfg, active_return_name_raw


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
