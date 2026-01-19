"""Utilities for synchronizing YAML configuration files over ZMQ."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

try:  # pragma: no cover - import guard for lightweight test envs
    import yaml
except ImportError:  # pragma: no cover - defer failure to actual use sites
    yaml = None  # type: ignore[assignment]
import zmq


_LOG = logging.getLogger(__name__)

# Default wait time for config sync clients (seconds).
DEFAULT_CONFIG_SYNC_TIMEOUT = 5.0


@dataclass(frozen=True)
class ConfigMetadata:
    """Metadata describing a configuration file snapshot."""

    mtime_ns: int
    size: int
    sha256: str

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, int | str]) -> "ConfigMetadata":
        try:
            mtime_ns = int(payload["mtime_ns"])
            size = int(payload["size"])
            sha256 = str(payload["sha256"])
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("invalid metadata payload") from exc
        return cls(mtime_ns=mtime_ns, size=size, sha256=sha256)

    def compare(self, other: "ConfigMetadata") -> int:
        """Return positive if ``self`` is newer than ``other``.

        Comparison order:
        1. ``mtime_ns``
        2. ``sha256`` (lexicographic)
        3. ``size``
        """

        if self.mtime_ns != other.mtime_ns:
            return 1 if self.mtime_ns > other.mtime_ns else -1
        if self.sha256 != other.sha256:
            return 1 if self.sha256 > other.sha256 else -1
        if self.size != other.size:
            return 1 if self.size > other.size else -1
        return 0


@dataclass(frozen=True)
class ConfigSnapshot:
    """File contents plus metadata captured at a moment in time."""

    text: str
    metadata: ConfigMetadata


class ConfigSyncError(RuntimeError):
    """Raised when the config synchronization handshake fails."""


_MARKER_SUFFIX = ".config_sync_marker.json"


def read_snapshot(path: Path) -> ConfigSnapshot:
    """Read ``path`` and return its contents + metadata.

    Missing files are treated as empty strings with zeroed metadata so that
    whichever peer has a real copy automatically wins.
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


def atomic_write(path: Path, text: str) -> None:
    """Persist ``text`` to ``path`` using a write-to-temp + replace strategy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    prefix = f".{path.name}."
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


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
    """Merge top-level config mappings in order, later values overriding earlier ones."""

    merged: Dict[str, Any] = {}
    for cfg in configs:
        merged.update(cfg)
    return merged


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


def resolve_config_sync_endpoint(cfg: Mapping[str, Any]) -> str:
    """Extract and validate ``net.config_sync`` from ``cfg``."""

    net_section = cfg.get("net")
    if not isinstance(net_section, Mapping):
        raise SystemExit("config missing 'net' section")
    raw_endpoint = net_section.get("config_sync")
    if raw_endpoint is None:
        raise SystemExit("config missing net.config_sync endpoint")
    if not isinstance(raw_endpoint, str):
        raise SystemExit("net.config_sync must be a string endpoint")
    endpoint = raw_endpoint.strip()
    if not endpoint:
        raise SystemExit("net.config_sync must be a non-empty tcp endpoint")
    if not endpoint.startswith("tcp://"):
        raise SystemExit(
            f"net.config_sync must be a tcp://HOST:PORT endpoint, got {endpoint!r}"
        )
    host_port = endpoint[len("tcp://"):]
    if ":" not in host_port:
        raise SystemExit(f"net.config_sync is missing a port: {endpoint!r}")
    host, port_str = host_port.rsplit(":", 1)
    if not host:
        raise SystemExit(f"net.config_sync is missing a host: {endpoint!r}")
    try:
        port = int(port_str)
    except ValueError as exc:  # pragma: no cover - validated at runtime
        raise SystemExit(f"net.config_sync has an invalid port: {endpoint!r}") from exc
    if not (0 < port < 65536):
        raise SystemExit(f"net.config_sync port out of range: {port}")
    return endpoint


def config_sync_marker_path(config_path: Path | str) -> Path:
    """Return the marker path that records the last successful sync."""

    path = Path(config_path)
    return path.with_suffix(path.suffix + _MARKER_SUFFIX)


def write_sync_marker(
    config_path: Path | str,
    metadata: ConfigMetadata,
    *,
    timestamp_ns: Optional[int] = None,
) -> None:
    """Record ``metadata`` for ``config_path`` so other clients can skip syncing."""

    payload = {
        "metadata": metadata.to_dict(),
        "timestamp_ns": int(timestamp_ns if timestamp_ns is not None else time.time_ns()),
    }
    marker_path = config_sync_marker_path(config_path)
    atomic_write(marker_path, json.dumps(payload, sort_keys=True))


def load_sync_marker(config_path: Path | str) -> Optional[Tuple[ConfigMetadata, int]]:
    """Return the metadata + timestamp stored for ``config_path``, if any."""

    marker_path = config_sync_marker_path(config_path)
    try:
        raw = marker_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        return None
    try:
        metadata = ConfigMetadata.from_dict(metadata_payload)
    except ValueError:
        return None
    timestamp_raw = payload.get("timestamp_ns")
    try:
        timestamp_ns = int(timestamp_raw)
    except (TypeError, ValueError):
        timestamp_ns = 0
    return metadata, timestamp_ns


def clear_sync_marker(config_path: Path | str) -> None:
    """Remove any existing sync marker for ``config_path``."""

    marker_path = config_sync_marker_path(config_path)
    try:
        marker_path.unlink()
    except FileNotFoundError:
        return


def sync_as_server(
    config_path: Path | str,
    bind_ep: str,
    *,
    config_id: str,
    wait_timeout: Optional[float] = None,
) -> Tuple[str, ConfigMetadata]:
    """Run the server side of the synchronization handshake."""

    path = Path(config_path)
    ctx = zmq.Context.instance()
    deadline = _deadline(wait_timeout)

    with ctx.socket(zmq.REP) as rep:
        rep.setsockopt(zmq.LINGER, 0)
        rep.bind(bind_ep)

        try:
            request = _recv_json(rep, deadline)
        except TimeoutError as exc:
            raise ConfigSyncError("timed out waiting for client metadata") from exc

        if request.get("type") != "metadata":
            raise ConfigSyncError("unexpected request type")
        if request.get("config_id") != config_id:
            raise ConfigSyncError(
                f"unexpected config_id {request.get('config_id')!r} (expected {config_id!r})"
            )

        snapshot = read_snapshot(path)
        client_meta = ConfigMetadata.from_dict(request.get("metadata", {}))
        cmp_result = snapshot.metadata.compare(client_meta)

        if cmp_result > 0:
            payload = snapshot.text if snapshot.metadata.sha256 != client_meta.sha256 else None
            rep.send_json(
                {
                    "status": "ok",
                    "config_id": config_id,
                    "winner": "server",
                    "metadata": snapshot.metadata.to_dict(),
                    "content": payload,
                }
            )
            return snapshot.text, snapshot.metadata

        if cmp_result == 0:
            rep.send_json(
                {
                    "status": "ok",
                    "config_id": config_id,
                    "winner": "equal",
                    "metadata": snapshot.metadata.to_dict(),
                }
            )
            return snapshot.text, snapshot.metadata

        # Client's copy is newer – request the payload and overwrite locally.
        rep.send_json(
            {
                "status": "need_payload",
                "config_id": config_id,
                "metadata": snapshot.metadata.to_dict(),
            }
        )

        try:
            payload_msg = _recv_json(rep, deadline)
        except TimeoutError as exc:
            raise ConfigSyncError("timed out waiting for client payload") from exc

        if payload_msg.get("type") != "content":
            raise ConfigSyncError("unexpected payload type")
        if payload_msg.get("config_id") != config_id:
            raise ConfigSyncError(
                f"unexpected config_id {payload_msg.get('config_id')!r} (expected {config_id!r})"
            )

        new_text = str(payload_msg.get("content", ""))
        atomic_write(path, new_text)
        final_snapshot = read_snapshot(path)
        rep.send_json(
            {
                "status": "ok",
                "config_id": config_id,
                "winner": "client",
                "metadata": final_snapshot.metadata.to_dict(),
            }
        )
        _LOG.info("Config sync: accepted client version for %%s", path)
        return final_snapshot.text, final_snapshot.metadata


def sync_as_client(
    config_path: Path | str,
    connect_ep: str,
    *,
    config_id: str,
    retry_interval: float = 1.0,
    max_wait: Optional[float] = DEFAULT_CONFIG_SYNC_TIMEOUT,
    max_attempts: Optional[int] = None,
) -> Tuple[str, ConfigMetadata]:
    """Run the client side of the synchronization handshake.

    Parameters
    ----------
    config_path:
        Local configuration file path to synchronize.
    connect_ep:
        ZMQ endpoint for the sync server.
    retry_interval:
        Seconds between retries after timeouts.
    max_wait:
        Maximum seconds to wait before aborting the handshake. ``None`` means
        no deadline.
    max_attempts:
        Optional cap on the number of retries. Useful for preventing infinite
        loops when ``max_wait`` is ``None``.
    """

    if retry_interval <= 0:
        raise ValueError("retry_interval must be > 0")
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive when provided")

    path = Path(config_path)
    ctx = zmq.Context.instance()
    deadline = _deadline(max_wait)
    attempts = 0

    while True:
        if _deadline_expired(deadline):
            raise ConfigSyncError("timed out waiting for server response")

        attempt_deadline = _merge_deadlines(time.monotonic() + retry_interval, deadline)
        snapshot = read_snapshot(path)

        attempts += 1
        if max_attempts is not None and attempts > max_attempts:
            raise ConfigSyncError("exceeded maximum handshake attempts")

        with ctx.socket(zmq.REQ) as req:
            req.setsockopt(zmq.LINGER, 0)
            req.connect(connect_ep)
            req.send_json(
                {
                    "type": "metadata",
                    "config_id": config_id,
                    "metadata": snapshot.metadata.to_dict(),
                }
            )

            try:
                reply = _recv_json(req, attempt_deadline)
            except TimeoutError:
                if _deadline_expired(deadline):
                    raise ConfigSyncError("timed out waiting for server response")
                time.sleep(min(retry_interval, _remaining(deadline)))
                continue

            status = reply.get("status")
            if status == "need_payload":
                if reply.get("config_id") != config_id:
                    raise ConfigSyncError(
                        "server replied with mismatched config_id: "
                        f"{reply.get('config_id')!r} != {config_id!r}"
                    )
                req.send_json(
                    {
                        "type": "content",
                        "config_id": config_id,
                        "metadata": snapshot.metadata.to_dict(),
                        "content": snapshot.text,
                    }
                )
                try:
                    ack = _recv_json(req, attempt_deadline)
                except TimeoutError as exc:
                    raise ConfigSyncError("timed out waiting for server ack") from exc

                final_meta = ConfigMetadata.from_dict(ack.get("metadata", {}))
                return snapshot.text, final_meta

            if status != "ok":
                raise ConfigSyncError(f"unexpected server status: {status!r}")
            if reply.get("config_id") != config_id:
                raise ConfigSyncError(
                    "server replied with mismatched config_id: "
                    f"{reply.get('config_id')!r} != {config_id!r}"
                )

            winner = reply.get("winner")
            remote_meta = ConfigMetadata.from_dict(reply.get("metadata", {}))
            content = reply.get("content")

            if winner == "server" and content is not None:
                atomic_write(path, str(content))
                final_snapshot = read_snapshot(path)
                return final_snapshot.text, final_snapshot.metadata

            if winner in {"equal", "server"}:
                return snapshot.text, remote_meta

            if winner == "client":
                # Our copy already matches the server.
                return snapshot.text, remote_meta

            raise ConfigSyncError(f"unexpected winner value: {winner!r}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _recv_json(socket: zmq.Socket, deadline: Optional[float]) -> Dict[str, object]:
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    timeout_ms = _timeout_ms(deadline)
    events = dict(poller.poll(timeout_ms))
    if events.get(socket) == zmq.POLLIN:
        return socket.recv_json()
    raise TimeoutError


def _deadline(timeout: Optional[float]) -> Optional[float]:
    if timeout is None:
        return None
    return time.monotonic() + timeout


def _merge_deadlines(*deadlines: Optional[float]) -> Optional[float]:
    filtered = [d for d in deadlines if d is not None]
    if not filtered:
        return None
    return min(filtered)


def _deadline_expired(deadline: Optional[float]) -> bool:
    if deadline is None:
        return False
    return time.monotonic() >= deadline


def _remaining(deadline: Optional[float]) -> float:
    if deadline is None:
        return float("inf")
    return max(0.0, deadline - time.monotonic())


def _timeout_ms(deadline: Optional[float]) -> int:
    if deadline is None:
        return -1
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 0
    return int(remaining * 1000)
