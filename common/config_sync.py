"""Utilities for synchronizing YAML configuration files over ZMQ."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

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
_LOCK_SUFFIX = ".config_sync_lock"


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


def expand_config_paths(
    config_path: Path | str,
    extra_paths: Optional[str] = None,
) -> list[Path]:
    """Return ``config_path`` plus any comma-separated extra config paths."""

    paths = [Path(config_path)]
    if extra_paths:
        for raw_path in extra_paths.split(","):
            path_text = raw_path.strip()
            if path_text:
                paths.append(Path(path_text))
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


def config_sync_lock_path(config_path: Path | str) -> Path:
    """Return the lock path that serializes config sync clients."""

    path = Path(config_path)
    return path.with_suffix(path.suffix + _LOCK_SUFFIX)


@contextmanager
def acquire_config_sync_lock(
    config_path: Path | str,
    timeout: Optional[float],
    *,
    poll_interval: float = 0.1,
) -> Iterator[None]:
    """Acquire a lock file to serialize config sync clients."""

    if poll_interval <= 0:
        raise ValueError("poll_interval must be > 0")
    lock_path = config_sync_lock_path(config_path)
    deadline = _deadline(timeout)

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner_pid = _read_lock_owner_pid(lock_path)
            if owner_pid is not None and not _pid_exists(owner_pid):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    pass
                else:
                    _LOG.warning(
                        "Config sync: removed stale lock %s owned by dead pid %s",
                        lock_path,
                        owner_pid,
                    )
                    continue
            if _deadline_expired(deadline):
                raise ConfigSyncError("timed out waiting for config sync lock")
            time.sleep(min(poll_interval, _remaining(deadline)))
            continue
        else:
            with os.fdopen(fd, "w") as handle:
                handle.write(str(os.getpid()))
            break

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


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


def request_startup_state(
    connect_ep: str,
    *,
    peer_id: Optional[str] = None,
    retry_interval: float = 1.0,
    max_wait: Optional[float] = DEFAULT_CONFIG_SYNC_TIMEOUT,
    max_attempts: Optional[int] = None,
) -> Dict[str, Any]:
    """Request startup state advertised by the config-sync server.

    The request is intentionally lightweight and does not synchronize any
    configuration file contents. It allows peers to branch startup behavior
    before beginning full config synchronization.
    """

    if retry_interval <= 0:
        raise ValueError("retry_interval must be > 0")
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive when provided")

    ctx = zmq.Context.instance()
    deadline = _deadline(max_wait)
    attempts = 0
    peer_id_clean = str(peer_id).strip() if peer_id is not None else ""

    while True:
        if _deadline_expired(deadline):
            raise ConfigSyncError("timed out waiting for startup state")

        attempt_deadline = _merge_deadlines(time.monotonic() + retry_interval, deadline)
        attempts += 1
        if max_attempts is not None and attempts > max_attempts:
            raise ConfigSyncError("exceeded maximum startup-state attempts")

        with ctx.socket(zmq.REQ) as req:
            req.setsockopt(zmq.LINGER, 0)
            req.connect(connect_ep)

            payload: Dict[str, object] = {
                "type": "startup_state",
                "config_id": "startup",
            }
            if peer_id_clean:
                payload["peer_id"] = peer_id_clean
            req.send_json(payload)

            try:
                reply = _recv_json(req, attempt_deadline)
            except TimeoutError:
                if _deadline_expired(deadline):
                    raise ConfigSyncError("timed out waiting for startup state")
                time.sleep(min(retry_interval, _remaining(deadline)))
                continue

            status = reply.get("status")
            if status == "retry_later":
                if _deadline_expired(deadline):
                    raise ConfigSyncError("timed out waiting for startup state")
                time.sleep(min(retry_interval, _remaining(deadline)))
                continue
            if status != "startup_state":
                raise ConfigSyncError(
                    f"unexpected startup-state status from server: {status!r}"
                )

            state_payload = reply.get("server_state")
            if not isinstance(state_payload, Mapping):
                raise ConfigSyncError("server startup state payload is missing or invalid")
            return dict(state_payload)


def sync_as_server(
    config_path: Path | str,
    bind_ep: str,
    *,
    config_id: str,
    required_peer_ids: Optional[Iterable[str]] = None,
    enforce_peer_match: bool = False,
    wait_timeout: Optional[float] = None,
    retry_interval: float = 1.0,
    max_attempts: Optional[int] = None,
    server_state: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, ConfigMetadata]:
    """Run the server side of the synchronization handshake.

    Parameters
    ----------
    config_path:
        Local configuration file path to synchronize.
    bind_ep:
        ZMQ endpoint to bind for the sync server.
    retry_interval:
        Seconds between retries after timeouts.
    wait_timeout:
        Maximum seconds to wait before aborting the handshake. ``None`` means
        no deadline.
    max_attempts:
        Optional cap on the number of retries. Useful for preventing infinite
        loops when ``wait_timeout`` is ``None``.
    """

    if retry_interval <= 0:
        raise ValueError("retry_interval must be > 0")
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive when provided")

    path = Path(config_path)
    ctx = zmq.Context.instance()
    deadline = _deadline(wait_timeout)
    required_peers = {
        str(peer).strip()
        for peer in (required_peer_ids or ())
        if str(peer).strip()
    }
    observed_required_peers: set[str] = set()
    announced_state = dict(server_state) if isinstance(server_state, Mapping) else {}

    def _reply_payload(payload: Mapping[str, object]) -> Dict[str, object]:
        data = dict(payload)
        if announced_state:
            data["server_state"] = announced_state
        return data

    with ctx.socket(zmq.REP) as rep:
        rep.setsockopt(zmq.LINGER, 0)
        rep.bind(bind_ep)

        def _recv_with_retry(error_message: str) -> Dict[str, object]:
            attempts = 0
            while True:
                if _deadline_expired(deadline):
                    raise ConfigSyncError(error_message)

                attempt_deadline = _merge_deadlines(
                    time.monotonic() + retry_interval, deadline
                )
                attempts += 1
                if max_attempts is not None and attempts > max_attempts:
                    raise ConfigSyncError("exceeded maximum handshake attempts")
                try:
                    return _recv_json(rep, attempt_deadline)
                except TimeoutError:
                    if _deadline_expired(deadline):
                        raise ConfigSyncError(error_message)
                    time.sleep(min(retry_interval, _remaining(deadline)))

        while True:
            request = _recv_with_retry("timed out waiting for client metadata")

            if request.get("type") == "startup_state":
                rep.send_json(
                    _reply_payload(
                        {
                            "status": "startup_state",
                            "config_id": config_id,
                        }
                    )
                )
                continue

            if request.get("type") != "metadata":
                rep.send_json(
                    _reply_payload(
                        {
                            "status": "retry_later",
                            "config_id": config_id,
                            "reason": "unexpected_request_type",
                        }
                    )
                )
                continue
            if request.get("config_id") != config_id:
                rep.send_json(
                    _reply_payload(
                        {
                            "status": "retry_later",
                            "config_id": request.get("config_id"),
                            "expected_config_id": config_id,
                            "reason": "config_id_out_of_order",
                        }
                    )
                )
                continue

            peer_id_raw = request.get("peer_id")
            peer_id = str(peer_id_raw).strip() if isinstance(peer_id_raw, str) else ""

            if enforce_peer_match and required_peers and peer_id not in required_peers:
                rep.send_json(
                    _reply_payload(
                        {
                            "status": "retry_later",
                            "config_id": config_id,
                            "reason": "unexpected_peer_id",
                            "expected_peer_ids": sorted(required_peers),
                        }
                    )
                )
                continue

            snapshot = read_snapshot(path)
            client_meta = ConfigMetadata.from_dict(request.get("metadata", {}))
            cmp_result = snapshot.metadata.compare(client_meta)

            final_snapshot = snapshot
            winner = "server"
            content: Optional[str] = None

            if cmp_result > 0:
                winner = "server"
                content = snapshot.text if snapshot.metadata.sha256 != client_meta.sha256 else None
                rep.send_json(
                    _reply_payload(
                        {
                            "status": "ok",
                            "config_id": config_id,
                            "winner": winner,
                            "metadata": snapshot.metadata.to_dict(),
                            "content": content,
                        }
                    )
                )
            elif cmp_result == 0:
                winner = "equal"
                rep.send_json(
                    _reply_payload(
                        {
                            "status": "ok",
                            "config_id": config_id,
                            "winner": winner,
                            "metadata": snapshot.metadata.to_dict(),
                        }
                    )
                )
            else:
                rep.send_json(
                    _reply_payload(
                        {
                            "status": "need_payload",
                            "config_id": config_id,
                            "metadata": snapshot.metadata.to_dict(),
                        }
                    )
                )

                while True:
                    payload_msg = _recv_with_retry("timed out waiting for client payload")

                    payload_type = payload_msg.get("type")
                    payload_config_id = payload_msg.get("config_id")

                    if payload_type == "content" and payload_config_id == config_id:
                        break

                    rep.send_json(
                        _reply_payload(
                            {
                                "status": "retry_later",
                                "config_id": payload_config_id,
                                "expected_config_id": config_id,
                                "reason": "waiting_for_payload",
                            }
                        )
                    )

                new_text = str(payload_msg.get("content", ""))
                atomic_write(path, new_text)
                final_snapshot = read_snapshot(path)
                winner = "client"
                rep.send_json(
                    _reply_payload(
                        {
                            "status": "ok",
                            "config_id": config_id,
                            "winner": winner,
                            "metadata": final_snapshot.metadata.to_dict(),
                        }
                    )
                )
                _LOG.info("Config sync: accepted client version for %%s", path)

            if not required_peers:
                return final_snapshot.text, final_snapshot.metadata

            if peer_id and peer_id in required_peers:
                observed_required_peers.add(peer_id)
            elif peer_id:
                _LOG.info(
                    "Config sync: observed non-required peer_id=%s for %s",
                    peer_id,
                    config_id,
                )
            else:
                _LOG.warning(
                    "Config sync: request missing peer_id for %s while waiting for peers %s",
                    config_id,
                    sorted(required_peers),
                )

            if observed_required_peers >= required_peers:
                return final_snapshot.text, final_snapshot.metadata


def sync_as_client(
    config_path: Path | str,
    connect_ep: str,
    *,
    config_id: str,
    peer_id: Optional[str] = None,
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
    peer_id_clean = str(peer_id).strip() if peer_id is not None else ""

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
            metadata_msg: Dict[str, object] = {
                "type": "metadata",
                "config_id": config_id,
                "metadata": snapshot.metadata.to_dict(),
            }
            if peer_id_clean:
                metadata_msg["peer_id"] = peer_id_clean
            req.send_json(metadata_msg)

            try:
                reply = _recv_json(req, attempt_deadline)
            except TimeoutError:
                if _deadline_expired(deadline):
                    raise ConfigSyncError("timed out waiting for server response")
                time.sleep(min(retry_interval, _remaining(deadline)))
                continue

            status = reply.get("status")
            if status == "retry_later":
                if _deadline_expired(deadline):
                    raise ConfigSyncError("timed out waiting for server response")
                time.sleep(min(retry_interval, _remaining(deadline)))
                continue

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
                        **({"peer_id": peer_id_clean} if peer_id_clean else {}),
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


def _read_lock_owner_pid(lock_path: Path) -> Optional[int]:
    try:
        raw_pid = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw_pid:
        return None
    try:
        pid = int(raw_pid)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True
