"""Host-local, validated live overlays and safe-point application.

No hardware is constructed here. Transport is local IPC; persistent overlay
polling also recovers updates missed during startup or PUB disconnection.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import queue
import threading
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import yaml

from common.config import (ConfigBundle, ConfigError, _freeze_mapping, load_config_bundle,
                           merge_config_layers, resolve_config_paths)
from common.config_constants import with_constants
from common.config_schema import tier, validate_config, validate_fields


def live_path() -> Path:
    return Path(os.environ.get("IDCS_LIVE_CONFIG", "~/.config/idcs/live.yaml")).expanduser().resolve()


def digest(value: Mapping) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_overlay(path: Path | None = None) -> dict:
    path = path or live_path()
    if not path.exists():
        return {}
    data = load_config_bundle([path]).mutable_copy()
    validate_fields(data, overlay=True)
    return data


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers briefly deny replacement while their handle is open.
        # Background polling makes this a normal writer/reader race. Bound the
        # retry and retain the same atomic-rename durability on every platform.
        deadline = time.monotonic() + 0.25
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                if (os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32)
                        or time.monotonic() >= deadline):
                    raise
                time.sleep(0.001)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def runtime_bundle(paths: Sequence[Path | str], *, required_sections=(), overlay=None) -> ConfigBundle:
    """Load release stack, then host overlay, then materialize fixed policy."""
    path = live_path()
    paths = tuple(Path(p) for p in paths if Path(p).resolve() != path)
    base = load_config_bundle(paths, required_sections=required_sections)
    authored = base.mutable_copy()
    validate_fields(authored)
    live_sources = ()
    if overlay is None and path.exists():
        live_bundle = load_config_bundle([path])
        live = live_bundle.mutable_copy()
        live_sources = live_bundle.sources
    else:
        live = {} if overlay is None else overlay
    validate_fields(live, overlay=True)
    merged = merge_config_layers(authored, live)
    validate_config(merged, bridge_limits=authored.get("gimbal"))
    effective = with_constants(merged)
    return replace(base, data=_freeze_mapping(effective), digest=digest(effective),
                   sources=base.sources + live_sources)


def flatten(value: Mapping, prefix="") -> dict:
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


def put(config: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"{key}: parent is not a mapping")
    node[parts[-1]] = value


def remove(config: dict, key: str) -> None:
    parts = key.split(".")
    node = config
    parents = []
    for part in parts[:-1]:
        if part not in node:
            return
        parents.append((node, part))
        node = node[part]
    node.pop(parts[-1], None)
    for parent, part in reversed(parents):
        if not parent[part]:
            parent.pop(part)


def changed(before: Mapping, after: Mapping) -> list[str]:
    left, right = flatten(before), flatten(after)
    return sorted(k for k in left.keys() | right.keys()
                  if (k in left) != (k in right) or left.get(k) != right.get(k))


def append_journal(record: dict) -> None:
    """One append syscall per record, shared by agent and service ACK writers."""
    path = live_path().with_name("live-journal.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        if os.write(fd, payload) != len(payload):
            raise OSError("short config journal append")
        os.fsync(fd)
    finally:
        os.close(fd)


# The deployed unit templates are the stack table; no duplicated overlay order.
DEFAULT_MODULES = {"jetson.deepstream.runtime", "jetson.operator_agent", "pc.hud_relay",
                   "rpi.runtime_control", "rpi.operator_display"}
CONSUMERS = {
    "jetson.control.video_runtime": {"controller", "control", "camera", "laser", "net", "video"},
    "jetson.deepstream.runtime": {"deepstream", "swarm_eval", "threat_eval", "camera", "laser",
                                   "control", "controller", "operator", "perception", "net", "video"},
    "jetson.operator_agent": {"operator", "net", "perception", "swarm_eval"},
    "pc.mode_agent": {"operator", "net"},
    "pc.streamer": {"sim", "video", "net", "gimbal", "camera", "laser", "swarm_eval", "threat_eval", "control", "controller"},
    "pc.ui": {"operator", "video", "net", "camera", "laser", "control"},
    "pc.hud_relay": {"operator", "net"},
    "rpi.runtime_control": {"rpi", "net", "gimbal"},
    "rpi.operator_display": {"rpi", "operator", "video", "net"},
    "jetson.gimbal_bridge": {"gimbal", "serial_io", "net"},
    "tools.serial_io_service": {"serial_io", "gimbal", "net"},
    "tools.flight_recorder": {"recorder"},
}


def uses_key(module: str, key: str) -> bool:
    if key.split(".")[0] not in CONSUMERS.get(module, set()):
        return False
    if key.startswith("controller.") and module != "jetson.control.video_runtime":
        return key in {"controller.rate_limit_rad_s", "controller.accel_limit_rad_s2"}
    if module == "jetson.operator_agent" and key.startswith("swarm_eval."):
        return key == "swarm_eval.excluded_target_classes"
    if module == "pc.mode_agent" and key.startswith("operator."):
        return key in {"operator.sim_activity_sub", "operator.sim_selection_bind", "operator.sim_score_endpoint"}
    if module == "jetson.deepstream.runtime" and key.startswith("operator."):
        return False  # operator authority arrives on its existing selection SUB
    # The current viewer's aiming objects are startup validation only; its HUD
    # consumes the controller's applied geometry through diagnostics.
    if module == "pc.ui" and key.startswith(("camera.", "laser.", "control.")):
        return False
    return True


def unit_stacks(root: Path, host: str | None = None) -> dict:
    result = {}
    for unit in sorted((root / "deploy/systemd").rglob("*.service")):
        role = unit.parent.name
        if host is not None and role != host:
            continue
        text = unit.read_text(encoding="utf-8")
        command = next((line for line in text.splitlines() if line.startswith("ExecStart=")), "")
        module = re.search(r" -m ([\w.]+)", command)
        primary = re.search(r"--config(?:=|\s+)(\S+)", command)
        extra = re.search(r"--config-extra(?:=|\s+)(\S+)", command)
        if module is None or (primary is None and module[1] not in DEFAULT_MODULES):
            continue
        paths = resolve_config_paths(root / (primary[1] if primary else "configs/base"),
            [root / p for p in extra[1].split(",")] if extra else None)
        result[f"{role}/{unit.name}"] = dict(paths=paths, module=module[1])
    return result


def service_unit(module: str, paths: Sequence[Path]) -> str:
    """Identify the current template from its module and exact ordered stack."""
    root = Path(__file__).resolve().parents[1]
    resolved = tuple(Path(p).resolve() for p in paths)
    for name, spec in unit_stacks(root).items():
        if spec["module"] == module and tuple(p.resolve() for p in spec["paths"]) == resolved:
            return name
    return "local/" + module


def receipt_path(unit: str) -> Path:
    return live_path().parent / "config-status" / (unit.replace("/", "--") + ".json")


@dataclass(frozen=True)
class PreparedSettings:
    epoch: int
    base: ConfigBundle
    bundle: ConfigBundle
    requested_digest: str
    payload: Any = None
    pending: Sequence[str] = ()
    restart: Sequence[str] = ()
    records: tuple = ()
    error: str | None = None


class LiveSettings:
    """One preparation worker, one owning tick/probe thread.

    ``prepare`` is pure with respect to running service state. ``apply`` only
    swaps its complete result (and may check current owner-only safety state).
    Neither callback may replace queues or reset controller sequencing.
    """
    def __init__(self, paths, *, unit: str, apply: Callable[[Any], None], prepare=lambda values: values,
                 hot_prefixes=(), initial_bundle=None):
        self.paths = tuple(paths)
        self.unit, self.apply, self.prepare = unit, apply, prepare
        self.hot_prefixes = tuple(hot_prefixes)
        self.module = unit_stacks(Path(__file__).resolve().parents[1]).get(unit, {}).get("module")
        self.bundle = runtime_bundle(paths) if initial_bundle is None else initial_bundle
        # Do not reread here: the caller's startup snapshot may already be
        # running while a newer (possibly invalid) overlay arrives. The first
        # safe point validates and acknowledges the desired revision.
        self.requested_digest = ""
        self.pending: list[str] = []
        self.restart: list[str] = []
        self.error: str | None = None
        self.extra_status = lambda: {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._acks = queue.SimpleQueue()
        self._engaged = None  # no idle proof until the first owning tick
        self._epoch = 0
        self._prepared = None
        self._worker_error = None
        self._subscribe = False
        self._ctx = self._sub = None
        self._worker = threading.Thread(target=self._work, name="live-settings:" + unit, daemon=True)
        self._worker.start()

    def subscribe(self):
        # Creation, receives and teardown all belong to the worker, including
        # when the caller is a GStreamer metadata probe.
        self._subscribe = True
        self._wake.set()

    def close(self):
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            raise RuntimeError("live settings worker did not stop")

    def status(self) -> dict:
        return dict(config_digest=self.bundle.digest, requested_digest=self.requested_digest,
                    pending_config_keys=self.pending, restart_config_keys=self.restart,
                    config_error=self.error, **self.extra_status())

    def tick(self, *, engaged: bool, force: bool = False) -> bool:
        if engaged != self._engaged:
            self._engaged = engaged
            force = True
        if force:
            self._epoch += 1
            self._wake.set()
        # Never wait behind preparation, disk I/O or receipt collection.
        if not self._lock.acquire(blocking=False):
            return False
        try:
            ready = self._prepared
            if ready is None:
                return False
            self._prepared = None
            if ready.epoch != self._epoch or ready.base is not self.bundle:
                self._acks.put((ready, False, time.time_ns()))
                return False
            self.requested_digest = ready.requested_digest
            self.pending, self.restart = ready.pending, ready.restart
            self.error = ready.error
            applied = False
            if ready.error is None and ready.bundle is not ready.base:
                try:
                    self.apply(ready.payload)
                    self.bundle = ready.bundle
                    applied = True
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    self.error = str(exc)
            # Retain the old bundle/payload until the worker can reclaim them.
            self._acks.put((ready, applied, time.time_ns()))
            self._wake.set()
            return applied
        finally:
            self._lock.release()

    def _prepare(self, overlay, signature, base, epoch, engaged):
        candidate = runtime_bundle(self.paths, overlay=overlay)
        old, new = base.mutable_copy(), candidate.mutable_copy()
        left = flatten(old)
        pending, restart = [], []
        for key in changed(old, new):
            if self.module is not None and not uses_key(self.module, key):
                continue
            kind = tier(key)
            hot = any(key == p or key.startswith(p + ".") for p in self.hot_prefixes)
            if kind == "restart" or not hot:
                restart.append(key)
            elif kind == "idle" and engaged:
                pending.append(key)
            else:
                continue
            if key in left:
                put(new, key, left[key])
            else:
                remove(new, key)
        keys = changed(old, new)
        payload, records, bundle = None, (), base
        if keys:
            # Validate the partially applied combination, not just the desired
            # whole: a live value may depend on a cap still held until idle.
            authored = merge_config_layers(new)
            from common.config_constants import FIXED_VALUES
            for key in FIXED_VALUES:
                remove(authored, key)
            validate_config(authored)
            payload = self.prepare(new)
            bundle = replace(candidate, data=_freeze_mapping(new), digest=digest(new))
            right = flatten(new)
            records = tuple(dict(type="application", actor=self.unit, key=key,
                old=left.get(key), new=right.get(key), tier=tier(key),
                resulting_digest=signature, config_digest=bundle.digest, units={self.unit: "applied"})
                for key in keys if self.module is None or uses_key(self.module, key))
        return PreparedSettings(epoch, base, bundle, signature, payload,
                                pending, restart, records)

    def _work(self):
        signature = None
        next_poll = next_receipt = 0.0
        journal_records = deque()
        receipt_due = False
        try:
            while True:
                notified = self._wake.wait(0.02)
                self._wake.clear()
                try:
                    if self._subscribe and self._sub is None:
                        import zmq
                        self._ctx = zmq.Context()
                        self._sub = self._ctx.socket(zmq.SUB)
                        self._sub.setsockopt(zmq.LINGER, 0)
                        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
                        self._sub.connect(update_endpoint())
                    if self._sub is not None:
                        import zmq
                        updated = False
                        # Bound notification draining; durable polling always
                        # reads the newest overlay even under an update flood.
                        for _ in range(64):
                            try:
                                self._sub.recv(zmq.NOBLOCK)
                                notified = True
                                updated = True
                            except zmq.Again:
                                break
                        if updated:
                            with self._lock:
                                self._prepared = None
                            signature = None
                    while not self._acks.empty():
                        ready, applied, applied_ns = self._acks.get_nowait()
                        journal_records.extend(dict(record, time_ns=applied_ns)
                                               for record in ready.records if applied)
                        receipt_due = True
                    while journal_records:
                        append_journal(journal_records[0])
                        journal_records.popleft()
                    now = time.monotonic()
                    if (not self._stop.is_set() and self._engaged is not None
                            and (notified or now >= next_poll)):
                        next_poll = now + 0.2
                        with self._lock:
                            base, epoch, engaged = self.bundle, self._epoch, self._engaged
                        desired = ""
                        try:
                            overlay = read_overlay()
                            desired = digest(overlay)
                            token = (desired, epoch)
                            if token != signature:
                                with self._lock:
                                    self._prepared = None
                                ready = self._prepare(overlay, desired, base, epoch, engaged)
                                # The file or engagement may change while a
                                # slow parser is running. Never publish that work.
                                if digest(read_overlay()) != desired:
                                    self._wake.set()
                                    continue
                                with self._lock:
                                    if epoch != self._epoch or base is not self.bundle:
                                        self._wake.set()
                                        continue
                                    self._prepared = ready
                                    signature = token
                        except (ValueError, KeyError, TypeError, OSError) as exc:
                            signature = None
                            with self._lock:
                                self._prepared = PreparedSettings(epoch, base, base, desired, error=str(exc))
                    if receipt_due or now >= next_receipt:
                        with self._lock:
                            status = dict(config_digest=self.bundle.digest,
                                requested_digest=self.requested_digest, pending_config_keys=self.pending,
                                restart_config_keys=self.restart, config_error=self.error)
                        status.update(self.extra_status())
                        atomic_write(receipt_path(self.unit), json.dumps(dict(unit=self.unit,
                            updated_ns=time.time_ns(), **status), sort_keys=True) + "\n")
                        next_receipt = now + 1.0
                        receipt_due = False
                        self._worker_error = None
                    if self._stop.is_set() and self._acks.empty():
                        break
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    # Preserve the running settings and retry receipt/IPC work;
                    # publication never claims an unconfirmed swap.
                    if self._worker_error != str(exc):
                        logging.getLogger(__name__).warning("%s live settings worker: %s", self.unit, exc)
                    self._worker_error = str(exc)
                    next_receipt = 0.0
        finally:
            if self._sub is not None:
                self._sub.close(0)
            if self._ctx is not None:
                self._ctx.term()


def update_endpoint() -> str:
    return "ipc://" + str(live_path().parent / "config-update.sock")


def command_endpoint() -> str:
    return "ipc://" + str(live_path().parent / "config-command.sock")


@contextmanager
def writer_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ConfigError(f"live config writer busy: {lock}") from exc
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink()


class ConfigAgent:
    """Revalidate all local unit stacks before any durable change or restart."""
    def __init__(self, root, host, *, run, publish=lambda event: None, active=lambda: set(),
                 stacks=None, result_wait_s=1.0, motors_live=None, controller_engaged=None):
        self.root, self.host = Path(root), host
        self.run, self.publish, self.active = run, publish, active
        self.stacks = unit_stacks(self.root, host) if stacks is None else stacks
        self.result_wait_s = result_wait_s
        self.motors_live = motors_live or (lambda: bool(set(self.active()) & {
            "idcs-hil.target", "idcs-bridge.service", "idcs-serial.service", "idcs-controller.service"}))
        self.controller_engaged = controller_engaged or self._controller_engaged

    def _controller_engaged(self):
        """Read existing controller health reports; unavailable active owners fail closed."""
        for unit, directory in (("idcs-controller.service", "idcs-controller"),
                                ("idcs-controller-camera.service", "idcs-controller-camera")):
            if unit not in self.active():
                continue
            try:
                health = json.loads((Path("/run") / directory / "controller-health.json").read_text())
                age = time.monotonic() - health["monotonic_s"]
                reason = health["last_reason"]
                if (not 0 <= age <= 3 or not isinstance(reason, str)
                        or reason in ("tracking", "coasting")):
                    return True
            except (OSError, ValueError, KeyError, TypeError):
                return True
        return False

    def bundles(self, overlay):
        result = {}
        for unit, spec in self.stacks.items():
            bundle = runtime_bundle(spec["paths"], overlay=overlay)
            if unit.startswith("pc/") and spec["module"] == "jetson.control.video_runtime":
                from common.sim_mode import require_simulation_loopback_endpoint
                from jetson.control.runtime_config import ControlRuntimeConfig
                cfg = ControlRuntimeConfig.from_config(bundle.data)
                require_simulation_loopback_endpoint(cfg.intent_bind, "controller.endpoints.intent_bind")
                if cfg.local_clock != "pc":
                    raise ConfigError("PC software controller requires controller.local_clock pc")
            if unit.startswith("pc/") and spec["module"] == "pc.streamer":
                release = load_config_bundle(spec["paths"]).data
                actual = bool(bundle.data.get("sim", {}).get("use_jetson_cam_state", False))
                expected = bool(release.get("sim", {}).get("use_jetson_cam_state", False))
                if actual != expected:
                    raise ConfigError("sim.use_jetson_cam_state motion-source changes require a bench-reviewed release")
            result[unit] = bundle
        return result

    def handle(self, request):
        try:
            if not isinstance(request, dict):
                raise ConfigError("config request must be an object")
            with writer_lock(live_path()):
                return self._handle(request)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return dict(type="ConfigReply", ok=False, error=str(exc))

    def _handle(self, request):
        actor = request.get("actor")
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 128:
            raise ConfigError("config actor must be a nonempty string up to 128 characters")
        old = read_overlay()
        old_digest = digest(old)
        if request.get("expected_digest", old_digest) != old_digest:
            raise ConfigError("overlay changed; refresh and retry")
        action = request.get("action")
        journal = live_path().with_name("live-journal.jsonl")
        if action in ("get", "status", "diff", "fold"):
            result = dict(type="ConfigReply", ok=True, overlay=old, digest=old_digest,
                          units=self.status())
            if action in ("get", "diff"):
                base, resolved = self.bundles({}), self.bundles(old)
                result["effective"] = {u: flatten(b.mutable_copy()) for u, b in resolved.items()}
                if action == "diff":
                    result["diff"] = {u: {k: dict(old=flatten(base[u].mutable_copy()).get(k),
                        new=flatten(b.mutable_copy()).get(k), tier=tier(k))
                        for k in changed(base[u].mutable_copy(), b.mutable_copy())}
                        for u, b in resolved.items()}
                if request.get("key"):
                    key = request["key"]
                    tier(key)
                    result["effective"] = {u: values.get(key) for u, values in result["effective"].items()}
            return result
        new = merge_config_layers(old)
        if action in ("set", "unset"):
            key = request["key"]
            kind = tier(key)
            from common.config_schema import FIELDS
            if kind == "fixed" or "map" in FIELDS[key]["types"] or "[]" in key:
                raise ConfigError(f"{key}: set/unset a configurable leaf, not a section or fixed value")
            if action == "set" and not any(uses_key(spec["module"], key) for spec in self.stacks.values()):
                raise ConfigError(f"{key}: no managed unit on this host consumes this key")
            if action == "set":
                put(new, key, request["value"])
            else:
                remove(new, key)
        elif action == "clear":
            new = {}
        elif action == "revert":
            n = request.get("n", 1)
            if type(n) is not int or n < 1:
                raise ConfigError("revert count must be a positive integer")
            records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
            transactions = [r for r in records if r.get("type") == "result"]
            if n > len(transactions):
                raise ConfigError("not enough journal transactions to revert")
            new = transactions[-n]["overlay_before"]
        else:
            raise ConfigError(f"unknown config action: {action}")
        validate_fields(new, overlay=True)
        before, after = self.bundles(old), self.bundles(new)
        overlay_keys = changed(old, new)
        if not overlay_keys:
            return dict(type="ConfigReply", ok=True, digest=old_digest, units={}, changed_keys=[])
        resulting_digest = digest(new)
        affected = {}
        for unit, candidate in after.items():
            keys = [k for k in changed(before[unit].mutable_copy(), candidate.mutable_copy())
                    if uses_key(self.stacks[unit]["module"], k)]
            if keys:
                affected[unit] = keys
        # Save journal intent before publishing the atomic overlay. A crash
        # after this append is recognizable by comparing resulting_digest with
        # the actual overlay; the host never claims an application without ACK.
        transaction = uuid.uuid4().hex
        left, right = flatten(old), flatten(new)
        records = [dict(time_ns=time.time_ns(), actor=actor, transaction=transaction,
                        key=k, old=left.get(k), new=right.get(k), old_present=k in left,
                        new_present=k in right, tier=tier(k), resulting_digest=resulting_digest,
                        overlay_before=old, overlay_after=new,
                        units={u: "awaiting_service" for u, ks in affected.items() if k in ks})
                   for k in overlay_keys]
        journal.parent.mkdir(parents=True, exist_ok=True)
        for record in records:
            append_journal(record)
        atomic_write(live_path(), yaml.safe_dump(new, sort_keys=True))
        self.publish(dict(type="config_update", digest=resulting_digest, changed_keys=overlay_keys))
        active = set(self.active())
        results = {}
        restart_units = []
        for unit, keys in affected.items():
            name = unit.split("/")[-1]
            module = self.stacks[unit]["module"]
            restart = [k for k in keys if tier(k) == "restart" or not supports_hot(module, k)]
            if name not in active:
                results[unit] = dict(state="inactive", keys=keys)
            elif restart:
                # New/unknown units are denied even with NOPASSWD: ALL.
                if unit not in RESTART_ALLOWED_UNITS:
                    results[unit] = dict(state="cannot_restart", keys=keys,
                                         reason="unit is not in the restart allow-list")
                elif (self.host == "jetson" and name in PERCEPTION_UNITS
                      and (self.motors_live() or self.controller_engaged())):
                    results[unit] = dict(state="cannot_restart", keys=keys, reason="motors live")
                else:
                    restart_units.append(name)
                    results[unit] = dict(state="restart_requested", keys=keys)
            else:
                results[unit] = dict(state="awaiting_service", keys=keys)
        # Viewer/source dependency changes restart the active software SIM
        # session as one systemd transaction. try-restart never starts an
        # inactive unit. Physical/HIL units are excluded from this set.
        if self.host == "pc" and set(restart_units) & SIM_DEPENDENCIES:
            restart_units = sorted(set(restart_units) | (active & SIM_SESSION_UNITS))
        # Dependency expansion is not authority. Future session members must
        # still be explicitly approved in the host-scoped allow-list.
        for name in tuple(restart_units):
            unit = f"{self.host}/{name}"
            if unit not in RESTART_ALLOWED_UNITS:
                restart_units.remove(name)
                results.setdefault(unit, dict(keys=[], dependency_restart=True)).update(
                    state="cannot_restart", reason="unit is not in the restart allow-list")
        # Recheck immediately before executing the transaction as well as at
        # planning time; a newly active motor stack must veto perception restart.
        perception = set(restart_units) & PERCEPTION_UNITS
        if (self.host == "jetson" and perception
                and (self.motors_live() or self.controller_engaged())):
            restart_units = [name for name in restart_units if name not in perception]
            for name in perception:
                results[f"jetson/{name}"].update(state="cannot_restart", reason="motors live")
        if restart_units:
            code, message = self.run(["try-restart", *sorted(restart_units)])
            for name in restart_units:
                unit = f"{self.host}/{name}"
                results.setdefault(unit, dict(keys=[], dependency_restart=True))
                results[unit].update(state="restarted" if code == 0 else "restart_failed", message=message)
        deadline = time.monotonic() + self.result_wait_s
        while any(r["state"] == "awaiting_service" for r in results.values()):
            statuses = self.status()
            for unit, result in results.items():
                status = statuses.get(unit, {})
                if (result["state"] == "awaiting_service" and not status.get("stale", True)
                        and status.get("requested_digest") == resulting_digest):
                    result.update(status)
                    result["state"] = ("rejected" if status.get("config_error") or status.get("selection_config_error") else
                                       "pending_worker" if status.get("pending_selection_config_digest") else
                                       "pending" if status.get("pending_config_keys") else
                                       "restart_required" if status.get("restart_config_keys") else "applied")
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        append_journal(dict(type="result", transaction=transaction,
                time_ns=time.time_ns(), actor=actor, resulting_digest=resulting_digest,
                overlay_before=old, overlay_after=new, units=results))
        return dict(type="ConfigReply", ok=True, digest=resulting_digest,
                    changed_keys=overlay_keys, units=results)

    def status(self):
        result = {}
        for unit in self.stacks:
            try:
                data = json.loads(receipt_path(unit).read_text(encoding="utf-8"))
                data["stale"] = time.time_ns() - data["updated_ns"] > 3_000_000_000
                result[unit] = data
            except (OSError, ValueError, KeyError):
                result[unit] = dict(state="no_receipt")
        return result


HOT_PREFIXES = {
    "jetson.control.video_runtime": ("controller", "laser", "camera.known_size_ranging.class_sizes_m"),
    "jetson.deepstream.runtime": ("deepstream.shadow_tracks", "deepstream.id_stitching", "swarm_eval",
                                  "threat_eval.zones", "camera.known_size_ranging.class_sizes_m", "laser"),
    "jetson.operator_agent": ("operator.lock_lost_s",),
    "pc.streamer": ("laser", "threat_eval.zones"),
}


def supports_hot(module, key):
    return any(key == p or key.startswith(p + ".") for p in HOT_PREFIXES.get(module, ()))
SIM_DEPENDENCIES = {"idcs-sim-streamer.service", "idcs-sim-swarm-streamer.service", "idcs-sim-ui.service"}
SIM_SESSION_UNITS = SIM_DEPENDENCIES | {"idcs-sim-controller.service", "idcs-sim-swarm-controller.service",
                                      "idcs-sim-panel.service", "idcs-sim-recorder.service"}
PERCEPTION_UNITS = {"idcs-deepstream-camera.service", "idcs-deepstream-video.service"}
RESTART_ALLOWED_UNITS = frozenset({
    "jetson/idcs-deepstream-camera.service", "jetson/idcs-deepstream-video.service",
    "jetson/idcs-recorder.service", "pc/idcs-camera-ui.service", "pc/idcs-ui.service", "pc/idcs-hud-relay.service",
    "pc/idcs-sim-streamer.service", "pc/idcs-sim-swarm-streamer.service", "pc/idcs-sim-ui.service",
    "pc/idcs-sim-controller.service", "pc/idcs-sim-swarm-controller.service",
    "pc/idcs-sim-panel.service", "pc/idcs-sim-recorder.service",
})


class AgentChannel:
    """Owned by an existing agent loop, using host-local IPC endpoints."""
    def __init__(self, context, agent):
        import zmq
        live_path().parent.mkdir(parents=True, exist_ok=True)
        self.agent = agent
        self.rep = context.socket(zmq.REP)
        self.pub = context.socket(zmq.PUB)
        for socket in (self.rep, self.pub):
            socket.setsockopt(zmq.LINGER, 0)
        self.rep.bind(command_endpoint())
        self.pub.bind(update_endpoint())
        agent.publish = self.pub.send_json

    def poll(self):
        import zmq
        if self.rep.poll(0):
            try:
                request = self.rep.recv_json()
                response = self.agent.handle(request)
            except (ValueError, TypeError) as exc:
                response = dict(type="ConfigReply", ok=False, error=str(exc))
            self.rep.send_json(response)

    def close(self):
        self.rep.close(0)
        self.pub.close(0)
