"""Host-local, validated live overlays and safe-point application.

No hardware is constructed here. Transport is local IPC; persistent overlay
polling also recovers updates missed during startup or PUB disconnection.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
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
        os.replace(temporary, path)
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


class LiveSettings:
    """One service's settings, swapped only by its owning tick/probe thread.

    Callbacks must construct/validate new objects before assigning them. They
    must not replace observation/intent queues or reset controller sequencing.
    """
    def __init__(self, paths, *, unit: str, apply: Callable[[dict], None], hot_prefixes=(), initial_bundle=None):
        self.paths = tuple(paths)
        self.unit, self.apply = unit, apply
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
        self._signature = None
        self._next_poll = 0.0
        self._next_receipt = 0.0
        self._ctx = self._sub = None

    def subscribe(self):
        import zmq
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sub.connect(update_endpoint())

    def close(self):
        if self._ctx is not None:
            self._ctx.destroy(linger=0)

    def status(self) -> dict:
        return dict(config_digest=self.bundle.digest, requested_digest=self.requested_digest,
                    pending_config_keys=self.pending, restart_config_keys=self.restart,
                    config_error=self.error, **self.extra_status())

    def tick(self, *, engaged: bool, force: bool = False) -> bool:
        now = time.monotonic()
        if self._sub is not None:
            import zmq
            try:
                while True:
                    self._sub.recv(zmq.NOBLOCK)
                    force = True
            except zmq.Again:
                pass
        # Held idle-tier keys are re-evaluated only once the service is no longer
        # engaged; while engaged, only a changed overlay triggers a rebuild, so
        # the owning tick does not re-parse the config stack every period.
        retry_pending = bool(self.pending) and not engaged
        if not force and now < self._next_poll and not retry_pending:
            return False
        self._next_poll = now + 0.2
        applied = False
        try:
            overlay = read_overlay()
            self.requested_digest = digest(overlay)
            signature = self.requested_digest
            if force or signature != self._signature or retry_pending:
                candidate = runtime_bundle(self.paths, overlay=overlay)
                old, new = self.bundle.mutable_copy(), candidate.mutable_copy()
                keys = changed(old, new)
                self.pending, self.restart = [], []
                for key in keys:
                    if self.module is not None and not uses_key(self.module, key):
                        continue
                    kind = tier(key)
                    hot = any(key == p or key.startswith(p + ".") for p in self.hot_prefixes)
                    if kind == "restart" or not hot:
                        self.restart.append(key)
                    elif kind == "idle" and engaged:
                        self.pending.append(key)
                    else:
                        continue
                    if key in flatten(old):
                        put(new, key, flatten(old)[key])
                    else:
                        remove(new, key)
                if changed(old, new):
                    # Revalidate the *partially* applied combination as well.
                    # A live value depending on a pending cap must wait, rather
                    # than create an invalid intermediate setting.
                    authored = merge_config_layers(new)
                    from common.config_constants import FIXED_VALUES
                    for key in FIXED_VALUES:
                        remove(authored, key)
                    validate_config(authored)
                    self.apply(new)
                    self.bundle = replace(candidate, data=_freeze_mapping(new), digest=digest(new))
                    applied = True
                    for key in changed(old, new):
                        if self.module is None or uses_key(self.module, key):
                            append_journal(dict(type="application", time_ns=time.time_ns(),
                                actor=self.unit, key=key, old=flatten(old).get(key),
                                new=flatten(new).get(key), tier=tier(key),
                                resulting_digest=self.requested_digest,
                                config_digest=self.bundle.digest, units={self.unit: "applied"}))
                self._signature = signature
                self.error = None
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.error = str(exc)
        if force or applied or now >= self._next_receipt:
            atomic_write(receipt_path(self.unit), json.dumps(dict(unit=self.unit,
                         updated_ns=time.time_ns(), **self.status()), sort_keys=True) + "\n")
            self._next_receipt = now + 1.0
        return applied


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
                 stacks=None, result_wait_s=1.0):
        self.root, self.host = Path(root), host
        self.run, self.publish, self.active = run, publish, active
        self.stacks = unit_stacks(self.root, host) if stacks is None else stacks
        self.result_wait_s = result_wait_s

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
                # Never automatically restart physical owners, HIL, or the
                # command agent itself. These require a reviewed operator step.
                if name in PROTECTED_UNITS:
                    results[unit] = dict(state="cannot_restart", keys=keys,
                                         reason="physical/HIL owner or current command agent")
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
PROTECTED_UNITS = {"idcs-bridge.service", "idcs-serial.service", "idcs-controller.service",
                   "idcs-controller-camera.service",
                   "idcs-hil-streamer.service", "idcs-operator-agent.service", "idcs-pc-mode-agent.service"}
SIM_DEPENDENCIES = {"idcs-sim-streamer.service", "idcs-sim-swarm-streamer.service", "idcs-sim-ui.service"}
SIM_SESSION_UNITS = SIM_DEPENDENCIES | {"idcs-sim-controller.service", "idcs-sim-swarm-controller.service",
                                      "idcs-sim-panel.service", "idcs-sim-recorder.service"}


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
