"""Operator agent: the one service that acts on the panel screen's commands.

The panel display (``rpi.operator_display``) is read-only; every change it
asks for comes here as an ``OperatorCommand`` (``common.operator_commands``)
and is answered with the resulting state:

- target lock / release: held here and published as ``OperatorSelection``
  to the DeepStream target selector, which then selects the locked track
  while it exists. A locked track missing from perception for
  ``operator.lock_lost_s`` is released.
- target classes: which detector classes the selector may choose; persisted.
- mode: the configured units started and stopped (``operator.modes``), never
  while a motor unit (``operator.motor_units``) is active.
- recording: the flight recorder unit started or stopped.

    python -m jetson.operator_agent --config configs/base [--check]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import zmq

from common.config import ConfigError, load_config_bundle, resolve_config_paths
from common.operator_commands import CommandError, OperatorCommand, OperatorSelection, parse_command

log = logging.getLogger("jetson.operator_agent")

SELECTION_PERIOD_S = 0.2
UNIT_POLL_S = 2.0
PERCEPTION_STALE_S = 1.0


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    start: tuple[str, ...]
    stop: tuple[str, ...]


@dataclass(frozen=True)
class AgentConfig:
    command_bind: str
    selection_bind: str
    perception: str
    modes: tuple[Mode, ...]
    motor_units: tuple[str, ...]
    recorder_unit: str
    lock_lost_s: float
    state_file: Path
    class_names: tuple[str, ...]
    default_target_classes: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "AgentConfig":
        net = cfg.get("net") or {}
        op = cfg.get("operator") or {}
        if not op:
            raise ConfigError("operator section is required")
        command = str(net.get("zmq_operator_command") or "")
        selection = str(net.get("zmq_operator_selection") or "")
        if not command.startswith("tcp://") or not selection.startswith("tcp://"):
            raise ConfigError("net.zmq_operator_command and net.zmq_operator_selection must be tcp:// URLs")
        modes = []
        for name, raw in (op.get("modes") or {}).items():
            if not isinstance(raw, Mapping):
                raise ConfigError(f"operator.modes.{name} must be a mapping")
            modes.append(Mode(str(name), str(raw.get("label") or name),
                              tuple(str(u) for u in raw.get("start") or ()),
                              tuple(str(u) for u in raw.get("stop") or ())))
        if not modes:
            raise ConfigError("operator.modes must define at least one mode")
        labels = ((cfg.get("perception") or {}).get("class_labels") or {})
        class_names = tuple(sorted({str(v).strip().lower() for v in labels.values() if str(v).strip()}))
        if not class_names:
            raise ConfigError("perception.class_labels is required (the classes the operator can target)")
        excluded = {str(c).strip().lower() for c in (cfg.get("swarm_eval") or {}).get("excluded_target_classes") or ()}
        defaults = tuple(c for c in class_names if c not in excluded) or class_names
        lock_lost_s = float(op.get("lock_lost_s", 1.0))
        if not 0.1 <= lock_lost_s <= 10.0:
            raise ConfigError("operator.lock_lost_s must be in [0.1, 10]")
        return cls(
            command_bind=_bind_address(command), selection_bind=_bind_address(selection),
            perception=str(net.get("zmq_perception_v2") or ""), modes=tuple(modes),
            motor_units=tuple(str(u) for u in op.get("motor_units") or ()),
            recorder_unit=str(op.get("recorder_unit") or "idcs-recorder.service"),
            lock_lost_s=lock_lost_s,
            state_file=Path(str(op.get("state_file") or "~/.local/state/idcs/operator_agent.json")).expanduser(),
            class_names=class_names, default_target_classes=defaults,
        )

    def mode(self, name: str) -> Mode | None:
        return next((m for m in self.modes if m.name == name), None)

    def units(self) -> tuple[str, ...]:
        names = {self.recorder_unit, *self.motor_units}
        for mode in self.modes:
            names.update(mode.start)
            names.update(mode.stop)
        return tuple(sorted(names))


def _bind_address(endpoint: str) -> str:
    """tcp://192.168.0.5:5590 -> tcp://*:5590; loopback endpoints stay loopback."""
    host_port = endpoint.removeprefix("tcp://")
    host, _, port = host_port.rpartition(":")
    return endpoint if host in ("127.0.0.1", "localhost") else f"tcp://*:{port}"


def systemctl_runner(args: Sequence[str], *, timeout_s: float = 60.0) -> tuple[int, str]:
    """``systemctl`` (state queries) or ``sudo -n systemctl`` (start/stop)."""
    privileged = bool(args) and args[0] in ("start", "stop")
    argv = (["sudo", "-n", "systemctl"] if privileged else ["systemctl"]) + list(args)
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return done.returncode, (done.stdout + done.stderr).strip()


@dataclass
class OperatorAgent:
    config: AgentConfig
    run: Callable[..., tuple[int, str]] = systemctl_runner
    clock: Callable[[], float] = time.monotonic
    target_classes: tuple[str, ...] = ()
    lock_track_id: int | None = None
    active_units: frozenset[str] = frozenset()
    busy: str | None = None
    last_message: str = ""
    sequence: int = 0
    _track_ids: frozenset[int] = frozenset()
    _perception_s: float | None = None
    _lock_seen_s: float | None = None
    _job: threading.Thread | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.target_classes:
            self.target_classes = self._load_classes()

    # --- state ---------------------------------------------------------------

    def refresh_units(self) -> None:
        units = self.config.units()
        code, out = self.run(["is-active", *units])
        states = out.split()
        if len(states) != len(units):
            log.warning("systemctl is-active returned %r (code %d)", out, code)
            return
        self.active_units = frozenset(u for u, s in zip(units, states) if s in ("active", "activating", "reloading"))

    def current_mode(self) -> str:
        active = self.active_units
        for mode in self.config.modes:
            if mode.start and all(u in active for u in mode.start):
                return mode.name
        if any(u in active for mode in self.config.modes for u in mode.start):
            return "mixed"
        return next((m.name for m in self.config.modes if not m.start), "none")

    def motors_live(self) -> bool:
        return any(u in self.active_units for u in self.config.motor_units)

    def state(self) -> dict[str, Any]:
        return {
            "mode": self.current_mode(),
            "modes": [{"name": m.name, "label": m.label} for m in self.config.modes],
            "recording": self.config.recorder_unit in self.active_units,
            "motors_live": self.motors_live(),
            "busy": self.busy,
            "lock_track_id": self.lock_track_id,
            "target_classes": list(self.target_classes),
            "class_names": list(self.config.class_names),
            "message": self.last_message,
        }

    def selection(self) -> OperatorSelection:
        return OperatorSelection(self.sequence, self.lock_track_id, self.target_classes)

    # --- perception (lock lifetime) -------------------------------------------

    def on_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        now = self.clock()
        self._perception_s = now
        self._track_ids = frozenset(int(t["track_id"]) for t in snapshot.get("tracks") or ()
                                    if isinstance(t, Mapping) and isinstance(t.get("track_id"), int))
        if self.lock_track_id is not None and self.lock_track_id in self._track_ids:
            self._lock_seen_s = now

    def expire_lock(self) -> None:
        if self.lock_track_id is None or self._lock_seen_s is None:
            return
        if self.clock() - self._lock_seen_s > self.config.lock_lost_s:
            self._set_message(f"lock on #{self.lock_track_id} lost")
            self.lock_track_id = None
            self._lock_seen_s = None
            self.sequence += 1

    def _perception_live(self) -> bool:
        return self._perception_s is not None and self.clock() - self._perception_s <= PERCEPTION_STALE_S

    # --- commands --------------------------------------------------------------

    def handle(self, command: OperatorCommand) -> tuple[bool, str]:
        if command.command == "status":
            return True, self.last_message
        if command.command == "lock":
            assert command.track_id is not None
            if not self._perception_live():
                return self._refuse("no perception: nothing to lock")
            if command.track_id not in self._track_ids:
                return self._refuse(f"track #{command.track_id} is not present")
            self.lock_track_id = command.track_id
            self._lock_seen_s = self.clock()
            self.sequence += 1
            return self._accept(f"locked #{command.track_id}")
        if command.command == "release":
            if self.lock_track_id is None:
                return self._accept("no lock")
            released = self.lock_track_id
            self.lock_track_id = None
            self._lock_seen_s = None
            self.sequence += 1
            return self._accept(f"released #{released}")
        if command.command == "target_classes":
            unknown = [c for c in command.classes if c not in self.config.class_names]
            if unknown:
                return self._refuse(f"unknown classes {unknown}; known {list(self.config.class_names)}")
            self.target_classes = tuple(command.classes)
            self.sequence += 1
            self._save_classes()
            return self._accept("targets: " + ", ".join(self.target_classes))
        if command.command == "set_mode":
            mode = self.config.mode(command.mode or "")
            if mode is None:
                return self._refuse(f"unknown mode {command.mode!r}")
            if self.motors_live():
                return self._refuse("motor stack running: switch modes at the bench")
            return self._start_job(f"mode {mode.label}", [("start", mode.start), ("stop", mode.stop)])
        if command.command == "recording":
            unit = self.config.recorder_unit
            return self._start_job("recording " + ("on" if command.on else "off"),
                                   [("start" if command.on else "stop", (unit,))])
        return self._refuse(f"unsupported command {command.command}")

    def _start_job(self, label: str, steps: list[tuple[str, tuple[str, ...]]]) -> tuple[bool, str]:
        with self._lock:
            if self.busy is not None:
                return self._refuse(f"busy: {self.busy}")
            self.busy = label

        def work() -> None:
            failed = None
            for verb, units in steps:
                if not units:
                    continue
                code, out = self.run([verb, *units])
                if code != 0:
                    failed = f"{verb} {' '.join(units)}: {out or code}"
                    break
            self.refresh_units()
            with self._lock:
                self.busy = None
            self._set_message(f"{label} failed: {failed}" if failed else f"{label} done")

        self._job = threading.Thread(target=work, name="operator-job", daemon=True)
        self._job.start()
        return self._accept(f"{label}: started")

    def wait_idle(self, timeout_s: float = 5.0) -> None:
        if self._job is not None:
            self._job.join(timeout_s)

    def _accept(self, message: str) -> tuple[bool, str]:
        self._set_message(message)
        return True, message

    def _refuse(self, message: str) -> tuple[bool, str]:
        log.info("refused: %s", message)
        return False, message

    def _set_message(self, message: str) -> None:
        self.last_message = message
        log.info("%s", message)

    # --- persistence -----------------------------------------------------------

    def _load_classes(self) -> tuple[str, ...]:
        try:
            stored = json.loads(self.config.state_file.read_text())
            classes = tuple(c for c in stored.get("target_classes", ()) if c in self.config.class_names)
            if classes:
                return classes
        except (OSError, ValueError, AttributeError):
            pass
        return self.config.default_target_classes

    def _save_classes(self) -> None:
        path = self.config.state_file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"target_classes": list(self.target_classes)}))
            tmp.replace(path)
        except OSError as exc:
            log.warning("could not save %s: %s", path, exc)


def reply(agent: OperatorAgent, raw: bytes) -> str:
    try:
        command = parse_command(raw)
    except CommandError as exc:
        ok, message = False, str(exc)
    else:
        ok, message = agent.handle(command)
    return json.dumps({"type": "OperatorReply", "ok": ok, "message": message, "state": agent.state()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base")
    parser.add_argument("--config-extra", default=None)
    parser.add_argument("--check", action="store_true", help="validate the configuration and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    try:
        bundle = load_config_bundle(resolve_config_paths(args.config, args.config_extra))
        config = AgentConfig.from_config(bundle.data)
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from exc
    if args.check:
        print(json.dumps({"command": config.command_bind, "selection": config.selection_bind,
                          "modes": [m.name for m in config.modes], "classes": config.class_names}))
        return 0

    agent = OperatorAgent(config)
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(config.command_bind)
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(config.selection_bind)
    perception = None
    if config.perception:
        perception = ctx.socket(zmq.SUB)
        perception.setsockopt(zmq.CONFLATE, 1)
        perception.setsockopt(zmq.LINGER, 0)
        perception.setsockopt_string(zmq.SUBSCRIBE, "")
        perception.connect(config.perception)
    poller = zmq.Poller()
    poller.register(rep, zmq.POLLIN)
    if perception is not None:
        poller.register(perception, zmq.POLLIN)

    stop = threading.Event()
    import signal
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    agent.refresh_units()
    log.info("serving operator commands on %s; selection on %s; targets %s; mode %s",
             config.command_bind, config.selection_bind, list(agent.target_classes), agent.current_mode())
    next_units = time.monotonic() + UNIT_POLL_S
    next_selection = 0.0
    published_sequence = -1
    try:
        while not stop.is_set():
            events = dict(poller.poll(50))
            if perception is not None and perception in events:
                try:
                    snapshot = json.loads(perception.recv(zmq.NOBLOCK))
                except (zmq.Again, ValueError):
                    snapshot = None
                if isinstance(snapshot, dict) and snapshot.get("type") == "PerceptionSnapshot":
                    agent.on_snapshot(snapshot)
            if rep in events:
                rep.send_string(reply(agent, rep.recv()))
            agent.expire_lock()
            now = time.monotonic()
            if now >= next_units and agent.busy is None:
                agent.refresh_units()
                next_units = now + UNIT_POLL_S
            if now >= next_selection or agent.sequence != published_sequence:
                pub.send_string(agent.selection().to_json())
                published_sequence = agent.sequence
                next_selection = now + SELECTION_PERIOD_S
    finally:
        ctx.destroy(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
