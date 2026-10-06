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
from common.pc_mode import PC_ACTIONS, PC_STATUS_UNITS
from jetson.pc_mode_channel import PcModeChannel
from common.control_activity import SimActivityRelay

log = logging.getLogger("jetson.operator_agent")

SELECTION_PERIOD_S = 0.2
UNIT_POLL_S = 2.0
PERCEPTION_STALE_S = 1.0
MODE_UNITS = frozenset({"idcs-camera.target", "idcs-deepstream-camera.service",
                        "idcs-controller-camera.service", "idcs-deepstream-video.service"})
MOTOR_UNITS = frozenset({"idcs-hil.target", "idcs-bridge.service", "idcs-serial.service",
                         "idcs-controller.service"})


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    start: tuple[str, ...]
    stop: tuple[str, ...]
    pc_start: tuple[str, ...] = ()
    pc_stop: tuple[str, ...] = ()
    ready_units: tuple[str, ...] = ()
    pc_ready_units: tuple[str, ...] = ()
    source_clock: str = ""


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
    pc_bind: str = "tcp://*:5592"
    readiness_timeout_s: float = 30.0
    health_file: Path = Path("/run/idcs/deepstream-video-health.json")
    activity_bind: str = "tcp://*:5593"

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
            def units(key: str) -> tuple[str, ...]:
                value = raw.get(key) or ()
                if not isinstance(value, (list, tuple)) or not all(isinstance(u, str) for u in value):
                    raise ConfigError(f"operator.modes.{name}.{key} must be a unit list")
                return tuple(value)

            mode = Mode(str(name), str(raw.get("label") or name), units("start"), units("stop"),
                        units("pc_start"), units("pc_stop"), units("ready_units"),
                        units("pc_ready_units"), str(raw.get("source_clock") or ""))
            if any(u not in MODE_UNITS for u in (*mode.start, *mode.stop, *mode.ready_units)):
                raise ConfigError(f"operator.modes.{name}: non-video Jetson unit")
            if any((verb, u) not in PC_ACTIONS for verb, us in
                   (("start", mode.pc_start), ("stop", mode.pc_stop)) for u in us):
                raise ConfigError(f"operator.modes.{name}: non-allowlisted PC action")
            if any(u not in PC_STATUS_UNITS for u in mode.pc_ready_units):
                raise ConfigError(f"operator.modes.{name}: unknown PC readiness unit")
            if mode.source_clock not in ("", "pc_monotonic", "jetson_monotonic"):
                raise ConfigError(f"operator.modes.{name}: unknown source clock")
            if mode.start and not mode.source_clock:
                raise ConfigError(f"operator.modes.{name}: video source clock required")
            modes.append(mode)
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
        pc_endpoint = str(net.get("zmq_operator_pc") or "")
        if not pc_endpoint.startswith("tcp://"):
            raise ConfigError("net.zmq_operator_pc must be a tcp:// URL")
        timeout = float(op.get("readiness_timeout_s", 30.0))
        if not 1 <= timeout <= 120:
            raise ConfigError("operator.readiness_timeout_s must be in [1, 120]")
        return cls(
            command_bind=_bind_address(command), selection_bind=_bind_address(selection),
            perception=str(net.get("zmq_perception_v2") or ""), modes=tuple(modes),
            motor_units=tuple(sorted(MOTOR_UNITS | set(op.get("motor_units") or ()))),
            recorder_unit=str(op.get("recorder_unit") or "idcs-recorder.service"),
            lock_lost_s=lock_lost_s,
            state_file=Path(str(op.get("state_file") or "~/.local/state/idcs/operator_agent.json")).expanduser(),
            class_names=class_names, default_target_classes=defaults,
            pc_bind=_bind_address(pc_endpoint), readiness_timeout_s=timeout,
            health_file=Path(str(op.get("health_file") or "/run/idcs/deepstream-video-health.json")),
            activity_bind=_bind_address(str(net.get("zmq_sim_activity") or "tcp://192.168.0.5:5593")),
        )

    def mode(self, name: str) -> Mode | None:
        return next((m for m in self.modes if m.name == name), None)

    def units(self) -> tuple[str, ...]:
        names = {self.recorder_unit, *self.motor_units}
        for mode in self.modes:
            names.update(mode.start)
            names.update(mode.stop)
            names.update(mode.ready_units)
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
    pc: PcModeChannel = field(default_factory=PcModeChannel)
    ready_probe: Callable[[Mode, float], bool] | None = None
    target_classes: tuple[str, ...] = ()
    lock_track_id: int | None = None
    active_units: frozenset[str] = frozenset()
    busy: str | None = None
    last_message: str = ""
    sequence: int = 0
    _track_ids: frozenset[int] = frozenset()
    _perception_s: float | None = None
    _source_clock: str = ""
    _source_frame: tuple[Any, ...] | None = None
    _fresh_frames: int = 0
    _transition_s: float = 0.0
    _unit_states: dict[str, str] = field(default_factory=dict)
    _units_known: bool = False
    _lock_seen_s: float | None = None
    _job: threading.Thread | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.target_classes:
            self.target_classes = self._load_classes()

    # --- state ---------------------------------------------------------------

    def refresh_units(self) -> bool:
        units = self.config.units()
        code, out = self.run(["is-active", *units])
        states = out.split()
        if (len(states) != len(units) or any(s not in
                {"active", "inactive", "activating", "deactivating", "reloading", "failed"} for s in states)):
            log.warning("systemctl is-active returned %r (code %d)", out, code)
            self._units_known = False
            return False
        self._unit_states = dict(zip(units, states))
        self._units_known = True
        self.active_units = frozenset(u for u, s in zip(units, states)
                                      if s in ("active", "activating", "reloading", "deactivating"))
        return True

    def current_mode(self) -> str:
        if not self._units_known:
            return "unknown"
        active = self.active_units
        for mode in self.config.modes:
            if (mode.start and all(self._unit_states.get(u) == "active" for u in
                                  (*mode.start, *mode.ready_units))
                    and self.pc.online() and all(u in self.pc.active_units for u in mode.pc_ready_units)):
                return mode.name
        if (any(u in active for mode in self.config.modes for u in mode.start)
                or (self.pc.online() and any(u in self.pc.active_units for m in self.config.modes for u in m.pc_start))):
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
            "pc_online": self.pc.online(),
            "pc_action_pending": self.pc.pending(),
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
        frame = snapshot.get("frame") or {}
        source_clock = str(frame.get("source_clock_domain") or "")
        if source_clock != self._source_clock:
            self._fresh_frames = 0
        self._source_clock = source_clock
        identity = (self._source_clock, frame.get("frame_id"), frame.get("source_time_ns"))
        observed_ns = frame.get("observed_time_ns")
        if (identity != self._source_frame and frame.get("frame_id") is not None
                and isinstance(observed_ns, int) and observed_ns >= self._transition_s * 1e9
                and frame.get("observation_clock_domain") == "jetson_monotonic"):
            self._fresh_frames += 1
        self._source_frame = identity
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
            if self.busy is not None:
                return self._refuse("source transition in progress: lock unavailable")
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
            if self.busy is not None:
                return self._refuse(f"busy: {self.busy}")
            if not self.refresh_units():
                return self._refuse("Jetson unit status unavailable: no source changes")
            if self.motors_live():
                return self._refuse("motor stack running: switch modes at the bench")
            if not self.pc.online():
                return self._refuse("PC mode agent offline: no source changes")
            if "idcs-hil-streamer.service" in self.pc.active_units:
                return self._refuse("HIL streamer running: switch modes at the bench")
            if self.pc.pending():
                return self._refuse("PC action acknowledgement still pending: no source changes")
            return self._start_mode_job(mode)
        if command.command == "recording":
            unit = self.config.recorder_unit
            return self._start_job("recording " + ("on" if command.on else "off"),
                                   [("start" if command.on else "stop", (unit,))])
        return self._refuse(f"unsupported command {command.command}")

    def _video_ready(self, mode: Mode, began: float) -> bool:
        if not mode.source_clock:
            return True
        if (not self._perception_live() or self._source_clock != mode.source_clock
                or self._fresh_frames < 3):
            return False
        try:
            health = json.loads(self.config.health_file.read_text())
            last = float(health["last_frame_monotonic_s"])
            return (began <= last <= self.clock() and self.clock() - last <= 2.0
                    and int(health["frames"]) >= 3 and int(health["return_frames"]) > 0)
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _transition(self, mode: Mode) -> None:
        self._transition_s = began = self.clock()
        self._fresh_frames = 0
        self._perception_s = None
        self._source_frame = None
        # Outgoing sources release shared UDP/ZMQ ports before incoming ones
        # start. PC acknowledgements cannot be overtaken by later operations.
        for host, verb, units in (("PC", "stop", mode.pc_stop),
                                  ("Jetson", "stop", mode.stop),
                                  ("Jetson", "start", mode.start),
                                  ("PC", "start", mode.pc_start)):
            for unit in units:
                self.busy = f"{mode.label}: {host} {verb} {unit}"
                if not self.refresh_units() or self.motors_live():
                    raise RuntimeError("motor stack active or status unavailable")
                if host == "PC":
                    code, out = self.pc.run([verb, unit])
                else:
                    code, out = self.run([verb, unit], timeout_s=25.0)
                if code:
                    raise RuntimeError(f"{host} {verb} {unit}: {out or code}")
            if host == "Jetson" and verb == "stop":
                # Only observations after all outgoing pipelines stop may
                # qualify readiness, including a restart of the same source.
                self._transition_s = began = self.clock()
                self._fresh_frames = 0
                self._source_frame = None
                self._perception_s = None
        self.busy = f"{mode.label}: waiting for video and return stream"
        deadline = time.monotonic() + self.config.readiness_timeout_s
        while time.monotonic() < deadline:
            if not self.refresh_units() or self.motors_live():
                raise RuntimeError("motor stack active or status unavailable")
            if (self.pc.online() and all(self._unit_states.get(u) == "active" for u in (*mode.start, *mode.ready_units))
                    and all(self._unit_states.get(u) in ("inactive", "failed") for u in mode.stop
                            if u not in (*mode.start, *mode.ready_units))
                    and all(u not in self.pc.active_units for u in mode.pc_stop
                            if u not in (*mode.pc_start, *mode.pc_ready_units))
                    and all(u in self.pc.active_units for u in mode.pc_ready_units)
                    and (self.ready_probe or self._video_ready)(mode, began)):
                return
            time.sleep(0.1)
        raise RuntimeError("incoming source/return video did not become ready")

    def _start_mode_job(self, mode: Mode) -> tuple[bool, str]:
        previous = self.config.mode(self.current_mode()) or self.config.mode("standby")
        with self._lock:
            if self.busy is not None:
                return self._refuse(f"busy: {self.busy}")
            self.busy = f"mode {mode.label}"
        self.lock_track_id = None
        self._lock_seen_s = None
        self._track_ids = frozenset()
        self.sequence += 1
        message = f"mode {mode.label}: started"
        self._set_message(message)

        def work() -> None:
            try:
                self._transition(mode)
                result = f"mode {mode.label} ready"
            except Exception as exc:
                result = f"mode {mode.label} failed: {exc}"
                # Never overtake a timed-out PC task, or interrupt a bench
                # session. Report recovery needed instead of claiming success.
                if (previous is not None and self.pc.online() and not self.pc.pending()
                        and self.refresh_units() and not self.motors_live()
                        and "idcs-hil-streamer.service" not in self.pc.active_units):
                    try:
                        self._transition(previous)
                        result += f"; restored {previous.label}"
                    except Exception as recovery:
                        result += f"; recovery failed: {recovery}"
                else:
                    result += "; recovery required (source state unconfirmed)"
            finally:
                self.refresh_units()
                self._set_message(result)
                with self._lock:
                    self.busy = None

        self._job = threading.Thread(target=work, name="operator-mode", daemon=True)
        self._job.start()
        return True, message

    def _start_job(self, label: str, steps: list[tuple[str, tuple[str, ...]]]) -> tuple[bool, str]:
        with self._lock:
            if self.busy is not None:
                return self._refuse(f"busy: {self.busy}")
            self.busy = label
        message = f"{label}: started"
        self._set_message(message)

        def work() -> None:
            try:
                for verb, units in steps:
                    if not units:
                        continue
                    code, out = self.run([verb, *units], timeout_s=25.0)
                    if code != 0:
                        raise RuntimeError(f"{verb} {' '.join(units)}: {out or code}")
                self._set_message(f"{label} done")
            except Exception as exc:
                self._set_message(f"{label} failed: {exc}")
            finally:
                self.refresh_units()
                with self._lock:
                    self.busy = None

        self._job = threading.Thread(target=work, name="operator-job", daemon=True)
        self._job.start()
        return True, message

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
        print(json.dumps({"command": config.command_bind, "selection": config.selection_bind, "pc": config.pc_bind,
                          "modes": [m.name for m in config.modes], "classes": config.class_names}))
        return 0

    agent = OperatorAgent(config)
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(config.command_bind)
    pc_rep = ctx.socket(zmq.REP)
    pc_rep.setsockopt(zmq.LINGER, 0)
    pc_rep.bind(config.pc_bind)
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(config.selection_bind)
    activity_pub = ctx.socket(zmq.PUB)
    activity_pub.setsockopt(zmq.LINGER, 0)
    activity_pub.setsockopt(zmq.SNDHWM, 1)
    activity_pub.bind(config.activity_bind)
    perception = None
    if config.perception:
        perception = ctx.socket(zmq.SUB)
        perception.setsockopt(zmq.CONFLATE, 1)
        perception.setsockopt(zmq.LINGER, 0)
        perception.setsockopt_string(zmq.SUBSCRIBE, "")
        perception.connect(config.perception)
    poller = zmq.Poller()
    poller.register(rep, zmq.POLLIN)
    poller.register(pc_rep, zmq.POLLIN)
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
            if pc_rep in events:
                try:
                    response = agent.pc.poll(pc_rep.recv())
                except ValueError as exc:
                    response = json.dumps({"type": "PcModeError", "message": str(exc)})
                pc_rep.send_string(response)
            agent.expire_lock()
            now = time.monotonic()
            activity = (agent.pc.activity.forward(now)
                        if agent.busy is None and agent.pc.activity.fresh(now) else None)
            try:
                activity_pub.send_string(SimActivityRelay(activity=activity).model_dump_json(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
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
