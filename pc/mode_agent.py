"""PC simulator service agent: polls the Jetson, with no inbound PC endpoint."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Sequence

import zmq

from common.live_config import runtime_bundle
from common.config import resolve_config_paths
from common.pc_mode import PC_ACTIONS, PC_STATUS_UNITS, PcModePoll, PcModeTask, parse_task, sim_activity_units_running
from common.control_activity import ActivityCache, ActivityEnvelope
from common.schemas import CamState
from common.sim_mode import require_simulation_loopback_endpoint
from common.target_authority import CatalogCache, CatalogEnvelope
from common.operator_commands import parse_selection
from pc.sim_score import clear_score
from common.sim_score import DEFAULT_SCORE_ENDPOINT
from functools import partial

log = logging.getLogger("pc.mode_agent")


def systemctl_runner(args: Sequence[str], *, timeout_s: float = 25.0) -> tuple[int, str]:
    try:
        result = subprocess.run(["systemctl", "--user", *args], capture_output=True,
                                text=True, timeout=timeout_s)
        return result.returncode, (result.stdout + result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)


def active_units(run: Callable = systemctl_runner) -> frozenset[str]:
    _, output = run(["is-active", *PC_STATUS_UNITS], timeout_s=2.0)
    states = output.split()
    if (len(states) != len(PC_STATUS_UNITS) or any(s not in
            {"active", "inactive", "activating", "deactivating", "reloading", "failed"} for s in states)):
        raise RuntimeError("PC unit-state query failed: " + output[:200])
    # Activating is not ready. A target with failed Wants is not a healthy sim.
    return frozenset(unit for unit, state in zip(PC_STATUS_UNITS, states)
                     if state == "active" or (unit == "idcs-hil-streamer.service"
                                             and state in ("activating", "reloading", "deactivating")))


@dataclass
class PcModeWorker:
    run: Callable = systemctl_runner
    reset_score: Callable = clear_score
    session: str = field(default_factory=lambda: uuid.uuid4().hex)
    _task: PcModeTask | None = None
    _result: tuple[int, bool, str] | None = None
    _job: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def poll(self, activity: ActivityEnvelope | Callable[[], ActivityEnvelope | None] | None = None,
             targets: Callable[[], CatalogEnvelope | None] | None = None) -> PcModePoll:
        units = active_units(self.run)
        if callable(activity):
            activity = activity()  # sample after a potentially slow systemd query
        if not sim_activity_units_running(units):
            activity = None
        catalog = targets() if targets is not None and sim_activity_units_running(units) else None
        with self._lock:
            result = self._result
        return PcModePoll(units, *(result or (None, None, "")), session=self.session, activity=activity, targets=catalog)

    def accept(self, task: PcModeTask | None) -> None:
        if task is None:
            return
        with self._lock:
            # Keep the completion until a different task arrives. Repeated
            # replies after a dropped acknowledgement never repeat the action.
            if self._task is not None and self._task.task_id == task.task_id:
                return
            if self._job is not None and self._job.is_alive():
                raise RuntimeError("received another PC task while one is running")
            self._task, self._result = task, None

        def work() -> None:
            try:
                if (task.verb, task.unit) not in PC_ACTIONS:
                    raise ValueError("PC action is not allowlisted")
                # A HIL source is a bench session; the menu never interrupts it.
                units = active_units(self.run)
                if "idcs-hil-streamer.service" in units:
                    raise RuntimeError("HIL streamer is active")
                if task.verb == 'clear_score':
                    if task.unit not in units or 'idcs-sim-streamer.service' in units:
                        raise RuntimeError('exclusive Swarm streamer is not active')
                    code, output = self.reset_score(task, self.session)
                else:
                    code, output = self.run([task.verb, task.unit])
                if code == 0 and task.verb != 'clear_score':
                    units = active_units(self.run)
                    expected = task.verb == "start"
                    if (task.unit in units) != expected:
                        code, output = 1, "unit did not reach the requested state"
                result = (task.task_id, code == 0, output[:300])
            except Exception as exc:
                result = (task.task_id, False, str(exc)[:300])
            with self._lock:
                self._result = result
            log.info("PC task %s %s: %s", task.verb, task.unit, "done" if result[1] else result[2])

        self._job = threading.Thread(target=work, name="pc-mode-task", daemon=True)
        self._job.start()

    def wait_idle(self, timeout_s: float = 30.0) -> None:
        if self._job is not None:
            self._job.join(timeout_s)


class SimActivitySampler:
    """Read exact same-host sink timestamps; never infer activity from systemd."""

    def __init__(self):
        self.cache = ActivityCache("sim")
        self.targets = CatalogCache()

    def receive(self, raw: bytes, now_ns: int) -> None:
        state = CamState.model_validate_json(raw)
        catalog = state.target_catalog
        if catalog is None or catalog.source != "sim" or catalog.created_monotonic_ns > now_ns:
            self.targets.invalidate()
        else:
            self.targets.accept(CatalogEnvelope(catalog=catalog,
                age_ms=(now_ns - catalog.created_monotonic_ns) / 1e6), now_ns / 1e9)
        activity = state.control_activity
        if activity is None or activity.backend != "sim":
            self.cache.invalidate()
            return
        age_ns = now_ns - activity.created_monotonic_ns
        if age_ns < 0:
            self.cache.invalidate()
            return
        self.cache.accept(ActivityEnvelope(activity=activity, age_ms=age_ns / 1e6), now_ns / 1e9)

    def sample(self, now_ns: int) -> ActivityEnvelope | None:
        return self.cache.forward(now_ns / 1e9) if self.cache.fresh(now_ns / 1e9) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base")
    parser.add_argument("--config-extra")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args()
    bundle = runtime_bundle(resolve_config_paths(args.config, args.config_extra))
    cfg = bundle.data
    endpoint = str((cfg.get("net") or {}).get("zmq_operator_pc") or "")
    sim_endpoint = require_simulation_loopback_endpoint(
        str((cfg.get("operator") or {}).get("sim_activity_sub") or "tcp://127.0.0.1:5572"),
        "operator.sim_activity_sub")
    selection_endpoint = require_simulation_loopback_endpoint(
        str((cfg.get("operator") or {}).get("sim_selection_bind") or "tcp://127.0.0.1:5579"),
        "operator.sim_selection_bind")
    score_endpoint = require_simulation_loopback_endpoint(
        str((cfg.get('operator') or {}).get('sim_score_endpoint') or DEFAULT_SCORE_ENDPOINT),
        'operator.sim_score_endpoint')
    if not endpoint.startswith("tcp://"):
        parser.error("net.zmq_operator_pc must be a tcp:// endpoint")
    if args.check:
        print(json.dumps({"endpoint": endpoint, "actions": sorted(PC_ACTIONS), 'score_endpoint': score_endpoint,
                          **bundle.provenance(), "pending_config_keys": []}))
        return 0
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    worker = PcModeWorker(reset_score=partial(clear_score, endpoint=score_endpoint))
    ctx = zmq.Context()
    from pathlib import Path
    from common.live_config import ConfigAgent, AgentChannel
    config_agent = ConfigAgent(Path(__file__).resolve().parents[1], "pc",
                               run=worker.run, active=lambda: active_units(worker.run))
    config_channel = AgentChannel(ctx, config_agent)
    from common.live_config import LiveSettings, service_unit
    paths = resolve_config_paths(args.config, args.config_extra)
    settings = LiveSettings(paths, unit=service_unit("pc.mode_agent", paths), apply=lambda _: None,
                            initial_bundle=bundle)
    settings.subscribe()
    selection_pub = ctx.socket(zmq.PUB)
    selection_pub.setsockopt(zmq.LINGER, 0)
    selection_pub.setsockopt(zmq.SNDHWM, 1)
    selection_pub.bind(selection_endpoint)
    sim_sub = ctx.socket(zmq.SUB)
    sim_sub.setsockopt(zmq.CONFLATE, 1)
    sim_sub.setsockopt(zmq.LINGER, 0)
    sim_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sim_sub.connect(sim_endpoint)
    activity_sampler = SimActivitySampler()
    socket = None
    deadline = None if args.duration_s is None else time.monotonic() + args.duration_s
    last_error = ""
    try:
        while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
            config_channel.poll()
            settings.tick(engaged=False)
            try:
                if socket is None:
                    socket = ctx.socket(zmq.REQ)
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.setsockopt(zmq.SNDTIMEO, 1000)
                    socket.setsockopt(zmq.RCVTIMEO, 1000)
                    socket.connect(endpoint)
                try:
                    activity_sampler.receive(sim_sub.recv(zmq.NOBLOCK), time.monotonic_ns())
                except zmq.Again:
                    pass
                socket.send_string(worker.poll(lambda: activity_sampler.sample(time.monotonic_ns()),
                    lambda: activity_sampler.targets.forward(time.monotonic())).to_json())
                raw_reply = socket.recv()
                worker.accept(parse_task(raw_reply))
                selection = (json.loads(raw_reply) or {}).get("selection")
                if selection is not None and parse_selection(json.dumps(selection)) is not None:
                    selection_pub.send_string(json.dumps(selection), flags=zmq.NOBLOCK)
                last_error = ""
            except (zmq.ZMQError, ValueError, RuntimeError) as exc:
                error = str(exc)
                if error != last_error:
                    log.warning("PC mode channel: %s", error)
                    last_error = error
                if socket is not None:
                    socket.close(0)
                    socket = None
            stop.wait(0.2)
    finally:
        if socket is not None:
            socket.close(0)
        config_channel.close()
        settings.close()
        sim_sub.close(0)
        selection_pub.close(0)
        ctx.term()
        worker.wait_idle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
