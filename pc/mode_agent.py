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

from common.config import load_config_bundle, resolve_config_paths
from common.pc_mode import PC_ACTIONS, PC_STATUS_UNITS, PcModePoll, PcModeTask, parse_task

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
    session: str = field(default_factory=lambda: uuid.uuid4().hex)
    _task: PcModeTask | None = None
    _result: tuple[int, bool, str] | None = None
    _job: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def poll(self) -> PcModePoll:
        units = active_units(self.run)
        with self._lock:
            result = self._result
        return PcModePoll(units, *(result or (None, None, "")), session=self.session)

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
                if "idcs-hil-streamer.service" in active_units(self.run):
                    raise RuntimeError("HIL streamer is active")
                code, output = self.run([task.verb, task.unit])
                if code == 0:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base")
    parser.add_argument("--config-extra")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args()
    cfg = load_config_bundle(resolve_config_paths(args.config, args.config_extra)).data
    endpoint = str((cfg.get("net") or {}).get("zmq_operator_pc") or "")
    if not endpoint.startswith("tcp://"):
        parser.error("net.zmq_operator_pc must be a tcp:// endpoint")
    if args.check:
        print(json.dumps({"endpoint": endpoint, "actions": sorted(PC_ACTIONS)}))
        return 0
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    worker = PcModeWorker()
    ctx = zmq.Context()
    socket = None
    deadline = None if args.duration_s is None else time.monotonic() + args.duration_s
    last_error = ""
    try:
        while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
            try:
                if socket is None:
                    socket = ctx.socket(zmq.REQ)
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.setsockopt(zmq.SNDTIMEO, 1000)
                    socket.setsockopt(zmq.RCVTIMEO, 1000)
                    socket.connect(endpoint)
                socket.send_string(worker.poll().to_json())
                worker.accept(parse_task(socket.recv()))
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
        ctx.term()
        worker.wait_idle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
