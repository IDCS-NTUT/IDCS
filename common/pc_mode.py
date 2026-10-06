"""Narrow, PC-initiated operator-mode protocol.

The PC polls the Jetson for one allowlisted simulator action at a time. No
Jetson-to-PC connection or PC address is needed, and this channel has no motor
or HIL start action. A task ID makes a lost reply safe to retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


PC_STATUS_UNITS = (
    "idcs-sim.target",
    "idcs-sim-session.target",
    "idcs-sim-streamer.service",
    "idcs-sim-controller.service",
    "idcs-sim-panel.service",
    "idcs-sim-ui.service",
    "idcs-sim-swarm.target",
    "idcs-sim-swarm-streamer.service",
    "idcs-sim-swarm-controller.service",
    "idcs-hil-streamer.service",
    "idcs-camera-ui.service",
)
PC_ACTIONS = frozenset({
    ("start", "idcs-sim.target"),
    ("stop", "idcs-sim.target"),
    ("stop", "idcs-sim-session.target"),
    ("stop", "idcs-sim-swarm.target"),
    ("start", "idcs-camera-ui.service"),
    ("stop", "idcs-camera-ui.service"),
})


@dataclass(frozen=True)
class PcModeTask:
    task_id: int
    verb: str
    unit: str

    def __post_init__(self) -> None:
        if (not isinstance(self.task_id, int) or isinstance(self.task_id, bool)
                or self.task_id < 1 or (self.verb, self.unit) not in PC_ACTIONS):
            raise ValueError("PC mode task is not allowlisted")

    def to_json(self) -> str:
        return json.dumps({"type": "PcModeTask", "id": self.task_id,
                           "verb": self.verb, "unit": self.unit})


def parse_task(raw: bytes | str) -> PcModeTask | None:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        raise ValueError("invalid PC mode task JSON") from None
    if not isinstance(payload, Mapping) or payload.get("type") != "PcModeTask":
        raise ValueError("not a PC mode task")
    task_id = payload.get("id")
    if task_id is None:
        return None
    if not isinstance(task_id, int) or isinstance(task_id, bool):
        raise ValueError("invalid PC mode task ID")
    return PcModeTask(task_id, str(payload.get("verb")), str(payload.get("unit")))


@dataclass(frozen=True)
class PcModePoll:
    active_units: frozenset[str]
    completed_id: int | None = None
    ok: bool | None = None
    detail: str = ""
    session: str = ""

    def to_json(self) -> str:
        return json.dumps({"type": "PcModePoll", "active_units": sorted(self.active_units),
                           "completed_id": self.completed_id, "ok": self.ok,
                           "detail": self.detail[:300], "session": self.session})


def parse_poll(raw: bytes | str) -> PcModePoll:
    try:
        payload: Any = json.loads(raw)
    except (TypeError, ValueError):
        raise ValueError("invalid PC mode poll JSON") from None
    if not isinstance(payload, Mapping) or payload.get("type") != "PcModePoll":
        raise ValueError("not a PC mode poll")
    active = payload.get("active_units")
    if (not isinstance(active, list) or not all(isinstance(u, str) for u in active)
            or any(u not in PC_STATUS_UNITS for u in active)):
        raise ValueError("invalid PC active units")
    task_id = payload.get("completed_id")
    ok = payload.get("ok")
    if task_id is not None and (not isinstance(task_id, int) or isinstance(task_id, bool)
                                or task_id < 1 or not isinstance(ok, bool)):
        raise ValueError("invalid PC completion")
    if task_id is None and ok is not None:
        raise ValueError("completion result without task ID")
    detail = payload.get("detail", "")
    if not isinstance(detail, str) or len(detail) > 300:
        raise ValueError("invalid PC completion detail")
    session = payload.get("session")
    if not isinstance(session, str) or not 1 <= len(session) <= 80:
        raise ValueError("invalid PC session")
    return PcModePoll(frozenset(active), task_id, ok, detail, session)
