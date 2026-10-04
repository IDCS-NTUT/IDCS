"""Operator commands from the panel screen and the selection they produce.

Two plain-JSON messages:

``OperatorCommand`` (panel display REQ -> ``jetson.operator_agent`` REP)::

    {"type": "OperatorCommand", "command": "status"}
    {"type": "OperatorCommand", "command": "lock", "track_id": 12}
    {"type": "OperatorCommand", "command": "release"}
    {"type": "OperatorCommand", "command": "target_classes", "classes": ["drone"]}
    {"type": "OperatorCommand", "command": "set_mode", "mode": "camera"}
    {"type": "OperatorCommand", "command": "recording", "on": true}

    reply: {"type": "OperatorReply", "ok": bool, "message": str, "state": {...}}

``OperatorSelection`` (agent PUB -> DeepStream target selector, loopback),
republished at a steady rate so a restarted subscriber converges::

    {"type": "OperatorSelection", "version": 1, "sequence": n,
     "lock_track_id": 12 | null, "target_classes": ["drone"]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

COMMANDS = ("status", "lock", "release", "target_classes", "set_mode", "recording")


class CommandError(ValueError):
    pass


@dataclass(frozen=True)
class OperatorCommand:
    command: str
    track_id: int | None = None
    classes: tuple[str, ...] = ()
    mode: str | None = None
    on: bool | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {"type": "OperatorCommand", "command": self.command}
        if self.command == "lock":
            payload["track_id"] = self.track_id
        elif self.command == "target_classes":
            payload["classes"] = list(self.classes)
        elif self.command == "set_mode":
            payload["mode"] = self.mode
        elif self.command == "recording":
            payload["on"] = self.on
        return json.dumps(payload)


def parse_command(raw: bytes | str) -> OperatorCommand:
    try:
        message = json.loads(raw)
    except ValueError as exc:
        raise CommandError(f"not JSON: {exc}") from exc
    if not isinstance(message, Mapping) or message.get("type") != "OperatorCommand":
        raise CommandError("not an OperatorCommand")
    command = message.get("command")
    if command not in COMMANDS:
        raise CommandError(f"unknown command {command!r}")
    if command == "lock":
        track_id = message.get("track_id")
        if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0:
            raise CommandError("lock needs a non-negative integer track_id")
        return OperatorCommand(command, track_id=track_id)
    if command == "target_classes":
        classes = message.get("classes")
        if (not isinstance(classes, list) or not classes
                or not all(isinstance(c, str) and c.strip() for c in classes)):
            raise CommandError("target_classes needs a non-empty list of class names")
        return OperatorCommand(command, classes=tuple(sorted({c.strip().lower() for c in classes})))
    if command == "set_mode":
        mode = message.get("mode")
        if not isinstance(mode, str) or not mode:
            raise CommandError("set_mode needs a mode name")
        return OperatorCommand(command, mode=mode)
    if command == "recording":
        on = message.get("on")
        if not isinstance(on, bool):
            raise CommandError("recording needs on: true or false")
        return OperatorCommand(command, on=on)
    return OperatorCommand(command)


@dataclass(frozen=True)
class OperatorSelection:
    sequence: int
    lock_track_id: int | None
    target_classes: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps({
            "type": "OperatorSelection", "version": 1, "sequence": self.sequence,
            "lock_track_id": self.lock_track_id, "target_classes": list(self.target_classes),
        })


def parse_selection(raw: bytes | str) -> OperatorSelection | None:
    """The selection, or None for anything malformed (the subscriber keeps its last)."""
    try:
        message = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(message, Mapping) or message.get("type") != "OperatorSelection":
        return None
    lock = message.get("lock_track_id")
    classes = message.get("target_classes")
    sequence = message.get("sequence")
    if lock is not None and (not isinstance(lock, int) or isinstance(lock, bool) or lock < 0):
        return None
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        return None
    if not isinstance(sequence, int):
        return None
    return OperatorSelection(sequence, lock, tuple(c.strip().lower() for c in classes if c.strip()))
