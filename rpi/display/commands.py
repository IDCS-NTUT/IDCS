"""Panel display -> Jetson operator agent: one request in flight, never blocking.

A REQ socket polled from the display tick. A request without a reply within
``timeout_s`` is abandoned: the socket is rebuilt (a REQ socket cannot send
again until it receives) and the agent is reported offline. While idle, a
``status`` request every ``status_period_s`` keeps the agent state fresh.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any

import zmq

from common.operator_commands import OperatorCommand


@dataclass(frozen=True)
class Reply:
    command: str
    ok: bool
    message: str
    state: dict[str, Any] | None


class CommandClient:
    def __init__(self, ctx: zmq.Context, endpoint: str, *, timeout_s: float = 1.5,
                 status_period_s: float = 2.0) -> None:
        self._ctx = ctx
        self._endpoint = endpoint
        self._timeout_s = timeout_s
        self._status_period_s = status_period_s
        self._socket: zmq.Socket | None = None
        self._queue: deque[OperatorCommand] = deque(maxlen=4)
        self._in_flight: tuple[OperatorCommand, float] | None = None
        self._next_status_s = 0.0
        self.online = False
        self._connect()

    def submit(self, command: OperatorCommand) -> None:
        self._queue.append(command)

    def poll(self, now_s: float) -> list[Reply]:
        """Replies received (or timed out) this tick; sends the next request."""
        replies: list[Reply] = []
        if self._in_flight is not None:
            command, sent_s = self._in_flight
            reply = self._receive(command)
            if reply is not None:
                replies.append(reply)
                self._in_flight = None
            elif now_s - sent_s > self._timeout_s:
                self.online = False
                self._in_flight = None
                self._connect()
                if command.command != "status":
                    replies.append(Reply(command.command, False, "operator agent not answering", None))
        if self._in_flight is None:
            command = self._queue.popleft() if self._queue else None
            if command is None and now_s >= self._next_status_s:
                command = OperatorCommand("status")
            if command is not None:
                if command.command == "status":
                    self._next_status_s = now_s + self._status_period_s
                assert self._socket is not None
                self._socket.send_string(command.to_json())
                self._in_flight = (command, now_s)
        return replies

    def _receive(self, command: OperatorCommand) -> Reply | None:
        assert self._socket is not None
        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None
        self.online = True
        try:
            message = json.loads(raw)
        except ValueError:
            return Reply(command.command, False, "unreadable reply", None)
        state = message.get("state") if isinstance(message, dict) else None
        return Reply(command.command, bool(message.get("ok")), str(message.get("message") or ""),
                     state if isinstance(state, dict) else None)

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._endpoint)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None

