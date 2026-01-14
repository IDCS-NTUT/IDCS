"""ZMQ helpers for serial I/O update publishing and reply subscription."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import zmq


_LOG = logging.getLogger(__name__)


def _nowait_send(socket: zmq.Socket, payload: str) -> bool:
    try:
        socket.send_string(payload, flags=zmq.NOBLOCK)
    except zmq.Again:
        return False
    return True


def _split_topic(message: str) -> Tuple[str, str]:
    if " " not in message:
        return "", message
    topic, rest = message.split(" ", 1)
    return topic, rest


class SerialUpdatePublisher:
    """Non-blocking publisher for SerialUpdate messages."""

    def __init__(self, endpoint: str, *, ctx: Optional[zmq.Context] = None) -> None:
        self._ctx = ctx or zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

    def send_update(self, update: Dict[str, Any]) -> bool:
        payload = json.dumps(update)
        sent = _nowait_send(self._socket, payload)
        if not sent:
            _LOG.debug("SerialUpdate dropped (PUB socket not ready)")
        return sent

    def close(self) -> None:
        self._socket.close(linger=0)


class SerialReplySubscriber:
    """Non-blocking subscriber for SerialReplyData messages."""

    def __init__(
        self,
        endpoint: str,
        *,
        topics: Optional[Iterable[str]] = None,
        ctx: Optional[zmq.Context] = None,
    ) -> None:
        self._ctx = ctx or zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        if topics is None:
            topics = ["serial.reply."]
        for topic in topics:
            self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    def recv_nowait(self) -> List[Dict[str, Any]]:
        replies: List[Dict[str, Any]] = []
        while True:
            try:
                raw = self._socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            topic, body = _split_topic(raw)
            try:
                payload = json.loads(body)
            except Exception as exc:  # noqa: BLE001
                _LOG.debug("Failed to decode reply (%s): %s", topic, exc)
                continue
            replies.append(payload)
        return replies

    def close(self) -> None:
        self._socket.close(linger=0)
