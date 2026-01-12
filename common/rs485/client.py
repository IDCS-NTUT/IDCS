"""Client helpers for interacting with the RS485 service over IPC."""

from __future__ import annotations

import logging
from typing import List, Optional

import zmq

from .ipc import (
    RS485CommandRequest,
    RS485CommandResponse,
    RS485HealthQuery,
    RS485HealthResponse,
    RS485HistoryQuery,
    RS485HistoryResponse,
    RS485LatestQuery,
    RS485LatestResponse,
    rs485_response_from_json,
)

logger = logging.getLogger(__name__)


class RS485Client:
    """ZMQ client for the RS485 service."""

    def __init__(self, endpoint: str, *, timeout_ms: int = 1000) -> None:
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._sock.connect(endpoint)

    def close(self) -> None:
        self._sock.close(0)

    def latest(
        self,
        *,
        addr: int,
        func: int,
        fresh: bool = False,
        max_age_ms: Optional[int] = None,
    ) -> RS485LatestResponse:
        request = RS485LatestQuery(
            addr=addr,
            func=func,
            fresh=fresh,
            max_age_ms=max_age_ms,
        )
        return self._send(request)

    def history(
        self,
        *,
        addr: int,
        func: int,
        fresh: bool = False,
        max_items: Optional[int] = None,
    ) -> RS485HistoryResponse:
        request = RS485HistoryQuery(
            addr=addr,
            func=func,
            fresh=fresh,
            max_items=max_items,
        )
        return self._send(request)

    def command(
        self,
        *,
        addr: int,
        func: int,
        data: Optional[List[int]] = None,
        response_expected: bool = True,
        expected_response_len: Optional[int] = None,
        fresh: bool = True,
    ) -> RS485CommandResponse:
        request = RS485CommandRequest(
            addr=addr,
            func=func,
            data=data,
            response_expected=response_expected,
            expected_response_len=expected_response_len,
            fresh=fresh,
        )
        return self._send(request)

    def health(self) -> RS485HealthResponse:
        request = RS485HealthQuery()
        return self._send(request)

    def _send(self, request):
        payload = request.model_dump_json(exclude_none=True)
        self._sock.send_string(payload)
        response = self._sock.recv()
        decoded = rs485_response_from_json(response)
        return decoded
