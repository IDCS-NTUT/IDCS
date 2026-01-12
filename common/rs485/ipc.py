"""IPC schema for RS485 service requests and responses."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, Field


class RS485Request(BaseModel):
    """Base class for RS485 IPC requests."""

    version: int = Field(1, description="Schema version for RS485 IPC messages")
    type: str
    fresh: bool = Field(
        False,
        description="If true, request a fresh device read; otherwise cached data is acceptable",
    )


class RS485LatestQuery(RS485Request):
    """Request the latest cached frame for a device/function."""

    type: Literal["latest"] = "latest"
    addr: int
    func: int
    max_age_ms: Optional[int] = Field(
        None,
        description="Optional maximum acceptable age for cached data in milliseconds",
    )


class RS485HistoryQuery(RS485Request):
    """Request recent history for a device/function."""

    type: Literal["history"] = "history"
    addr: int
    func: int
    max_items: Optional[int] = Field(
        None,
        description="Optional maximum number of records to return from the history buffer",
    )


class RS485CommandRequest(RS485Request):
    """Request the service to send a command over RS485."""

    type: Literal["command"] = "command"
    addr: int
    func: int
    data: Optional[List[int]] = None
    response_expected: bool = True
    expected_response_len: Optional[int] = None


class RS485HealthQuery(RS485Request):
    """Request current RS485 service health metrics."""

    type: Literal["health"] = "health"


class RS485FramePayload(BaseModel):
    raw: List[int]
    received_ts: float
    addr: Optional[int]
    func: Optional[int]


class RS485Response(BaseModel):
    """Base class for RS485 IPC responses."""

    version: int = 1
    type: str
    ok: bool = True
    cached: bool = True
    error: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class RS485LatestResponse(RS485Response):
    type: Literal["latest"] = "latest"
    cached: bool = True
    payload: Optional[RS485FramePayload] = None


class RS485HistoryResponse(RS485Response):
    type: Literal["history"] = "history"
    cached: bool = True
    payload: Optional[List[RS485FramePayload]] = None


class RS485CommandResponse(RS485Response):
    type: Literal["command"] = "command"
    cached: bool = False
    payload: Optional[List[int]] = None


class RS485HealthResponse(RS485Response):
    type: Literal["health"] = "health"
    cached: bool = True
    payload: Optional[Dict[str, int]] = None


def rs485_request_from_json(
    payload: Union[str, bytes, bytearray, Mapping[str, Any]]
) -> RS485Request:
    """Decode JSON payload into an RS485 request."""

    data = _normalize_json_payload(payload)
    req_type = data.get("type")
    if req_type == "latest":
        return RS485LatestQuery(**data)
    if req_type == "history":
        return RS485HistoryQuery(**data)
    if req_type == "command":
        return RS485CommandRequest(**data)
    if req_type == "health":
        return RS485HealthQuery(**data)
    raise ValueError(f"Unknown RS485 request type: {req_type!r}")


def rs485_response_from_json(
    payload: Union[str, bytes, bytearray, Mapping[str, Any]]
) -> RS485Response:
    """Decode JSON payload into an RS485 response."""

    data = _normalize_json_payload(payload)
    resp_type = data.get("type")
    if resp_type == "latest":
        return RS485LatestResponse(**data)
    if resp_type == "history":
        return RS485HistoryResponse(**data)
    if resp_type == "command":
        return RS485CommandResponse(**data)
    if resp_type == "health":
        return RS485HealthResponse(**data)
    raise ValueError(f"Unknown RS485 response type: {resp_type!r}")


def _normalize_json_payload(
    payload: Union[str, bytes, bytearray, Mapping[str, Any]]
) -> Dict[str, Any]:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(f"RS485 payload must be mapping-like, got {type(payload)!r}")
    return dict(payload)
