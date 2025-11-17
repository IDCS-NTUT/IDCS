from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel

class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float
    cls: str
    conf: float
    distance_m: Optional[float] = None
    distance_src: Optional[Literal["height", "width", "average"]] = None

class DetectionMsg(BaseModel):
    frame_id: int
    src_ts_ms: int
    rx_ts_ms: int
    infer_ts_ms: int
    img_w: int
    img_h: int
    boxes: List[Box]
    target_idx: Optional[int] = None
    target_distance_smoothed_m: Optional[float] = None
    laser_origin_px: Optional[Tuple[float, float]] = None
    laser_dot_px: Optional[Tuple[float, float]] = None
    laser_on_target: Optional[bool] = None
    laser_range_m: Optional[float] = None
    laser_range_source: Optional[str] = None
    parallax_compensation_active: Optional[bool] = None
    target_velocity_px_s: Optional[Tuple[float, float]] = None
    target_lead_uv: Optional[Tuple[float, float]] = None
    target_lead_time_s: Optional[float] = None
    predictive_active: Optional[bool] = None
    predictive_target_uv: Optional[Tuple[float, float]] = None
    predictive_box_px: Optional[Tuple[float, float, float, float]] = None


class MpcAxisDiagnostic(BaseModel):
    status: str
    cost: Optional[float] = None
    u0: Optional[float] = None
    slack: Optional[Dict[str, float]] = None
    solver: Optional[Dict[str, float]] = None
    terms: Optional[Dict[str, float]] = None


class ControlCmd(BaseModel):
    """Jetson → PC control command payload."""

    type: Literal["ControlCmd"] = "ControlCmd"
    frame_id: int
    src_ts_ms: int
    cmd_ts_ms: int
    target_ok: bool
    target_uv: Tuple[float, float]
    err_uv: Tuple[float, float]
    err_rad: Tuple[float, float]
    pan_rate_cmd: float
    tilt_rate_cmd: float
    pan_abs_cmd: Optional[float] = None
    tilt_abs_cmd: Optional[float] = None
    laser_origin_px: Optional[Tuple[float, float]] = None
    laser_dot_px: Optional[Tuple[float, float]] = None
    laser_on_target: Optional[bool] = None
    laser_range_m: Optional[float] = None
    laser_range_source: Optional[str] = None
    parallax_compensation_active: Optional[bool] = None
    controller_mode: Optional[Literal["pid", "mpc"]] = None
    mpc: Optional[Dict[str, MpcAxisDiagnostic]] = None


class CamState(BaseModel):
    """PC → Jetson camera pose/state header."""

    type: Literal["CamState"] = "CamState"
    frame_id: int
    src_ts_ms: int
    pan: float
    tilt: float
    pan_rate: Optional[float] = None
    tilt_rate: Optional[float] = None
    home_pan: Optional[float] = None
    home_tilt: Optional[float] = None


def detection_msg_to_json(msg: DetectionMsg) -> str:
    """Serialize a :class:`DetectionMsg` without emitting ``null`` placeholders.

    Older consumers expect the payload to omit fields that are not in use.
    ``model_dump_json(exclude_none=True)`` preserves backward compatibility by
    leaving newly-added optional keys out of the JSON when they have no value.
    """

    return msg.model_dump_json(exclude_none=True)


def detection_msg_from_json(payload: Union[str, bytes, bytearray, Mapping[str, Any]]) -> DetectionMsg:
    """Decode a JSON payload into a :class:`DetectionMsg` instance.

    Accepts raw JSON strings/bytes or a pre-parsed mapping so callers can pass
    data directly from ZMQ without worrying about the intermediate type.
    """

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(f"DetectionMsg payload must be mapping-like, got {type(payload)!r}")
    return DetectionMsg(**payload)
