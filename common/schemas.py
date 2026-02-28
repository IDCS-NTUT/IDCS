"""Shared Pydantic schemas for ZMQ metadata payloads between PC and Jetson.

These models define the JSON payloads exchanged over the metadata sockets.
They aim to stay backward compatible by treating newly added fields as
optional and by omitting ``None`` values during serialization so older
consumers do not receive unexpected ``null`` keys.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel

class Box(BaseModel):
    """Normalized detection box in image coordinates.

    Coordinates are expressed as fractions of the full image size. ``x``/``y``
    are the top-left corner of the box (normalized to ``[0, 1]``), while
    ``w``/``h`` are the normalized width/height. The invariant is that
    ``w``/``h`` are positive for valid detections; downstream consumers should
    treat values outside ``[0, 1]`` as invalid and clamp if needed.

    ``cls`` is the class label or ID (string), ``conf`` is the detector
    confidence in ``[0, 1]``. ``distance_m`` is the estimated range in meters
    when known-size ranging is enabled, and ``distance_src`` identifies the
    measurement method (height/width/average). These ranging fields are
    optional and omitted from serialized payloads for backward compatibility.
    """

    x: float
    y: float
    w: float
    h: float
    cls: str
    conf: float
    distance_m: Optional[float] = None
    distance_src: Optional[Literal["height", "width", "average"]] = None

class DetectionMsg(BaseModel):
    """Jetson → PC detection payload with optional overlays and target metadata.

    Required fields capture the originating frame identifiers and timing
    information in milliseconds (``src_ts_ms`` from the PC, ``rx_ts_ms`` when
    the Jetson received the frame, ``infer_ts_ms`` after inference), plus the
    original image dimensions in pixels. ``boxes`` contains normalized
    detections (see :class:`Box`).

    Optional fields are populated when specific features are enabled:

    - ``target_idx``: index into ``boxes`` for the currently selected target.
    - ``target_distance_smoothed_m``: EMA-smoothed target range in meters.
    - ``laser_*`` fields: laser overlay info in pixels and meters.
    - ``target_velocity_px_s``: target velocity estimate in pixels/second.
    - ``target_lead_uv``/``predictive_*``: lead or predicted aim points in
      pixel coordinates, with ``predictive_box_px`` as an ``(x, y, w, h)`` box
      in pixel units.

    When serialized via :func:`detection_msg_to_json`, unset optional values
    are omitted using ``exclude_none=True`` to preserve backward compatibility
    for older consumers that expect the legacy schema.
    """

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
    infer_source: Optional[Literal["search", "track"]] = None
    tracker_mode: Optional[Literal["search", "slew", "track", "recover"]] = None


class MpcAxisDiagnostic(BaseModel):
    """Compact MPC diagnostics for a single axis (yaw or pitch).

    ``status`` mirrors the solver status string. ``cost`` is the objective
    value when available. ``u0`` is the first control command in the MPC
    sequence (typically a rate command in rad/s). ``slack``, ``solver``, and
    ``terms`` are optional diagnostic dictionaries containing solver and cost
    breakdowns. ``refs`` and ``pred`` optionally expose compact reference and
    prediction snapshots (for example ``theta_ref0`` or ``theta_pred0``).
    Optional fields are omitted when empty or non-finite to keep payloads
    compact and backward compatible.
    """

    status: str
    cost: Optional[float] = None
    u0: Optional[float] = None
    slack: Optional[Dict[str, float]] = None
    solver: Optional[Dict[str, float]] = None
    terms: Optional[Dict[str, float]] = None
    refs: Optional[Dict[str, float]] = None
    pred: Optional[Dict[str, float]] = None


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


def control_cmd_from_json(payload: Union[str, bytes, bytearray, Mapping[str, Any]]) -> ControlCmd:
    """Decode serialized control commands into :class:`ControlCmd` objects."""

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(f"ControlCmd payload must be mapping-like, got {type(payload)!r}")
    return ControlCmd(**payload)
