from typing import List, Literal, Optional, Tuple

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
