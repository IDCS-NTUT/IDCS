from pydantic import BaseModel
from typing import List

class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float
    cls: str
    conf: float

class DetectionMsg(BaseModel):
    frame_id: int
    src_ts_ms: int
    rx_ts_ms: int
    infer_ts_ms: int
    img_w: int
    img_h: int
    boxes: List[Box]
