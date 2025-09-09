import time
from typing import List
from common.schemas import Box

class YoloEngine:
    def __init__(self, conf_thres=0.25, iou_thres=0.45):
        self.conf = conf_thres; self.iou = iou_thres

    def infer(self, bgr_frame) -> List[Box]:
        # TODO: swap with real TensorRT engine
        time.sleep(0.005)  # simulate work
        h, w = bgr_frame.shape[:2]
        # return a dummy box to prove the path
        return [Box(x=0.3, y=0.3, w=0.2, h=0.2, cls="dummy", conf=0.9)]
