# TensorRT YOLO v8n engine loader (FP16 fixed 1x3x640x640)
import os, time, math
from typing import List, Tuple
import numpy as np
import cv2

import tensorrt as trt
import pycuda.autoinit        # noqa: F401
import pycuda.driver as cuda

from common.schemas import Box

def letterbox(img, new_shape=640, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top

def nms(boxes, scores, classes, iou_thres=0.45, conf_thres=0.25):
    idx = scores >= conf_thres
    boxes, scores, classes = boxes[idx], scores[idx], classes[idx]
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return [(boxes[i], scores[i], int(classes[i])) for i in keep]

class YoloEngine:
    def __init__(self, engine_path: str, conf_thres=0.25, iou_thres=0.45, input_size=640):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")
        self.conf = conf_thres
        self.iou = iou_thres
        self.sz = int(input_size)

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Bindings (assume 1 input, 1 output; YOLOv8n INT/FP16 dynamic heads already fused)
        self.stream = cuda.Stream()
        self.bindings = [None] * self.engine.num_bindings
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.in_idx = i
                in_shape = self.engine.get_binding_shape(i)  # e.g. (1,3,640,640)
                self.n_input = int(np.prod(in_shape))
                self.d_input = cuda.mem_alloc(self.n_input * np.float32().nbytes)
                self.bindings[i] = int(self.d_input)
            else:
                self.out_idx = i
                out_shape = self.engine.get_binding_shape(i)  # e.g. (1,84,8400) or similar
                self.out_size = int(np.prod(out_shape))
                self.d_output = cuda.mem_alloc(self.out_size * np.float32().nbytes)
                self.bindings[i] = int(self.d_output)
        self.h_output = np.empty(self.out_size, dtype=np.float32)

        # Minimal class map (COCO); update if your engine differs
        self.class_names = None  # optional; you can load names from a file

    def infer(self, bgr_frame) -> List[Box]:
        # Preprocess
        img, r, left, top = letterbox(bgr_frame, self.sz)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2,0,1)).copy()  # CHW
        x = np.expand_dims(x, 0)             # NCHW
        cuda.memcpy_htod_async(self.d_input, x, self.stream)

        # Infer
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        # Post (assume (1, No, 85): x,y,w,h,conf,80 cls scores OR merged (84,8400))
        out = self.h_output
        # Try to guess layout (common export formats)
        # Case A: (1,84,8400) → reshape to (8400,84)
        # Case B: (1,8400,84) → already row-major per anchor
        # We attempt A first:
        try:
            n_anchors = out.size // 84
            preds = out.reshape(1, 84, n_anchors).transpose(0,2,1)[0]
        except Exception:
            # Fallback: trust last dim 84
            preds = out.reshape(-1, 84)

        # xywh
        xywh = preds[:, :4]
        obj = preds[:, 4:5]
        cls = preds[:, 5:]
        scores = (obj * cls).max(axis=1)
        cls_ids = (obj * cls).argmax(axis=1)

        # Convert xywh (in 640 space) → x1y1x2y2 in original frame
        # Undo letterbox + scale
        boxes = []
        for (cx, cy, w, h), sc, cid in zip(xywh, scores, cls_ids):
            if sc < self.conf: continue
            x1 = (cx - w/2) - left
            y1 = (cy - h/2) - top
            x2 = (cx + w/2) - left
            y2 = (cy + h/2) - top
            # scale back
            boxes.append([x1, y1, x2, y2])
        if not boxes:
            return []
        boxes = np.array(boxes)
        # Clip to frame
        H, W = bgr_frame.shape[:2]
        boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, W-1)
        boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, H-1)

        # NMS
        kept = nms(boxes, scores, cls_ids, iou_thres=self.iou, conf_thres=self.conf)
        out_boxes: List[Box] = []
        for (x1,y1,x2,y2), sc, cid in kept:
            # to normalized x,y,w,h
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            x = x1 / W
            y = y1 / H
            name = str(cid)
            out_boxes.append(Box(x=x, y=y, w=w, h=h, cls=name, conf=float(sc)))
        return out_boxes

