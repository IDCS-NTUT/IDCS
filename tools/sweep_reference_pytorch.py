"""PyTorch reference on the detector sweep with controlled preprocessing.

Runs the .pt model on the sweep AVI (tools.generate_detector_sim_sweep) with
its own letterbox and with square 736 letterboxes of chosen padding and
interpolation; writes one analyzer JSONL per variant into the sweep directory
(tools.analyze_detector_sim_sweep scores them).

    python tools/sweep_reference_pytorch.py <sweep dir> <model.pt>
"""
import json, sys
import cv2, numpy as np
from ultralytics import YOLO

S = sys.argv[1]
model = YOLO(sys.argv[2])
SIZE = 736
VARIANTS = {
    "ultralytics": None,  # model's own preprocessing
    "sq_gray_linear": (114, cv2.INTER_LINEAR),
    "sq_black_linear": (0, cv2.INTER_LINEAR),
    "sq_gray_nearest": (114, cv2.INTER_NEAREST),
    "sq_black_nearest": (0, cv2.INTER_NEAREST),
    "sq_black_area": (0, cv2.INTER_AREA),
}
cap = cv2.VideoCapture(f"{S}/sweep.avi")
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
h, w = frames[0].shape[:2]
scale = SIZE / max(w, h)
nw, nh = round(w * scale), round(h * scale)
px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
for name, spec in VARIANTS.items():
    out = open(f"{S}/ref_{name}.jsonl", "w")
    for i, f in enumerate(frames):
        if spec is None:
            r = model.predict(f, imgsz=SIZE, conf=0.30, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy(); conf = r.boxes.conf.cpu().numpy()
            norm = [(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h) for x0, y0, x1, y1 in boxes]
        else:
            pad, interp = spec
            sq = np.full((SIZE, SIZE, 3), pad, np.uint8)
            sq[py:py + nh, px:px + nw] = cv2.resize(f, (nw, nh), interpolation=interp)
            r = model.predict(sq, imgsz=SIZE, conf=0.30, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy(); conf = r.boxes.conf.cpu().numpy()
            norm = []
            for x0, y0, x1, y1 in boxes:
                x0, x1 = np.clip([(x0 - px) / scale, (x1 - px) / scale], 0, w)
                y0, y1 = np.clip([(y0 - py) / scale, (y1 - py) / scale], 0, h)
                norm.append((x0 / w, y0 / h, max(x1 - x0, 1) / w, max(y1 - y0, 1) / h))
        rec = {"frame_id": i + 1, "boxes": [
            {"cls": str(int(c)), "x": float(b[0]), "y": float(b[1]), "w": float(min(b[2], 1 - b[0])),
             "h": float(min(b[3], 1 - b[1])), "conf": float(cf)} for b, c, cf in zip(norm, cls, conf)]}
        out.write(json.dumps(rec) + "\n")
    out.close()
    print(name, "done", flush=True)
