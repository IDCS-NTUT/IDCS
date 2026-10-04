"""Track a recorded clip with OpenCV CSRT, seeded from YOLO detections.

Needs an OpenCV build with the contrib trackers (cv2.TrackerCSRT_create).
Detections come from a jetson/tools/tracker_clip_probe.py run on the same
clip, so CSRT and NvDCF start from identical YOLO output. Two modes:

    standalone  initialize on the first drone detection and track alone;
                re-initialize only when CSRT reports failure
    hybrid      YOLO's box when there is one (re-initializing CSRT when it has
                drifted from it, IoU < 0.5); CSRT's box on frames YOLO misses

Writes one JSON line per frame ({"i", "standalone", "hybrid"}, normalized
[x, y, w, h] or null) and prints timing per update and per initialization.

    python tools/csrt_clip_run.py clip.mp4 probe.jsonl out.jsonl [--scale 1.0]
"""
from __future__ import annotations

import argparse
import json
import time

import cv2

DRONE = 0


def iou(a, b) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    iy = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class Csrt:
    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.tracker = None
        self.update_ms: list[float] = []
        self.init_ms: list[float] = []

    def _px(self, box, w, h):
        return (int(box[0] * w), int(box[1] * h), max(2, int(box[2] * w)), max(2, int(box[3] * h)))

    def init(self, frame, box) -> None:
        h, w = frame.shape[:2]
        t = time.perf_counter()
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, self._px(box, w, h))
        self.init_ms.append((time.perf_counter() - t) * 1000)

    def update(self, frame):
        if self.tracker is None:
            return None
        h, w = frame.shape[:2]
        t = time.perf_counter()
        ok, (x, y, bw, bh) = self.tracker.update(frame)
        self.update_ms.append((time.perf_counter() - t) * 1000)
        if not ok:
            self.tracker = None
            return None
        return [x / w, y / h, bw / w, bh / h]


def pct(values, q):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 2) if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("probe")
    ap.add_argument("output")
    ap.add_argument("--scale", type=float, default=1.0, help="resize frames before tracking")
    args = ap.parse_args()

    dets = {}
    for line in open(args.probe, encoding="utf-8"):
        rec = json.loads(line)
        dets[rec["i"]] = [d for d in rec["det"] if d[0] == DRONE]
    cap = cv2.VideoCapture(args.clip)
    alone, hyb = Csrt(args.scale), Csrt(args.scale)
    alone_box = hyb_box = None
    out = open(args.output, "w", encoding="utf-8")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        frame_dets = dets.get(i, [])
        best = max(frame_dets, key=lambda d: d[1])[2:] if frame_dets else None

        # standalone
        if alone.tracker is not None:
            alone_box = alone.update(frame)
        if alone.tracker is None:
            alone_box = None
            if best is not None:
                alone.init(frame, best)
                alone_box = best

        # hybrid
        tracked = hyb.update(frame) if hyb.tracker is not None else None
        if frame_dets:
            ref = tracked or hyb_box
            det = max(frame_dets, key=lambda d: iou(d[2:], ref)) if ref else max(frame_dets, key=lambda d: d[1])
            det_box = det[2:]
            if tracked is None or iou(tracked, det_box) < 0.5:
                hyb.init(frame, det_box)
            hyb_box = det_box
        else:
            hyb_box = tracked

        out.write(json.dumps({"i": i, "standalone": alone_box, "hybrid": hyb_box}) + "\n")
        i += 1
    out.close()
    print(json.dumps({
        "frames": i, "scale": args.scale,
        "standalone": {"update_ms_p50": pct(alone.update_ms, 0.5), "update_ms_p95": pct(alone.update_ms, 0.95),
                       "inits": len(alone.init_ms), "init_ms_p50": pct(alone.init_ms, 0.5)},
        "hybrid": {"update_ms_p50": pct(hyb.update_ms, 0.5), "update_ms_p95": pct(hyb.update_ms, 0.95),
                   "inits": len(hyb.init_ms), "init_ms_p50": pct(hyb.init_ms, 0.5)},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
