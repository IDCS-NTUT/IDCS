"""Score DeepStream's detector and tracker against the simulator's ground truth.

Subscribes to the simulator's exact truth snapshots and to DeepStream's
published snapshots (whose ``detections`` are YOLO's raw output for the frame
and ``tracks`` the tracker's) and joins them by source frame id. Per frame
with a truth target it records whether YOLO detected it, whether the tracker
covered it, whether that coverage was tracker-only (YOLO missed), and the box
centre error. The summary answers: how often YOLO misses, how many misses the
tracker bridges, and how accurate tracker-only boxes are compared with
detections. Read-only: it never publishes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import zmq

MATCH_IOU = 0.3


def _box(box: dict) -> tuple[float, float, float, float]:
    return box["x"], box["y"], box["w"], box["h"]


def iou(a: dict, b: dict) -> float:
    ax, ay, aw, ah = _box(a)
    bx, by, bw, bh = _box(b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def centre_error_px(a: dict, b: dict, width: int, height: int) -> float:
    ax, ay, aw, ah = _box(a)
    bx, by, bw, bh = _box(b)
    return math.hypot((ax + aw / 2 - bx - bw / 2) * width, (ay + ah / 2 - by - bh / 2) * height)


def score_frame(truth: dict, observed: dict) -> dict | None:
    """One frame: truth target vs YOLO's raw detections and the tracker's tracks."""
    tracks = truth.get("tracks") or []
    if not tracks:
        return None
    target = tracks[0]["box"]
    width, height = observed["frame"]["width"], observed["frame"]["height"]
    detections = observed.get("detections") or []
    best_det = max(detections, key=lambda d: iou(d["box"], target), default=None)
    detected = best_det is not None and iou(best_det["box"], target) >= MATCH_IOU
    tracker_tracks = observed.get("tracks") or []
    best_track = max(tracker_tracks, key=lambda t: iou(t["box"], target), default=None)
    covered = best_track is not None and iou(best_track["box"], target) >= MATCH_IOU
    return {
        "frame_id": observed["frame"]["frame_id"],
        "yolo_detected": detected,
        "yolo_confidence": best_det["confidence"] if detected else None,
        "yolo_error_px": centre_error_px(best_det["box"], target, width, height) if detected else None,
        "tracker_covered": covered,
        "tracker_only": covered and best_track.get("missed_frames", 0) > 0,
        "tracker_error_px": centre_error_px(best_track["box"], target, width, height) if covered else None,
        "track_id": best_track["track_id"] if best_track is not None else None,
        # A track that no longer overlaps the target: drift or a wrong lock.
        "stray_tracks": sum(1 for t in tracker_tracks if iou(t["box"], target) < MATCH_IOU),
    }


def summarize(frames: list[dict], *, px_to_mrad: float) -> dict:
    n = len(frames)
    if not n:
        return {"frames": 0}
    missed = [f for f in frames if not f["yolo_detected"]]
    bridged = [f for f in missed if f["tracker_covered"]]

    def stats(values: list[float]) -> dict | None:
        if not values:
            return None
        values = sorted(values)
        return {"n": len(values), "median_px": statistics.median(values),
                "p95_px": values[min(len(values) - 1, int(0.95 * len(values)))],
                "median_mrad": statistics.median(values) * px_to_mrad}

    runs, run = [], 0
    for f in frames:
        if not f["yolo_detected"]:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)  # a miss still open when the window ended
    return {
        "frames": n,
        "yolo_detection_rate": 1 - len(missed) / n,
        "tracker_coverage": sum(f["tracker_covered"] for f in frames) / n,
        "yolo_misses": len(missed),
        "misses_bridged_by_tracker": len(bridged),
        "longest_yolo_miss_frames": max(runs, default=0),
        "yolo_error": stats([f["yolo_error_px"] for f in frames if f["yolo_error_px"] is not None]),
        "tracker_error_when_detected": stats([f["tracker_error_px"] for f in frames
                                              if f["tracker_covered"] and not f["tracker_only"]]),
        "tracker_error_tracker_only": stats([f["tracker_error_px"] for f in frames if f["tracker_only"]]),
        "frames_with_stray_tracks": sum(f["stray_tracks"] > 0 for f in frames),
        "track_ids": sorted({f["track_id"] for f in frames if f["track_id"] is not None}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", default="tcp://127.0.0.1:5574", help="simulator truth PUB")
    parser.add_argument("--observed", default="tcp://192.168.0.5:5564", help="DeepStream snapshot PUB")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--hfov-deg", type=float, default=135.0, help="for the px -> mrad conversion")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ctx = zmq.Context()
    subs = {}
    for name, endpoint in (("truth", args.truth), ("observed", args.observed)):
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 10000)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(endpoint)
        subs[name] = sock
    poller = zmq.Poller()
    for sock in subs.values():
        poller.register(sock, zmq.POLLIN)
    truth: dict[int, dict] = {}
    observed: dict[int, dict] = {}
    width = None
    deadline = time.monotonic() + args.duration_s
    while time.monotonic() < deadline:
        for sock, _ in poller.poll(100):
            message = json.loads(sock.recv())
            frame_id = message["frame"]["frame_id"]
            (truth if sock is subs["truth"] else observed)[frame_id] = message
            width = width or message["frame"]["width"]
    frames = [f for f in (score_frame(truth[i], observed[i]) for i in sorted(truth.keys() & observed.keys()))
              if f is not None]
    px_to_mrad = math.radians(args.hfov_deg) / width * 1000 if width else float("nan")
    report = {"truth_frames": len(truth), "observed_frames": len(observed),
              "summary": summarize(frames, px_to_mrad=px_to_mrad)}
    if args.output:
        args.output.write_text(json.dumps({**report, "frames": frames}, indent=1))
    print(json.dumps(report, indent=2))
    for sock in subs.values():
        sock.close(0)
    ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
