"""Score tracker runs on a recorded clip against the simulator's truth.

Inputs: the clip's sidecar (pc.streamer --dump-frames: per-frame truth boxes and
camera pose), NvDCF probe runs (jetson/tools/tracker_clip_probe.py) and
optionally a CSRT run (tools/csrt_clip_run.py). For each method, per frame with
a truth target:

    covered   some published box's centre lies within max(8 px, half the truth
              width) of the truth centre (what matters for aiming a small target;
              IoU penalizes a well-centred box of the wrong height)
    covered_iou  the same with IoU >= 0.3 instead
    stray     some published box is off every truth target (centre criterion)
    error     centre error of the closest box, px

NvDCF probe runs are scored under several publication policies: YOLO alone,
everything the tracker outputs, and coast gates (tracker-only and shadow boxes
published while tracker confidence >= C and consecutive missed frames <= N).
Results are also split by camera angular speed.

    python tools/tracker_clip_eval.py frames.jsonl --nvdcf NAME=probe.jsonl ... [--csrt csrt.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict

W, H = 1280, 720
MATCH = 0.3
SPEED_BINS = ((0.0, 0.1, "camera still (<0.1 rad/s)"), (0.1, 0.3, "0.1-0.3 rad/s"),
              (0.3, 99.0, "slewing (>0.3 rad/s)"))
GATES = {"production gate 0.3 / 30": (0.3, 30), "gate 0.2 / 30": (0.2, 30),
         "gate 0.3 / 60": (0.3, 60), "gate 0.0 / 60": (0.0, 60)}


def iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def centre_px(b):
    return ((b[0] + b[2] / 2) * W, (b[1] + b[3] / 2) * H)


def centre_dist(b, t) -> float:
    (bx, by), (tx, ty) = centre_px(b), centre_px(t)
    return math.hypot(bx - tx, by - ty)


def on_target(b, t) -> bool:
    return centre_dist(b, t) <= max(8.0, 0.5 * t[2] * W)


def load_truth(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    speeds = [0.0]
    for a, b in zip(rows, rows[1:]):
        dt = (b["source_time_ns"] - a["source_time_ns"]) / 1e9
        speeds.append(math.hypot(b["pan"] - a["pan"], b["tilt"] - a["tilt"]) / dt if dt > 0 else 0.0)
    truth = {r["index"]: [[t["x"], t["y"], t["w"], t["h"]] for t in r["truth"]] for r in rows}
    return truth, dict(zip((r["index"] for r in rows), speeds))


def nvdcf_policies(path):
    """Published boxes per frame under each policy, from one ungated probe run."""
    out = defaultdict(dict)
    last_detected: dict[int, int] = {}
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        i = rec["i"]
        out["YOLO alone"][i] = [d[2:] for d in rec["det"] if d[0] == 0]
        matched, coasting = [], []
        for oid, cls, dconf, tconf, *box in rec["obj"]:
            if cls != 0:
                continue
            if dconf >= 0:
                last_detected[oid] = i
                matched.append(box)
            else:
                coasting.append((oid, tconf, box))
        for oid, cls, conf, *box in rec["shadow"]:
            if cls == 0:
                coasting.append((oid, conf, box))
        out["tracker, ungated"][i] = matched + [b for _, _, b in coasting]
        for name, (cmin, nmax) in GATES.items():
            out[name][i] = matched + [b for oid, c, b in coasting
                                      if c >= cmin and i - last_detected.get(oid, -10**9) <= nmax]
    return out


def csrt_policies(path):
    out = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        for mode in ("standalone", "hybrid"):
            out[f"CSRT {mode}"][rec["i"]] = [rec[mode]] if rec[mode] else []
    return out


def score(published, truth, speeds):
    agg = defaultdict(lambda: {"frames": 0, "covered": 0, "covered_iou": 0, "stray": 0, "err": []})
    for i, tboxes in truth.items():
        if not tboxes:
            continue
        boxes = published.get(i, [])
        covered = any(on_target(b, t) for b in boxes for t in tboxes)
        covered_iou = max((iou(b, t) for b in boxes for t in tboxes), default=0.0) >= MATCH
        stray = any(not any(on_target(b, t) for t in tboxes) for b in boxes)
        keys = ["all"] + [label for lo, hi, label in SPEED_BINS if lo <= speeds.get(i, 0.0) < hi]
        for k in keys:
            a = agg[k]
            a["frames"] += 1
            a["stray"] += stray
            a["covered_iou"] += covered_iou
            if covered:
                a["covered"] += 1
                a["err"].append(min(centre_dist(b, t) for b in boxes for t in tboxes))
    return {k: {"frames": v["frames"], "coverage": v["covered"] / v["frames"],
                "coverage_iou": v["covered_iou"] / v["frames"],
                "stray_rate": v["stray"] / v["frames"],
                "centre_err_px_median": statistics.median(v["err"]) if v["err"] else None}
            for k, v in agg.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sidecar")
    ap.add_argument("--nvdcf", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--csrt")
    ap.add_argument("--json", help="also write the full results here")
    args = ap.parse_args()
    truth, speeds = load_truth(args.sidecar)
    results = {}
    for spec in args.nvdcf:
        name, path = spec.split("=", 1)
        for policy, published in nvdcf_policies(path).items():
            key = "YOLO alone" if policy == "YOLO alone" else f"NvDCF {name}: {policy}"
            if key not in results:
                results[key] = score(published, truth, speeds)
    if args.csrt:
        for policy, published in csrt_policies(args.csrt).items():
            results[policy] = score(published, truth, speeds)
    bins = [label for _, _, label in SPEED_BINS]
    print(f"{'method':48} {'coverage':>9} {'IoU cov':>8} {'stray':>7} {'err px':>7} | " + " | ".join(f"{b[:22]:>22}" for b in bins))
    for name, r in results.items():
        a = r["all"]
        per = " | ".join(f"{r[b]['coverage']:>11.1%} cov {r[b]['stray_rate']:>5.1%} st" if b in r else f"{'-':>22}"
                         for b in bins)
        err = f"{a['centre_err_px_median']:.1f}" if a["centre_err_px_median"] is not None else "-"
        print(f"{name:48} {a['coverage']:>9.1%} {a['coverage_iou']:>8.1%} {a['stray_rate']:>7.1%} {err:>7} | {per}")
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
