"""Offline scoring only. Simulator truth is NEVER fed to the visual tracker.

Scene projections describe an object centre, not a tightly labeled mesh mask.
Report raw centre error AND drift of the operator's initially selected ROI point.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def score(rows,truth_rows,target_id,*,max_p95_px=3.,max_peak_px=8.):
    truth = {r["frame_id"]:r for r in truth_rows}
    if len(truth) != len(truth_rows) or not rows: raise ValueError("empty or duplicate truth evidence")
    seed = rows[0]
    if not seed.get("seed") or seed.get("box") is None: raise ValueError("explicit seed evidence required")
    def match(row):
        ref = truth.get(row["frame_id"])
        if ref is None or ref["source_time_ns"] != row["source_time_ns"]:
            raise ValueError("truth/frame identity join incomplete")
        targets = [t for t in ref["truth"] if t["id"] == target_id]
        if len(targets) != 1: raise ValueError("chosen truth identity absent or ambiguous")
        t = targets[0]
        center = np.array([(t["x"]+t["w"]/2)*ref["width"],(t["y"]+t["h"]/2)*ref["height"]])
        return center,t["w"]*ref["width"]
    initial_center,initial_width = match(seed)
    if initial_width <= 0: raise ValueError("invalid truth width")
    initial_offset = np.array(seed["box"][:2])+np.array(seed["box"][2:])/2-initial_center
    raw,relative = [],[]
    selected = 0
    for row in rows:
        center,width = match(row)
        if not row["selected"]: continue
        selected += 1
        point = np.array(row["box"][:2])+np.array(row["box"][2:])/2
        raw.append(float(np.linalg.norm(point-center)))
        expected_roi_point = center+initial_offset*(width/initial_width)
        relative.append(float(np.linalg.norm(point-expected_roi_point)))
    coverage = selected/max(1,len(rows)-3)
    percentile = lambda values,p:float(np.percentile(values,p)) if values else None
    p95,peak = percentile(relative,95),max(relative) if relative else None
    return dict(frames=len(rows),selected=selected,coverage_after_initial_acquisition=coverage,
        target_truth_id=target_id,initial_roi_offset_px=initial_offset.tolist(),
        centre_error_px_p50=percentile(raw,50),centre_error_px_p95=percentile(raw,95),
        seed_relative_error_px_p50=percentile(relative,50),seed_relative_error_px_p95=p95,
        seed_relative_error_px_max=peak,truth_scope="projected scene centre/width, not labeled mesh silhouette",
        source_span_s=(rows[-1]["source_time_ns"]-seed["source_time_ns"])/1e9,
        acceptance="pass" if coverage>=.95 and p95 is not None and p95<=max_p95_px and peak<=max_peak_px else "fail",
        gates=dict(min_coverage=.95,max_seed_relative_p95_px=max_p95_px,max_seed_relative_peak_px=max_peak_px),
        truth_used_only_after_tracking=True,hardware_or_latency_qualification=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames",type=Path,required=True)
    p.add_argument("--truth",type=Path,required=True)
    p.add_argument("--target-id",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    load = lambda path:[json.loads(line) for line in path.read_text().splitlines()]
    report = score(load(args.frames),load(args.truth),args.target_id)
    with args.output.open("x") as out: out.write(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["acceptance"] == "pass" else 1


if __name__ == "__main__": raise SystemExit(main())
