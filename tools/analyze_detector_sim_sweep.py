"""Score V2 detector/tracker output against a rendered-simulator manifest."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.perception import NormalizedBoxV2, perception_snapshot_from_json


CLASS_NAMES = {"0": "drone", "1": "person"}


def _class_name(value: str) -> str:
    return CLASS_NAMES.get(str(value), str(value).lower())


def _iou(left: NormalizedBoxV2, expected: Sequence[float]) -> float:
    ax0, ay0, aw, ah = float(left.x), float(left.y), float(left.w), float(left.h)
    bx0, by0, bw, bh = (float(value) for value in expected)
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _read_observation(payload: str) -> tuple[int, list[tuple[str, NormalizedBoxV2]]]:
    """Read either authoritative V2 snapshots or detector-only shadow records."""
    decoded = json.loads(payload)
    if isinstance(decoded, Mapping) and isinstance(decoded.get("boxes"), list):
        objects = []
        for item in decoded["boxes"]:
            objects.append((
                _class_name(str(item["cls"])),
                NormalizedBoxV2(x=item["x"], y=item["y"], w=item["w"], h=item["h"]),
            ))
        return int(decoded["frame_id"]), objects
    snapshot = perception_snapshot_from_json(decoded)
    source = snapshot.tracks if snapshot.tracks else snapshot.detections
    return snapshot.frame.frame_id, [(_class_name(item.class_id), item.box) for item in source]


def analyze(manifest: Mapping[str, Any], payloads: Sequence[str], *, min_iou: float = 0.2) -> dict[str, Any]:
    cases = list(manifest.get("cases", ()))
    blanks = list(manifest.get("blanks", ()))
    by_frame: dict[int, Mapping[str, Any]] = {}
    blank_frames: set[int] = set()
    for case in cases:
        for frame_id in range(int(case["start_frame"]), int(case["end_frame"]) + 1):
            by_frame[frame_id] = case
    for blank in blanks:
        blank_frames.update(range(int(blank["start_frame"]), int(blank["end_frame"]) + 1))
    stats: dict[str, dict[str, Any]] = {
        str(case["case_id"]): {"observed": 0, "hits": 0, "confused": 0,
                               "predictions": Counter(), "ious": []}
        for case in cases
    }
    invalid = 0
    nonmonotonic = 0
    blank_observed = 0
    blank_false_positive = 0
    last_frame: int | None = None
    for payload in payloads:
        try:
            frame_id, objects = _read_observation(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid += 1
            continue
        if last_frame is not None and frame_id <= last_frame:
            nonmonotonic += 1
        last_frame = frame_id
        case = by_frame.get(frame_id)
        if frame_id in blank_frames:
            blank_observed += 1
            if objects:
                blank_false_positive += 1
        if case is None:
            continue
        record = stats[str(case["case_id"])]
        record["observed"] += 1
        expected_class = str(case["expected_class"])
        expected_box = case["expected_box"]
        best_expected_iou = 0.0
        overlap_wrong = False
        for name, box in objects:
            overlap = _iou(box, expected_box)
            record["predictions"][name] += 1
            if name == expected_class:
                best_expected_iou = max(best_expected_iou, overlap)
            elif overlap >= min_iou:
                overlap_wrong = True
        if best_expected_iou >= min_iou:
            record["hits"] += 1
            record["ious"].append(best_expected_iou)
        elif overlap_wrong:
            record["confused"] += 1
    case_reports: list[dict[str, Any]] = []
    aggregate = {name: {"observed": 0, "hits": 0, "confused": 0} for name in CLASS_CASES}
    for case in cases:
        record = stats[str(case["case_id"])]
        observed = int(record["observed"])
        hits = int(record["hits"])
        confused = int(record["confused"])
        class_name = str(case["expected_class"])
        aggregate.setdefault(class_name, {"observed": 0, "hits": 0, "confused": 0})
        aggregate[class_name]["observed"] += observed
        aggregate[class_name]["hits"] += hits
        aggregate[class_name]["confused"] += confused
        case_reports.append({
            **dict(case),
            "observed_frames": observed,
            "hit_frames": hits,
            "confused_frames": confused,
            "recall": hits / observed if observed else 0.0,
            "confusion_rate": confused / observed if observed else 0.0,
            "mean_iou": sum(record["ious"]) / len(record["ious"]) if record["ious"] else 0.0,
            "predictions": dict(sorted(record["predictions"].items())),
        })
    class_reports = {}
    for name, record in aggregate.items():
        observed = int(record["observed"])
        class_reports[name] = {
            **record,
            "recall": int(record["hits"]) / observed if observed else 0.0,
            "confusion_rate": int(record["confused"]) / observed if observed else 0.0,
        }
    return {"schema": "idcs.detector_sim_sweep_report", "version": 1,
            "invalid_messages": invalid, "nonmonotonic_frame_ids": nonmonotonic,
            "blank": {
                "observed": blank_observed,
                "false_positive_frames": blank_false_positive,
                "false_positive_rate": blank_false_positive / blank_observed if blank_observed else 0.0,
            },
            "min_iou": min_iou, "classes": class_reports, "cases": case_reports}


CLASS_CASES = ("drone", "person")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.2)
    parser.add_argument("--min-person-recall", type=float, default=0.8)
    parser.add_argument("--min-drone-recall", type=float, default=0.8)
    parser.add_argument("--max-confusion-rate", type=float, default=0.1)
    parser.add_argument("--max-blank-false-positive-rate", type=float, default=0.1)
    args = parser.parse_args(argv)
    if not 0 <= args.min_iou <= 1:
        parser.error("--min-iou must be between 0 and 1")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payloads = args.jsonl.read_text(encoding="utf-8").splitlines()
    report = analyze(manifest, payloads, min_iou=args.min_iou)
    failures: list[str] = []
    for name, threshold in (("person", args.min_person_recall), ("drone", args.min_drone_recall)):
        result = report["classes"].get(name, {})
        if not result.get("observed"):
            failures.append(f"{name}: no observed validation frames")
        elif float(result.get("recall", 0.0)) < threshold:
            failures.append(f"{name}: recall below {threshold:.3f}")
        if float(result.get("confusion_rate", 0.0)) > args.max_confusion_rate:
            failures.append(f"{name}: confusion above {args.max_confusion_rate:.3f}")
    missing_cases = [item["case_id"] for item in report["cases"] if not item["observed_frames"]]
    if missing_cases:
        failures.append("unobserved cases: " + ", ".join(missing_cases))
    blank = report["blank"]
    if not blank["observed"]:
        failures.append("no observed blank validation frames")
    elif float(blank["false_positive_rate"]) > args.max_blank_false_positive_rate:
        failures.append(
            f"blank false-positive rate above {args.max_blank_false_positive_rate:.3f}"
        )
    if report["invalid_messages"]:
        failures.append("invalid V2 messages observed")
    if report["nonmonotonic_frame_ids"]:
        failures.append("non-monotonic V2 frame IDs observed")
    report["failures"] = failures
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
