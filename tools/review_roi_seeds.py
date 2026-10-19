"""Offline seed review suite, serialized under ONE software trial guard.

No sockets or control. The manifest names six local clutter videos (optional
truth JSONL) or exact-frame capture directories. No input discovery/downloads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2

from jetson.visual_lock.egl_lifetime import StandaloneEGL
from tools.verify_visual_lock import evaluate


HAND_CASES = (
    ("thumb", "camera_hand_thumb_1280x720.png", (486, 298, 62, 89)),
    ("open", "camera_hand_open_1280x720.png", (438, 299, 124, 91)),
)
LOOSE_HAND = (426, 281, 176, 145)
FIXTURES = ("translation", "scale", "distractor", "occlusion", "loss", "textureless")


def load_clips(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 6:
        raise ValueError("manifest must name exactly six clutter clips")
    names = set()
    for entry in data:
        name = entry.get("name", "")
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("clip name must be a safe unique directory name")
        if name in names: raise ValueError("duplicate clip name")
        names.add(name)
        if ("video" in entry) == ("capture" in entry):
            raise ValueError("each clip needs either video or capture")
        if "video" in entry and len(entry.get("seed_xywh", [])) != 4:
            raise ValueError("video needs an explicit seed_xywh")
        for field in ("video", "capture", "truth"):
            if field in entry:
                source = Path(entry[field])
                if not source.is_absolute(): source = Path(path).resolve().parent/source
                if not source.exists(): raise ValueError(f"missing {field}: {source}")
                entry[field] = str(source)
    return data


def summarize(directory, name=None, *, labeled=True):
    """Old and new verifier/capture reports; never invent unlabeled errors."""
    directory = Path(directory)
    path = directory/"report.json"
    if not path.exists(): return {"missing": True}
    document = json.loads(path.read_text(encoding="utf-8"))
    reports = document.get("report")
    report = next((r for r in reports if r["name"] == name), None) if reports else document
    if report is None: return {"missing": True}
    row_path = directory/(f"{name}.jsonl" if reports else "frames.jsonl")
    rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines()] if row_path.exists() else []
    seed = rows[0] if rows else {}
    refinement = seed.get("seed_report") or report.get("seed_report") or {}
    loss = next((dict(frame=r.get("i", r.get("frame_id")), reason=r.get("reason"))
                 for r in rows if r.get("state") == "lost"), report.get("first_loss"))
    coverage = report.get("tracking_coverage")
    if not labeled: coverage = report.get("selection_coverage_after_acquisition")
    if coverage is None and rows:
        coverage = sum(bool(r.get("selected")) for r in rows[3:])/max(1, len(rows)-3)
    return dict(seed_accepted=report.get("seed_accepted", seed.get("state") not in (None, "lost")),
        refined_box=refinement.get("refined_box", report.get("refined_box_px", seed.get("box"))),
        operator_box=refinement.get("operator_box", report.get("seed_box_px")),
        coverage=coverage, error_p50=report.get("centre_error_px_median") if labeled else None,
        error_p95=report.get("centre_error_px_p95") if labeled else None,
        false_tracking=report.get("false_tracking_absent") if labeled else None,
        first_loss=loss, gates_sha256=report.get("gates_sha256"),
        pixels_sha256=report.get("pixels_sha256", report.get("input_sha256")),
        label_scope="provided labels" if labeled else "unlabeled; errors/false tracking unavailable")


def table(rows):
    columns = ("seed_accepted", "refined_box", "coverage", "error_p50", "error_p95", "false_tracking", "first_loss")
    lines = ["| Case | Version | Seed | Refined xywh | Coverage | Error p50 | Error p95 | False tracking | First loss |",
             "|---|---|---|---|---|---|---|---|---|"]
    for case in rows:
        for version in ("baseline", "candidate"):
            values = case[version]
            def cell(key):
                value = values.get(key)
                if value is None: return "n/a"
                if isinstance(value, float): return f"{value:.4f}"
                return json.dumps(value, separators=(",", ":")).replace("|", "\\|")
            lines.append("| "+" | ".join([case["name"], version]+[cell(c) for c in columns])+" |")
    return "\n".join(lines)+"\n"


def run_child(command, log, *, timeout, runner=subprocess.run):
    with Path(log).open("w", encoding="utf-8") as stream:
        done = runner(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    return done.returncode


def run_suite(args, clips):
    output = Path(args.output)
    runs = output/"runs"
    runs.mkdir(parents=True, exist_ok=False)
    commands, comparisons = [], []
    def compare(name, relative, report_name=None, labeled=True, baseline=None):
        candidate = summarize(runs/relative, report_name, labeled=labeled)
        paths = [Path(args.baseline)/(baseline or "runs/"+relative)]
        if baseline is None: paths += [Path(args.baseline)/relative, Path(args.baseline)]
        previous = {"missing": True}
        for path in paths:
            previous = summarize(path, report_name, labeled=labeled)
            if not previous.get("missing"): break
        comparisons.append(dict(name=name, baseline=previous, candidate=candidate,
            same_pixels=(candidate["pixels_sha256"] == previous["pixels_sha256"])
                if candidate.get("pixels_sha256") and previous.get("pixels_sha256") else None,
            same_gates=(candidate.get("gates_sha256") == previous.get("gates_sha256")) if candidate.get("gates_sha256") and previous.get("gates_sha256") else None))
    def child(name, module, extra, *, timeout=360):
        command = [sys.executable, "-m", module, "--library", args.library,
                   "--output", str(runs/name), "--provenance", args.provenance, *extra]
        try: code = run_child(command, output/f"{name}.log", timeout=timeout)
        except subprocess.TimeoutExpired: code = "timeout"
        commands.append(dict(name=name, command=command, returncode=code))
    # One native process for the six unchanged deterministic fixtures.
    child("fixtures", "tools.verify_visual_lock", ["--standalone-egl"])
    for name in FIXTURES: compare(name, "fixtures", name)
    # Frozen hand repeats exercise initialization/retention only, not live motion.
    hand_output = runs/"hands"
    hand_output.mkdir()
    hands = []
    with StandaloneEGL():
        for label, filename, tight in HAND_CASES:
            path = Path(args.hand_frames)/filename
            image = cv2.imread(str(path))
            if image is None: raise ValueError(f"cannot decode {path}")
            for kind, box in (("loose", LOOSE_HAND), ("tight", tight)):
                name = f"{label}-{kind}"
                report = evaluate(name, ((image, None) for _ in range(120)), args.library,
                                  hand_output, box)
                report["fixture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                hands.append(report)
            if label == "thumb":
                hands.append(evaluate("wall", ((image, None) for _ in range(120)), args.library,
                                      hand_output, (200, 180, 120, 100)))
    (hand_output/"report.json").write_text(json.dumps(dict(report=hands), indent=2)+"\n", encoding="utf-8")
    for report in hands: compare(report["name"], "hands", report["name"], labeled=False)
    for entry in clips:
        name = "clutter-"+entry["name"]
        if "capture" in entry:
            child(name, "tools.replay_visual_lock_capture", ["--capture", entry["capture"], "--max-wall-s", "300"])
            report_name = None
        else:
            extra = ["--standalone-egl", "--video", entry["video"], "--fps", str(entry.get("fps", 60)),
                     "--seed-xywh", *map(str, entry["seed_xywh"])]
            if "truth" in entry: extra += ["--truth", entry["truth"]]
            child(name, "tools.verify_visual_lock", extra)
            report_name = "replay"
        compare(name, name, report_name, labeled="truth" in entry and "video" in entry,
                baseline=entry.get("baseline"))
    fixture_doc = json.loads((runs/"fixtures/report.json").read_text()) if (runs/"fixtures/report.json").exists() else {}
    by_name = {r["name"]: r for r in hands}
    hand_pass = all(by_name[f"{label}-loose"]["seed_accepted"]
                    and by_name[f"{label}-tight"]["seed_accepted"]
                    and by_name[f"{label}-loose"]["selected_frames"]
                        >= by_name[f"{label}-tight"]["selected_frames"] > 0
                    and (by_name[f"{label}-loose"]["first_loss"] or {"frame":120})["frame"]
                        >= (by_name[f"{label}-tight"]["first_loss"] or {"frame":120})["frame"]
                    for label, _, _ in HAND_CASES)
    wall = by_name["wall"]
    wall_pass = (not wall["seed_accepted"] and wall["selected_frames"] == 0
                 and wall["first_loss"] is not None and wall["first_loss"]["frame"] == 0)
    native_fault = any("backend fault" in str(r["candidate"].get("first_loss")) for r in comparisons)
    complete = all(not row["candidate"].get("missing") for row in comparisons)
    gates = dict(fixtures=fixture_doc.get("acceptance") == "pass", hands=hand_pass,
                 wall=wall_pass, no_native_fault=not native_fault, reports_complete=complete,
                 children_normal=all(c["returncode"] in (0, 1) for c in commands))
    result = dict(provenance=args.provenance, commands=commands, comparison=comparisons,
                  acceptance_gates=gates, acceptance="pass" if all(gates.values()) else "fail",
                  baseline_complete=all(not row["baseline"].get("missing") for row in comparisons),
                  control_disabled=True, scope="offline; hand frames repeated, no motion qualification")
    (output/"comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    (output/"comparison.md").write_text(table(comparisons), encoding="utf-8")
    print(table(comparisons))
    return 0 if result["acceptance"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--clips", required=True, help="JSON list of six clutter inputs")
    parser.add_argument("--hand-frames", default="tests/fixtures/camera_roi")
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--inside-guard", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    clips = load_clips(args.clips)
    if args.inside_guard: return run_suite(args, clips)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "tools.run_visual_lock_trial_guard", "--host", "jetson",
               "--output", str(output/"guard"), "--", sys.executable, "-m", "tools.review_roi_seeds",
               *sys.argv[1:], "--inside-guard"]
    return subprocess.run(command, timeout=3600).returncode


if __name__ == "__main__":
    raise SystemExit(main())
