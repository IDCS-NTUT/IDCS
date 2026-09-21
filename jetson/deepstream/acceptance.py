"""Evaluate a DeepStream runtime report for passive-video acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def evaluate_report(report: Mapping[str, Any], *, min_steady_fps: float | None = None,
                    receiver_report: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if int(report.get("frames", 0)) <= 0:
        failures.append("no DeepStream frames")
    if not report.get("gpu_osd_enabled"):
        failures.append("GPU OSD was not enabled")
    if not report.get("h264_return_enabled") or int(report.get("encoded_h264_buffers", 0)) <= 0:
        failures.append("return-video encoder produced no H.264 buffers")
    output = report.get("return_output")
    if not isinstance(output, Mapping) or output.get("control_disabled") is not True:
        failures.append("runtime report does not prove control-disabled return output")
    transport = report.get("snapshot_transport")
    if not isinstance(transport, Mapping):
        failures.append("no passive perception transport report")
    else:
        if int(transport.get("published", 0)) <= 0:
            failures.append("no perception records published")
        # Reports predating the explicit flag are RTP/header-correlated.
        if transport.get("header_correlation", True):
            if int(transport.get("invalid_headers", 0)):
                failures.append("invalid PC frame headers observed")
            if int(transport.get("dropped_nonmonotonic", 0)):
                failures.append("non-monotonic PC frame headers observed")
    steady = report.get("steady_pipeline_fps")
    if min_steady_fps is not None:
        if not isinstance(steady, (int, float)) or float(steady) < min_steady_fps:
            failures.append(f"steady pipeline FPS below {min_steady_fps:g}")
    elif isinstance(steady, (int, float)) and float(steady) < 55.0:
        warnings.append("steady pipeline FPS below the CPU-fallback canary floor of 55")
    if receiver_report is not None:
        if int(receiver_report.get("messages", 0)) <= 0:
            failures.append("PC V2 receiver observed no records")
        if int(receiver_report.get("invalid", 0)):
            failures.append("PC V2 receiver observed invalid records")
        if int(receiver_report.get("nonmonotonic_frame_ids", 0)):
            failures.append("PC receiver observed non-monotonic frame IDs")
        if int(receiver_report.get("nonmonotonic_source_timestamps", 0)):
            failures.append("PC receiver observed non-monotonic source timestamps")
    return {"failures": failures, "warnings": warnings}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-steady-fps", type=float)
    parser.add_argument("--receiver-report", type=Path, help="optional PC metadata-monitor report")
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    receiver = json.loads(args.receiver_report.read_text(encoding="utf-8")) if args.receiver_report else None
    outcome = evaluate_report(
        report,
        min_steady_fps=args.min_steady_fps,
        receiver_report=receiver,
    )
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 1 if outcome["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
