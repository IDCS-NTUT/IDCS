#!/usr/bin/env python3
"""Score a frozen gimbal fit report against an independent sweep CSV.

This never fits, tunes, publishes, or opens hardware.  Rows affected by motor
limits or transport-quality faults are excluded by the same quality filter as
the fitter; their counts remain in the report so a seemingly good score cannot
hide an invalid experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from tools.fit_gimbal_response import _axis_samples, _metrics, load_sweep_samples


REPORT_FORMAT = "idcs.gimbal_frozen_fit_validation"
REPORT_VERSION = 1


def build_validation_report(
    fit_report: Mapping[str, Any],
    *,
    fit_report_path: Path,
    validation_csv: Path,
) -> dict[str, Any]:
    samples, quality_filter = load_sweep_samples(validation_csv)
    limit_blocks = int(quality_filter.get("rejected_limit_blocked", 0))
    axes_raw = fit_report.get("axes")
    if not isinstance(axes_raw, Mapping):
        raise ValueError("fit report has no axes mapping")
    axes: dict[str, Any] = {}
    for axis in ("yaw", "pitch"):
        entry = axes_raw.get(axis)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("parameters"), Mapping):
            raise ValueError(f"fit report has no usable {axis} parameters")
        parameters = entry["parameters"]
        params = tuple(float(parameters[name]) for name in ("a_u", "a_f", "bias"))
        delay_s = float(parameters.get("delay_s", 0.0))
        axis_samples = _axis_samples(samples, axis)
        axes[axis] = {
            "selected_model": entry.get("selected_model"),
            "parameters": {"a_u": params[0], "a_f": params[1], "bias": params[2]},
            "delay_s": delay_s,
            "accepted_sample_count": len(axis_samples),
            "metrics": _metrics(axis_samples, params=params, delay_s=delay_s),
        }
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "fit_report": str(fit_report_path),
        "validation_csv": str(validation_csv),
        "quality_filter": quality_filter,
        "qualification": {
            "qualified": bool(samples) and limit_blocks == 0,
            "reason": (
                "qualified_transport_and_limits"
                if samples and limit_blocks == 0
                else "not_qualified_limit_blocked" if limit_blocks else "not_qualified_no_accepted_samples"
            ),
        },
        "axes": axes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    fit_report = json.loads(args.fit_report.read_text(encoding="utf-8"))
    if not isinstance(fit_report, Mapping):
        raise SystemExit("fit report must be a JSON object")
    report = build_validation_report(
        fit_report, fit_report_path=args.fit_report, validation_csv=args.validation_csv
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "qualification": report["qualification"], "quality_filter": report["quality_filter"],
                      "axes": {axis: values["metrics"] for axis, values in report["axes"].items()}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
