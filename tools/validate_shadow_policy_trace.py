#!/usr/bin/env python3
"""Independently replay and qualify a recorded shadow-controller trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.schemas import ControlIntent, ControlObservation
from jetson.qualified_controller_profile import load_qualified_shadow_policy_config
from jetson.shadow_rate_policy import ShadowRatePolicy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_trace(
    trace_path: Path,
    report_path: Path,
    *,
    min_tracking_intents: int = 1,
    min_error_span_rad: float = 0.0,
    max_decode_errors: int = 0,
    max_missed_periods: int = 0,
    max_deadline_lateness_ms: float = math.inf,
    require_no_external_bearing_rate: bool = False,
) -> dict[str, Any]:
    config = load_qualified_shadow_policy_config(report_path)
    policy = ShadowRatePolicy(config)
    observations: list[ControlObservation] = []
    recorded_intents: list[ControlIntent] = []
    meta: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    invalid_records = 0
    with trace_path.open(encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
                record_type = record.get("type")
                if record_type == "meta":
                    meta = record
                elif record_type == "summary":
                    summary = record
                elif record_type == "observation":
                    observations.append(ControlObservation.model_validate(record["observation"]))
                elif record_type in {"intent", "replayed_intent"}:
                    recorded_intents.append(ControlIntent.model_validate(record["intent"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_records += 1

    replayed = [policy.decide(observation) for observation in observations]
    mismatch_indices = [
        index for index, (actual, expected) in enumerate(zip(recorded_intents, replayed), start=1)
        if actual != expected
    ]
    valid_errors = [
        observation.target.bearing_error_rad for observation in observations
        if observation.target.valid and observation.target.bearing_error_rad is not None
    ]
    yaw_span = max((value[0] for value in valid_errors), default=0.0) - min(
        (value[0] for value in valid_errors), default=0.0
    )
    pitch_span = max((value[1] for value in valid_errors), default=0.0) - min(
        (value[1] for value in valid_errors), default=0.0
    )
    tracking = sum(intent.reason == "tracking" for intent in recorded_intents)
    external_rates = sum(
        observation.target.bearing_rate_rad_s is not None for observation in observations
    )
    provenance_complete = sum(
        observation.source_frame_id is not None
        and observation.source_time_ns is not None
        and observation.source_clock_domain is not None
        for observation in observations
    )
    tracking_provenance_complete = sum(
        intent.reason == "tracking"
        and observation.source_frame_id is not None
        and observation.source_time_ns is not None
        and observation.source_clock_domain is not None
        for observation, intent in zip(observations, recorded_intents)
    )
    report_hash = _sha256(report_path)
    failures: list[str] = []
    if invalid_records:
        failures.append(f"invalid_records={invalid_records}")
    if len(observations) != len(recorded_intents):
        failures.append(
            f"observation_intent_count={len(observations)}/{len(recorded_intents)}"
        )
    if mismatch_indices:
        failures.append(f"intent_mismatches={len(mismatch_indices)}")
    if tracking < min_tracking_intents:
        failures.append(f"tracking_intents={tracking}<{min_tracking_intents}")
    if min(yaw_span, pitch_span) < min_error_span_rad:
        failures.append(
            f"error_span_rad=({yaw_span:.6g},{pitch_span:.6g})<{min_error_span_rad:.6g}"
        )
    if require_no_external_bearing_rate and external_rates:
        failures.append(f"external_bearing_rate_observations={external_rates}")
    if int(summary.get("decode_errors", -1)) > max_decode_errors:
        failures.append(f"decode_errors={summary.get('decode_errors')}>{max_decode_errors}")
    if int(summary.get("missed_periods", -1)) > max_missed_periods:
        failures.append(f"missed_periods={summary.get('missed_periods')}>{max_missed_periods}")
    if float(summary.get("max_deadline_lateness_ms", math.inf)) > max_deadline_lateness_ms:
        failures.append(
            f"max_deadline_lateness_ms={summary.get('max_deadline_lateness_ms')}"
            f">{max_deadline_lateness_ms}"
        )
    if meta.get("physical_control_disabled") is not True or summary.get("physical_control_disabled") is not True:
        failures.append("physical_control_not_explicitly_disabled")
    if meta.get("policy") != "shadow_rate_qualified_los" or summary.get("policy") != "shadow_rate_qualified_los":
        failures.append("trace_policy_is_not_qualified_los")
    if meta.get("qualified_controller_report_sha256") != report_hash or summary.get(
        "qualified_controller_report_sha256"
    ) != report_hash:
        failures.append("qualified_controller_report_hash_mismatch")
    if tracking_provenance_complete != tracking:
        failures.append(
            f"tracking_source_provenance={tracking_provenance_complete}/{tracking}"
        )
    max_yaw = max((abs(intent.yaw_rate_rad_s) for intent in recorded_intents), default=0.0)
    max_pitch = max((abs(intent.pitch_rate_rad_s) for intent in recorded_intents), default=0.0)
    if max_yaw > config.yaw_rate_limit_rad_s + 1e-12 or max_pitch > config.pitch_rate_limit_rad_s + 1e-12:
        failures.append("recorded_intent_exceeds_rate_limit")

    return {
        "format": "idcs.shadow_policy_trace_validation",
        "qualified": not failures,
        "failures": failures,
        "observations": len(observations),
        "recorded_intents": len(recorded_intents),
        "exact_intent_matches": len(replayed) - len(mismatch_indices),
        "tracking_intents": tracking,
        "external_bearing_rate_observations": external_rates,
        "source_provenance_observations": provenance_complete,
        "tracking_source_provenance_observations": tracking_provenance_complete,
        "yaw_error_span_rad": yaw_span,
        "pitch_error_span_rad": pitch_span,
        "max_abs_yaw_rate_rad_s": max_yaw,
        "max_abs_pitch_rate_rad_s": max_pitch,
        "decode_errors": summary.get("decode_errors"),
        "missed_periods": summary.get("missed_periods"),
        "max_deadline_lateness_ms": summary.get("max_deadline_lateness_ms"),
        "physical_control_disabled": summary.get("physical_control_disabled") is True,
        "qualified_controller_report_sha256": report_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--qualified-controller-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-tracking-intents", type=int, default=1)
    parser.add_argument("--min-error-span-rad", type=float, default=0.0)
    parser.add_argument("--max-decode-errors", type=int, default=0)
    parser.add_argument("--max-missed-periods", type=int, default=0)
    parser.add_argument("--max-deadline-lateness-ms", type=float, default=math.inf)
    parser.add_argument("--require-no-external-bearing-rate", action="store_true")
    args = parser.parse_args(argv)
    result = validate_trace(
        args.trace,
        args.qualified_controller_report,
        min_tracking_intents=args.min_tracking_intents,
        min_error_span_rad=args.min_error_span_rad,
        max_decode_errors=args.max_decode_errors,
        max_missed_periods=args.max_missed_periods,
        max_deadline_lateness_ms=args.max_deadline_lateness_ms,
        require_no_external_bearing_rate=args.require_no_external_bearing_rate,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
