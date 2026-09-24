#!/usr/bin/env python3
"""Attribute estimator/feedforward decisions on one recorded V2 trace.

This is an identical-input, non-causal ablation. It can compare commands and
safety decisions, but cannot claim closed-loop tracking improvement because the
recorded observations were generated under one active policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.schemas import ControlDiagnostics, ControlIntent, ControlObservation
from jetson.qualified_controller_profile import load_qualified_shadow_policy_config
from jetson.shadow_rate_policy import ShadowRatePolicy, ShadowRatePolicyConfig


REPORT_FORMAT = "idcs.estimator_feedforward_trace_ablation"
REPORT_VERSION = 1


def load_trace(path: Path) -> tuple[list[ControlObservation], list[ControlIntent]]:
    observations: list[ControlObservation] = []
    intents: list[ControlIntent] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "tick":
            continue
        try:
            observations.append(ControlObservation.model_validate(record["observation"]))
            intents.append(ControlIntent.model_validate(record["intent"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid tick at {path}:{line_number}: {exc}") from exc
    if not observations:
        raise ValueError(f"trace contains no tick records: {path}")
    return observations, intents


def _rms(values: Iterable[float]) -> float:
    materialized = list(values)
    return math.sqrt(mean(value * value for value in materialized)) if materialized else 0.0


def _axis_metrics(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "rms_rad_s": _rms(values),
        "mean_abs_rad_s": mean(abs(value) for value in values) if values else 0.0,
        "max_abs_rad_s": max((abs(value) for value in values), default=0.0),
        "total_variation_rad_s": sum(abs(b - a) for a, b in zip(values, values[1:])),
        "sign_changes": sum(a * b < 0.0 for a, b in zip(values, values[1:])),
    }


def _optional_axis_metrics(values: Iterable[float | None]) -> dict[str, float | int]:
    present = [value for value in values if value is not None]
    return {"samples": len(present), **_axis_metrics(present)}


def _commanding(reason: str) -> bool:
    return reason in {"tracking", "position_limit_hold"}


def _run_variant(
    observations: Sequence[ControlObservation],
    config: ShadowRatePolicyConfig,
) -> tuple[list[ControlIntent], list[ControlDiagnostics]]:
    policy = ShadowRatePolicy(config)
    intents: list[ControlIntent] = []
    diagnostics: list[ControlDiagnostics] = []
    for observation in observations:
        intent = policy.decide(observation)
        if policy.last_diagnostics is None:
            raise RuntimeError("policy produced no diagnostics")
        intents.append(intent)
        diagnostics.append(policy.last_diagnostics)
    return intents, diagnostics


def _variant_metrics(
    intents: Sequence[ControlIntent], diagnostics: Sequence[ControlDiagnostics]
) -> dict[str, Any]:
    yaw = [intent.yaw_rate_rad_s for intent in intents]
    pitch = [intent.pitch_rate_rad_s for intent in intents]
    reasons = Counter(intent.reason for intent in intents)
    limits = Counter()
    for intent in intents:
        for key, value in intent.limits.model_dump().items():
            if value:
                limits[key] += 1
    return {
        "ticks": len(intents),
        "reasons": dict(sorted(reasons.items())),
        "limit_counts": dict(sorted(limits.items())),
        "commanding_ticks": sum(_commanding(intent.reason) for intent in intents),
        "yaw_command": _axis_metrics(yaw),
        "pitch_command": _axis_metrics(pitch),
        "yaw_feedback": _optional_axis_metrics(item.yaw.feedback_term_rad_s for item in diagnostics),
        "pitch_feedback": _optional_axis_metrics(item.pitch.feedback_term_rad_s for item in diagnostics),
        "yaw_damping": _optional_axis_metrics(item.yaw.damping_term_rad_s for item in diagnostics),
        "pitch_damping": _optional_axis_metrics(item.pitch.damping_term_rad_s for item in diagnostics),
        "yaw_feedforward": _optional_axis_metrics(item.yaw.feedforward_term_rad_s for item in diagnostics),
        "pitch_feedforward": _optional_axis_metrics(item.pitch.feedforward_term_rad_s for item in diagnostics),
        "measurement_updates": sum(item.yaw.measurement_updated for item in diagnostics),
        "measurement_rejections": {
            "yaw": sum(item.yaw.measurement_accepted is False for item in diagnostics),
            "pitch": sum(item.pitch.measurement_accepted is False for item in diagnostics),
        },
        "measurement_reinitializations": {
            "yaw": sum(item.yaw.measurement_reinitialized for item in diagnostics),
            "pitch": sum(item.pitch.measurement_reinitialized for item in diagnostics),
        },
        "estimator_reinitializations_max": {
            "yaw": max((item.yaw.reinitialized_updates for item in diagnostics), default=0),
            "pitch": max((item.pitch.reinitialized_updates for item in diagnostics), default=0),
        },
    }


def _comparison(
    reference: Sequence[ControlIntent], candidate: Sequence[ControlIntent]
) -> dict[str, Any]:
    yaw_delta = [
        value.yaw_rate_rad_s - baseline.yaw_rate_rad_s
        for baseline, value in zip(reference, candidate)
    ]
    pitch_delta = [
        value.pitch_rate_rad_s - baseline.pitch_rate_rad_s
        for baseline, value in zip(reference, candidate)
    ]
    return {
        "yaw_delta": _axis_metrics(yaw_delta),
        "pitch_delta": _axis_metrics(pitch_delta),
        "reason_mismatches": sum(a.reason != b.reason for a, b in zip(reference, candidate)),
        "command_authority_mismatches": sum(
            _commanding(a.reason) != _commanding(b.reason)
            for a, b in zip(reference, candidate)
        ),
    }


def analyze_trace(
    trace_path: Path,
    report_path: Path,
    *,
    yaw_position_limits_rad: tuple[float, float] | None = None,
    pitch_position_limits_rad: tuple[float, float] | None = None,
) -> dict[str, Any]:
    observations, recorded = load_trace(trace_path)
    first = recorded[0]
    valid_for_ns = first.valid_until_monotonic_ns - first.issued_monotonic_ns
    base = load_qualified_shadow_policy_config(
        report_path,
        yaw_position_limits_rad=yaw_position_limits_rad,
        pitch_position_limits_rad=pitch_position_limits_rad,
        valid_for_ns=valid_for_ns,
        intent_mode=first.mode,
        sequence_base=first.sequence - 1,
    )
    variants = {
        "raw_pd": replace(
            base,
            yaw_los_kalman=None,
            pitch_los_kalman=None,
            yaw_feedforward_gain=0.0,
            pitch_feedforward_gain=0.0,
            raw_gimbal_damping=True,
        ),
        "estimated_no_feedforward": replace(
            base,
            yaw_feedforward_gain=0.0,
            pitch_feedforward_gain=0.0,
        ),
        "qualified_estimator_feedforward": base,
    }
    replayed: dict[str, list[ControlIntent]] = {}
    results: dict[str, Any] = {}
    for name, config in variants.items():
        intents, diagnostics = _run_variant(observations, config)
        replayed[name] = intents
        results[name] = _variant_metrics(intents, diagnostics)

    qualified = replayed["qualified_estimator_feedforward"]
    exact_recorded_matches = sum(
        expected.model_dump(mode="json") == actual.model_dump(mode="json")
        for expected, actual in zip(recorded, qualified)
    )
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "causal_performance_claim_allowed": False,
        "trace": str(trace_path.resolve()),
        "qualified_controller_report": str(report_path.resolve()),
        "ticks": len(observations),
        "recorded_qualified_exact_matches": exact_recorded_matches,
        "recorded_qualified_exact_fraction": exact_recorded_matches / len(recorded),
        "variants": results,
        "comparisons_to_qualified": {
            name: _comparison(qualified, intents)
            for name, intents in replayed.items()
            if name != "qualified_estimator_feedforward"
        },
    }


def _bounds(minimum: float | None, maximum: float | None, name: str) -> tuple[float, float] | None:
    if minimum is None and maximum is None:
        return None
    if minimum is None or maximum is None or minimum >= maximum:
        raise ValueError(f"{name} limits require finite min < max")
    return float(minimum), float(maximum)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--qualified-controller-report", required=True, type=Path)
    parser.add_argument("--yaw-min-rad", type=float)
    parser.add_argument("--yaw-max-rad", type=float)
    parser.add_argument("--pitch-min-rad", type=float)
    parser.add_argument("--pitch-max-rad", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze_trace(
            args.trace,
            args.qualified_controller_report,
            yaw_position_limits_rad=_bounds(args.yaw_min_rad, args.yaw_max_rad, "yaw"),
            pitch_position_limits_rad=_bounds(args.pitch_min_rad, args.pitch_max_rad, "pitch"),
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
