#!/usr/bin/env python3
"""Qualify a controller observation/intent trace for shadow-capture evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common.schemas import ControlObservation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--min-observations", type=int, default=500)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--require-auto", action="store_true",
                        help="require fresh safety authority allowing automatic shadow tracking")
    parser.add_argument(
        '--require-scheduler-health',
        action='store_true',
        help='require a scheduler summary with zero decode errors and missed periods',
    )
    args = parser.parse_args()
    observations: list[ControlObservation] = []
    reasons: Counter[str] = Counter()
    scheduler_health = None
    for line in args.trace.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "observation":
            observations.append(ControlObservation.model_validate(record["observation"]))
        elif record.get("type") == "intent":
            reasons[str(record["intent"].get("reason", "unknown"))] += 1
        elif record.get('type') == 'summary':
            scheduler_health = record
    total = len(observations)
    target = sum(value.target.valid for value in observations)
    gimbal = sum(value.gimbal.valid for value in observations)
    safety = sum(value.safety.valid for value in observations)
    automatic = sum(value.safety.auto_allowed for value in observations)
    complete = sum(
        value.target.valid and value.gimbal.valid and value.safety.valid
        for value in observations
    )
    automatic_tracking_ready = sum(
        value.target.valid
        and value.gimbal.valid
        and value.safety.valid
        and value.safety.auto_allowed
        for value in observations
    )
    required = automatic_tracking_ready if args.require_auto else complete
    fraction = required / total if total else 0.0
    failures: list[str] = []
    if total < args.min_observations:
        failures.append("insufficient_observations")
    if fraction < args.min_valid_fraction:
        failures.append("insufficient_valid_input_fraction")
    if args.require_scheduler_health:
        if scheduler_health is None:
            failures.append('missing_scheduler_summary')
        else:
            if int(scheduler_health.get('decode_errors', -1)) != 0:
                failures.append('scheduler_decode_errors')
            if int(scheduler_health.get('missed_periods', -1)) != 0:
                failures.append('scheduler_missed_periods')
    print(json.dumps({
        "qualified": not failures,
        "failures": failures,
        "observations": total,
        "target_valid": target,
        "gimbal_valid": gimbal,
        "safety_valid": safety,
        "auto_allowed": automatic,
        "required_valid_fraction": fraction,
        "intent_reasons": dict(sorted(reasons.items())),
        "physical_control_disabled": True,
        'complete_valid': complete,
        'automatic_tracking_ready': automatic_tracking_ready,
        'scheduler_health': scheduler_health,
    }, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
