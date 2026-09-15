from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.validate_shadow_policy_trace import validate_trace


def test_qualified_golden_trace_replays_with_exact_intent_parity(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    observations = [
        json.loads(line) for line in
        (root / "tests/fixtures/control_protocol_qualified_los_trace.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    intents = [
        json.loads(line) for line in
        (root / "tests/fixtures/control_protocol_qualified_los_expected.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    report = root / "artifacts/controller_sim/los_kalman_feedforward_wire_20260914/los_estimator_validation_report.json"
    report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    trace = tmp_path / "paired.jsonl"
    records = [{
        "type": "meta", "policy": "shadow_rate_qualified_los",
        "qualified_controller_report_sha256": report_hash,
        "physical_control_disabled": True,
    }]
    for observation, intent in zip(observations, intents):
        intent["type"] = "intent"
        records.extend((observation, intent))
    records.append({
        "type": "summary", "policy": "shadow_rate_qualified_los",
        "qualified_controller_report_sha256": report_hash,
        "decode_errors": 0, "missed_periods": 0, "max_deadline_lateness_ms": 0.5,
        "physical_control_disabled": True,
    })
    trace.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    result = validate_trace(
        trace, report,
        min_tracking_intents=6,
        min_error_span_rad=0.05,
        max_deadline_lateness_ms=1.0,
        require_no_external_bearing_rate=True,
    )

    assert result["qualified"] is True
    assert result["exact_intent_matches"] == 6
    assert result["external_bearing_rate_observations"] == 0
    assert result["tracking_source_provenance_observations"] == 6
