from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

from common.schemas import (ControlGimbalObservation, ControlObservation, ControlSafetyObservation,
                            ControlTargetObservation, ControlTransportObservation)
from jetson.control_replay import HoldPolicy, replay_observations


def _observation(sequence: int, timestamp: int) -> ControlObservation:
    return ControlObservation(sequence=sequence, created_monotonic_ns=timestamp,
                              target=ControlTargetObservation(valid=False),
                              gimbal=ControlGimbalObservation(valid=False),
                              transport=ControlTransportObservation(),
                              safety=ControlSafetyObservation(valid=True, auto_allowed=True,
                                                               manual_active=False, emergency_active=False))


def test_replay_is_deterministic_and_rejects_out_of_order_snapshots() -> None:
    intents, result = replay_observations([_observation(1, 100), _observation(3, 300), _observation(2, 200)], HoldPolicy())
    assert result.observations == 3 and result.intents == 2 and result.rejected == 1
    assert [intent.observation_sequence for intent in intents] == [1, 3]
    assert [intent.sequence for intent in intents] == [1, 2]
    assert all(intent.yaw_rate_rad_s == 0.0 and intent.reason == "hold_valid" for intent in intents)


def test_shadow_rate_cli_matches_golden_trace(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "intents.jsonl"
    completed = subprocess.run(
        [sys.executable, str(root / "tools/replay_control_protocol_trace.py"),
         str(root / "tests/fixtures/control_protocol_shadow_rate_trace.jsonl"),
         "--policy", "shadow-rate", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    expected = (root / "tests/fixtures/control_protocol_shadow_rate_expected.jsonl").read_text(encoding="utf-8")
    assert output.read_text(encoding="utf-8") == expected
    assert json.loads(completed.stdout) == {
        "intents": 5, "invalid_records": 0, "observations": 6,
        "physical_control_disabled": True, "policy": "shadow_rate", "rejected": 1,
    }


def test_qualified_los_cli_matches_native_v2_golden_without_external_rate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    trace = root / "tests/fixtures/control_protocol_qualified_los_trace.jsonl"
    report = root / "artifacts/controller_sim/los_kalman_feedforward_wire_20260914/los_estimator_validation_report.json"
    expected = (root / "tests/fixtures/control_protocol_qualified_los_expected.jsonl").read_text(encoding="utf-8")
    trace_records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert all("bearing_rate_rad_s" not in record["observation"]["target"] for record in trace_records)

    outputs = []
    summaries = []
    for run in range(2):
        output = tmp_path / f"qualified-intents-{run}.jsonl"
        completed = subprocess.run(
            [sys.executable, str(root / "tools/replay_control_protocol_trace.py"), str(trace),
             "--policy", "shadow-rate", "--qualified-controller-report", str(report),
             "--output", str(output)],
            check=True, capture_output=True, text=True,
        )
        outputs.append(output.read_text(encoding="utf-8"))
        summaries.append(json.loads(completed.stdout))

    assert outputs == [expected, expected]
    assert summaries[0] == summaries[1]
    assert summaries[0] == {
        "intents": 6, "invalid_records": 0, "observations": 6,
        "physical_control_disabled": True, "policy": "shadow_rate_qualified_los", "rejected": 0,
        "qualified_controller_report": str(report.resolve()),
        "qualified_controller_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
