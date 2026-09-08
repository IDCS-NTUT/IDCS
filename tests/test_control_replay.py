from __future__ import annotations

import json
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
