from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.replay_hardware_trials import load_trial


def test_load_trial_uses_active_axis_home_and_tracking_intents(tmp_path: Path) -> None:
    run = tmp_path / "v3_single_pitch_p4_18"
    run.mkdir()
    meta = {"type": "meta", "axis": "pitch_a", "active_axis_gains": {"kp": 4.0, "ki": 0.0, "kd": 0.0},
            "observation_delay_ms": 60, "yaw_rate_limit_rad_s": 0.2, "yaw_acceleration_limit_rad_s2": 3.5}
    ticks = [
        {"type": "tick", "elapsed_s": 1.0, "measured_pitch_rad": 1.5, "reference_pitch_rad": 1.5,
         "intent": {"reason": "tracking", "pitch_rate_rad_s": 0.1}},
        {"type": "tick", "elapsed_s": 1.02, "measured_pitch_rad": 1.502, "reference_pitch_rad": 1.56,
         "intent": {"reason": "safety_stale", "pitch_rate_rad_s": 0.3}},
        {"type": "tick", "elapsed_s": 1.04, "measured_pitch_rad": None, "reference_pitch_rad": 1.56},
    ]
    (run / "trial.jsonl").write_text("\n".join(json.dumps(r) for r in [meta, *ticks]))
    trial = load_trial(run / "trial.jsonl")
    assert trial["axis"] == "pitch" and trial["gains"]["kp"] == 4.0
    assert trial["observation_delay_s"] == pytest.approx(0.06)
    assert list(trial["t"]) == pytest.approx([0.0, 0.02])
    assert list(trial["measured"]) == pytest.approx([0.0, 0.002])
    assert list(trial["reference"]) == pytest.approx([0.0, 0.06])
    assert list(trial["command"]) == pytest.approx([0.1, 0.0])  # non-tracking intents are zero
