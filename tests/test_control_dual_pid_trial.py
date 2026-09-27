from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jetson.control.dual_pid_trial import target_offset


def test_dual_target_is_bounded_and_continuous() -> None:
    assert target_offset(2.9) == 0.0
    assert target_offset(3.0) == 0.0
    assert target_offset(3.75) == pytest.approx(0.06)
    assert target_offset(4.5) == pytest.approx(0.0, abs=1e-12)
    assert target_offset(4.0, period_s=4.0) == pytest.approx(0.06)


def test_dual_trial_check_has_no_live_authority(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, "-m", "jetson.control.dual_pid_trial",
        "--gimbal-sub", "tcp://127.0.0.1:55001",
        "--manual-bind", "tcp://127.0.0.1:55002",
        "--intent-bind", "tcp://127.0.0.1:55003",
        "--duration-s", "20", "--feedforward-scale", "0.5",
        "--trace", str(tmp_path / "trace.jsonl"), "--check",
    ], cwd=repo, text=True, capture_output=True, check=True)
    config = json.loads(result.stdout)
    assert config["live"] is False
    assert config["yaw_gains"]["kp"] == 8.0
    assert config["pitch_gains"]["kp"] == 4.0
    assert config["feedforward_scale"] == 0.5
