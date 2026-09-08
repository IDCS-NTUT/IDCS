from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_parity_comparator_reports_shadow_decisions() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools/compare_control_policy_trace.py"),
         str(root / "tests/fixtures/control_protocol_shadow_rate_trace.jsonl")],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(result.stdout)
    assert report["physical_control_disabled"]
    assert report["observations"] == 6
    assert report["shadow"]["rejected"] == 1
    assert report["shadow_reasons"]["manual_active"] == 1
    assert report["max_shadow_rate_delta_rad_s"] > 0
