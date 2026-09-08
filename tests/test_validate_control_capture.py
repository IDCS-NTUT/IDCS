from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_capture_qualification_accepts_complete_observations() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools/validate_control_capture.py"),
         str(root / "tests/fixtures/control_protocol_shadow_rate_trace.jsonl"),
         "--min-observations", "1"], capture_output=True, text=True,
    )
    report = json.loads(result.stdout)
    assert result.returncode == 0
    assert report["qualified"]
