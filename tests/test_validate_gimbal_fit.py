from __future__ import annotations

from pathlib import Path

from tools.validate_gimbal_fit import build_validation_report


def test_validation_requires_both_axis_parameter_sets(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("axis\n", encoding="utf-8")
    bad_report = {"axes": {"yaw": {"parameters": {"a_u": 1, "a_f": 1, "bias": 0}}}}
    try:
        build_validation_report(bad_report, fit_report_path=tmp_path / "fit.json", validation_csv=csv_path)
    except ValueError as exc:
        assert "pitch" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("missing pitch parameters were accepted")
