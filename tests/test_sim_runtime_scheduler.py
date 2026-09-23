from __future__ import annotations

import pytest

from jetson.sim_control_runtime import _advance_fixed_deadline


def test_fixed_deadline_does_not_accumulate_small_wakeup_delays() -> None:
    next_tick = 10.0

    next_tick, skipped = _advance_fixed_deadline(next_tick, 10.001, 0.02)
    assert next_tick == pytest.approx(10.02)
    assert skipped == 0

    next_tick, skipped = _advance_fixed_deadline(next_tick, 10.0215, 0.02)
    assert next_tick == pytest.approx(10.04)
    assert skipped == 0


def test_fixed_deadline_skips_overdue_periods_without_catchup_burst() -> None:
    next_tick, skipped = _advance_fixed_deadline(10.0, 10.105, 0.02)

    assert next_tick == pytest.approx(10.12)
    assert skipped == 5


def test_fixed_deadline_rejects_nonpositive_period() -> None:
    with pytest.raises(ValueError, match="positive"):
        _advance_fixed_deadline(10.0, 10.0, 0.0)
