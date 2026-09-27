from __future__ import annotations

import pytest

from jetson.tools.limit_probe import calibrate_sign, main, step_loss_counts, summarize


def test_disagreement_uses_the_calibrated_sign() -> None:
    assert step_loss_counts(100, -512, -1) == pytest.approx(0.0)
    assert step_loss_counts(100, 400, 1) == pytest.approx(112.0)
    assert calibrate_sign(100, -510) == -1
    with pytest.raises(ValueError, match="no motion"):
        calibrate_sign(0, 3)
    with pytest.raises(ValueError, match="scale"):
        calibrate_sign(100, 200)  # half the expected counts: wrong steps_per_rev


def test_levels_above_the_first_failure_are_not_credited() -> None:
    def m(acc, level, lost):
        return {"acc": acc, "level": level, "lost_steps": lost}

    moves = [m(10, 1, False), m(10, 1, False), m(10, 2, False), m(10, 2, True), m(10, 4, False),
             m(50, 1, True)]
    summary = summarize(moves, [1, 2, 4], [10, 50])
    assert summary["10"]["max_passing_level"] == 1 and summary["10"]["first_failing_level"] == 2
    assert summary["50"]["max_passing_level"] is None


def test_plan_refuses_a_move_that_could_pass_the_guard(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--addr", "1", "--levels", "1,12", "--move-s", "1.0"])
    assert main(["--addr", "1"]) == 0  # dry run: plan only
    assert '"event": "plan"' in capsys.readouterr().out
