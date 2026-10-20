"""Bounded DCF-only hold trial switch (IDCS_VPI_DCF_HOLD), no GPU."""
import numpy as np
import pytest

from common.perception import NormalizedBoxV2
from jetson.visual_lock.fusion import VisualLock
from tests.test_visual_lock import frame, proposal


def _tracking(monkeypatch, hold):
    if hold is None: monkeypatch.delenv("IDCS_VPI_DCF_HOLD", raising=False)
    else: monkeypatch.setenv("IDCS_VPI_DCF_HOLD", hold)
    lock = VisualLock()
    lock.seed(frame(), NormalizedBoxV2(x=.25, y=.25, w=.25, h=.25), 16)
    for i in range(1, 4):
        d = lock.evaluate(frame(i), proposal(lock))
    assert d.state == "tracking"
    return lock


def _starved(lock, psr=15., dx=2.):
    # Only three points pass FB: below min_points (8).
    p = proposal(lock, dx=dx, psr=psr)
    fb = np.full(len(p.before), 5.); fb[:3] = 0
    return proposal(lock, dx=dx, psr=psr, fb=fb)


def test_off_by_default_starvation_rejects(monkeypatch):
    lock = _tracking(monkeypatch, None)
    d = lock.evaluate(frame(4), _starved(lock))
    assert not d.accepted and d.state == "uncertain"


def test_hold_follows_dcf_with_frozen_size_and_no_learning(monkeypatch):
    lock = _tracking(monkeypatch, "12,5")
    size = lock.box[2:]
    d = lock.evaluate(frame(4), _starved(lock))
    assert d.accepted and d.state == "tracking" and d.reason.startswith("DCF hold")
    assert not d.update_template and not d.geometry_confident and d.box[2:] == size
    assert bin(d.retain_mask).count("1") == 3  # surviving points kept for replenishment


def test_hold_is_bounded_then_the_normal_loss_path_applies(monkeypatch):
    lock = _tracking(monkeypatch, "12,2")
    states = [lock.evaluate(frame(i), _starved(lock)).state for i in range(4, 9)]
    assert states == ["tracking", "tracking", "uncertain", "uncertain", "lost"]


def test_good_frame_resets_the_hold(monkeypatch):
    lock = _tracking(monkeypatch, "12,2")
    lock.evaluate(frame(4), _starved(lock)); lock.evaluate(frame(5), _starved(lock))
    assert lock.evaluate(frame(6), proposal(lock)).reason != "DCF hold"
    assert lock.hold == 0
    assert lock.evaluate(frame(7), _starved(lock)).accepted


@pytest.mark.parametrize("psr,dx", [(11., 2.), (15., 30.)])
def test_weak_dcf_or_jump_is_never_held(monkeypatch, psr, dx):
    lock = _tracking(monkeypatch, "12,5")
    assert not lock.evaluate(frame(4), _starved(lock, psr=psr, dx=dx)).accepted


def test_acquiring_locks_are_never_held(monkeypatch):
    monkeypatch.setenv("IDCS_VPI_DCF_HOLD", "12,5")
    lock = VisualLock()
    lock.seed(frame(), NormalizedBoxV2(x=.25, y=.25, w=.25, h=.25), 16)
    assert not lock.evaluate(frame(1), _starved(lock)).accepted


def test_invalid_switch(monkeypatch):
    for raw in ("x", "3,10", "12,0"):
        monkeypatch.setenv("IDCS_VPI_DCF_HOLD", raw)
        with pytest.raises(ValueError):
            VisualLock()
