from dataclasses import replace

import numpy as np
import pytest

from jetson.visual_lock.fusion import Proposal
from test_visual_lock import frame,seeded,proposal


def test_absolute_scale_does_not_compound_biased_adjacent_measurement():
    lock = seeded()
    p = proposal(lock,dx=0,dy=0)
    ids = np.arange(len(p.before),dtype=np.uint64)
    base = p.before.copy()
    center = np.array(lock.box[:2])+np.array(lock.box[2:])/2
    assert lock.evaluate(frame(1),replace(p,point_ids=ids)).accepted
    for i in range(2,1002):
        # Systematic 0.1% adjacent-frame scale bias must not become a 2.7x box.
        before = center+(base-center)*.999
        d = lock.evaluate(frame(i),replace(p,before=before,after=base,point_ids=ids))
        assert d.accepted and d.geometry_confident
    assert lock.box[2:] == pytest.approx((80,60),abs=1e-6)


def test_reference_scale_follows_actual_change_and_keeps_anchors():
    lock = seeded()
    p = proposal(lock,dx=0,dy=0)
    ids = np.arange(16,dtype=np.uint64)
    base = p.before.copy()
    center = np.array([120,90])
    before = base
    for i in range(1,101):
        scale = 1+i*.002
        after = center+(base-center)*scale
        d = lock.evaluate(frame(i),replace(p,box=lock.box,before=before,after=after,point_ids=ids))
        assert d.accepted and d.geometry_confident
        before = after
    assert lock.box[2:] == pytest.approx((96,72))
    assert np.array([lock.anchors[k] for k in range(16)]) == pytest.approx(base)


def test_unknown_geometry_holds_size_and_template_without_reseeding():
    lock = seeded()
    p = proposal(lock,dx=0,dy=0,point_ids=np.arange(16))
    for i in range(1,4): assert lock.evaluate(frame(i),p).accepted
    anchors = {k:v.copy() for k,v in lock.anchors.items()}
    d = lock.evaluate(frame(4),replace(p,point_ids=np.arange(100,116)))
    assert d.accepted and not d.geometry_confident and not d.update_template
    assert d.box[2:] == pytest.approx((80,60))
    assert lock.anchors.keys() == anchors.keys()
    assert lock.track_id == 1


def test_retain_mask_excludes_fb_and_ransac_outliers():
    lock = seeded()
    p = proposal(lock)
    fb = p.fb.copy(); fb[0] = 2
    after = p.after.copy(); after[1:4] += [30,20]
    d = lock.evaluate(frame(1),replace(p,fb=fb,after=after))
    assert d.accepted and d.inliers == 12
    assert d.retain_mask == sum(1<<i for i in range(4,16))


def test_position_quality_does_not_imply_geometry_or_learning_quality():
    lock = seeded()
    p = proposal(lock,dx=0,dy=0)
    for i in range(1,4): assert lock.evaluate(frame(i),p).accepted
    d = lock.evaluate(frame(4),replace(p,fb=np.full(16,.75)))
    assert d.accepted and d.state == "tracking"
    assert not d.geometry_confident and not d.update_template


def test_immutable_reference_position_overrides_accumulated_local_flow_bias():
    lock = seeded()
    p = proposal(lock,dx=0,dy=0)
    base = p.before.copy()
    p = replace(p,after=base+[5,0],reference_before=base,reference_after=base,reference_fb=np.zeros(16))
    d = lock.evaluate(frame(1),p)
    assert d.accepted and d.geometry_confident
    assert d.box == pytest.approx((80,60,80,60))


def test_appearance_anchor_bounds_drift_when_reference_geometry_unavailable():
    lock = seeded()
    for i in range(1,501):
        p = proposal(lock,dx=5,dy=0,box=(80,60,80,60),reference_before=np.empty((0,2)),
                     reference_after=np.empty((0,2)),reference_fb=np.empty(0))
        d = lock.evaluate(frame(i),p)
        assert d.accepted and not d.geometry_confident and not d.update_template
    assert lock.box[0] == pytest.approx(80+5/3)
    assert lock.box[2:] == pytest.approx((80,60))


def test_reference_corruption_never_becomes_selected():
    lock = seeded()
    p = proposal(lock)
    p = replace(p,reference_before=p.before,reference_after=np.full((16,2),np.nan),reference_fb=np.zeros(16))
    assert lock.evaluate(frame(1),p).state == "lost"


@pytest.mark.parametrize("ids",[np.zeros(16,dtype=int),np.arange(15),np.arange(16,dtype=float),np.arange(-1,15)])
def test_feature_identity_corruption_fails_closed(ids):
    lock = seeded()
    assert lock.evaluate(frame(1),proposal(lock,point_ids=ids)).state == "lost"
