from copy import deepcopy

import pytest

from tools.score_visual_lock import score


def fixture():
    rows,truth = [],[]
    for i in range(10):
        size = 20+i
        center = 50+i
        # Initial ROI point lies 5px to the right of the scene centre. It is a
        # fixed selected point, not an unannounced claim of detector centring.
        offset = 5*size/20
        rows.append(dict(frame_id=i,source_time_ns=1_000_000_000+i*20_000_000,seed=i==0,
                         box=[center+offset-size/2,50-size/2,size,size],selected=i>=3))
        truth.append(dict(frame_id=i,source_time_ns=rows[-1]["source_time_ns"],width=100,height=100,
            truth=[dict(id=1,x=(center-size/2)/100,y=(50-size/2)/100,w=size/100,h=size/100)]))
    return rows,truth


def test_scoring_separates_initial_seed_offset_from_drift():
    rows,truth = fixture()
    report = score(rows,truth,1)
    assert report["acceptance"] == "pass"
    assert report["initial_roi_offset_px"] == pytest.approx([5,0])
    assert report["centre_error_px_p95"] > 5
    assert report["seed_relative_error_px_p95"] == pytest.approx(0,abs=1e-10)


def test_false_positive_state_alone_cannot_pass_localization():
    rows,truth = fixture()
    rows[-1]["box"][0] += 50
    assert score(rows,truth,1)["acceptance"] == "fail"


def test_scoring_never_guesses_missing_or_wrong_timestamp_identity():
    rows,truth = fixture()
    with pytest.raises(ValueError,match="join"): score(rows,truth[:-1],1)
    wrong = deepcopy(truth); wrong[5]["source_time_ns"] += 1
    with pytest.raises(ValueError,match="join"): score(rows,wrong,1)
    with pytest.raises(ValueError,match="absent"): score(rows,truth,2)
