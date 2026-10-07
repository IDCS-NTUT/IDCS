import pytest
from tools.verify_nvdcf_handoff import fixture_score


def test_fixture_truth_is_only_scored_and_absence_grace_is_reported():
    rows = [dict(frame_id=i,selected=i<7,box=(10,10,20,20)) for i in range(9)]
    truth = [(10,10,20,20)]*5+[None]*4
    measured = fixture_score(rows,truth)
    assert measured["selected_absent_frames"] == 2
    assert measured["selected_absent_after_three_frames"] == 0
    assert measured["truth_input_after_seed"] is False
    rows[7]["selected"] = True
    assert fixture_score(rows,truth)["selected_absent_after_three_frames"] == 1


def test_fixture_join_fails_closed():
    with pytest.raises(ValueError): fixture_score([],[(1,2,3,4)])
    with pytest.raises(ValueError): fixture_score([dict(frame_id=1)],[(1,2,3,4)])
