import json

import pytest

from pc.truth_capture import TruthCapture
from common.perception import PerceptionFrameV2,PerceptionSnapshotV2


def frame(i):
    return PerceptionFrameV2(tracker_session="fixture",frame_id=i,source_time_ns=1_000_000_000+i,
        observed_time_ns=1_000_000_000+i,source_clock_domain="pc_monotonic",
        observation_clock_domain="pc_monotonic",source_identity_verified=True,width=320,height=240)


def test_truth_capture_is_bounded_exclusive_and_has_no_pixels(tmp_path):
    path = tmp_path/"truth.jsonl"
    cap = TruthCapture(path,max_frames=2)
    for i in range(2):
        f = frame(i)
        assert cap.append(f.frame_id,f.source_time_ns,320,240,PerceptionSnapshotV2(sequence=i,frame=f))
    assert not cap.append(3,2_000_000_000,320,240,None)
    report = cap.close()
    assert report["limit_reached"] and not report["pixels_recorded"] and not report["tracking_input"]
    assert len(path.read_text().splitlines()) == 2
    assert json.loads(path.read_text().splitlines()[0])["truth"] == []
    with pytest.raises(FileExistsError): TruthCapture(path)


def test_truth_capture_rejects_identity_mismatch_or_reordering(tmp_path):
    cap = TruthCapture(tmp_path/"truth.jsonl")
    f = frame(1)
    with pytest.raises(ValueError,match="describe"):
        cap.append(0,f.source_time_ns,320,240,PerceptionSnapshotV2(sequence=1,frame=f))
    cap.append(1,f.source_time_ns,320,240,None)
    with pytest.raises(ValueError,match="order"): cap.append(1,f.source_time_ns,320,240,None)
    cap.close()
