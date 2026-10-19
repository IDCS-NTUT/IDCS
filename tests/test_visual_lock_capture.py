import json

import pytest

from jetson.visual_lock.capture import InputCapture
from tools.replay_visual_lock_capture import annexb_nals,load_manifest,operator_seed_box


def test_capture_whole_units_and_exact_index(tmp_path):
    cap = InputCapture(tmp_path,max_bytes=12)
    packet = b"\x00\x00\x00\x01\x65a"
    cap.append(packet,10)
    cap.append(packet,20)
    cap.append(packet,30)
    cap.close()
    assert (tmp_path/"input.h264").read_bytes() == packet*2
    rows = [json.loads(x) for x in (tmp_path/"input.jsonl").read_text().splitlines()]
    assert [(r["encoded_index"],r["decoded_pts_ns"],r["offset"]) for r in rows] == [(0,10,0),(1,20,6)]
    assert cap.report()["stop_reason"] == "byte limit"


def test_replay_uses_original_operator_box_not_the_refined_or_missing_decision_box():
    original, refined = [40,25,200,150], [80,60,96,72]
    for decision in (refined, None):
        assert operator_seed_box(dict(box=decision,seed_report=dict(operator_box=original))) == original
    assert operator_seed_box(dict(box=refined)) == refined  # old captures
    assert operator_seed_box(dict(box=refined,seed_report=None)) == refined


def test_duration_and_invalid_input_fail_closed(tmp_path):
    cap = InputCapture(tmp_path,max_duration_ns=10)
    with pytest.raises(ValueError): cap.append(b"not annex b",1)
    packet = b"\x00\x00\x01\x65a"
    cap.append(packet,10)
    with pytest.raises(ValueError): cap.append(packet,10)
    cap.append(packet,21)
    cap.close()
    assert cap.count == 1
    assert cap.stop_reason == "duration limit"


def test_matched_manifest_rejects_digest_and_missing_seed(tmp_path):
    cap = InputCapture(tmp_path)
    cap.append(b"\x00\x00\x01\x65a",10)
    cap.append(b"\x00\x00\x01\x65b",20)
    cap.close()
    row = dict(decoded_pts_ns=20,seed=True,box=[10,10,12,12])
    (tmp_path/"frames.jsonl").write_text(json.dumps(row)+"\n")
    index,records,seed = load_manifest(tmp_path)
    assert len(index) == 2 and len(records) == 1 and seed == row
    (tmp_path/"input.h264").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="digest"): load_manifest(tmp_path)


def test_matched_manifest_rejects_duplicate_frame_pts(tmp_path):
    cap = InputCapture(tmp_path)
    cap.append(b"\x00\x00\x01\x65a",10)
    cap.close()
    row = dict(decoded_pts_ns=10,seed=True)
    (tmp_path/"frames.jsonl").write_text((json.dumps(row)+"\n")*2)
    with pytest.raises(ValueError,match="duplicate"): load_manifest(tmp_path)


def test_keyframe_and_parameter_sets_preserved_for_independent_replay(tmp_path):
    cap = InputCapture(tmp_path)
    cap.append(b"\x00\x00\x00\x01\x67sps\x00\x00\x01\x68pps\x00\x00\x01\x61p",10)
    cap.append(b"\x00\x00\x01\x65idr",20)
    cap.close()
    (tmp_path/"frames.jsonl").write_text(json.dumps(dict(decoded_pts_ns=20,seed=True))+"\n")
    index,_,_ = load_manifest(tmp_path)
    assert not index[0]["keyframe"] and index[1]["keyframe"]
    assert [n[0] & 31 for n in annexb_nals(index[1]["codec_prefix"])] == [7,8]
