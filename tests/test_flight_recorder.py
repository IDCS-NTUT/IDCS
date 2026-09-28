from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path

import pytest
import zmq

from tools.flight_log import iter_records, sessions, summarize
from tools.flight_recorder import RecorderConfig, SegmentWriter, StreamSpec, decode, prune, run


def test_decode_plain_json_and_serial_topic_messages() -> None:
    assert decode(b'{"type":"CamState"}') == (None, {"type": "CamState"})
    assert decode(b'serial.reply.gimbal {"addr":1}') == ("serial.reply.gimbal", {"addr": 1})
    assert decode(b"not json") == (None, "not json")


def test_segments_rotate_and_a_flushed_open_segment_is_readable(tmp_path: Path) -> None:
    session = tmp_path / "s1"
    session.mkdir()
    (session / "session.json").write_text("{}")
    writer = SegmentWriter(session, segment_s=10)
    assert writer.write({"stream": "a", "wall_ns": 1, "n": 1}, 0.0) is None
    writer.write({"stream": "a", "wall_ns": 2, "n": 2}, 3.0)  # forces a sync flush
    # Not closed yet: the reader still sees what was flushed.
    assert [r["n"] for r in iter_records(session)][:1] == [1]
    closed = writer.write({"stream": "b", "wall_ns": 3, "n": 3}, 11.0)
    assert closed is not None and closed.name == "segment-000001.jsonl.gz"
    writer.close()
    assert [r["n"] for r in iter_records(session)] == [1, 2, 3]
    assert [r["n"] for r in iter_records(session, streams={"b"})] == [3]
    assert summarize(session)["messages"] == {"a": 2, "b": 1}


def test_prune_removes_oldest_segments_first_and_empty_sessions(tmp_path: Path) -> None:
    for name in ("20260101T000000-h", "20260102T000000-h"):
        d = tmp_path / name
        d.mkdir()
        (d / "session.json").write_text("{}")
        for i in (1, 2):
            with gzip.open(d / f"segment-{i:06d}.jsonl.gz", "wb") as f:
                f.write(bytes(range(256)) * 40)
    each = (tmp_path / "20260101T000000-h" / "segment-000001.jsonl.gz").stat().st_size
    removed = prune(tmp_path, max_total_bytes=int(each * 2.5))
    assert [p.parent.name for p in removed] == ["20260101T000000-h"] * 2
    assert not (tmp_path / "20260101T000000-h").exists()
    assert [s.name for s in sessions(tmp_path)] == ["20260102T000000-h"]


def test_recorder_captures_a_live_stream(tmp_path: Path) -> None:
    context = zmq.Context.instance()
    pub = context.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    cfg = RecorderConfig(tmp_path, 60.0, 10**9,
                         (StreamSpec("gimbal", f"tcp://127.0.0.1:{port}"),))
    stop = threading.Event()
    thread = threading.Thread(target=run, args=(cfg,), kwargs={"stop": stop})
    thread.start()
    time.sleep(0.5)  # slow joiner
    for i in range(20):
        pub.send_string(json.dumps({"type": "CamState", "i": i}))
    time.sleep(0.5)
    stop.set()
    thread.join(timeout=5)
    pub.close(0)
    (session,) = sessions(tmp_path)
    records = list(iter_records(session))
    assert [r["msg"]["i"] for r in records] == list(range(20))
    assert all(r["stream"] == "gimbal" and r["rx_ns"] > 0 for r in records)


def test_config_rejects_duplicate_stream_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        RecorderConfig.from_mapping({"recorder": {"root": "/tmp/x", "streams": [
            {"name": "a", "endpoint": "tcp://127.0.0.1:1"}, {"name": "a", "endpoint": "tcp://127.0.0.1:2"}]}})
