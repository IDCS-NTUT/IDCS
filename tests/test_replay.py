import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from common.clock import StepClock
from common.replay import DetectionReplay, FrameReplayCapture


def _write_detection(path: Path, frame_id: int = 1) -> None:
    payload = {
        "frame_id": frame_id,
        "src_ts_ms": 1000,
        "rx_ts_ms": 1010,
        "infer_ts_ms": 1020,
        "img_w": 640,
        "img_h": 480,
        "boxes": [
            {
                "x": 0.1,
                "y": 0.2,
                "w": 0.3,
                "h": 0.4,
                "cls": "0",
                "conf": 0.9,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_detection_replay_materialize_adjusts_timestamps(tmp_path):
    detections_path = tmp_path / "detections.json"
    _write_detection(detections_path, frame_id=42)

    replay = DetectionReplay(detections_path)
    clock = StepClock()
    header = {"frame_id": 42, "src_ts_ms": 5000}
    msg = replay.materialize(42, header=header, clock=clock, image_size=(1280, 720))

    assert msg is not None
    assert msg.frame_id == 42
    assert msg.src_ts_ms == 5000
    assert msg.rx_ts_ms == 5010
    assert msg.infer_ts_ms == 5020
    assert msg.img_w == 1280
    assert msg.img_h == 720


def test_detection_replay_missing_frame_returns_none(tmp_path):
    detections_path = tmp_path / "detections.jsonl"
    detections_path.write_text("", encoding="utf-8")
    with open(detections_path, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "frame_id": 1,
                    "src_ts_ms": 1000,
                    "rx_ts_ms": 1010,
                    "infer_ts_ms": 1020,
                    "img_w": 640,
                    "img_h": 480,
                    "boxes": [
                        {
                            "x": 0.1,
                            "y": 0.2,
                            "w": 0.3,
                            "h": 0.4,
                            "cls": "0",
                            "conf": 0.9,
                        }
                    ],
                }
            )
            + "\n"
        )

    replay = DetectionReplay(detections_path)
    clock = StepClock()
    assert replay.materialize(5, header={"frame_id": 5, "src_ts_ms": 1234}, clock=clock) is None


def test_frame_replay_capture_directory(tmp_path):
    for idx in range(2):
        arr = np.full((4, 6, 3), idx * 40, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{idx}.png"), arr)

    cap = FrameReplayCapture(tmp_path, loop=False)
    ok1, frame1 = cap.read()
    ok2, frame2 = cap.read()
    ok3, frame3 = cap.read()

    assert ok1 and ok2
    assert frame1 is not None and frame2 is not None
    assert not ok3 and frame3 is None
    assert frame1.shape == frame2.shape == (4, 6, 3)
    assert np.all(frame1 == 0)
    assert np.all(frame2 == 40)
