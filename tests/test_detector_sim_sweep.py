from common.perception import (
    NormalizedBoxV2,
    PerceptionDetectionV2,
    PerceptionFrameV2,
    PerceptionSnapshotV2,
)
from tools.analyze_detector_sim_sweep import analyze


def _payload(frame_id: int, class_id: str | None) -> str:
    detections = ()
    if class_id is not None:
        detections = (PerceptionDetectionV2(
            detection_id=0,
            box=NormalizedBoxV2(x=0.2, y=0.2, w=0.2, h=0.4),
            class_id=class_id,
            confidence=0.9,
        ),)
    return PerceptionSnapshotV2(
        sequence=frame_id,
        frame=PerceptionFrameV2(
            frame_id=frame_id,
            source_time_ns=frame_id,
            observed_time_ns=frame_id,
            source_clock_domain="test",
            observation_clock_domain="test",
            width=100,
            height=100,
        ),
        detections=detections,
    ).model_dump_json()


def test_analyze_scores_classes_confusion_and_blank_false_positives():
    manifest = {
        "cases": [
            {"case_id": "drone-a", "expected_class": "drone", "start_frame": 2,
             "end_frame": 2, "expected_box": [0.2, 0.2, 0.2, 0.4]},
            {"case_id": "person-a", "expected_class": "person", "start_frame": 4,
             "end_frame": 5, "expected_box": [0.2, 0.2, 0.2, 0.4]},
        ],
        "blanks": [
            {"blank_id": "blank-a", "start_frame": 1, "end_frame": 1},
            {"blank_id": "blank-b", "start_frame": 3, "end_frame": 3},
        ],
    }
    report = analyze(manifest, [
        _payload(1, None),
        _payload(2, "0"),
        _payload(3, "1"),
        _payload(4, "1"),
        _payload(5, "0"),
    ])

    assert report["classes"]["drone"]["recall"] == 1.0
    assert report["classes"]["person"]["recall"] == 0.5
    assert report["classes"]["person"]["confusion_rate"] == 0.5
    assert report["blank"] == {
        "observed": 2,
        "false_positive_frames": 1,
        "false_positive_rate": 0.5,
    }


def test_analyze_accepts_detector_only_shadow_records():
    manifest = {
        "cases": [
            {"case_id": "person-a", "expected_class": "person", "start_frame": 1,
             "end_frame": 1, "expected_box": [0.2, 0.2, 0.2, 0.4]},
        ],
        "blanks": [],
    }
    payload = '{"frame_id":1,"boxes":[{"x":0.2,"y":0.2,"w":0.2,"h":0.4,"cls":"1"}]}'

    report = analyze(manifest, [payload])

    assert report["classes"]["person"]["recall"] == 1.0
    assert report["invalid_messages"] == 0
