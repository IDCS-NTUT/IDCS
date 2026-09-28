from __future__ import annotations

from tools.detection_truth_compare import score_frame, summarize


def _snap(frame_id, detections=(), tracks=()):
    return {"frame": {"frame_id": frame_id, "width": 1280, "height": 720},
            "detections": [{"box": b, "confidence": c} for b, c in detections],
            "tracks": [{"box": b, "track_id": i, "missed_frames": m} for b, i, m in tracks]}


BOX = {"x": 0.5, "y": 0.5, "w": 0.05, "h": 0.05}
NEAR = {"x": 0.502, "y": 0.5, "w": 0.05, "h": 0.05}
FAR = {"x": 0.1, "y": 0.1, "w": 0.05, "h": 0.05}


def test_bridged_miss_and_stray_track_are_classified() -> None:
    truth = _snap(1, tracks=[(BOX, 0, 0)])
    detected = score_frame(truth, _snap(1, detections=[(NEAR, 0.4)], tracks=[(NEAR, 3, 0)]))
    bridged = score_frame(truth, _snap(1, tracks=[(NEAR, 3, 2)]))
    lost = score_frame(truth, _snap(1, tracks=[(FAR, 3, 4)]))
    assert detected["yolo_detected"] and detected["tracker_covered"] and not detected["tracker_only"]
    assert not bridged["yolo_detected"] and bridged["tracker_only"]
    assert bridged["tracker_error_px"] == 0.002 * 1280 or abs(bridged["tracker_error_px"] - 2.56) < 1e-9
    assert not lost["tracker_covered"] and lost["stray_tracks"] == 1
    summary = summarize([detected, bridged, lost], px_to_mrad=1.84)
    assert summary["yolo_misses"] == 2 and summary["misses_bridged_by_tracker"] == 1
    assert summary["longest_yolo_miss_frames"] == 2  # an open run at the end counts


def test_box_size_ratio_offset_and_range_ratio() -> None:
    truth = _snap(1, tracks=[(BOX, 0, 0)])
    truth["assessments"] = [{"track_id": 0, "distance_m": 3.0}]
    wide = {"x": 0.49, "y": 0.51, "w": 0.075, "h": 0.04}
    observed = _snap(1, detections=[(wide, 0.5)], tracks=[(wide, 7, 0)])
    observed["assessments"] = [{"track_id": 7, "distance_m": 2.0}]
    frame = score_frame(truth, observed)
    assert abs(frame["yolo_width_ratio"] - 1.5) < 1e-9 and abs(frame["yolo_height_ratio"] - 0.8) < 1e-9
    assert abs(frame["yolo_offset_px"][1] - 0.005 * 720) < 1e-9  # box centre below the truth
    assert abs(frame["range_ratio"] - 2.0 / 3.0) < 1e-9
    summary = summarize([frame], px_to_mrad=1.0)
    assert abs(summary["range_ratio_median"] - 2.0 / 3.0) < 1e-9
