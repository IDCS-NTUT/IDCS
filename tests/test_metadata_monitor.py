from common.perception import (
    NormalizedBoxV2,
    PerceptionDetectionV2,
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
    TargetSelectionV2,
)
from pc.metadata_monitor import MetadataMetrics


def _snapshot(frame: int, timestamp: int) -> PerceptionSnapshotV2:
    box = NormalizedBoxV2(x=0, y=0, w=.1, h=.2)
    return PerceptionSnapshotV2(
        sequence=frame,
        frame=PerceptionFrameV2(
            frame_id=frame,
            source_time_ns=timestamp,
            observed_time_ns=timestamp + 2,
            source_clock_domain="test.monotonic",
            observation_clock_domain="test.monotonic",
            width=1280,
            height=720,
        ),
        detections=(PerceptionDetectionV2(
            detection_id=0, box=box, class_id="person", confidence=.9,
        ),),
        tracks=(PerceptionTrackV2(
            track_id=4, box=box, class_id="person", confidence=.9, missed_frames=0,
        ),),
        selection=TargetSelectionV2(
            track_id=4,
            source_frame_id=frame,
            applied_frame_id=frame,
            selected_time_ns=timestamp + 3,
            selection_clock_domain="test.monotonic",
            policy="test",
        ),
    )


def test_metadata_monitor_reports_receiver_order_and_gaps():
    metrics = MetadataMetrics()
    metrics.ingest(_snapshot(10, 100))
    metrics.ingest(_snapshot(12, 120))
    metrics.ingest(_snapshot(11, 110))
    report = metrics.report()
    assert report["schema"] == "PerceptionSnapshotV2"
    assert report["frame_gaps"] == 1
    assert report["nonmonotonic_frame_ids"] == 1
    assert report["nonmonotonic_source_timestamps"] == 1
    assert report["detections"] == 3
    assert report["tracks"] == 3
    assert report["classes"] == {"person": 3}
    assert report["track_classes"] == {"person": 3}
    assert report["selected"] == 3
    assert report["tracker_observations"] == {4: 3}
    assert report["selected_tracker_ids"] == [4]
