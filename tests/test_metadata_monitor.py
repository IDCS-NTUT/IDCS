from common.schemas import Box, DetectionMsg
from pc.metadata_monitor import MetadataMetrics


def _msg(frame: int, timestamp: int) -> DetectionMsg:
    return DetectionMsg(frame_id=frame, src_ts_ms=timestamp, rx_ts_ms=timestamp + 1,
                        infer_ts_ms=timestamp + 2, img_w=1280, img_h=720,
                        boxes=[Box(x=0, y=0, w=.1, h=.2, cls="person", conf=.9, track_id=4)], target_idx=0)


def test_metadata_monitor_reports_receiver_order_and_gaps():
    metrics = MetadataMetrics(); metrics.ingest(_msg(10, 100)); metrics.ingest(_msg(12, 120)); metrics.ingest(_msg(11, 110))
    report = metrics.report()
    assert report["frame_gaps"] == 1
    assert report["nonmonotonic_frame_ids"] == 1
    assert report["nonmonotonic_source_timestamps"] == 1
    assert report["selected"] == 3
    assert report["tracker_observations"] == {4: 3}
    assert report["selected_tracker_ids"] == [4]
