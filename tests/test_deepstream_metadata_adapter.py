import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.perception import TrackAssessmentV2
from jetson.deepstream import metadata_adapter
from jetson.deepstream.metadata_adapter import (
    FrameTiming,
    UNTRACKED_OBJECT_ID,
    object_meta_to_observation_v2,
    perception_snapshot_from_metadata,
    pts_ns_to_ms,
)
from jetson.deepstream.pipeline import StageClock, _load_nvinfer_labels, _pipeline_description, _target_osd_suffix
from jetson.deepstream.header_correlation import HeaderCorrelator


def _object(*, left, top, width, height, class_id=4, confidence=0.8, object_id=UNTRACKED_OBJECT_ID):
    return SimpleNamespace(
        class_id=class_id,
        confidence=confidence,
        object_id=object_id,
        rect_params=SimpleNamespace(left=left, top=top, width=width, height=height),
    )


def test_metadata_enters_v2_as_separate_detections_and_tracks():
    timing = FrameTiming(
        frame_id=22,
        src_ts_ms=100,
        rx_ts_ms=108,
        infer_ts_ms=115,
        img_w=1280,
        img_h=720,
        source_clock_domain="pc_monotonic",
    )
    snapshot = perception_snapshot_from_metadata(timing, [
        _object(left=100, top=100, width=100, height=100),
        _object(left=300, top=100, width=100, height=100, object_id=77),
    ])

    assert snapshot.frame.source_clock_domain == "pc_monotonic"
    assert snapshot.frame.observation_clock_domain == "jetson_monotonic"
    assert snapshot.frame.source_time_ns == 100_000_000
    assert snapshot.frame.observed_time_ns == 115_000_000
    assert len(snapshot.detections) == 1
    assert snapshot.detections[0].detection_id == 0
    assert len(snapshot.tracks) == 1
    assert snapshot.tracks[0].track_id == 77
    assert snapshot.tracks[0].age_frames is None


def test_verified_rtp_timing_keeps_nanosecond_source_time():
    timing = FrameTiming(
        frame_id=22, src_ts_ms=100, src_ts_ns=100_123_456,
        rx_ts_ms=108, infer_ts_ms=115, img_w=1280, img_h=720,
        source_clock_domain="pc_monotonic", source_identity_verified=True,
    )
    snapshot = perception_snapshot_from_metadata(timing, [])
    assert snapshot.frame.source_time_ns == 100_123_456
    assert snapshot.frame.source_identity_verified is True


def test_v2_metadata_module_has_no_legacy_schema_dependency():
    source = Path(metadata_adapter.__file__).read_text(encoding="utf-8")
    assert "common.schemas" not in source
    assert "perception_compat" not in source

    timing = FrameTiming(
        frame_id=23,
        src_ts_ms=100,
        rx_ts_ms=108,
        infer_ts_ms=115,
        img_w=1280,
        img_h=720,
    )

    snapshot = perception_snapshot_from_metadata(
        timing,
        [_object(left=300, top=100, width=100, height=100, object_id=77)],
    )

    assert snapshot.tracks[0].track_id == 77


def test_v2_object_metadata_clips_without_legacy_schema():
    observation = object_meta_to_observation_v2(
        _object(left=-10, top=700, width=100, height=50, object_id=23),
        img_w=1280,
        img_h=720,
    )

    assert observation is not None
    assert observation.box.x == 0.0
    assert observation.box.h == 20 / 720
    assert observation.track_id == 23


def test_pts_conversion_is_relative_milliseconds():
    assert pts_ns_to_ms(1_234_567_890) == 1234
    assert pts_ns_to_ms(-1) == 0


def test_nvinfer_label_loader_reads_configured_labels(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("drone\nperson\n", encoding="utf-8")
    config = tmp_path / "nvinfer.txt"
    config.write_text(f"labelfile-path={labels}\n", encoding="utf-8")

    assert _load_nvinfer_labels(config) == {0: "drone", 1: "person"}


def test_target_osd_suffix_uses_only_controller_independent_metadata():
    assessment = TrackAssessmentV2(
        track_id=7,
        distance_m=3.8,
        threat_level="threatening",
        engagement_rank=1,
    )

    assert _target_osd_suffix(assessment) == " r=3.8m threatening rank=1"


def test_stage_clock_uses_ordered_single_source_buffers():
    clock = StageClock()
    clock.record_decode(10.000)
    clock.record_decode(10.010)
    clock.record_infer_input(10.004)
    clock.record_infer_input(10.015)

    first = clock.consume(10.012)
    second = clock.consume(10.023)

    assert first[0] == pytest.approx(4.0)
    assert first[1] == pytest.approx(8.0)
    assert first[2] == pytest.approx(10_004.0)
    assert second[0] == pytest.approx(5.0)
    assert second[1] == pytest.approx(8.0)
    assert second[2] == pytest.approx(10_015.0)
    assert clock.report()["matched_infer_inputs"] == 2


def test_first_metadata_frame_writes_readiness_file(tmp_path):
    from jetson.deepstream.pipeline import VerificationStats

    ready = tmp_path / "ready.json"
    stats = VerificationStats(ready_file=ready)
    stats.record_frame(SimpleNamespace(buf_pts=0), decode_to_infer_input_ms=None, infer_stage_ms=None)

    assert ready.is_file()


def test_metadata_frame_refreshes_health_file(tmp_path):
    from jetson.deepstream.pipeline import VerificationStats

    health = tmp_path / "health.json"
    stats = VerificationStats(health_file=health)
    stats.record_frame(SimpleNamespace(buf_pts=0), decode_to_infer_input_ms=None, infer_stage_ms=None)

    assert json.loads(health.read_text(encoding="utf-8"))["frames"] == 1


def test_stats_reports_current_pipeline_fps():
    from jetson.deepstream.pipeline import VerificationStats

    stats = VerificationStats(started_at_s=1.0, frames=30)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("jetson.deepstream.pipeline.time.monotonic", lambda: 2.0)
        assert stats.current_pipeline_fps() == pytest.approx(30.0)


def test_gpu_osd_h264_tail_stays_on_nvmm_and_defaults_to_local_sink(tmp_path):
    pipeline = _pipeline_description(
        input_file=tmp_path / "input.mp4",
        live_argus=False,
        rtp_input_port=None,
        argus_sensor_id=0,
        argus_sensor_mode=4,
        argus_width=1280,
        argus_height=720,
        argus_fps=60,
        nvinfer_config=tmp_path / "nvinfer.txt",
        paced=True,
        nvsort=True,
        gpu_osd=True,
        return_h264=True,
        return_udp_host=None,
        return_udp_port=None,
        return_h264_file=None,
    )

    assert "nvdsosd name=osd process-mode=1" in pipeline
    assert "nvvideoconvert name=osd_rgba_convert" in pipeline
    assert "video/x-raw(memory:NVMM),format=RGBA" in pipeline
    assert "video/x-raw(memory:NVMM),format=NV12" in pipeline
    assert "nvv4l2h264enc name=encoder" in pipeline
    assert "h264parse name=h264parse" in pipeline
    assert "fakesink name=sink sync=false" in pipeline
    assert "appsrc" not in pipeline


def test_gpu_osd_h264_udp_tail_uses_idcs_return_payload_type(tmp_path):
    pipeline = _pipeline_description(
        input_file=None,
        live_argus=True,
        rtp_input_port=None,
        argus_sensor_id=0,
        argus_sensor_mode=4,
        argus_width=1280,
        argus_height=720,
        argus_fps=60,
        nvinfer_config=tmp_path / "nvinfer.txt",
        paced=False,
        nvsort=False,
        gpu_osd=True,
        return_h264=True,
        return_udp_host="127.0.0.1",
        return_udp_port=5601,
        return_h264_file=None,
        return_width=1280,
        return_height=720,
        return_fps=30,
        return_bitrate_kbps=7000,
    )

    assert "rtph264pay name=rtp_pay pt=97 config-interval=1" in pipeline
    assert "udpsink name=return_udp host=127.0.0.1 port=5601" in pipeline
    assert "videorate name=return_rate drop-only=true max-rate=30" in pipeline
    assert "framerate=30/1" in pipeline
    assert "bitrate=7000000" in pipeline
    assert "iframeinterval=1 idrinterval=1" in pipeline
    assert "queue leaky=downstream max-size-buffers=1" in pipeline


def test_header_correlator_is_ordered_bounded_and_never_fabricates_identity():
    correlator = HeaderCorrelator(capacity=2)
    assert correlator.push_mapping({"frame_id": 10, "src_ts_ms": 100})
    assert correlator.push_mapping({"type": "CamState", "frame_id": 11, "src_ts_ms": 117})
    assert correlator.push_mapping({"frame_id": 12, "src_ts_ms": 133})
    assert not correlator.push_mapping({"frame_id": 12, "src_ts_ms": 134})

    first = correlator.match_next()
    second = correlator.match_next()
    empty = correlator.match_next()

    assert first.header is not None and first.header.frame_id == 11
    assert first.dropped_stale == 1
    assert second.header is not None and second.header.src_ts_ms == 133
    assert empty.header is None
    assert correlator.dropped_nonmonotonic == 1
