from pathlib import Path

from tools.analyze_control_trace import Event, _build_summary


def test_v2_trace_summary_matches_nested_frame_and_clock_domains() -> None:
    perception = Event(
        stream="perception_v2",
        rx_monotonic_ns=1_000_000_000,
        payload={
            "type": "PerceptionSnapshot",
            "version": 2,
            "frame": {
                "frame_id": 7,
                "received_time_ns": 100_000_000,
                "observed_time_ns": 115_000_000,
                "receive_clock_domain": "jetson_monotonic",
                "observation_clock_domain": "jetson_monotonic",
            },
        },
    )
    control = Event(
        stream="control",
        rx_monotonic_ns=1_005_000_000,
        payload={"frame_id": 7, "cmd_ts_ms": 120, "target_ok": True},
    )

    summary = _build_summary(
        Path("synthetic.jsonl"),
        [perception, control],
        decode_errors=0,
        rate_limits=None,
        settle_threshold_rad=0.02,
        settle_hold_s=0.2,
        segment_gap_s=0.25,
        match_window_ms=500.0,
    )

    assert summary["counts"]["perception_v2"] == 1
    assert summary["timing"]["jetson_observed_minus_received"]["mean"] == 15.0
    assert summary["timing"]["control_cmd_minus_observed"]["mean"] == 5.0
    assert summary["timing"]["recorder_control_after_perception"]["mean"] == 5.0


def test_v2_trace_does_not_compare_different_clock_domains() -> None:
    perception = Event(
        stream="perception_v2",
        rx_monotonic_ns=1_000_000_000,
        payload={
            "frame": {
                "frame_id": 8,
                "received_time_ns": 100_000_000,
                "observed_time_ns": 115_000_000,
                "receive_clock_domain": "pc_monotonic",
                "observation_clock_domain": "jetson_monotonic",
            },
        },
    )

    summary = _build_summary(
        Path("synthetic.jsonl"),
        [perception],
        decode_errors=0,
        rate_limits=None,
        settle_threshold_rad=0.02,
        settle_hold_s=0.2,
        segment_gap_s=0.25,
        match_window_ms=500.0,
    )

    assert summary["timing"]["jetson_observed_minus_received"]["count"] == 0
