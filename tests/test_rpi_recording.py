"""Pi return-video recording: retention, keyframe gate, pipeline and config (no GStreamer)."""
import os

import pytest

from rpi.display.recording import KeyframeGate, prune


def test_prune_removes_oldest_recordings_only(tmp_path):
    other = tmp_path / "notes.txt"; other.write_bytes(b"x" * 500)
    files = []
    for i in range(4):
        f = tmp_path / f"2026102{i}T000000-0000{i}-return.mkv"; f.write_bytes(b"x" * 100)
        os.utime(f, (i, i)); files.append(f)
    deleted = prune(tmp_path, budget_bytes=300, reserve_bytes=100)
    assert deleted == files[:2] and all(f.exists() for f in files[2:]) and other.exists()


def test_each_recording_starts_at_a_keyframe():
    g = KeyframeGate()
    assert not g.accept(is_delta=True) and not g.accept(is_delta=True)
    assert g.accept(is_delta=False) and g.accept(is_delta=True) and g.passed == 2


def test_pipeline_has_recording_branch_only_when_configured():
    pytest.importorskip("gi")
    try:
        from rpi.display.video import receiver_description
    except (ImportError, ValueError) as exc:
        pytest.skip(f"GStreamer bindings unavailable: {exc}")
    plain = receiver_description(port=5002, jitter_ms=20)
    assert "tee" not in plain and "splitmuxsink" not in plain
    rec = receiver_description(port=5002, jitter_ms=20, record_segment_s=60)
    assert "h264parse ! tee name=return_tee allow-not-linked=true ! v4l2h264dec" in rec


def test_display_config_recording_section():
    try:
        from rpi.operator_display import DisplayConfig
    except (ImportError, ValueError) as exc:
        pytest.skip(f"display import unavailable: {exc}")
    base = {"net": {"rtp_return_port": 5002}, "rpi": {"operator_display": {"panel_state_endpoint": "tcp://x:1"}}}
    assert DisplayConfig(base, None).recording is None
    base["rpi"]["operator_display"]["recording"] = {"dir": "/tmp/r", "segment_s": 60, "max_total_gb": 15}
    assert DisplayConfig(base, None).recording == {"dir": "/tmp/r", "segment_s": 60, "max_total_gb": 15.0}
    base["rpi"]["operator_display"]["recording"]["segment_s"] = 5
    from common.config import ConfigError
    with pytest.raises(ConfigError):
        DisplayConfig(base, None)
