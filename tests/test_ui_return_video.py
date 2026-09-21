from types import SimpleNamespace
from pathlib import Path

import pc.ui as ui


class _Pipeline:
    def __init__(self):
        self.calls = []

    def set_state(self, state):
        self.calls.append(("set_state", state))

    def get_state(self, timeout):
        self.calls.append(("get_state", timeout))


def test_return_video_release_waits_for_null_transition(monkeypatch):
    fake_gst = SimpleNamespace(State=SimpleNamespace(NULL="null"), SECOND=1_000)
    monkeypatch.setattr(ui, "Gst", fake_gst)
    pipeline = _Pipeline()
    video = ui.GstReturnVideo.__new__(ui.GstReturnVideo)
    video._pipeline = pipeline
    video._appsink = object()
    video._bus = object()
    video._eos = False

    video.release()

    assert pipeline.calls == [("set_state", "null"), ("get_state", 2_000)]
    assert video._pipeline is None
    assert video._appsink is None
    assert video._bus is None
    assert video._eos is True


def test_return_video_release_is_idempotent(monkeypatch):
    fake_gst = SimpleNamespace(State=SimpleNamespace(NULL="null"), SECOND=1_000)
    monkeypatch.setattr(ui, "Gst", fake_gst)
    video = ui.GstReturnVideo.__new__(ui.GstReturnVideo)
    video._pipeline = None
    video._appsink = None
    video._bus = None
    video._eos = False

    video.release()
    video.release()

    assert video._eos is True


def test_report_write_failure_is_nonfatal(monkeypatch, capsys, tmp_path):
    report_path = tmp_path / "report.json"

    def fail_write(_self, *_args, **_kwargs):
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(Path, "write_text", fail_write)

    assert ui.write_ui_report(report_path, {"metadata_messages": 12}) is False
    assert "Disk quota exceeded" in capsys.readouterr().out
