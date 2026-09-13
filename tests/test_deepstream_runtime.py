import json
from pathlib import Path

import pytest
from jetson.deepstream.runtime import build_pipeline_argv, load_settings, run


def test_runtime_resolves_rtp_contract(tmp_path):
    model = tmp_path / "model.txt"; model.write_text("x", encoding="utf-8")
    cfg = {"net": {"rtp_port": 5000, "header_push": "tcp://192.168.0.5:5555", "zmq_results": "tcp://192.168.0.5:5556", "zmq_perception_v2": "tcp://192.168.0.5:5564", "return_ip": "192.168.0.1", "rtp_return_port": 5002}, "deepstream": {"input_mode": "rtp", "nvinfer_config": model.name, "target_selection": True}}
    settings = load_settings(cfg, base_dir=tmp_path)
    argv = build_pipeline_argv(settings, [Path("configs/network.yaml")], ready_file=tmp_path / "ready", health_file=tmp_path / "health")
    assert settings.header_bind == "tcp://0.0.0.0:5555"
    assert settings.snapshot_bind == "tcp://0.0.0.0:5564"
    assert "--snapshot-result-bind" in argv
    assert "--shadow-target-selection" in argv and "--return-h264" in argv and "--ready-file" in argv and "--health-file" in argv


def test_runtime_rejects_non_tcp_metadata_endpoint(tmp_path):
    model = tmp_path / "model.txt"; model.write_text("x", encoding="utf-8")
    cfg = {"net": {"rtp_port": 5000, "header_push": "ipc:///tmp/headers", "zmq_results": "tcp://x:5556", "zmq_perception_v2": "tcp://x:5564", "return_ip": "pc", "rtp_return_port": 5002}, "deepstream": {"input_mode": "rtp", "nvinfer_config": model.name}}
    with pytest.raises(ValueError, match="net.header_push"):
        load_settings(cfg, base_dir=tmp_path)


def test_runtime_builds_headerless_argus_metadata_contract(tmp_path):
    model = tmp_path / "model.txt"; model.write_text("x", encoding="utf-8")
    cfg = {"net": {"rtp_port": 5000, "header_push": "tcp://jetson:5555", "zmq_results": "tcp://jetson:5556", "zmq_perception_v2": "tcp://jetson:5564", "return_ip": "pc", "rtp_return_port": 5002}, "deepstream": {"input_mode": "argus", "nvinfer_config": model.name, "argus_sensor_id": 0, "argus_sensor_mode": 4, "argus_width": 1280, "argus_height": 720, "argus_fps": 60}}
    settings = load_settings(cfg, base_dir=tmp_path)
    argv = build_pipeline_argv(settings, [])
    assert settings.rtp_input_port is None
    assert "--live-argus" in argv
    assert "--shadow-result-bind" in argv
    assert "--shadow-header-bind" not in argv


def test_runtime_check_reports_immutable_config_provenance(tmp_path, capsys, monkeypatch):
    nvinfer = _write(tmp_path / "nvinfer.txt", "model-engine-file=model.engine\n")
    network = _write(
        tmp_path / "network.yaml",
        "net:\n"
        "  rtp_port: 5000\n"
        "  header_push: tcp://jetson:5555\n"
        "  zmq_results: tcp://jetson:5556\n"
        "  zmq_perception_v2: tcp://jetson:5564\n"
        "  return_ip: pc\n"
        "  rtp_return_port: 5002\n",
    )
    runtime = _write(
        tmp_path / "runtime.yaml",
        "deepstream:\n"
        "  input_mode: rtp\n"
        f"  nvinfer_config: {nvinfer.name}\n",
    )
    monkeypatch.chdir(tmp_path)

    assert run([
        "--config", str(network),
        "--config-extra", str(runtime),
        "--check",
    ]) == 0

    result = json.loads(capsys.readouterr().out)
    assert len(result["config_digest"]) == 64
    assert [item["path"] for item in result["config_sources"]] == [
        str(network.resolve()), str(runtime.resolve())
    ]
    assert result["pipeline_argv"][0] == "--nvsort"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
