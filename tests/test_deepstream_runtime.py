import json
from pathlib import Path

import pytest
from jetson.deepstream.runtime import (
    build_pipeline_argv,
    load_settings,
    resolve_runtime_base_dir,
    run,
)


def test_runtime_base_dir_follows_active_config_tree(tmp_path):
    overlay = tmp_path / "overlay"
    primary = overlay / "configs" / "network.yaml"
    primary.parent.mkdir(parents=True)
    primary.write_text("video: {}\n", encoding="utf-8")

    assert resolve_runtime_base_dir(primary, cwd=tmp_path / "other") == overlay.resolve()


def test_runtime_base_dir_falls_back_for_nonstandard_layout(tmp_path):
    primary = tmp_path / "network.yaml"
    primary.write_text("video: {}\n", encoding="utf-8")
    cwd = tmp_path / "checkout"

    assert resolve_runtime_base_dir(primary, cwd=cwd) == cwd.resolve()


def test_runtime_resolves_rtp_contract(tmp_path):
    model = tmp_path / "model.txt"; model.write_text("x", encoding="utf-8")
    cfg = {"net": {"rtp_port": 5000, "header_push": "tcp://192.168.0.5:5555", "zmq_perception_v2": "tcp://192.168.0.5:5564", "return_ip": "192.168.0.1", "rtp_return_port": 5002}, "deepstream": {"input_mode": "rtp", "nvinfer_config": model.name, "target_selection": True}}
    settings = load_settings(cfg, base_dir=tmp_path)
    argv = build_pipeline_argv(settings, [Path("configs/network.yaml")], ready_file=tmp_path / "ready", health_file=tmp_path / "health")
    assert settings.header_bind == "tcp://0.0.0.0:5555"
    assert settings.snapshot_bind == "tcp://0.0.0.0:5564"
    assert "--snapshot-result-bind" in argv
    assert argv.count("--snapshot-result-bind") == 1
    assert "--target-selection" in argv and "--return-h264" in argv and "--ready-file" in argv and "--health-file" in argv


def test_runtime_resolves_independent_return_video_profile(tmp_path):
    model = tmp_path / "model.txt"; model.write_text("x", encoding="utf-8")
    cfg = {
        "net": {
            "rtp_port": 5000,
            "header_push": "tcp://jetson:5555",
            "zmq_perception_v2": "tcp://jetson:5564",
            "return_ip": "pc",
            "rtp_return_port": 5002,
        },
        "video": {
            "active_profile": "input60",
            "active_return_profile": "return30",
            "profiles": {
                "input60": {"width": 1920, "height": 1080, "fps": 60, "bitrate_kbps": 9000},
                "return30": {"width": 1280, "height": 720, "fps": 30, "bitrate_kbps": 7000},
            },
        },
        "deepstream": {"input_mode": "rtp", "nvinfer_config": model.name},
    }
    settings = load_settings(cfg, base_dir=tmp_path)
    argv = build_pipeline_argv(settings, [])

    assert (settings.return_width, settings.return_height, settings.return_fps) == (1280, 720, 30)
    assert settings.return_bitrate_kbps == 7000
    assert argv[argv.index("--return-fps") + 1] == "30"
    assert argv[argv.index("--return-bitrate-kbps") + 1] == "7000"


def test_runtime_rejects_non_tcp_metadata_endpoint(tmp_path):
    model = tmp_path / "model.txt"; model.write_text("x", encoding="utf-8")
    cfg = {"net": {"rtp_port": 5000, "header_push": "ipc:///tmp/headers", "zmq_perception_v2": "tcp://x:5564", "return_ip": "pc", "rtp_return_port": 5002}, "deepstream": {"input_mode": "rtp", "nvinfer_config": model.name}}
    with pytest.raises(ValueError, match="net.header_push"):
        load_settings(cfg, base_dir=tmp_path)


def test_runtime_builds_headerless_argus_v2_metadata_contract(tmp_path):
    model = tmp_path / "model.txt"; model.write_text("x", encoding="utf-8")
    cfg = {"net": {"rtp_port": 5000, "header_push": "tcp://jetson:5555", "zmq_perception_v2": "tcp://jetson:5564", "return_ip": "pc", "rtp_return_port": 5002}, "deepstream": {"input_mode": "argus", "nvinfer_config": model.name, "argus_sensor_id": 0, "argus_sensor_mode": 4, "argus_width": 1280, "argus_height": 720, "argus_fps": 60}}
    settings = load_settings(cfg, base_dir=tmp_path)
    argv = build_pipeline_argv(settings, [])
    assert settings.rtp_input_port is None
    assert "--live-argus" in argv
    assert "--snapshot-result-bind" in argv
    assert "--header-bind" not in argv


def test_runtime_check_reports_immutable_config_provenance(tmp_path, capsys, monkeypatch):
    nvinfer = _write(tmp_path / "nvinfer.txt", "model-engine-file=model.engine\n")
    network = _write(
        tmp_path / "network.yaml",
        "net:\n"
        "  rtp_port: 5000\n"
        "  header_push: tcp://jetson:5555\n"
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
