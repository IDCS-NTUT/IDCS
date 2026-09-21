from pathlib import Path

from common.config import load_config_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_base_configuration_exposes_only_v2_perception_surface():
    config = load_config_bundle(
        [
            ROOT / "configs/network.yaml",
            ROOT / "configs/perception.yaml",
            ROOT / "configs/system.yaml",
        ]
    ).data

    assert "yolo" not in config
    assert "argus" not in config["camera"]
    assert config["perception"]["class_labels"] == {"0": "drone", "1": "person"}
    assert "logging" not in config
    assert "perf" not in config


def test_base_network_has_no_removed_or_placeholder_keys():
    config = load_config_bundle([ROOT / "configs/network.yaml"]).data
    net = config["net"]

    assert {
        "rpi_ip",
        "zmq_manual_state_trace",
        "pc_ip",
        "pc_iface",
    }.isdisjoint(net)
    assert net["return_ip"] == "192.168.0.1"


def test_obsolete_config_profiles_are_removed():
    assert not (ROOT / "configs/shadow_yolo26s_best_current_736.yaml").exists()
    assert not (ROOT / "configs/dev_validation_file.yaml").exists()


def test_serial_service_startup_cannot_enable_or_zero_motors():
    config = load_config_bundle([ROOT / "configs/control.yaml"]).data
    startup = config["serial_io"]["startup"]

    assert all(str(command["func"]).upper() != "F3" for command in startup)
    assert all(str(command["func"]).lower() != "0x92" for command in startup)
    assert config["gimbal"]["startup_calibration_enabled"] is False
    assert config["gimbal"]["startup_encoder_zero_enabled"] is False
