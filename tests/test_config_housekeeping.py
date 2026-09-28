from pathlib import Path

from common.config import load_config_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_base_configuration_exposes_only_v2_perception_surface():
    config = load_config_bundle(
        [
            ROOT / "configs/base/network.yaml",
            ROOT / "configs/base/camera.yaml",
            ROOT / "configs/base/sim.yaml",
        ]
    ).data

    assert "yolo" not in config
    assert "argus" not in config["camera"]
    assert config["perception"]["class_labels"] == {"0": "drone", "1": "person"}
    assert "logging" not in config
    assert "perf" not in config


def test_base_network_has_no_removed_or_placeholder_keys():
    config = load_config_bundle([ROOT / "configs/base/network.yaml"]).data
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
    config = load_config_bundle([ROOT / "configs/base/gimbal.yaml"]).data
    startup = config["serial_io"]["startup"]

    assert all(str(command["func"]).upper() != "F3" for command in startup)
    assert all(str(command["func"]).lower() != "0x92" for command in startup)
    assert config["gimbal"]["startup_calibration_enabled"] is False
    assert config["gimbal"]["startup_encoder_zero_enabled"] is False


def test_serial_schedule_reserves_bus_for_control_commands():
    config = load_config_bundle([ROOT / "configs/base/gimbal.yaml"]).data
    schedule = config["serial_io"]["schedule"]
    encoders = [command for command in schedule if command["func"] == "0x31"]
    statuses = [command for command in schedule if command["func"] == "F1"]

    assert len(encoders) == 3
    assert all(command["interval_ms"] >= 100 for command in encoders)
    assert all(command["priority"] == "low" for command in encoders)
    assert all(command["interval_ms"] >= 5000 for command in statuses)
    assert all(command["priority"] == "low" for command in statuses)


def test_base_files_hold_disjoint_sections_so_their_order_does_not_matter():
    import yaml
    seen = {}
    for path in sorted((ROOT / "configs/base").glob("*.yaml")):
        for key in yaml.safe_load(path.read_text(encoding="utf-8")):
            assert key not in seen, f"{key} in both {seen.get(key)} and {path.name}"
            seen[key] = path.name


def test_every_config_a_unit_loads_exists():
    import re
    for unit in (ROOT / "deploy/systemd").glob("*/*.service"):
        for match in re.finditer(r"--config(?:-extra)? (\S+)", unit.read_text(encoding="utf-8")):
            for path in match.group(1).split(","):
                assert (ROOT / path).exists(), f"{unit.name}: {path}"


def test_targets_and_conflicts_name_units_that_exist_on_that_host():
    import re
    for host in (ROOT / "deploy/systemd").iterdir():
        names = {p.name for p in host.iterdir()}
        for unit in host.iterdir():
            for match in re.finditer(r"^(?:Wants|Requires|Conflicts|PartOf)=(.*)$", unit.read_text(), re.M):
                for name in match.group(1).split():
                    if name.startswith("idcs-"):
                        assert name in names, f"{host.name}/{unit.name}: {name}"
