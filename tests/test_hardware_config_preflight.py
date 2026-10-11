"""Config preflight must reject omitted motor settings without opening devices."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from jetson import gimbal_bridge
from tools import serial_io_service


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("loader", "prefix"),
    [
        (gimbal_bridge._load_config, "invalid bridge configuration"),
        (serial_io_service._load_config, "invalid serial configuration"),
    ],
)
def test_hardware_loaders_reject_missing_overlay(loader, prefix, tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("net: {}\ngimbal: {}\nserial_io: {}\n", encoding="utf-8")
    missing = tmp_path / "missing-bench-overlay.yaml"

    with pytest.raises(SystemExit, match=prefix) as error:
        loader([base, missing])
    assert str(missing) in str(error.value)


@pytest.mark.parametrize("loader", [gimbal_bridge._load_config, serial_io_service._load_config])
def test_hardware_loaders_reject_duplicate_keys(loader, tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("net: {}\ngimbal: {}\nserial_io: {}\n", encoding="utf-8")
    overlay = tmp_path / "bench.yaml"
    overlay.write_text("gimbal:\n  yaw_rate_limit_rad_s: 0.8\n  yaw_rate_limit_rad_s: 10.0\n",
                       encoding="utf-8")

    with pytest.raises(SystemExit, match="duplicate configuration key"):
        loader([base, overlay])


@pytest.mark.parametrize("loader", [gimbal_bridge._load_config, serial_io_service._load_config])
def test_hardware_loaders_preserve_ordered_overlay(loader, tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("net: {}\ngimbal:\n  yaw_rate_limit_rad_s: 10.0\n"
                    "serial_io:\n  schedule: [{name: old}]\n", encoding="utf-8")
    overlay = tmp_path / "bench.yaml"
    overlay.write_text("gimbal:\n  yaw_rate_limit_rad_s: 0.8\n"
                       "serial_io:\n  schedule: [{name: new}]\n", encoding="utf-8")

    result = loader([base, overlay])
    assert result["gimbal"]["yaw_rate_limit_rad_s"] == 0.8
    assert result["serial_io"]["schedule"] == [{"name": "new"}]


def test_explicit_parameter_file_must_exist_and_have_unique_keys(tmp_path):
    missing = tmp_path / "missing-parameters.yaml"
    with pytest.raises(SystemExit, match="cannot read motor parameter file") as error:
        gimbal_bridge._load_parameter_map(missing)
    assert str(missing) in str(error.value)

    duplicate = tmp_path / "duplicate-parameters.yaml"
    duplicate.write_text("motors:\n  1: {}\n  1: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate configuration key"):
        gimbal_bridge._load_parameter_map(duplicate)

    assert set(gimbal_bridge._load_parameter_map(ROOT / "configs/bench/mks_parameters.yaml")) == {1, 2, 3}


@pytest.mark.parametrize(
    ("module", "command"),
    [(gimbal_bridge, "gimbal_bridge"), (serial_io_service, "serial_io_service")],
)
@pytest.mark.parametrize("bad_overlay", ["missing", "duplicate"])
def test_check_command_rejects_bad_overlay_before_runtime(
    module, command, bad_overlay, tmp_path, monkeypatch
):
    monkeypatch.chdir(ROOT)
    overlay = tmp_path / "bench.yaml"
    if bad_overlay == "duplicate":
        overlay.write_text("gimbal:\n  yaw_rate_limit_rad_s: 0.8\n"
                           "  yaw_rate_limit_rad_s: 10.0\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [command, "--config", "configs/base",
                                      "--config-extra", str(overlay), "--check"])
    monkeypatch.setattr(module.zmq, "Context", lambda: pytest.fail("opened ZMQ context"))

    reason = "does not exist" if bad_overlay == "missing" else "duplicate configuration key"
    with pytest.raises(SystemExit, match=reason):
        module.main()


def test_managed_bridge_preflight_uses_no_sockets_or_serial(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["gimbal_bridge", "--config", "configs/base",
        "--config-extra", "configs/bench/uncoupled.yaml,configs/bench/tuned.yaml",
        "--enable-live-intent-actuation", "--check"])
    monkeypatch.setattr(gimbal_bridge.zmq, "Context", lambda: pytest.fail("opened ZMQ context"))
    monkeypatch.setattr(gimbal_bridge, "install_signal_handlers",
                        lambda: pytest.fail("installed runtime signal handlers"))

    assert gimbal_bridge.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["check_only"] is True
    assert result["pitch_motor_b_enabled"] is False
    assert result["yaw_limits_rad"] == [-2.9, -1.9]


def test_managed_serial_preflight_uses_no_sockets_or_bus(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["serial_io_service", "--config", "configs/base",
        "--config-extra", "configs/bench/uncoupled.yaml", "--check"])
    monkeypatch.setattr(serial_io_service.zmq, "Context", lambda: pytest.fail("opened ZMQ context"))
    monkeypatch.setattr(serial_io_service, "RS485Bus", lambda **_kwargs: pytest.fail("opened serial bus"))

    assert serial_io_service.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["check_only"] is True
    assert result["startup_commands"] > 0
    assert result["schedule"]
