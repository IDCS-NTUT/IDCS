from __future__ import annotations

from pathlib import Path

import yaml

from common.gimbal.mks_servo42_rs485 import DEFAULT_BAUDRATE


ROOT = Path(__file__).resolve().parents[1]


def test_qualified_ch341_serial_settings_are_consistent() -> None:
    control = yaml.safe_load((ROOT / "configs/base/gimbal.yaml").read_text(encoding="utf-8"))
    gimbal = control["gimbal"]

    assert gimbal["serial_port"] == "/dev/ttyCH341USB0"
    assert gimbal["baudrate"] == 38_400
    assert gimbal["encoder_imu_horizon_enabled"] is False
    assert DEFAULT_BAUDRATE == 38_400


def test_motor_parameter_template_preserves_qualified_baud() -> None:
    parameter_map = yaml.safe_load(
        (ROOT / "configs/bench/mks_parameters.yaml").read_text(encoding="utf-8")
    )["motors"]

    for address in (1, 2, 3):
        parameters = parameter_map[address]["parameters"]
        assert parameters[10] == 0x04  # Byte14: 38,400-baud UART selector.
        assert parameters[11] == address  # Byte15: motor address.
