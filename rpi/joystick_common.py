"""Shared utilities for Raspberry Pi joystick-driven gimbal control."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import smbus
import yaml

from jetson.gimbal.mks_servo42_rs485 import (
    GimbalInterface,
    MksServo42Axis,
    PitchAxisGroup,
    RS485Bus,
)


_LOG = logging.getLogger(__name__)


@dataclass
class JoystickConfig:
    bus_id: int = 1
    adc_addr: int = 0x48
    yaw_channel: int = 0
    pitch_channel: int = 1
    deadzone: int = 8
    max_rate_yaw: float = 10.0
    max_rate_pitch: float = 10.0
    poll_interval_s: float = 0.05


@dataclass
class GimbalConfig:
    serial_port: str
    baudrate: int = 38400
    timeout: float = 0.1
    retries: int = 1
    use_group_writes: bool = True
    yaw_addr: int = 1
    yaw_group_addr: Optional[int] = None
    yaw_accel_byte: int = 10
    yaw_rate_limit_rad_s: float = 10.0
    pitch_group_addr: int = 0x50
    pitch_motor_a_addr: int = 2
    pitch_motor_b_addr: int = 3
    pitch_encoder_authority: str = "a"
    pitch_accel_byte: int = 10
    pitch_rate_limit_rad_s: float = 10.0
    counts_per_rev: int = 0x4000
    yaw_gear_ratio: float = 1.0
    pitch_gear_ratio: float = 1.0


def load_config(path: Path) -> Tuple[GimbalConfig, JoystickConfig]:
    """Load gimbal + joystick config from YAML."""

    with path.open("r") as fp:
        data = yaml.safe_load(fp) or {}

    gimbal_cfg = data.get("gimbal") or {}
    joystick_cfg = data.get("joystick") or {}

    try:
        serial_port = str(gimbal_cfg["serial_port"])
    except KeyError as exc:  # noqa: BLE001
        raise SystemExit("gimbal.serial_port is required in the config") from exc

    authority = str(
        gimbal_cfg.get("pitch_encoder_authority", GimbalConfig.pitch_encoder_authority)
    ).lower()
    if authority not in {"a", "b"}:
        raise SystemExit("gimbal.pitch_encoder_authority must be 'a' or 'b'")

    gimbal = GimbalConfig(
        serial_port=serial_port,
        baudrate=int(gimbal_cfg.get("baudrate", GimbalConfig.baudrate)),
        timeout=float(gimbal_cfg.get("timeout", GimbalConfig.timeout)),
        retries=int(gimbal_cfg.get("retries", GimbalConfig.retries)),
        use_group_writes=bool(gimbal_cfg.get("use_group_writes", GimbalConfig.use_group_writes)),
        yaw_addr=int(gimbal_cfg.get("yaw_addr", GimbalConfig.yaw_addr)),
        yaw_group_addr=(
            None
            if gimbal_cfg.get("yaw_group_addr") is None
            else int(gimbal_cfg.get("yaw_group_addr"))
        ),
        yaw_accel_byte=int(gimbal_cfg.get("yaw_accel_byte", GimbalConfig.yaw_accel_byte)),
        yaw_rate_limit_rad_s=float(
            gimbal_cfg.get("yaw_rate_limit_rad_s", GimbalConfig.yaw_rate_limit_rad_s)
        ),
        pitch_group_addr=int(gimbal_cfg.get("pitch_group_addr", GimbalConfig.pitch_group_addr)),
        pitch_motor_a_addr=int(
            gimbal_cfg.get("pitch_motor_a_addr", GimbalConfig.pitch_motor_a_addr)
        ),
        pitch_motor_b_addr=int(
            gimbal_cfg.get("pitch_motor_b_addr", GimbalConfig.pitch_motor_b_addr)
        ),
        pitch_encoder_authority=authority,
        pitch_accel_byte=int(gimbal_cfg.get("pitch_accel_byte", GimbalConfig.pitch_accel_byte)),
        pitch_rate_limit_rad_s=float(
            gimbal_cfg.get("pitch_rate_limit_rad_s", GimbalConfig.pitch_rate_limit_rad_s)
        ),
        counts_per_rev=int(gimbal_cfg.get("counts_per_rev", GimbalConfig.counts_per_rev)),
        yaw_gear_ratio=float(gimbal_cfg.get("yaw_gear_ratio", GimbalConfig.yaw_gear_ratio)),
        pitch_gear_ratio=float(gimbal_cfg.get("pitch_gear_ratio", GimbalConfig.pitch_gear_ratio)),
    )

    adc_cfg = joystick_cfg.get("adc_addr", JoystickConfig.adc_addr)
    adc_addr = int(str(adc_cfg), 0) if isinstance(adc_cfg, str) else int(adc_cfg)

    joystick = JoystickConfig(
        bus_id=int(joystick_cfg.get("bus_id", JoystickConfig.bus_id)),
        adc_addr=adc_addr,
        yaw_channel=int(joystick_cfg.get("yaw_channel", JoystickConfig.yaw_channel)),
        pitch_channel=int(joystick_cfg.get("pitch_channel", JoystickConfig.pitch_channel)),
        deadzone=int(joystick_cfg.get("deadzone", JoystickConfig.deadzone)),
        max_rate_yaw=float(joystick_cfg.get("max_rate_yaw", JoystickConfig.max_rate_yaw)),
        max_rate_pitch=float(joystick_cfg.get("max_rate_pitch", JoystickConfig.max_rate_pitch)),
        poll_interval_s=float(joystick_cfg.get("poll_interval_s", JoystickConfig.poll_interval_s)),
    )

    return gimbal, joystick


class JoystickReader:
    """Thin helper for PCF8591 ADC joystick."""

    def __init__(self, cfg: JoystickConfig) -> None:
        self._bus = smbus.SMBus(cfg.bus_id)
        self._addr = cfg.adc_addr

    def read_channel(self, ch: int) -> int:
        ctrl = 0x40 | (ch & 0x03)
        self._bus.write_byte(self._addr, ctrl)
        # Throw away first conversion after channel change.
        self._bus.read_byte(self._addr)
        return self._bus.read_byte(self._addr)


def build_gimbal(cfg: GimbalConfig) -> tuple[RS485Bus, GimbalInterface]:
    bus = RS485Bus(
        cfg.serial_port,
        baudrate=cfg.baudrate,
        timeout=cfg.timeout,
        max_retries=max(cfg.retries, 0),
    )
    yaw_axis = MksServo42Axis(
        bus,
        cfg.yaw_addr,
        group_addr=cfg.yaw_group_addr,
        counts_per_rev=cfg.counts_per_rev,
        gear_ratio=cfg.yaw_gear_ratio,
        use_group_writes=cfg.use_group_writes,
    )
    pitch_a = MksServo42Axis(
        bus,
        cfg.pitch_motor_a_addr,
        group_addr=cfg.pitch_group_addr,
        counts_per_rev=cfg.counts_per_rev,
        gear_ratio=cfg.pitch_gear_ratio,
        use_group_writes=cfg.use_group_writes,
    )
    pitch_b = MksServo42Axis(
        bus,
        cfg.pitch_motor_b_addr,
        group_addr=cfg.pitch_group_addr,
        counts_per_rev=cfg.counts_per_rev,
        gear_ratio=cfg.pitch_gear_ratio,
        use_group_writes=cfg.use_group_writes,
    )
    pitch_axis = PitchAxisGroup(
        bus,
        cfg.pitch_group_addr,
        motor_a=pitch_a,
        motor_b=pitch_b,
        authority=cfg.pitch_encoder_authority,
    )
    gimbal = GimbalInterface(
        yaw_axis,
        pitch_axis,
        max_rate_rad_s=max(cfg.yaw_rate_limit_rad_s, cfg.pitch_rate_limit_rad_s),
        yaw_accel_byte=cfg.yaw_accel_byte,
        pitch_accel_byte=cfg.pitch_accel_byte,
    )
    return bus, gimbal


def map_adc_to_rate(value: int, max_rate: float, deadzone: int = 8, center: int = 128) -> float:
    diff = value - center
    if abs(diff) < deadzone:
        return 0.0
    scale = max(min(diff / 127.0, 1.0), -1.0)
    return scale * max_rate


def log_sample(raw_yaw: int, raw_pitch: int, yaw_rate: float, pitch_rate: float) -> None:
    _LOG.info(
        "joystick raw yaw=%3d pitch=%3d | rates pan=%.3f rad/s tilt=%.3f rad/s",
        raw_yaw,
        raw_pitch,
        yaw_rate,
        pitch_rate,
    )
