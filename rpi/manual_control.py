"""Joystick-driven manual gimbal control using the shared RS485 driver.

This script runs on a Raspberry Pi (or any Linux SBC with I2C + RS485 USB)
and translates joystick ADC readings into pan/tilt rate commands using the
same MKS SERVO42 RS485 abstractions as the Jetson bridge. The goal is to keep
manual control consistent with the autonomous controller’s limits and
acceleration settings.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

from rpi.joystick_common import (
    GimbalConfig,
    JoystickConfig,
    JoystickReader,
    build_gimbal,
    load_config,
    log_sample,
    map_adc_to_rate,
)


_LOG = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="rpi/manual_config.yaml",
        type=Path,
        help="Path to joystick + gimbal YAML config",
    )
    ap.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    gimbal_cfg, joystick_cfg = load_config(args.config)
    _LOG.info("loaded config from %s", args.config)

    joystick = JoystickReader(joystick_cfg)
    bus, gimbal = build_gimbal(gimbal_cfg)

    poll_interval = max(joystick_cfg.poll_interval_s, 0.01)
    _LOG.info(
        "starting manual control | poll=%.2f s deadzone=%d max_rate yaw=%.2f pitch=%.2f",
        poll_interval,
        joystick_cfg.deadzone,
        joystick_cfg.max_rate_yaw,
        joystick_cfg.max_rate_pitch,
    )

    with bus:
        _LOG.info("RS485 bus opened on %s @ %d", bus.port, bus.baudrate)
        gimbal.yaw_axis.enable(True)
        if hasattr(gimbal.pitch_axis, "enable"):
            gimbal.pitch_axis.enable(True)  # type: ignore[union-attr]
        _LOG.info("zeroing axes at startup")
        gimbal.zero_axes()

        try:
            while True:
                raw_yaw = joystick.read_channel(joystick_cfg.yaw_channel)
                raw_pitch = joystick.read_channel(joystick_cfg.pitch_channel)

                yaw_rate = map_adc_to_rate(raw_yaw, joystick_cfg.max_rate_yaw, joystick_cfg.deadzone)
                pitch_rate = map_adc_to_rate(
                    raw_pitch, joystick_cfg.max_rate_pitch, joystick_cfg.deadzone
                )

                gimbal.apply_rate_commands(
                    yaw_rate,
                    pitch_rate,
                    yaw_accel_byte=gimbal_cfg.yaw_accel_byte,
                    pitch_accel_byte=gimbal_cfg.pitch_accel_byte,
                )
                log_sample(raw_yaw, raw_pitch, yaw_rate, pitch_rate)
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            _LOG.info("manual control interrupted; stopping gimbal")
        finally:
            try:
                gimbal.stop()
            except Exception:  # noqa: BLE001
                _LOG.warning("failed to send stop command", exc_info=True)
            try:
                if hasattr(gimbal.pitch_axis, "enable"):
                    gimbal.pitch_axis.enable(False)  # type: ignore[union-attr]
                gimbal.yaw_axis.enable(False)
            except Exception:  # noqa: BLE001
                _LOG.debug("failed to disable axes", exc_info=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
