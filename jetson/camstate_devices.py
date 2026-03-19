"""Publish Jetson I2C IMU/magnetometer orientation as CamState telemetry.

This process is intended to be run on Jetson when MPU/HMC modules are wired
locally. It publishes CamState on ``net.zmq_gimbal_state`` so the existing
Jetson server path can consume it as the primary camera-state source.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml
import zmq

from common.config_sync import merge_config_maps, parse_config_text, read_snapshot
from common.schemas import CamState
from common.shutdown import install_signal_handlers

_LOG = logging.getLogger(__name__)

try:
    from smbus2 import SMBus  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    try:
        from smbus import SMBus  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Neither smbus2 nor smbus is available. Install one on Jetson before running camstate_devices."
        ) from exc


@dataclass
class SensorConfig:
    mpu_bus: int = 1
    mag_bus: int = 7
    mpu_addr: int = 0x68
    mag_addr: int = 0x1E
    pwr_mgmt_1_reg: int = 0x6B
    accel_xout_reg: int = 0x3B
    gyro_xout_reg: int = 0x43
    mag_data_reg: int = 0x03
    accel_scale: float = 16384.0
    gyro_scale: float = 131.0
    alpha: float = 0.98
    publish_hz: float = 50.0
    pan_sign: float = 1.0
    tilt_sign: float = 1.0
    pan_offset_rad: float = 0.0
    tilt_offset_rad: float = 0.0
    home_pan: float = 0.0
    home_tilt: float = 0.0


def _wrapped_delta(now_angle: float, prev_angle: float) -> float:
    return math.atan2(math.sin(now_angle - prev_angle), math.cos(now_angle - prev_angle))


def _parse_tcp_port(endpoint: str, name: str) -> int:
    try:
        port = int(endpoint.rsplit(":", 1)[1])
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"invalid {name} endpoint: {endpoint!r}") from exc
    if port <= 0 or port > 65535:
        raise SystemExit(f"{name} port must be in 1..65535 (got {port})")
    return port


def _read_word(bus: SMBus, addr: int, reg: int) -> int:
    hi = bus.read_byte_data(addr, reg)
    lo = bus.read_byte_data(addr, reg + 1)
    value = (hi << 8) | lo
    if value >= 0x8000:
        value -= 65536
    return value


def _load_config(paths: Iterable[Path]) -> Mapping[str, Any]:
    maps = []
    for path in paths:
        snapshot = read_snapshot(path)
        maps.append(parse_config_text(snapshot.text, str(path)))
    return merge_config_maps(*maps)


def _float_cfg(block: Mapping[str, Any], key: str, default: float) -> float:
    raw = block.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"camstate_devices.{key} must be numeric, got {raw!r}") from exc
    return value


def _int_cfg(block: Mapping[str, Any], key: str, default: int) -> int:
    raw = block.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"camstate_devices.{key} must be an integer, got {raw!r}") from exc
    return value


class SensorReader:
    def __init__(self, cfg: SensorConfig) -> None:
        self._cfg = cfg
        self._mpu = SMBus(cfg.mpu_bus)
        self._mag = SMBus(cfg.mag_bus)

    def close(self) -> None:
        for bus in (self._mpu, self._mag):
            try:
                bus.close()
            except Exception:
                pass

    def init(self) -> None:
        self._mpu.write_byte_data(self._cfg.mpu_addr, self._cfg.pwr_mgmt_1_reg, 0)
        self._mag.write_byte_data(self._cfg.mag_addr, 0x00, 0x70)
        self._mag.write_byte_data(self._cfg.mag_addr, 0x01, 0x20)
        self._mag.write_byte_data(self._cfg.mag_addr, 0x02, 0x00)

    def read_accel(self) -> tuple[float, float, float]:
        ax = _read_word(self._mpu, self._cfg.mpu_addr, self._cfg.accel_xout_reg) / self._cfg.accel_scale
        ay = _read_word(self._mpu, self._cfg.mpu_addr, self._cfg.accel_xout_reg + 2) / self._cfg.accel_scale
        az = _read_word(self._mpu, self._cfg.mpu_addr, self._cfg.accel_xout_reg + 4) / self._cfg.accel_scale
        return ax, ay, az

    def read_gyro(self) -> tuple[float, float, float]:
        gx = _read_word(self._mpu, self._cfg.mpu_addr, self._cfg.gyro_xout_reg) / self._cfg.gyro_scale
        gy = _read_word(self._mpu, self._cfg.mpu_addr, self._cfg.gyro_xout_reg + 2) / self._cfg.gyro_scale
        gz = _read_word(self._mpu, self._cfg.mpu_addr, self._cfg.gyro_xout_reg + 4) / self._cfg.gyro_scale
        return gx, gy, gz

    def read_mag(self) -> tuple[int, int, int]:
        data = self._mag.read_i2c_block_data(self._cfg.mag_addr, self._cfg.mag_data_reg, 6)
        x = (data[0] << 8) | data[1]
        z = (data[2] << 8) | data[3]
        y = (data[4] << 8) | data[5]
        if x >= 32768:
            x -= 65536
        if y >= 32768:
            y -= 65536
        if z >= 32768:
            z -= 65536
        return x, y, z


def _build_sensor_cfg(block: Mapping[str, Any], *, args: argparse.Namespace) -> SensorConfig:
    cfg = SensorConfig(
        mpu_bus=_int_cfg(block, "mpu_bus", 1),
        mag_bus=_int_cfg(block, "mag_bus", 7),
        mpu_addr=_int_cfg(block, "mpu_addr", 0x68),
        mag_addr=_int_cfg(block, "mag_addr", 0x1E),
        pwr_mgmt_1_reg=_int_cfg(block, "pwr_mgmt_1_reg", 0x6B),
        accel_xout_reg=_int_cfg(block, "accel_xout_reg", 0x3B),
        gyro_xout_reg=_int_cfg(block, "gyro_xout_reg", 0x43),
        mag_data_reg=_int_cfg(block, "mag_data_reg", 0x03),
        accel_scale=_float_cfg(block, "accel_scale", 16384.0),
        gyro_scale=_float_cfg(block, "gyro_scale", 131.0),
        alpha=_float_cfg(block, "alpha", 0.98),
        publish_hz=_float_cfg(block, "publish_hz", 50.0),
        pan_sign=_float_cfg(block, "pan_sign", 1.0),
        tilt_sign=_float_cfg(block, "tilt_sign", 1.0),
        pan_offset_rad=_float_cfg(block, "pan_offset_rad", 0.0),
        tilt_offset_rad=_float_cfg(block, "tilt_offset_rad", 0.0),
        home_pan=_float_cfg(block, "home_pan", 0.0),
        home_tilt=_float_cfg(block, "home_tilt", 0.0),
    )
    if args.publish_hz is not None:
        cfg.publish_hz = float(args.publish_hz)
    if cfg.publish_hz <= 0.0:
        raise SystemExit("camstate_devices.publish_hz must be > 0")
    if not (0.0 <= cfg.alpha <= 1.0):
        raise SystemExit("camstate_devices.alpha must be in [0, 1]")
    if cfg.pan_sign == 0.0 or cfg.tilt_sign == 0.0:
        raise SystemExit("camstate_devices.pan_sign and tilt_sign must be non-zero")
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dev.yaml", help="Base config path")
    parser.add_argument(
        "--config-extra",
        default="configs/dev_extra.yaml",
        help="Optional overlay config path",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override CamState PUB endpoint (default: net.zmq_gimbal_state)",
    )
    parser.add_argument("--publish-hz", type=float, default=None, help="Override publish frequency")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    stop_event = install_signal_handlers()

    config_paths = [Path(args.config)]
    if args.config_extra:
        config_paths.append(Path(args.config_extra))
    cfg = _load_config(config_paths)

    net_cfg = cfg.get("net") if isinstance(cfg, Mapping) else None
    if not isinstance(net_cfg, Mapping):
        raise SystemExit("config missing net section")
    endpoint = args.endpoint or net_cfg.get("zmq_gimbal_state")
    if not endpoint:
        raise SystemExit("camstate endpoint is missing (net.zmq_gimbal_state)")

    devices_cfg_raw = cfg.get("camstate_devices") if isinstance(cfg, Mapping) else None
    devices_cfg = devices_cfg_raw if isinstance(devices_cfg_raw, Mapping) else {}
    sensor_cfg = _build_sensor_cfg(devices_cfg, args=args)

    reader = SensorReader(sensor_cfg)
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)
    port = _parse_tcp_port(str(endpoint), "net.zmq_gimbal_state")
    bind_ep = f"tcp://0.0.0.0:{port}"
    pub.bind(bind_ep)

    _LOG.info(
        "publishing device CamState on %s (hz=%.1f, alpha=%.3f, pan_sign=%.1f, tilt_sign=%.1f)",
        bind_ep,
        sensor_cfg.publish_hz,
        sensor_cfg.alpha,
        sensor_cfg.pan_sign,
        sensor_cfg.tilt_sign,
    )

    frame_id = 0
    period_s = 1.0 / sensor_cfg.publish_hz
    next_tick = time.monotonic()
    prev_t: Optional[float] = None
    prev_pan: Optional[float] = None
    prev_tilt: Optional[float] = None
    roll = 0.0
    pitch = 0.0
    last_err_log = 0.0

    try:
        reader.init()
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.005))
                continue

            dt = 0.0 if prev_t is None else max(1e-4, now - prev_t)
            prev_t = now

            try:
                ax, ay, az = reader.read_accel()
                gx, gy, _gz = reader.read_gyro()
                mx, my, _mz = reader.read_mag()
            except OSError as exc:
                if (now - last_err_log) >= 1.0:
                    _LOG.warning("sensor read error: %s", exc)
                    last_err_log = now
                next_tick += period_s
                continue

            accel_roll = math.atan2(ay, az)
            accel_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))

            roll += math.radians(gx) * dt
            pitch += math.radians(gy) * dt

            roll = sensor_cfg.alpha * roll + (1.0 - sensor_cfg.alpha) * accel_roll
            pitch = sensor_cfg.alpha * pitch + (1.0 - sensor_cfg.alpha) * accel_pitch
            heading = math.atan2(float(my), float(mx))

            pan = sensor_cfg.pan_sign * heading + sensor_cfg.pan_offset_rad
            tilt = sensor_cfg.tilt_sign * pitch + sensor_cfg.tilt_offset_rad

            pan_rate: Optional[float] = None
            tilt_rate: Optional[float] = None
            if prev_pan is not None and dt > 0.0:
                pan_rate = _wrapped_delta(pan, prev_pan) / dt
            if prev_tilt is not None and dt > 0.0:
                tilt_rate = (tilt - prev_tilt) / dt
            prev_pan = pan
            prev_tilt = tilt

            frame_id += 1
            payload = CamState(
                frame_id=frame_id,
                src_ts_ms=int(time.time() * 1000),
                pan=float(pan),
                tilt=float(tilt),
                pan_rate=pan_rate,
                tilt_rate=tilt_rate,
                home_pan=sensor_cfg.home_pan,
                home_tilt=sensor_cfg.home_tilt,
            )
            try:
                pub.send_string(payload.model_dump_json(exclude_none=True), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            next_tick += period_s
            if next_tick < time.monotonic():
                next_tick = time.monotonic()
    except Exception as exc:  # noqa: BLE001
        _LOG.error("camstate device publisher failed: %s", exc)
        return 1
    finally:
        reader.close()
        try:
            pub.close(0)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
