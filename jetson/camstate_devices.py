"""Publish Jetson I2C IMU/magnetometer orientation as CamState telemetry.

This process is intended to be run on Jetson when MPU9255/AK8963 modules are
wired locally. It publishes CamState on ``net.zmq_gimbal_state`` so the
existing Jetson server path can consume it as the primary camera-state source.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

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
    except Exception:  # noqa: BLE001
        SMBus = None  # type: ignore[assignment]


@dataclass
class SensorConfig:
    mpu_bus: int = 7
    mpu_addr: int = 0x68
    mag_addr: int = 0x0C
    publish_hz: float = 50.0


_PWR_MGMT_1 = 0x6B
_INT_PIN_CFG = 0x37
_INT_PIN_CFG_BYPASS_VAL = 0x02
_ACCEL_XOUT = 0x3B
_MAG_ST1 = 0x02
_MAG_DATA = 0x03
_MAG_ST2 = 0x09
_MAG_CNTL1 = 0x0A
_MAG_POWER_DOWN = 0x00
_MAG_CONTINUOUS_100HZ = 0x16
_ACCEL_SCALE = 16384.0

_SUPPORTED_CAMSTATE_KEYS = {
    "mpu_bus",
    "mpu_addr",
    "mag_addr",
    "publish_hz",
}
_REMOVED_CAMSTATE_KEYS = {
    "mag_bus",
    "pwr_mgmt_1_reg",
    "int_pin_cfg_reg",
    "int_pin_cfg_bypass_val",
    "accel_xout_reg",
    "gyro_xout_reg",
    "mag_st1_reg",
    "mag_data_reg",
    "mag_st2_reg",
    "mag_cntl1_reg",
    "mag_mode_val",
    "accel_scale",
    "gyro_scale",
    "alpha",
    "pan_sign",
    "tilt_sign",
    "pan_offset_rad",
    "tilt_offset_rad",
    "pitch_gyro_axis",
    "pitch_gyro_sign",
    "pitch_accel_axis",
    "pitch_accel_sign",
    "home_pan",
    "home_tilt",
}


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


def _validate_camstate_keys(block: Mapping[str, Any]) -> None:
    keys = {str(key) for key in block.keys()}
    unknown = sorted(keys - _SUPPORTED_CAMSTATE_KEYS)
    if not unknown:
        return
    removed = [key for key in unknown if key in _REMOVED_CAMSTATE_KEYS]
    if removed:
        raise SystemExit(
            "camstate_devices contains removed keys: "
            + ", ".join(removed)
            + ". Keep only: "
            + ", ".join(sorted(_SUPPORTED_CAMSTATE_KEYS))
        )
    raise SystemExit(
        "camstate_devices contains unsupported keys: "
        + ", ".join(unknown)
        + ". Keep only: "
        + ", ".join(sorted(_SUPPORTED_CAMSTATE_KEYS))
    )


def _accel_pitch_roll(ax: float, ay: float, az: float) -> tuple[float, float]:
    pitch = math.atan2(float(ax), math.sqrt(float(ay) * float(ay) + float(az) * float(az)))
    roll = math.atan2(-float(ay), float(az))
    return pitch, roll


def _tilt_compensated_heading(
    *,
    pitch: float,
    roll: float,
    mx: float,
    my: float,
    mz: float,
) -> float:
    # Canonical test.py alignment: swap X/Y and flip Z.
    mx_aligned = float(my)
    my_aligned = float(mx)
    mz_aligned = -float(mz)

    mx2 = mx_aligned * math.cos(pitch) + mz_aligned * math.sin(pitch)
    my2 = (
        mx_aligned * math.sin(roll) * math.sin(pitch)
        + my_aligned * math.cos(roll)
        - mz_aligned * math.sin(roll) * math.cos(pitch)
    )
    return math.atan2(my2, mx2)


def _compute_orientation(ax: float, ay: float, az: float, mx: float, my: float, mz: float) -> tuple[float, float]:
    pitch, roll = _accel_pitch_roll(ax, ay, az)
    heading = _tilt_compensated_heading(pitch=pitch, roll=roll, mx=mx, my=my, mz=mz)
    return pitch, heading


class SensorReader:
    def __init__(self, cfg: SensorConfig) -> None:
        if SMBus is None:
            raise SystemExit("camstate_devices requires smbus2 or smbus to be installed")
        self._cfg = cfg
        self._mpu = SMBus(cfg.mpu_bus)
        self._mag = SMBus(cfg.mpu_bus)

    def close(self) -> None:
        for bus in (self._mpu, self._mag):
            try:
                bus.close()
            except Exception:
                pass

    def init(self) -> None:
        self._mpu.write_byte_data(self._cfg.mpu_addr, _PWR_MGMT_1, 0)
        time.sleep(0.1)
        self._mpu.write_byte_data(self._cfg.mpu_addr, _INT_PIN_CFG, _INT_PIN_CFG_BYPASS_VAL)
        self._mag.write_byte_data(self._cfg.mag_addr, _MAG_CNTL1, _MAG_POWER_DOWN)
        time.sleep(0.01)
        self._mag.write_byte_data(self._cfg.mag_addr, _MAG_CNTL1, _MAG_CONTINUOUS_100HZ)
        time.sleep(0.01)

    def read_accel(self) -> tuple[float, float, float]:
        ax = _read_word(self._mpu, self._cfg.mpu_addr, _ACCEL_XOUT) / _ACCEL_SCALE
        ay = _read_word(self._mpu, self._cfg.mpu_addr, _ACCEL_XOUT + 2) / _ACCEL_SCALE
        az = _read_word(self._mpu, self._cfg.mpu_addr, _ACCEL_XOUT + 4) / _ACCEL_SCALE
        return ax, ay, az

    def read_mag(self) -> Optional[tuple[int, int, int]]:
        st1 = self._mag.read_byte_data(self._cfg.mag_addr, _MAG_ST1)
        if not (st1 & 0x01):
            return None
        data = self._mag.read_i2c_block_data(self._cfg.mag_addr, _MAG_DATA, 6)
        self._mag.read_byte_data(self._cfg.mag_addr, _MAG_ST2)
        x = (data[1] << 8) | data[0]
        y = (data[3] << 8) | data[2]
        z = (data[5] << 8) | data[4]
        if x >= 32768:
            x -= 65536
        if y >= 32768:
            y -= 65536
        if z >= 32768:
            z -= 65536
        return x, y, z


def _build_sensor_cfg(block: Mapping[str, Any], *, args: argparse.Namespace) -> SensorConfig:
    _validate_camstate_keys(block)
    mpu_bus = _int_cfg(block, "mpu_bus", 7)
    cfg = SensorConfig(
        mpu_bus=mpu_bus,
        mpu_addr=_int_cfg(block, "mpu_addr", 0x68),
        mag_addr=_int_cfg(block, "mag_addr", 0x0C),
        publish_hz=_float_cfg(block, "publish_hz", 50.0),
    )
    if args.publish_hz is not None:
        cfg.publish_hz = float(args.publish_hz)
    if cfg.publish_hz <= 0.0:
        raise SystemExit("camstate_devices.publish_hz must be > 0")
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
        "publishing device CamState on %s (hz=%.1f, mpu_bus=%d, mpu_addr=0x%02x, mag_addr=0x%02x)",
        bind_ep,
        sensor_cfg.publish_hz,
        sensor_cfg.mpu_bus,
        sensor_cfg.mpu_addr,
        sensor_cfg.mag_addr,
    )

    frame_id = 0
    period_s = 1.0 / sensor_cfg.publish_hz
    next_tick = time.monotonic()
    prev_t: Optional[float] = None
    prev_pan: Optional[float] = None
    prev_tilt: Optional[float] = None
    last_err_log = 0.0
    last_heading: Optional[float] = None
    home_pan: Optional[float] = None
    home_tilt: Optional[float] = None

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
                mag = reader.read_mag()
            except OSError as exc:
                if (now - last_err_log) >= 1.0:
                    _LOG.warning("sensor read error: %s", exc)
                    last_err_log = now
                next_tick += period_s
                continue

            pitch, roll = _accel_pitch_roll(ax, ay, az)
            if mag is not None:
                mx, my, mz = mag
                last_heading = _tilt_compensated_heading(
                    pitch=pitch,
                    roll=roll,
                    mx=mx,
                    my=my,
                    mz=mz,
                )
            if last_heading is None:
                next_tick += period_s
                continue

            heading = last_heading
            pan = heading
            tilt = pitch
            if home_pan is None:
                home_pan = float(pan)
            if home_tilt is None:
                home_tilt = float(tilt)

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
                home_pan=home_pan,
                home_tilt=home_tilt,
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
