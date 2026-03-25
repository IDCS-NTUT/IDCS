"""Live MPU9255 orientation probe (canonical test.py model).

This tool prints:
- pitch_deg: pitch from accelerometer gravity vector
- azimuth_deg: tilt-compensated heading with canonical mag axis alignment

It intentionally matches the runtime IMU path used by the pipeline.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from common.config_sync import merge_config_maps, parse_config_text, read_snapshot
from common.shutdown import install_signal_handlers

try:
    from mpu9250_jmdev import registers as jm_regs
    from mpu9250_jmdev.mpu_9250 import MPU9250
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "mpu9250-jmdev is required for this tool. Install with: pip install mpu9250-jmdev"
    ) from exc


def _reg(name: str, default: int) -> int:
    return int(getattr(jm_regs, name, default))


AK8963_ADDRESS = _reg("AK8963_ADDRESS", 0x0C)
MPU_ADDR_68 = _reg("MPU9050_ADDRESS_68", 0x68)
MPU_ADDR_69 = _reg("MPU9050_ADDRESS_69", 0x69)
GFS_250 = _reg("GFS_250", 0)
AFS_2G = _reg("AFS_2G", 0)
AK8963_BIT_16 = _reg("AK8963_BIT_16", 1)
AK8963_MODE_C100HZ = _reg("AK8963_MODE_C100HZ", 6)

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


@dataclass
class ProbeConfig:
    mpu_bus: int = 7
    mpu_addr: int = 0x68
    mag_addr: int = 0x0C


def _load_config(paths: Iterable[Path]) -> Mapping[str, Any]:
    maps = []
    for path in paths:
        snapshot = read_snapshot(path)
        maps.append(parse_config_text(snapshot.text, str(path)))
    return merge_config_maps(*maps)


def _int_cfg(block: Mapping[str, Any], key: str, default: int) -> int:
    raw = block.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"camstate_devices.{key} must be an integer, got {raw!r}") from exc


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


def _build_probe_cfg(block: Mapping[str, Any]) -> ProbeConfig:
    _validate_camstate_keys(block)
    return ProbeConfig(
        mpu_bus=_int_cfg(block, "mpu_bus", 7),
        mpu_addr=_int_cfg(block, "mpu_addr", 0x68),
        mag_addr=_int_cfg(block, "mag_addr", 0x0C),
    )


def _yaw_deg(rad: float) -> float:
    deg = math.degrees(rad) % 360.0
    if deg < 0.0:
        deg += 360.0
    return deg


def _compute_orientation(ax: float, ay: float, az: float, mx: float, my: float, mz: float) -> tuple[float, float]:
    pitch = math.atan2(float(ax), math.sqrt(float(ay) * float(ay) + float(az) * float(az)))
    roll = math.atan2(-float(ay), float(az))
    mx_aligned = float(my)
    my_aligned = float(mx)
    mz_aligned = -float(mz)
    mx2 = mx_aligned * math.cos(pitch) + mz_aligned * math.sin(pitch)
    my2 = (
        mx_aligned * math.sin(roll) * math.sin(pitch)
        + my_aligned * math.cos(roll)
        - mz_aligned * math.sin(roll) * math.cos(pitch)
    )
    heading = math.atan2(my2, mx2)
    return pitch, heading


class JmdevSensorReader:
    def __init__(self, cfg: ProbeConfig) -> None:
        self._cfg = cfg
        self._mpu = MPU9250(
            address_ak=int(cfg.mag_addr or AK8963_ADDRESS),
            address_mpu_master=int(cfg.mpu_addr),
            address_mpu_slave=None,
            bus=int(cfg.mpu_bus),
            gfs=GFS_250,
            afs=AFS_2G,
            mfs=AK8963_BIT_16,
            mode=AK8963_MODE_C100HZ,
        )

    def init(self) -> None:
        self._mpu.configure()

    def close(self) -> None:
        bus = getattr(self._mpu, "bus", None)
        if bus is not None and hasattr(bus, "close"):
            try:
                bus.close()
            except Exception:
                pass

    def read_accel(self) -> tuple[float, float, float]:
        vals = self._mpu.readAccelerometerMaster()
        return float(vals[0]), float(vals[1]), float(vals[2])

    def read_mag(self) -> tuple[float, float, float] | None:
        vals = self._mpu.readMagnetometerMaster()
        mx = float(vals[0])
        my = float(vals[1])
        mz = float(vals[2])
        if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(mz)):
            return None
        if mx == 0.0 and my == 0.0 and mz == 0.0:
            return None
        return mx, my, mz


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dev.yaml")
    ap.add_argument("--config-extra", default="configs/dev_extra.yaml")
    ap.add_argument("--dt", type=float, default=0.1, help="Print interval in seconds")
    ap.add_argument("--samples", type=int, default=0, help="0 = run until Ctrl+C")
    ap.add_argument("--show-raw", action="store_true", help="Also print raw accel/mag values")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    stop_event = install_signal_handlers()

    cfg_paths = [Path(args.config)]
    if args.config_extra:
        cfg_paths.append(Path(args.config_extra))
    cfg_map = _load_config(cfg_paths)
    camstate_block = cfg_map.get("camstate_devices")
    camstate_block = camstate_block if isinstance(camstate_block, Mapping) else {}
    cfg = _build_probe_cfg(camstate_block)

    if cfg.mpu_addr == 0x68:
        mpu_addr_name = "0x68/default"
        expected_const = MPU_ADDR_68
    elif cfg.mpu_addr == 0x69:
        mpu_addr_name = "0x69/ad0-high"
        expected_const = MPU_ADDR_69
    else:
        mpu_addr_name = f"0x{cfg.mpu_addr:02x}/custom"
        expected_const = cfg.mpu_addr

    if expected_const != cfg.mpu_addr:
        print(
            f"[note] jmdev register constant for selected address differs "
            f"(const={expected_const}, cfg={cfg.mpu_addr})"
        )

    reader = JmdevSensorReader(cfg)
    count = 0
    print(
        "using jmdev: "
        f"bus={cfg.mpu_bus} mpu_addr={mpu_addr_name} mag_addr=0x{cfg.mag_addr:02x} "
        "GFS_250 AFS_2G AK8963_BIT_16 AK8963_MODE_C100HZ"
    )

    try:
        reader.init()
        while not stop_event.is_set():
            try:
                ax, ay, az = reader.read_accel()
                mag = reader.read_mag()
            except OSError as exc:
                print(f"[warn] sensor read error: {exc}")
                time.sleep(args.dt)
                continue

            if mag is None:
                print("waiting for first valid mag sample...")
                time.sleep(args.dt)
                continue

            mx, my, mz = mag
            pitch, heading = _compute_orientation(ax, ay, az, mx, my, mz)

            line = (
                f"pitch_deg={math.degrees(pitch):7.2f} "
                f"azimuth_deg={_yaw_deg(heading):7.2f}"
            )
            if args.show_raw:
                line += (
                    f"  raw:acc=({ax:+.3f},{ay:+.3f},{az:+.3f})"
                    f" mag=({mx:+.1f},{my:+.1f},{mz:+.1f})"
                )
            print(line)

            count += 1
            if args.samples > 0 and count >= args.samples:
                break
            time.sleep(max(0.0, args.dt))
    finally:
        reader.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
