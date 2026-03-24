"""Live MPU9255 orientation probe (mpu9250-jmdev backend).

This tool prints:
- yaw_raw_deg: heading from magnetometer (atan2(my, mx))
- pitch_est_deg: complementary-filter pitch estimate
- tilt_out_deg: pitch after tilt_sign / tilt_offset mapping
- yaw_out_deg: yaw after pan_sign / pan_offset mapping

Use CLI overrides/sweeps to quickly find the correct configuration.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

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
GFS_500 = _reg("GFS_500", 1)
GFS_1000 = _reg("GFS_1000", 2)
GFS_2000 = _reg("GFS_2000", 3)
AFS_2G = _reg("AFS_2G", 0)
AFS_4G = _reg("AFS_4G", 1)
AFS_8G = _reg("AFS_8G", 2)
AFS_16G = _reg("AFS_16G", 3)
AK8963_BIT_16 = _reg("AK8963_BIT_16", 1)
AK8963_MODE_C100HZ = _reg("AK8963_MODE_C100HZ", 6)


@dataclass
class ProbeConfig:
    mpu_bus: int = 7
    mpu_addr: int = 0x68
    mag_addr: int = 0x0C
    accel_scale: float = 16384.0
    gyro_scale: float = 131.0
    alpha: float = 0.98
    pan_sign: float = 1.0
    tilt_sign: float = 1.0
    pan_offset_rad: float = 0.0
    tilt_offset_rad: float = 0.0
    pitch_gyro_axis: str = "y"
    pitch_gyro_sign: float = 1.0
    pitch_accel_axis: str = "x"
    pitch_accel_sign: float = -1.0


def _load_config(paths: Iterable[Path]) -> Mapping[str, Any]:
    maps = []
    for path in paths:
        snapshot = read_snapshot(path)
        maps.append(parse_config_text(snapshot.text, str(path)))
    return merge_config_maps(*maps)


def _parse_float_list(raw: str) -> list[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("expected at least one numeric value")
    return vals


def _wrap_rad(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _yaw_deg(rad: float) -> float:
    deg = math.degrees(rad) % 360.0
    if deg < 0.0:
        deg += 360.0
    return deg


def _float_cfg(block: Mapping[str, Any], key: str, default: float) -> float:
    raw = block.get(key, default)
    return float(raw)


def _int_cfg(block: Mapping[str, Any], key: str, default: int) -> int:
    raw = block.get(key, default)
    return int(raw)


def _axis_cfg(block: Mapping[str, Any], key: str, default: str) -> str:
    axis = str(block.get(key, default)).strip().lower()
    if axis not in {"x", "y", "z"}:
        raise SystemExit(f"camstate_devices.{key} must be one of x/y/z, got {axis!r}")
    return axis


def _build_probe_cfg(block: Mapping[str, Any]) -> ProbeConfig:
    cfg = ProbeConfig(
        mpu_bus=_int_cfg(block, "mpu_bus", 7),
        mpu_addr=_int_cfg(block, "mpu_addr", 0x68),
        mag_addr=_int_cfg(block, "mag_addr", 0x0C),
        accel_scale=_float_cfg(block, "accel_scale", 16384.0),
        gyro_scale=_float_cfg(block, "gyro_scale", 131.0),
        alpha=_float_cfg(block, "alpha", 0.98),
        pan_sign=_float_cfg(block, "pan_sign", 1.0),
        tilt_sign=_float_cfg(block, "tilt_sign", 1.0),
        pan_offset_rad=_float_cfg(block, "pan_offset_rad", 0.0),
        tilt_offset_rad=_float_cfg(block, "tilt_offset_rad", 0.0),
        pitch_gyro_axis=_axis_cfg(block, "pitch_gyro_axis", "y"),
        pitch_gyro_sign=_float_cfg(block, "pitch_gyro_sign", 1.0),
        pitch_accel_axis=_axis_cfg(block, "pitch_accel_axis", "x"),
        pitch_accel_sign=_float_cfg(block, "pitch_accel_sign", -1.0),
    )
    if cfg.pan_sign == 0.0 or cfg.tilt_sign == 0.0:
        raise SystemExit("pan_sign and tilt_sign must be non-zero")
    if cfg.pitch_gyro_sign == 0.0 or cfg.pitch_accel_sign == 0.0:
        raise SystemExit("pitch_gyro_sign and pitch_accel_sign must be non-zero")
    return cfg


def _select_gyro_range(gyro_scale: float) -> tuple[int, str]:
    candidates = [
        (131.0, GFS_250, "GFS_250"),
        (65.5, GFS_500, "GFS_500"),
        (32.8, GFS_1000, "GFS_1000"),
        (16.4, GFS_2000, "GFS_2000"),
    ]
    best = min(candidates, key=lambda row: abs(gyro_scale - row[0]))
    return best[1], best[2]


def _select_accel_range(accel_scale: float) -> tuple[int, str]:
    candidates = [
        (16384.0, AFS_2G, "AFS_2G"),
        (8192.0, AFS_4G, "AFS_4G"),
        (4096.0, AFS_8G, "AFS_8G"),
        (2048.0, AFS_16G, "AFS_16G"),
    ]
    best = min(candidates, key=lambda row: abs(accel_scale - row[0]))
    return best[1], best[2]


def _pitch_from_accel(ax: float, ay: float, az: float, *, axis: str, sign: float) -> float:
    accel_vals = {"x": float(ax), "y": float(ay), "z": float(az)}
    num = sign * accel_vals[axis]
    den = math.sqrt(sum(v * v for k, v in accel_vals.items() if k != axis))
    return math.atan2(num, den)


class JmdevSensorReader:
    def __init__(self, cfg: ProbeConfig) -> None:
        self._cfg = cfg
        self._gfs, self.gfs_label = _select_gyro_range(cfg.gyro_scale)
        self._afs, self.afs_label = _select_accel_range(cfg.accel_scale)
        self._mpu = MPU9250(
            address_ak=int(cfg.mag_addr or AK8963_ADDRESS),
            address_mpu_master=int(cfg.mpu_addr),
            address_mpu_slave=None,
            bus=int(cfg.mpu_bus),
            gfs=self._gfs,
            afs=self._afs,
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

    def read_gyro(self) -> tuple[float, float, float]:
        vals = self._mpu.readGyroscopeMaster()
        return float(vals[0]), float(vals[1]), float(vals[2])

    def read_mag(self) -> Optional[tuple[float, float, float]]:
        vals = self._mpu.readMagnetometerMaster()
        mx = float(vals[0])
        my = float(vals[1])
        mz = float(vals[2])
        if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(mz)):
            return None
        # jmdev can return all zeros on transient read failures; treat as missing sample.
        if mx == 0.0 and my == 0.0 and mz == 0.0:
            return None
        return mx, my, mz


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dev.yaml")
    ap.add_argument("--config-extra", default="configs/dev_extra.yaml")
    ap.add_argument("--dt", type=float, default=0.1, help="Print interval in seconds")
    ap.add_argument("--samples", type=int, default=0, help="0 = run until Ctrl+C")

    # Single mapping overrides.
    ap.add_argument("--pan-sign", type=float, default=None)
    ap.add_argument("--pan-offset-deg", type=float, default=None)
    ap.add_argument("--tilt-sign", type=float, default=None)
    ap.add_argument("--tilt-offset-deg", type=float, default=None)
    ap.add_argument("--pitch-gyro-axis", choices=("x", "y", "z"), default=None)
    ap.add_argument("--pitch-gyro-sign", type=float, default=None)
    ap.add_argument("--pitch-accel-axis", choices=("x", "y", "z"), default=None)
    ap.add_argument("--pitch-accel-sign", type=float, default=None)

    # Sweep options for fast comparison.
    ap.add_argument("--pan-sign-options", default="", help='CSV, e.g. "1,-1"')
    ap.add_argument("--pan-offset-options-deg", default="", help='CSV, e.g. "0,90,180,270"')
    ap.add_argument("--tilt-sign-options", default="", help='CSV, e.g. "1,-1"')
    ap.add_argument("--show-raw", action="store_true", help="Also print raw accel/gyro/mag")
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

    # Apply runtime overrides.
    if args.pan_sign is not None:
        cfg.pan_sign = float(args.pan_sign)
    if args.pan_offset_deg is not None:
        cfg.pan_offset_rad = math.radians(float(args.pan_offset_deg))
    if args.tilt_sign is not None:
        cfg.tilt_sign = float(args.tilt_sign)
    if args.tilt_offset_deg is not None:
        cfg.tilt_offset_rad = math.radians(float(args.tilt_offset_deg))
    if args.pitch_gyro_axis is not None:
        cfg.pitch_gyro_axis = str(args.pitch_gyro_axis)
    if args.pitch_gyro_sign is not None:
        cfg.pitch_gyro_sign = float(args.pitch_gyro_sign)
    if args.pitch_accel_axis is not None:
        cfg.pitch_accel_axis = str(args.pitch_accel_axis)
    if args.pitch_accel_sign is not None:
        cfg.pitch_accel_sign = float(args.pitch_accel_sign)

    pan_sign_options = (
        _parse_float_list(args.pan_sign_options)
        if args.pan_sign_options.strip()
        else [float(cfg.pan_sign)]
    )
    pan_offset_options_deg = (
        _parse_float_list(args.pan_offset_options_deg)
        if args.pan_offset_options_deg.strip()
        else [math.degrees(float(cfg.pan_offset_rad))]
    )
    tilt_sign_options = (
        _parse_float_list(args.tilt_sign_options)
        if args.tilt_sign_options.strip()
        else [float(cfg.tilt_sign)]
    )
    combos = [
        (ps, po, ts)
        for ps in pan_sign_options
        for po in pan_offset_options_deg
        for ts in tilt_sign_options
    ]

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
        # This is only informational; jmdev still accepts numeric address.
        print(
            f"[note] jmdev register constant for selected address differs "
            f"(const={expected_const}, cfg={cfg.mpu_addr})"
        )

    reader = JmdevSensorReader(cfg)
    roll = 0.0
    pitch = 0.0
    last_t: Optional[float] = None
    last_heading: Optional[float] = None
    count = 0

    print(
        "using jmdev: "
        f"bus={cfg.mpu_bus} mpu_addr={mpu_addr_name} mag_addr=0x{cfg.mag_addr:02x} "
        f"{reader.gfs_label} {reader.afs_label} AK8963_BIT_16 AK8963_MODE_C100HZ"
    )
    print(
        "mapping: "
        f"pan_sign={cfg.pan_sign:.3f} pan_offset_deg={math.degrees(cfg.pan_offset_rad):.3f} "
        f"tilt_sign={cfg.tilt_sign:.3f} tilt_offset_deg={math.degrees(cfg.tilt_offset_rad):.3f} "
        f"pitch_gyro={cfg.pitch_gyro_axis}*{cfg.pitch_gyro_sign:.3f} "
        f"pitch_accel={cfg.pitch_accel_axis}*{cfg.pitch_accel_sign:.3f}"
    )
    print(f"combinations: {len(combos)}")

    try:
        reader.init()
        while not stop_event.is_set():
            now = time.monotonic()
            dt = 0.0 if last_t is None else max(1e-4, now - last_t)
            last_t = now

            try:
                ax, ay, az = reader.read_accel()
                gx, gy, gz = reader.read_gyro()
                mag = reader.read_mag()
            except OSError as exc:
                print(f"[warn] sensor read error: {exc}")
                time.sleep(args.dt)
                continue

            accel_roll = math.atan2(ay, az)
            accel_pitch = _pitch_from_accel(
                ax,
                ay,
                az,
                axis=cfg.pitch_accel_axis,
                sign=cfg.pitch_accel_sign,
            )
            gyro_vals = {"x": gx, "y": gy, "z": gz}
            pitch_gyro_deg_s = cfg.pitch_gyro_sign * gyro_vals[cfg.pitch_gyro_axis]

            roll += math.radians(gx) * dt
            pitch += math.radians(pitch_gyro_deg_s) * dt
            roll = cfg.alpha * roll + (1.0 - cfg.alpha) * accel_roll
            pitch = cfg.alpha * pitch + (1.0 - cfg.alpha) * accel_pitch

            if mag is not None:
                mx, my, _mz = mag
                last_heading = math.atan2(my, mx)
            if last_heading is None:
                print("waiting for first valid mag sample...")
                time.sleep(args.dt)
                continue

            heading = _wrap_rad(last_heading)
            pan = cfg.pan_sign * heading + cfg.pan_offset_rad
            tilt = cfg.tilt_sign * pitch + cfg.tilt_offset_rad

            line = (
                f"yaw_raw_deg={_yaw_deg(heading):7.2f} "
                f"pitch_est_deg={math.degrees(pitch):7.2f} "
                f"tilt_out_deg={math.degrees(tilt):7.2f} "
                f"yaw_out_deg={_yaw_deg(_wrap_rad(pan)):7.2f}"
            )

            rendered = []
            for pan_sign, pan_offset_deg, tilt_sign in combos:
                yaw_candidate = _yaw_deg(
                    _wrap_rad(pan_sign * heading + math.radians(pan_offset_deg))
                )
                tilt_candidate = math.degrees(tilt_sign * pitch + cfg.tilt_offset_rad)
                rendered.append(
                    f"[ps={pan_sign:+.0f},po={pan_offset_deg:.0f},ts={tilt_sign:+.0f}"
                    f"->yaw={yaw_candidate:6.1f},tilt={tilt_candidate:6.1f}]"
                )
            if rendered:
                line = f"{line}  {' '.join(rendered)}"

            if args.show_raw:
                line += (
                    f"  raw:acc=({ax:+.3f},{ay:+.3f},{az:+.3f})"
                    f" gyro=({gx:+.3f},{gy:+.3f},{gz:+.3f})"
                    f" mag={mag}"
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
