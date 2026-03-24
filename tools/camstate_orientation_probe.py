"""Live MPU9255 orientation probe for tuning CamState mapping.

This tool reads the same device path used by Jetson CamState and prints:
- yaw_raw_deg: heading from AK8963 (magnetometer)
- pitch_est_deg: complementary-filter pitch estimate
- tilt_out_deg: pitch after tilt_sign / tilt_offset mapping

It also supports option sweeps so you can compare multiple sign/offset
combinations in one run.

Examples
--------
# Use config defaults and print one output mapping
python -m tools.camstate_orientation_probe

# Flip pitch and add +90 yaw offset
python -m tools.camstate_orientation_probe --tilt-sign -1 --pan-offset-deg 90

# Compare several yaw sign/offset candidates at once
python -m tools.camstate_orientation_probe \
  --pan-sign-options "1,-1" \
  --pan-offset-options-deg "0,90,180,270" \
  --tilt-sign-options "1,-1"
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional

from common.config_sync import merge_config_maps, parse_config_text, read_snapshot
from common.shutdown import install_signal_handlers
from jetson.camstate_devices import SensorReader, _build_sensor_cfg


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


def _pitch_from_accel(ax: float, ay: float, az: float, *, axis: str, sign: float) -> float:
    accel_vals = {"x": float(ax), "y": float(ay), "z": float(az)}
    num = sign * accel_vals[axis]
    den = math.sqrt(sum(v * v for k, v in accel_vals.items() if k != axis))
    return math.atan2(num, den)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dev.yaml")
    ap.add_argument("--config-extra", default="configs/dev_extra.yaml")
    ap.add_argument("--hz", type=float, default=None, help="Override publish_hz from config")
    ap.add_argument("--dt", type=float, default=0.1, help="Print interval in seconds")
    ap.add_argument("--samples", type=int, default=0, help="0 = run until Ctrl+C")

    # Single mapping overrides
    ap.add_argument("--pan-sign", type=float, default=None)
    ap.add_argument("--pan-offset-deg", type=float, default=None)
    ap.add_argument("--tilt-sign", type=float, default=None)
    ap.add_argument("--tilt-offset-deg", type=float, default=None)
    ap.add_argument("--pitch-gyro-axis", choices=("x", "y", "z"), default=None)
    ap.add_argument("--pitch-gyro-sign", type=float, default=None)
    ap.add_argument("--pitch-accel-axis", choices=("x", "y", "z"), default=None)
    ap.add_argument("--pitch-accel-sign", type=float, default=None)

    # Sweep options for fast comparison
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

    cfg = _build_sensor_cfg(camstate_block, args=SimpleNamespace(publish_hz=args.hz))

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

    combos = []
    for pan_sign in pan_sign_options:
        for pan_offset_deg in pan_offset_options_deg:
            for tilt_sign in tilt_sign_options:
                combos.append((pan_sign, pan_offset_deg, tilt_sign))
    if len(combos) > 16:
        print(f"[warn] {len(combos)} combinations requested; consider narrowing options.")

    reader = SensorReader(cfg)
    roll = 0.0
    pitch = 0.0
    last_t: Optional[float] = None
    last_heading: Optional[float] = None
    count = 0

    print(
        "using config: "
        f"mpu_bus={cfg.mpu_bus} mag_bus={cfg.mag_bus} "
        f"pitch_gyro={cfg.pitch_gyro_axis}*{cfg.pitch_gyro_sign:.3f} "
        f"pitch_accel={cfg.pitch_accel_axis}*{cfg.pitch_accel_sign:.3f} "
        f"alpha={cfg.alpha:.3f}"
    )
    print(
        "single-output mapping: "
        f"pan_sign={cfg.pan_sign:.3f} pan_offset_deg={math.degrees(cfg.pan_offset_rad):.3f} "
        f"tilt_sign={cfg.tilt_sign:.3f} tilt_offset_deg={math.degrees(cfg.tilt_offset_rad):.3f}"
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
            gyro_vals = {"x": float(gx), "y": float(gy), "z": float(gz)}
            pitch_gyro_deg_s = cfg.pitch_gyro_sign * gyro_vals[cfg.pitch_gyro_axis]

            roll += math.radians(gx) * dt
            pitch += math.radians(pitch_gyro_deg_s) * dt
            roll = cfg.alpha * roll + (1.0 - cfg.alpha) * accel_roll
            pitch = cfg.alpha * pitch + (1.0 - cfg.alpha) * accel_pitch

            if mag is not None:
                mx, my, _mz = mag
                last_heading = math.atan2(float(my), float(mx))
            if last_heading is None:
                print("waiting for first mag sample (ST1 ready)...")
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

            if combos:
                rendered = []
                for pan_sign, pan_offset_deg, tilt_sign in combos:
                    yaw_candidate = _yaw_deg(
                        _wrap_rad(pan_sign * heading + math.radians(pan_offset_deg))
                    )
                    tilt_candidate = math.degrees(
                        tilt_sign * pitch + cfg.tilt_offset_rad
                    )
                    rendered.append(
                        f"[ps={pan_sign:+.0f},po={pan_offset_deg:.0f},ts={tilt_sign:+.0f}"
                        f" -> yaw={yaw_candidate:6.1f},tilt={tilt_candidate:6.1f}]"
                    )
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
