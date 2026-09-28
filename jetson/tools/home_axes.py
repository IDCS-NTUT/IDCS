"""Drive each active axis back to the centre of its configured envelope.

On the uncoupled bench nothing stops an axis, so relative moves (probes,
sweeps) accumulate into turns of drift until the envelope blocks one
direction. This homes by the magnetic encoder (0x31 multi-turn counts): F4
relative moves toward the envelope centre, repeated until within tolerance.
F4 moves are offset-free on this firmware; the firmware's own homing (91H) is
not used: without an origin homing it ran as an endless endstop search that
F7 did not stop (2026-09-28 bench test).

Targets come from the gimbal config: centre of ``*_min_rad``/``*_max_rad``,
mapped to encoder counts through the CamState sign and gear ratio. A paired
pitch-B is refused until pair handling is qualified. Opens the RS485 port
directly (stop the gimbal stack first); moves only with ``--execute``; on any
failure the axes are stopped and de-energized.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common.config import load_config_bundle, resolve_config_paths  # noqa: E402
from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus  # noqa: E402

COUNTS_PER_REV = 16384


def home_targets(gimbal: dict) -> dict[str, dict]:
    """Encoder-count target per axis from the envelope centre."""
    if gimbal.get("pitch_motor_b_enabled", True):
        raise ValueError("homing a paired pitch-B is not qualified; set gimbal.pitch_motor_b_enabled: false")
    targets = {}
    for axis, addr_key, sign_key, gear_key in (("yaw", "yaw_addr", "camstate_yaw_sign", "yaw_gear_ratio"),
                                               ("pitch", "pitch_motor_a_addr", "camstate_pitch_sign",
                                                "pitch_gear_ratio")):
        low, high = gimbal.get(f"{axis}_min_rad"), gimbal.get(f"{axis}_max_rad")
        if low is None or high is None:
            raise ValueError(f"gimbal.{axis}_min_rad/{axis}_max_rad are required to home {axis}")
        centre = (float(low) + float(high)) / 2.0
        sign = float(gimbal.get(sign_key, 1.0))
        gear = float(gimbal.get(gear_key, 1.0))
        counts = round(centre / sign * gear * COUNTS_PER_REV / (2.0 * math.pi))
        targets[axis] = {"addr": int(gimbal[addr_key]), "centre_rad": centre, "target_counts": counts}
    return targets


def home_axis(bus: RS485Bus, addr: int, target: int, *, speed_rpm: int, acc: int,
              tolerance: int, attempts: int = 4) -> dict:
    def counts() -> int:
        return int.from_bytes(bus.send_command(addr, 0x31, expected_response_len=6), "big", signed=True)

    if bus.send_command(addr, 0xF3, [1], expected_response_len=1) != b"\x01":
        raise RuntimeError(f"addr {addr}: enable not acknowledged")
    time.sleep(0.3)
    start = here = counts()
    for _ in range(attempts):
        rel = target - here
        if abs(rel) <= tolerance:
            break
        bus.send_command(addr, 0xF4, MksServo42Axis._encode_relative_axis_payload(rel, speed_rpm, acc),
                         response_expected=False)
        deadline = time.monotonic() + abs(rel) / (speed_rpm * COUNTS_PER_REV / 60.0) + 3.0
        last = None
        while time.monotonic() < deadline:
            time.sleep(0.25)
            here = counts()
            if last is not None and here == last:
                break  # stopped
            last = here
        else:
            raise RuntimeError(f"addr {addr}: move did not settle before its deadline")
    return {"addr": addr, "start_counts": start, "target_counts": target, "final_counts": here,
            "error_counts": here - target, "homed": abs(here - target) <= tolerance}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base")
    parser.add_argument("--config-extra", required=True)
    parser.add_argument("--port", default="/dev/ttyCH341USB0")
    parser.add_argument("--speed-rpm", type=int, default=20)
    parser.add_argument("--acc", type=int, default=2)
    parser.add_argument("--tolerance-counts", type=int, default=30, help="30 counts = 11.5 mrad")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = load_config_bundle(resolve_config_paths(args.config, args.config_extra)).mutable_copy()
    try:
        targets = home_targets(config.get("gimbal") or {})
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"event": "plan", "targets": targets}), flush=True)
    if not args.execute:
        return 0
    results = {}
    with RS485Bus(args.port, baudrate=38400, timeout=0.2, max_retries=1) as bus:
        try:
            for axis, spec in targets.items():
                results[axis] = home_axis(bus, spec["addr"], spec["target_counts"], speed_rpm=args.speed_rpm,
                                          acc=args.acc, tolerance=args.tolerance_counts)
        except Exception as exc:  # noqa: BLE001
            for spec in targets.values():  # F7 alone does not stop every firmware motion; de-energize
                for func, payload in ((0xF7, []), (0xF3, [0])):
                    try:
                        bus.send_command(spec["addr"], func, payload, response_expected=False)
                    except Exception:  # noqa: BLE001
                        pass
            print(json.dumps({"event": "abort", "reason": str(exc), "results": results}), flush=True)
            return 3
    ok = all(r["homed"] for r in results.values())
    print(json.dumps({"event": "summary", "homed": ok, "results": results}), flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
