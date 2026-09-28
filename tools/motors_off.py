"""De-energize the gimbal motors (MKS F3 0) and confirm each acknowledged.

Run on the Jetson after stopping the motor stack, which owns the serial port:

    sudo systemctl stop idcs-hil.target
    python -m tools.motors_off

Addresses and port come from the gimbal config (yaw, pitch A, pitch B). The
motors hold their position while energized after a stop; this lets them
rest. Exits non-zero if any motor did not acknowledge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from common.config_sync import expand_config_paths, load_merged_config

ENABLE = 0xF3


def motor_addresses(gimbal: dict) -> list[int]:
    keys = ("yaw_addr", "pitch_motor_a_addr", "pitch_motor_b_addr")
    return sorted({int(gimbal[key]) for key in keys if gimbal.get(key) is not None})


def port_owner(port: str) -> str | None:
    """A process holding the port (the serial service would fight this)."""
    result = subprocess.run(["fuser", port], capture_output=True, text=True)
    owners = result.stdout.split()
    return owners[0] if owners else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base")
    parser.add_argument("--config-extra", default="configs/bench/uncoupled.yaml")
    parser.add_argument("--on", action="store_true", help="energize instead (F3 1)")
    args = parser.parse_args()
    gimbal = load_merged_config(expand_config_paths(args.config, args.config_extra)).get("gimbal") or {}
    port = str(gimbal.get("serial_port", "/dev/ttyCH341USB0"))
    owner = port_owner(port)
    if owner is not None:
        print(f"{port} is in use by pid {owner}; stop the motor stack first "
              "(sudo systemctl stop idcs-hil.target)", file=sys.stderr)
        return 2
    from common.gimbal.mks_servo42_rs485 import RS485Bus

    value = 1 if args.on else 0
    results = {}
    with RS485Bus(port, baudrate=int(gimbal.get("baudrate", 38400)), timeout=0.12, max_retries=1) as bus:
        for addr in motor_addresses(gimbal):
            try:
                reply = bus.send_command(addr, ENABLE, [value], expected_response_len=1)
            except Exception as exc:  # noqa: BLE001 - report every motor
                reply = None
                results[addr] = f"error: {exc}"
                continue
            results[addr] = "ack" if reply == b"\x01" else f"no ack ({reply!r})"
    print(json.dumps({"energized" if args.on else "de_energized": results}))
    return 0 if all(v == "ack" for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
