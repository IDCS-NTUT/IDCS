"""CLI tool to exercise MKS SERVO42D/57D_RS485 axes over a TTL serial link.

Example:
    python -m jetson.tools.test_mks_gimbal_serial speed --port /dev/ttyTHS0 --addr 1 --omega 0.5
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from typing import Optional

from jetson.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus


STATUS_DESCRIPTIONS = {
    0x00: "Query failed",
    0x01: "Motor stopped",
    0x02: "Speeding up",
    0x03: "Slowing down",
    0x04: "Full speed",
    0x05: "Homing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyTHS0", help="Serial port for the TTL link")
    parser.add_argument("--baud", default=38400, type=int, help="Baudrate for the serial link")
    parser.add_argument(
        "--timeout",
        default=0.1,
        type=float,
        help="Serial timeout (seconds) for reads/writes",
    )
    parser.add_argument(
        "--retries",
        default=1,
        type=int,
        help="Retry count for command/response errors",
    )
    parser.add_argument("--addr", default=1, type=int, help="Motor slave address")
    parser.add_argument(
        "--group-addr",
        default=None,
        type=int,
        help="Optional group address for write commands (no replies expected)",
    )
    parser.add_argument(
        "--no-group-writes",
        action="store_true",
        help="Force all commands to use the slave address instead of the group",
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    speed = subparsers.add_parser("speed", help="Run motor in speed mode")
    speed.add_argument("--omega", type=float, required=True, help="Speed command (rad/s)")
    speed.add_argument("--acc", type=int, default=10, help="Acceleration byte (0-255)")
    speed.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration to hold speed before stopping",
    )

    stop = subparsers.add_parser("stop", help="Send decelerating stop (speed 0)")
    stop.add_argument("--acc", type=int, default=30, help="Acceleration byte for stop")

    subparsers.add_parser("estop", help="Emergency stop the motor")

    subparsers.add_parser("read-enc", help="Read encoder counts and radians")

    subparsers.add_parser("status", help="Query status byte")

    return parser.parse_args()


def describe_status(code: int) -> str:
    return STATUS_DESCRIPTIONS.get(code, f"Unknown status 0x{code:02X}")


def main() -> int:
    args = parse_args()

    # Keep cleanup within the serial context so the stop/estop can still reach the motor.
    try:
        with RS485Bus(
            args.port,
            args.baud,
            timeout=args.timeout,
            max_retries=max(args.retries, 0),
        ) as bus:
            axis = MksServo42Axis(
                bus,
                args.addr,
                group_addr=args.group_addr,
                use_group_writes=not args.no_group_writes,
            )

            try:
                if args.cmd == "speed":
                    axis.command_speed(args.omega, acc=args.acc)
                    if args.duration is not None:
                        time.sleep(args.duration)
                        axis.command_speed(0.0, acc=args.acc)
                elif args.cmd == "stop":
                    axis.command_speed(0.0, acc=args.acc)
                elif args.cmd == "estop":
                    axis.emergency_stop()
                elif args.cmd == "read-enc":
                    counts = axis.read_axis_counts()
                    angle = axis.read_angle_rad()
                    print(f"counts={counts} angle_rad={angle:.6f}")
                elif args.cmd == "status":
                    code = axis.status()
                    print(f"status=0x{code:02X} ({describe_status(code)})")
            except Exception as exc:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    axis.command_speed(0.0)
                print(f"Error: {exc}", file=sys.stderr)
                return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error opening serial bus: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
