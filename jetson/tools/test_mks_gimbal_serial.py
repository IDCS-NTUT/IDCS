"""CLI tool to exercise MKS SERVO42D/57D_RS485 axes over an RS485 IPC link.

Example:
    python -m jetson.tools.test_mks_gimbal_serial speed --port /dev/ttyTHS0 --addr 1 --omega 0.5
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from typing import Optional

import yaml

from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485ClientBus


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
    parser.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5559",
        help="RS485 IPC endpoint (REQ/REP) for the drain service",
    )
    parser.add_argument(
        "--timeout",
        default=1.0,
        type=float,
        help="IPC timeout (seconds) for reads/writes",
    )
    parser.add_argument(
        "--retries",
        default=1,
        type=int,
        help="Retry count for command/response errors",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Optional serial port label to include in logs (no direct serial access)",
    )
    parser.add_argument(
        "--baud",
        default=None,
        type=int,
        help="Optional baudrate label to include in logs (no direct serial access)",
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

    read_params = subparsers.add_parser(
        "read-params", help="Read all configuration parameters (0x47)"
    )
    read_params.add_argument(
        "--format",
        choices=["hex", "int"],
        default="hex",
        help="How to print the returned parameters",
    )

    write_params = subparsers.add_parser(
        "write-params", help="Write all configuration parameters (0x46)"
    )
    write_params.add_argument(
        "--config",
        required=True,
        help="YAML file containing per-motor parameter lists (motors.<addr>.parameters)",
    )
    write_params.add_argument(
        "--save-byte",
        type=int,
        default=None,
        help="Optional override for Byte37 save flag (default: use YAML value)",
    )

    raw = subparsers.add_parser(
        "raw",
        help="Send a raw command: function code + data bytes as hex (e.g., raw --func 0xF3 01)",
    )
    raw.add_argument(
        "--func",
        required=True,
        help="Function code byte in hex/0x/decimal (e.g., 0xF3)",
    )
    raw.add_argument(
        "--expected-len",
        type=int,
        default=None,
        help="Expected number of response data bytes (before CRC)",
    )
    raw.add_argument(
        "--no-response",
        action="store_true",
        help="Skip waiting for a response (broadcast/group writes)",
    )
    raw.add_argument(
        "data",
        nargs="*",
        help="Data bytes as hex/0x/decimal tokens (e.g., 01 02 0x03)",
    )

    raw_frame = subparsers.add_parser(
        "raw-frame",
        help="Send a full frame exactly as typed (e.g., raw-frame fa 01 f3 00 f4)",
    )
    raw_frame.add_argument(
        "--expected-len",
        type=int,
        default=None,
        help="If provided, read this many response bytes after sending the frame",
    )
    raw_frame.add_argument(
        "frame",
        nargs="+",
        help="Complete frame bytes as hex/0x/decimal tokens (start+addr+func+data+crc)",
    )

    return parser.parse_args()


def describe_status(code: int) -> str:
    return STATUS_DESCRIPTIONS.get(code, f"Unknown status 0x{code:02X}")


def _parse_byte_tokens(tokens: list[str]) -> list[int]:
    """Convert a list of user tokens into byte values (supports 0x / hex / decimal)."""

    bytes_out: list[int] = []
    for tok in tokens:
        value = int(tok, 0)
        if not 0 <= value <= 0xFF:
            raise ValueError(f"Byte value out of range (0-255): {tok}")
        bytes_out.append(value)
    return bytes_out


def _render_bytes(resp: bytes) -> str:
    return "[" + ", ".join(f"0x{b:02X}" for b in resp) + "]"


def main() -> int:
    args = parse_args()

    # Keep cleanup within the IPC context so the stop/estop can still reach the motor.
    try:
        with RS485ClientBus(
            args.endpoint,
            timeout_ms=int(args.timeout * 1000),
            max_retries=max(args.retries, 0),
            port=args.port,
            baudrate=args.baud,
        ) as bus:
            axis = MksServo42Axis(
                bus,
                args.addr,
                group_addr=args.group_addr,
                use_group_writes=not args.no_group_writes,
            )

            try:
                if args.cmd == "raw-frame":
                    raise SystemExit(
                        "raw-frame is not supported over IPC; use 'raw' or direct serial tools"
                    )
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
                elif args.cmd == "read-params":
                    params = axis.read_all_parameters()
                    if args.format == "hex":
                        rendered = "[" + ", ".join(f"0x{b:02X}" for b in params) + "]"
                    else:
                        rendered = list(params)
                    print(rendered)
                elif args.cmd == "write-params":
                    with open(args.config, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f) or {}
                    motors = config.get("motors", {}) if isinstance(config, dict) else {}
                    entry = motors.get(args.addr) or motors.get(str(args.addr))
                    if not entry or "parameters" not in entry:
                        raise ValueError(
                            f"No parameters found for motor address {args.addr} in {args.config}"
                        )
                    params = list(entry["parameters"])
                    if args.save_byte is not None:
                        params = list(params)
                        if len(params) < 34:
                            raise ValueError(
                                "Parameter list must contain 34 entries for Byte4–Byte37"
                            )
                        params[33] = int(args.save_byte) & 0xFF
                    status = axis.write_all_parameters(params)
                    print(f"write_all_parameters status={status}")
                elif args.cmd == "raw":
                    func = int(args.func, 0)
                    data = _parse_byte_tokens(args.data)
                    resp = bus.send_command(
                        args.addr,
                        func,
                        data,
                        response_expected=not args.no_response,
                        expected_response_len=args.expected_len,
                    )
                    if resp:
                        rendered = _render_bytes(resp)
                    else:
                        rendered = "[]"
                    print(rendered)
            except Exception as exc:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    axis.command_speed(0.0)
                print(f"Error: {exc}", file=sys.stderr)
                return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error connecting to RS485 service: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
