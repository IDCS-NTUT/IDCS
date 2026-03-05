"""Interactive serial console for servo control boards.

This tool reads hex bytes from stdin, appends checksum-8 (sum & 0xFF), sends
them over a serial port, and prints the raw response in hex.

Example:
    python tools/serial_console.py --port /dev/ttyUSB0 --baud 256000
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import serial


def _checksum8(data: bytes) -> int:
    return sum(data) & 0xFF


def _parse_hex_input(text: str) -> bytes:
    cleaned = text.replace(",", " ").replace("_", " ")
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        raise ValueError("No hex bytes provided")

    values: List[int] = []
    for token in tokens:
        value = int(token, 16)
        if value < 0 or value > 0xFF:
            raise ValueError(f"Byte out of range: {token}")
        values.append(value)
    return bytes(values)


def _hex_bytes(data: bytes) -> str:
    if not data:
        return "<empty>"
    return " ".join(f"{byte:02X}" for byte in data)


def _read_response(
    ser: serial.Serial,
    *,
    initial_wait_s: float,
    idle_timeout_s: float,
    max_bytes: int,
) -> bytes:
    if initial_wait_s > 0:
        time.sleep(initial_wait_s)

    response = bytearray()
    last_rx = time.monotonic()

    while len(response) < max_bytes:
        chunk = ser.read(1)
        if chunk:
            response.extend(chunk)
            last_rx = time.monotonic()
            continue

        if response and (time.monotonic() - last_rx) >= idle_timeout_s:
            break
        if not response:
            break

    return bytes(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serial hex console with automatic checksum-8 append"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial device path")
    parser.add_argument("--baud", type=int, default=256000, help="Serial baudrate")
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.1,
        help="Serial read timeout in seconds",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.05,
        help="Stop reading response after this idle time once bytes are received",
    )
    parser.add_argument(
        "--initial-wait",
        type=float,
        default=0.02,
        help="Wait time after TX before reading response",
    )
    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=512,
        help="Maximum number of response bytes to read",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_response_bytes <= 0:
        print("--max-response-bytes must be > 0", file=sys.stderr)
        return 2
    if args.idle_timeout < 0 or args.initial_wait < 0:
        print("--idle-timeout and --initial-wait must be >= 0", file=sys.stderr)
        return 2

    try:
        with serial.Serial(
            args.port,
            args.baud,
            timeout=args.timeout,
            write_timeout=args.timeout,
        ) as ser:
            print(
                f"Connected to {args.port} @ {args.baud}. "
                "Enter hex bytes (e.g. 'FA 01 F3 01')."
            )
            print("Type 'exit' or 'quit' to leave.")

            while True:
                try:
                    line = input("hex> ").strip()
                except EOFError:
                    print()
                    break

                if not line:
                    continue
                if line.lower() in {"exit", "quit"}:
                    break

                try:
                    msg = _parse_hex_input(line)
                except ValueError as exc:
                    print(f"Input error: {exc}")
                    continue

                checksum = _checksum8(msg)
                frame = msg + bytes([checksum])

                try:
                    ser.reset_input_buffer()
                    ser.write(frame)
                    ser.flush()
                    response = _read_response(
                        ser,
                        initial_wait_s=args.initial_wait,
                        idle_timeout_s=args.idle_timeout,
                        max_bytes=args.max_response_bytes,
                    )
                except serial.SerialException as exc:
                    print(f"Serial error: {exc}")
                    continue

                print(f"TX: {_hex_bytes(frame)}")
                print(f"RX: {_hex_bytes(response)}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    except serial.SerialException as exc:
        print(f"Failed to open serial port: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
