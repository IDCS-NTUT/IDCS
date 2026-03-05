"""Simple utility to continuously send a message over a serial port.

Example:
    python -m jetson.tools.serial_send_loop --port /dev/ttyTHS0 --baud 256000 --message "ping" --interval 0.5
"""

import argparse
import sys
import time

import serial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously send a message over serial")
    parser.add_argument(
        "--port", default="/dev/ttyTHS0", help="Serial port device path (default: /dev/ttyTHS0)"
    )
    parser.add_argument("--baud", type=int, default=256000, help="Baud rate (default: 256000)")
    parser.add_argument(
        "--message",
        default="ping",
        help="Message string to transmit repeatedly (default: 'ping')",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between transmissions (default: 1.0)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding to use when sending the message (default: utf-8)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = args.message.encode(args.encoding)

    try:
        with serial.Serial(args.port, args.baud, timeout=1, write_timeout=1) as ser:
            print(
                f"Sending message every {args.interval}s to {args.port} at {args.baud} baud."
                " Press Ctrl+C to stop."
            )
            while True:
                ser.write(payload)
                ser.flush()
                print(f"[{time.strftime('%H:%M:%S')}] sent: {payload!r}")
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Interrupted, exiting...")
        return 0
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - safety net for unexpected issues
        print(f"Unexpected error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
