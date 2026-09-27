"""F5 step 2c: yaw-only F5 out / F4 relative back, logging counts and angle error.

Distinguishes an internal-position offset (0x39 error ~0 while counts miss
the target) from an uncorrected position error (0x39 error ~ the miss).
Same bounds as step 2: 26 counts at 1 RPM, 150-count guard, F5 stop + F7 +
disable on exit. Records instead of requiring settling.
"""

import json
import sys
import time

from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus

YAW = 1
STEP_COUNTS = 26
GUARD_COUNTS = 150
SPEED_RPM = 1
ACC = 2
OBSERVE_S = 1.5
ERROR_UNITS_PER_COUNT = 51200 / 16384  # 0x39 units per 0x31 count


def move_payload(counts: int) -> list[int]:
    axis = counts.to_bytes(4, "big", signed=True)
    return [(SPEED_RPM >> 8) & 0xFF, SPEED_RPM & 0xFF, ACC, *axis]


F5_STOP_NOW = [0, 0, 0, 0, 0, 0, 0]


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, "monotonic_ns": time.monotonic_ns(), **fields}), flush=True)


def main() -> int:
    with RS485Bus("/dev/ttyCH341USB0", baudrate=38400, timeout=0.12, max_retries=1) as bus:
        yaw = MksServo42Axis(bus, addr=YAW)

        def angle_error_counts() -> float:
            data = bus.send_command(YAW, 0x39, expected_response_len=4)
            return int.from_bytes(data, "big", signed=True) / ERROR_UNITS_PER_COUNT

        def snapshot(phase: str, **extra) -> int:
            counts = yaw.read_axis_counts()
            emit("sample", phase=phase, counts=counts, delta=counts - start,
                 angle_error_counts=round(angle_error_counts(), 2), **extra)
            if abs(counts - start) > GUARD_COUNTS:
                raise RuntimeError(f"guard exceeded at {counts}")
            return counts

        try:
            samples = [yaw.read_axis_counts() for _ in range(3)]
            if max(samples) - min(samples) > 2:
                emit("abort", reason="yaw not static", samples=samples)
                return 2
            start = samples[-1]
            snapshot("disabled")

            ack = bus.send_command(YAW, 0xF3, [1], response_expected=True, expected_response_len=1)
            if ack != b"\x01":
                emit("abort", reason="enable not acknowledged", ack=list(ack))
                return 2
            time.sleep(0.3)
            enabled = snapshot("enabled")

            legs = (
                ("f5_out", 0xF5, move_payload(enabled + STEP_COUNTS), enabled + STEP_COUNTS),
                ("f4_back", 0xF4, move_payload(-STEP_COUNTS), None),
            )
            for phase, func, payload, target in legs:
                before = yaw.read_axis_counts()
                expected = target if target is not None else before - STEP_COUNTS
                bus.send_command(YAW, func, payload, response_expected=False)
                emit("sent", phase=phase, func=f"0x{func:02X}", expected_counts=expected)
                deadline = time.monotonic() + OBSERVE_S
                while time.monotonic() < deadline:
                    counts = snapshot(phase, miss=expected - yaw.read_axis_counts())
                    time.sleep(0.05)
                emit("leg_end", phase=phase, expected_counts=expected, counts=counts, miss=expected - counts)
            return 0
        except RuntimeError as exc:
            emit("abort", reason=str(exc))
            return 3
        finally:
            for func, payload in ((0xF5, F5_STOP_NOW), (0xF7, []), (0xF3, [0])):
                try:
                    bus.send_command(YAW, func, payload, response_expected=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"cleanup 0x{func:02X} failed: {exc}", flush=True)
            try:
                emit("end", counts=yaw.read_axis_counts())
            except Exception as exc:  # noqa: BLE001
                print(f"final read failed: {exc}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
