"""F5 step 2: one bounded yaw F5 move and return, on an unloaded bench.

Touches yaw (addr 1) only. Moves +0.01 rad (26 counts) at 1 RPM, holds,
returns to the start counts. Any count outside start +/- GUARD_COUNTS
triggers F7 + disable. Ends with F5 stop and yaw disable.
"""

import json
import sys
import time

from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus

YAW = 1
STEP_COUNTS = 26          # 0.01 rad at 16384 counts/rev, 1:1
GUARD_COUNTS = 150        # ~0.058 rad hard software guard around start
SPEED_RPM = 1
ACC = 2
SETTLE_TOL = 6           # one microstep at subdivision 16 is 5.12 counts
MOVE_TIMEOUT_S = 3.0


def f5_move(target: int) -> list[int]:
    if not -(2**31) <= target < 2**31:
        raise ValueError("target outside int32")
    axis = target.to_bytes(4, "big", signed=True)
    return [(SPEED_RPM >> 8) & 0xFF, SPEED_RPM & 0xFF, ACC, *axis]


F5_STOP_NOW = [0, 0, 0, 0, 0, 0, 0]
log = []


def emit(event: str, **fields) -> None:
    record = {"event": event, "monotonic_ns": time.monotonic_ns(), **fields}
    log.append(record)
    print(json.dumps(record), flush=True)


def main() -> int:
    with RS485Bus("/dev/ttyCH341USB0", baudrate=38400, timeout=0.12, max_retries=1) as bus:
        yaw = MksServo42Axis(bus, addr=YAW)
        try:
            samples = [yaw.read_axis_counts() for _ in range(3)]
            if max(samples) - min(samples) > 2:
                emit("abort", reason="yaw not static before move", samples=samples)
                return 2
            start = samples[-1]
            emit("start", counts=start, status=yaw.status())

            ack = bus.send_command(YAW, 0xF3, [1], response_expected=True, expected_response_len=1)
            if ack != b"\x01":
                emit("abort", reason="enable not acknowledged", ack=list(ack))
                return 2

            for leg, target in (("out", start + STEP_COUNTS), ("back", start)):
                bus.send_command(YAW, 0xF5, f5_move(target), response_expected=False)
                emit("f5_sent", leg=leg, target=target)
                deadline = time.monotonic() + MOVE_TIMEOUT_S
                settled_since = None
                while time.monotonic() < deadline:
                    counts = yaw.read_axis_counts()
                    emit("sample", leg=leg, counts=counts, error=target - counts)
                    if abs(counts - start) > GUARD_COUNTS:
                        emit("abort", reason="guard exceeded", counts=counts)
                        return 3
                    if abs(target - counts) <= SETTLE_TOL:
                        settled_since = settled_since or time.monotonic()
                        if time.monotonic() - settled_since >= 0.3:
                            break
                    else:
                        settled_since = None
                    time.sleep(0.03)
                else:
                    emit("abort", reason="did not settle", leg=leg, target=target)
                    return 4
                emit("settled", leg=leg, target=target, counts=counts)
            return 0
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
