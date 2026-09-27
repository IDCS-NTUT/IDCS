"""Find the fastest F6 speed level and acceleration byte an axis follows without losing steps.

Step count (0x33) is the controller's position feedback, so a lost step is a
silent position error. This probe moves one motor through a grid of
acceleration bytes x speed levels and, for each move, compares the motor's own
step count with the independent magnetic encoder (0x31). A move whose two
deltas disagree by more than ``--loss-tolerance-counts`` lost steps.

Each grid point runs a forward and a reverse move of equal duration, so the
axis returns near its start; ``--guard-rad`` of encoder travel from the start
aborts. The first move (level 1, gentle acceleration) calibrates the sign and
scale between the two sensors instead of assuming them.

Opens the RS485 port directly: stop ``gimbal_bridge`` and ``serial_io_service``
first. Moves a motor only with ``--execute``. Every exit sends F6 zero, F7 and
disable.

Output: one JSON line per move plus a final ``summary`` line with, per
acceleration byte, the fastest level whose moves all kept step and encoder in
agreement.
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

from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus, f6_level_speed_rad_s  # noqa: E402

STEPS_PER_REV = 3200
COUNTS_PER_REV = 16384
COUNTS_PER_STEP = COUNTS_PER_REV / STEPS_PER_REV


def step_loss_counts(d_steps: int, d_counts: int, sign: int) -> float:
    """Disagreement between the step count and encoder deltas, in encoder counts."""
    return abs(sign * d_steps * COUNTS_PER_STEP - d_counts)


def calibrate_sign(d_steps: int, d_counts: int) -> int:
    """Sign mapping steps onto encoder counts, from a gentle reference move."""
    if d_steps == 0 or d_counts == 0:
        raise ValueError("calibration move produced no motion; check enable and wiring")
    ratio = d_counts / (d_steps * COUNTS_PER_STEP)
    if not 0.8 <= abs(ratio) <= 1.25:
        raise ValueError(f"calibration step/encoder scale {abs(ratio):.2f} is not ~1; check steps_per_rev")
    return 1 if ratio > 0 else -1


def summarize(moves: list[dict], levels: list[int], accs: list[int]) -> dict:
    """Per acceleration byte: fastest level with every move (both directions) in agreement.

    Levels are only credited up to the first failing level, so a pass above a
    failure does not count.
    """
    by_acc: dict[str, dict] = {}
    for acc in accs:
        passed_up_to = None
        first_fail = None
        for level in sorted(levels):
            trials = [m for m in moves if m["acc"] == acc and m["level"] == level]
            if not trials:
                break
            if all(not m["lost_steps"] for m in trials):
                passed_up_to = level
            else:
                first_fail = level
                break
        by_acc[str(acc)] = {
            "max_passing_level": passed_up_to,
            "first_failing_level": first_fail,
            "max_passing_speed_rad_s": None if passed_up_to is None else f6_level_speed_rad_s(passed_up_to),
        }
    return by_acc


class Motor:
    def __init__(self, bus: RS485Bus, addr: int) -> None:
        self.bus, self.addr = bus, addr

    def steps(self) -> int:
        return int.from_bytes(self.bus.send_command(self.addr, 0x33, expected_response_len=4), "big", signed=True)

    def counts(self) -> int:
        return int.from_bytes(self.bus.send_command(self.addr, 0x31, expected_response_len=6), "big", signed=True)

    def run(self, level: int, acc: int, runtime_s: float) -> None:
        """Timed F6: the firmware stops the move itself after ``runtime_s``,
        so a crashed probe cannot leave the motor running."""
        units = max(1, math.ceil(runtime_s * 100))  # 10 ms units
        payload = [*MksServo42Axis._encode_speed_level_payload(level, acc), *units.to_bytes(4, "big")]
        self.bus.send_command(self.addr, 0xF6, payload, response_expected=False)

    def stop(self, acc: int) -> None:
        self.bus.send_command(self.addr, 0xF6, [0, 0, acc, 0, 0, 0, 10], response_expected=False)


def move(motor: Motor, level: int, acc: int, *, move_s: float, settle_s: float) -> tuple[int, int]:
    s0, c0 = motor.steps(), motor.counts()
    motor.run(level, acc, move_s)
    time.sleep(move_s)
    motor.stop(acc)
    time.sleep(settle_s)
    return motor.steps() - s0, motor.counts() - c0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyCH341USB0")
    parser.add_argument("--baud", type=int, default=38400)
    parser.add_argument("--addr", type=int, required=True)
    parser.add_argument("--levels", default="1,2,4,6,8,10,12", help="F6 speed levels, ascending")
    parser.add_argument("--accel-bytes", default="10,50,150,255", help="F6 acceleration bytes")
    parser.add_argument("--move-s", type=float, default=0.3)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--loss-tolerance-counts", type=float, default=26.0,
                        help="step/encoder disagreement that counts as lost steps (26 ~ 5 microsteps)")
    parser.add_argument("--guard-rad", type=float, default=0.6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    levels = [int(v) for v in args.levels.split(",")]
    accs = [int(v) for v in args.accel_bytes.split(",")]
    if levels != sorted(levels) or min(levels) < 1 or not all(0 <= a <= 255 for a in accs):
        parser.error("--levels must ascend from >= 1; --accel-bytes in 0..255")
    worst_travel = max(f6_level_speed_rad_s(level) for level in levels) * args.move_s
    if worst_travel >= args.guard_rad:
        parser.error(f"one move can travel {worst_travel:.2f} rad >= --guard-rad; shorten --move-s")
    print(json.dumps({"event": "plan", "addr": args.addr, "levels": levels, "accel_bytes": accs,
                      "moves": 2 * len(levels) * len(accs) + 2,
                      "est_s": round((2 * len(levels) * len(accs) + 2) * (args.move_s + args.settle_s + 0.1))}),
          flush=True)
    if not args.execute:
        return 0
    moves: list[dict] = []
    with RS485Bus(args.port, baudrate=args.baud, timeout=0.12, max_retries=1) as bus:
        motor = Motor(bus, args.addr)
        try:
            if bus.send_command(args.addr, 0xF3, [1], expected_response_len=1) != b"\x01":
                raise RuntimeError("enable not acknowledged")
            time.sleep(0.3)
            start_counts = motor.counts()
            gentle = min(accs)
            d_steps, d_counts = move(motor, 1, gentle, move_s=args.move_s, settle_s=args.settle_s)
            sign = calibrate_sign(d_steps, d_counts)
            move(motor, -1, gentle, move_s=args.move_s, settle_s=args.settle_s)
            print(json.dumps({"event": "calibration", "sign": sign, "d_steps": d_steps, "d_counts": d_counts}),
                  flush=True)
            for acc in accs:
                for level in levels:
                    failed = False
                    for direction in (1, -1):
                        d_steps, d_counts = move(motor, direction * level, acc,
                                                 move_s=args.move_s, settle_s=args.settle_s)
                        loss = step_loss_counts(d_steps, d_counts, sign)
                        record = {"event": "move", "acc": acc, "level": level, "direction": direction,
                                  "d_steps": d_steps, "d_counts": d_counts, "disagreement_counts": loss,
                                  "lost_steps": loss > args.loss_tolerance_counts}
                        moves.append(record)
                        print(json.dumps(record), flush=True)
                        failed = failed or record["lost_steps"]
                        travel = abs(motor.counts() - start_counts) * 2 * math.pi / COUNTS_PER_REV
                        if travel > args.guard_rad:
                            raise RuntimeError(f"guard: {travel:.2f} rad from start")
                    if failed:
                        break  # faster levels at this acceleration are not credited
            print(json.dumps({"event": "summary", "addr": args.addr,
                              "loss_tolerance_counts": args.loss_tolerance_counts,
                              "by_accel_byte": summarize(moves, levels, accs)}), flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"event": "abort", "reason": str(exc)}), flush=True)
            return 3
        finally:
            for func, payload in ((0xF6, [0, 0, 0, 0, 0, 0, 10]), (0xF7, []), (0xF3, [0])):
                try:
                    bus.send_command(args.addr, func, payload, response_expected=False)
                except Exception as exc:  # noqa: BLE001
                    print(json.dumps({"event": "stop_failed", "reason": str(exc)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
