"""Hardware PID gain sweep on one uncoupled motor with step-count feedback.

Opens the RS485 port directly: stop ``gimbal_bridge`` and
``serial_io_service`` first. Moves a motor only with ``--execute``.

Each run tracks a Jetson-local synthetic target from
``tools.latency_gain_sweep`` search scenarios. The motor's 0x33 step count
(microsteps, 3200/rev at 16x) is read every tick and is the angle source. A
camera is emulated as in the simulation: every 1/``fps`` s the bearing error
target(t) - angle(t) is captured and becomes visible to the controller
``latency`` later. The real ``BasicPID`` runs at ``tick_hz``; its rate is
sent as a bridge-style timed F6 (integer RPM truncated toward zero, acc,
100 ms firmware timer). Each run's angle is relative to its own start and a
``--guard-rad`` excursion aborts with F7. Every exit sends F6 zero, F7, and
disable.

Output: one JSON line per run (settings and metrics) and optionally the
per-tick trace, so hardware cost curves can be compared with simulation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus, min_f6_speed_rad_s  # noqa: E402
from jetson.control.feedforward import TargetRateKalman  # noqa: E402
from jetson.control.pid import AxisPIDConfig, BasicPID, PIDInput  # noqa: E402
from jetson.control.timing import TimingVerdict  # noqa: E402
from tools.feedforward_sweep import fast_scenarios  # noqa: E402
from tools.latency_gain_sweep import search_scenarios  # noqa: E402

STEPS_PER_REV = 3200
RAD_PER_STEP = 2.0 * math.pi / STEPS_PER_REV
_VALID = TimingVerdict(True, "ok")


def f6_timed_payload(rate_rad_s: float, *, motor_sign: int, acc: int, max_rate: float,
                     runtime_ms: int = 100) -> list[int]:
    """Bridge-equivalent timed F6: measured-speed level, never above ``max_rate``."""
    speed = MksServo42Axis._encode_speed_payload(
        motor_sign * rate_rad_s, acc, 1.0, max_rate_rad_s=max_rate)
    units = max(1, math.ceil(runtime_ms / 10))
    return [*speed, *units.to_bytes(4, "big")]


class Motor:
    def __init__(self, bus: RS485Bus, addr: int) -> None:
        self.bus, self.addr = bus, addr

    def steps(self) -> int:
        return int.from_bytes(self.bus.send_command(self.addr, 0x33, expected_response_len=4), "big", signed=True)

    def send(self, func: int, payload: list[int]) -> None:
        self.bus.send_command(self.addr, func, payload, response_expected=False)

    def stop(self, acc: int) -> None:
        self.send(0xF6, [0, 0, acc, 0, 0, 0, 10])


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    for i in range(len(xs) - 1, 0, -1):
        if xs[i - 1] <= x:
            span = xs[i] - xs[i - 1]
            w = 0.0 if span <= 0 else min(max((x - xs[i - 1]) / span, 0.0), 1.0)
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def run_once(motor: Motor, scenario, *, kp: float, ki: float, kd: float, latency_s: float,
             args, trace: list | None, ff: tuple[float, float, float] = (0.0, 0.0, 0.4)) -> dict:
    """``ff`` = (rate_scale, predict, accel_sigma) as in tools.latency_gain_sweep."""
    pid = BasicPID(
        AxisPIDConfig(kp, ki, kd, integral_limit_rad_s=args.rate_limit,
                      rate_limit_rad_s=args.rate_limit, acceleration_limit_rad_s2=args.accel_limit),
        AxisPIDConfig(0, 0, 0, 1, 1, 1),
    )
    step_sign = args.step_sign
    origin = motor.steps()
    tick = 1.0 / args.tick_hz
    frame = 1.0 / args.fps
    rate_scale, predict, sigma = ff
    kalman = None
    if rate_scale or predict:
        kalman = TargetRateKalman(measurement_sigma_rad=0.002, acceleration_sigma_rad_s2=sigma,
                                  max_sample_age_s=latency_s + 0.002 + 2.0 / args.fps + tick,
                                  innovation_limit_rad=1.0)
    history_t: list[float] = []
    history_angle: list[float] = []
    pending: list[tuple[float, float, float, float]] = []
    latest_error = None
    latest_capture = None
    prev = None
    errors = []
    t0 = time.monotonic()
    next_tick = t0
    next_frame = t0
    angle = 0.0
    while True:
        now = time.monotonic()
        t = now - t0
        if t > scenario.duration_s:
            break
        if now < next_tick:
            time.sleep(min(next_tick - now, 0.002))
            continue
        next_tick += tick
        angle = step_sign * (motor.steps() - origin) * RAD_PER_STEP
        read_t = time.monotonic() - t0
        if abs(angle) > args.guard_rad:
            motor.send(0xF7, [])
            raise RuntimeError(f"guard exceeded: {angle:.3f} rad")
        history_t.append(read_t)
        history_angle.append(angle)
        while next_frame - t0 <= read_t:
            capture_t = next_frame - t0
            pending.append((capture_t + latency_s, scenario.target(capture_t) - angle, capture_t, angle))
            next_frame += frame
        while pending and pending[0][0] <= read_t:
            _, latest_error, latest_capture, cam_at_capture = pending.pop(0)
            if kalman is not None:
                kalman.observe(track_id=1, angle_rad=latest_error + cam_at_capture,
                               sample_ns=int(latest_capture * 1e9) + 1)
        rate = 0.0 if prev is None else (angle - prev[1]) / (read_t - prev[0])
        prev = (read_t, angle)
        command = 0.0
        if latest_error is not None:
            error, feedforward = latest_error, 0.0
            if kalman is not None and latest_capture is not None:
                eval_t = latest_capture + predict * (read_t - latest_capture)
                estimate = kalman.estimate(decision_ns=int(eval_t * 1e9) + 1, track_id=1)
                if estimate.valid:
                    if predict:
                        cam_eval = _interp(eval_t, history_t, history_angle)
                        error = estimate.position_rad - cam_eval
                    feedforward = rate_scale * estimate.rate_rad_s
            decision = pid.decide(PIDInput(
                decision_ns=int(read_t * 1e9) + 1, track_id=1, error_rad=(error, 0.0),
                gimbal_rate_rad_s=(rate, 0.0), timing=_VALID, safety_allowed=True, gimbal_valid=True,
                feedforward_rad_s=(feedforward, 0.0)))
            command = decision.yaw.final_rad_s
        motor.send(0xF6, f6_timed_payload(command, motor_sign=args.motor_sign, acc=args.acc,
                                          max_rate=args.rate_limit))
        true_error = scenario.target(read_t) - angle
        errors.append(true_error)
        if trace is not None:
            trace.append((round(read_t, 4), round(scenario.target(read_t), 6), round(angle, 6),
                          None if latest_error is None else round(latest_error, 6), round(command, 5)))
    motor.stop(args.acc)
    time.sleep(0.3)
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    return {"rms_error_rad": rms, "ticks": len(errors),
            "tick_hz_achieved": len(errors) / scenario.duration_s,
            "final_abs_error_rad": abs(errors[-1])}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/ttyCH341USB0")
    parser.add_argument("--baud", type=int, default=38400)
    parser.add_argument("--addr", type=int, required=True)
    parser.add_argument("--motor-sign", type=int, default=1, choices=(-1, 1),
                        help="sign mapping camera-axis rate to F6 direction")
    parser.add_argument("--step-sign", type=int, default=-1, choices=(-1, 1),
                        help="sign mapping 0x33 steps to angle (probe: +F6 lowers 0x33)")
    parser.add_argument("--kps", default="4,8,12,16")
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--latencies-ms", default="0,60,120")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--scenarios", default="step_pos_0p06,step_neg_0p06,ramp_0p05,sine_0p06_3s")
    parser.add_argument("--ff-configs", default="0:0:0.4",
                        help="comma list of rate_scale:predict:accel_sigma")
    parser.add_argument("--kp-by-latency", default="",
                        help="latency_ms:kp pairs; overrides --kps with one Kp per latency")
    parser.add_argument("--tick-hz", type=float, default=50.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--rate-limit", type=float, default=0.8,
                        help="cap on actual F6 speed (rad/s); must reach the slowest level")
    parser.add_argument("--accel-limit", type=float, default=3.5)
    parser.add_argument("--acc", type=int, default=10)
    parser.add_argument("--guard-rad", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if args.rate_limit < min_f6_speed_rad_s():
        parser.error(f"--rate-limit below the slowest nonzero F6 speed {min_f6_speed_rad_s():.3f} rad/s")
    by_name = {s.name: s for s in (*search_scenarios(), *fast_scenarios())}
    scenarios = [by_name[n] for n in args.scenarios.split(",")]
    ff_configs = [tuple(float(v) for v in item.split(":")) for item in args.ff_configs.split(",")]
    if args.kp_by_latency:
        pairs = [item.split(":") for item in args.kp_by_latency.split(",")]
        grid = [(float(kp), float(ms)) for ms, kp in pairs]
    else:
        grid = [(float(kp), float(ms)) for kp in args.kps.split(",") for ms in args.latencies_ms.split(",")]
    order = [(kp, ms, ff, rep) for kp, ms in grid for ff in ff_configs for rep in range(args.repeats)]
    random.Random(args.seed).shuffle(order)
    print(json.dumps({"event": "plan", "runs": len(order), "scenarios": [s.name for s in scenarios],
                      "est_s": round(len(order) * sum(s.duration_s + 0.5 for s in scenarios))}), flush=True)
    if not args.execute:
        return 0
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    with RS485Bus(args.port, baudrate=args.baud, timeout=0.12, max_retries=1) as bus:
        motor = Motor(bus, args.addr)
        try:
            if bus.send_command(args.addr, 0xF3, [1], expected_response_len=1) != b"\x01":
                raise RuntimeError("enable not acknowledged")
            time.sleep(0.3)
            for index, (kp, ms, ff, rep) in enumerate(order):
                per = {}
                for scenario in scenarios:
                    trace = [] if args.trace_dir else None
                    per[scenario.name] = run_once(motor, scenario, kp=kp, ki=args.ki, kd=args.kd,
                                                  latency_s=ms / 1000.0, args=args, trace=trace, ff=ff)
                    if trace is not None:
                        tag = "ff" + "_".join(f"{v:g}" for v in ff)
                        name = f"addr{args.addr}_kp{kp:g}_L{ms:g}_{tag}_r{rep}_{scenario.name}.json"
                        (args.trace_dir / name).write_text(json.dumps(trace))
                cost = sum(m["rms_error_rad"] for m in per.values()) / len(per)
                print(json.dumps({"event": "run", "index": index, "addr": args.addr, "kp": kp, "ki": args.ki,
                                  "kd": args.kd, "latency_ms": ms, "repeat": rep, "cost_rad": cost,
                                  "ff_rate_scale": ff[0], "ff_predict": ff[1], "ff_accel_sigma": ff[2],
                                  "scenarios": per}), flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"event": "abort", "reason": str(exc)}), flush=True)
            return 3
        finally:
            for func, payload in ((0xF6, [0, 0, 0, 0, 0, 0, 10]), (0xF7, []), (0xF3, [0])):
                try:
                    motor.send(func, payload)
                except Exception as exc:  # noqa: BLE001
                    print(f"cleanup 0x{func:02X} failed: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
