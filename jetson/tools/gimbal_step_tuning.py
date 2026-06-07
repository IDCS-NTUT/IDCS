"""Record a gimbal speed step response and fit MPC plant parameters.

This tool drives a short speed step on a single axis, samples encoder replies
via the serial I/O service, writes a CSV trace, and emits a YAML snippet for
control.mpc.plant (a_u, a_f) based on a least-squares fit.

Requires the serial I/O service to be running and exclusive control of the
motor commands.

Example:
    python -m jetson.tools.gimbal_step_tuning --axis yaw --rate 0.6 --step-s 1.0
    python -m jetson.tools.gimbal_step_tuning --axis yaw --rate 0.6 --step-s 1.0 --start-serial-io
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import zmq

from common.config_sync import expand_config_paths, load_merged_config
from common.gimbal.mks_servo42_rs485 import MksServo42Axis
from common.serial_io import SerialReplySubscriber, SerialUpdatePublisher
from common.shutdown import install_signal_handlers
from jetson.gimbal_bridge import (
    _DEFAULT_GIMBAL_ACCEL_LIMIT_RAD_S2,
    _mks_accel_byte_from_physical,
)

_LOG = logging.getLogger(__name__)


@dataclass
class AxisConfig:
    axis: str
    counts_per_rev: int
    gear_ratio: float
    rate_limit: float
    accel_byte: int
    command_addrs: list[int]
    command_signs: list[float]
    command_labels: list[str]
    encoder_addr: int
    encoder_sign: float
    angle_min_rad: Optional[float]
    angle_max_rad: Optional[float]
    respond_on_writes: bool


@dataclass
class EncoderState:
    last_angle: Optional[float] = None
    last_timestamp: Optional[float] = None
    last_counts: Optional[int] = None
    last_omega: Optional[float] = None
    last_reply_mono: Optional[float] = None


def _wrapped_delta(angle_now: float, angle_prev: float) -> float:
    return math.atan2(math.sin(angle_now - angle_prev), math.cos(angle_now - angle_prev))


def _counts_to_rad(counts: int, *, counts_per_rev: int, gear_ratio: float) -> float:
    motor_revs = counts / float(counts_per_rev)
    axis_revs = motor_revs / float(gear_ratio)
    return axis_revs * 2.0 * math.pi


def _encode_speed_cmd(
    omega_rad_s: float,
    *,
    acc: int,
    gear_ratio: float,
    max_rate: float,
) -> tuple[int, int, int]:
    omega = max(min(omega_rad_s, max_rate), -max_rate)
    return MksServo42Axis._encode_speed_payload(omega, acc, gear_ratio)


def _apply_hard_angle_limit(
    rate_cmd: float,
    current_angle: Optional[float],
    angle_min: Optional[float],
    angle_max: Optional[float],
    axis: str,
) -> float:
    if current_angle is None:
        return rate_cmd
    if angle_max is not None and current_angle >= angle_max and rate_cmd > 0.0:
        _LOG.debug(
            "hard angle limit: %s at %.4f rad >= max %.4f rad; blocking positive command %.4f rad/s",
            axis,
            current_angle,
            angle_max,
            rate_cmd,
        )
        return 0.0
    if angle_min is not None and current_angle <= angle_min and rate_cmd < 0.0:
        _LOG.debug(
            "hard angle limit: %s at %.4f rad <= min %.4f rad; blocking negative command %.4f rad/s",
            axis,
            current_angle,
            angle_min,
            rate_cmd,
        )
        return 0.0
    return rate_cmd


def _reply_func_byte(reply: Mapping[str, Any]) -> Optional[int]:
    func = reply.get("func")
    if func is None:
        return None
    if isinstance(func, int):
        return func
    if isinstance(func, str):
        if func.lower().startswith("0x"):
            return int(func, 16)
        if func.upper().startswith("F"):
            return int(func, 16)
        return int(func)
    return None


def _build_command(
    *,
    cmd_id: str,
    func: str,
    addr: int,
    payload: Iterable[int],
    expect_reply: bool,
    expected_len: Optional[int],
    priority: str,
    target: str,
) -> dict:
    return {
        "cmd_id": cmd_id,
        "func": func,
        "addr": addr,
        "payload": list(payload),
        "expect_reply": expect_reply,
        "expected_len": expected_len,
        "priority": priority,
        "target": target,
    }


def _build_update(*, source: str, target: str, commands: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "type": "SerialUpdate",
        "source": source,
        "target": target,
        "fields": {},
        "commands": list(commands),
        "update_ts_ms": int(time.time() * 1000),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml", help="Path to YAML config")
    parser.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
        help="Comma-separated YAML configs merged over --config.",
    )
    parser.add_argument(
        "--axis",
        choices=["yaw", "pitch", "both"],
        default="both",
        help="Which axis to test (default: both in succession)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Step command rate (rad/s). Use negative for reverse.",
    )
    parser.add_argument(
        "--accel-byte",
        type=int,
        default=None,
        help="Override low-level MKS accel byte (0-255); defaults to gimbal.*_accel_limit_rad_s2 conversion.",
    )
    parser.add_argument("--sample-hz", type=float, default=50.0, help="Sampling rate (Hz)")
    parser.add_argument("--pre-roll-s", type=float, default=0.5, help="Pre-step duration (s)")
    parser.add_argument("--step-s", type=float, default=1.0, help="Step duration (s)")
    parser.add_argument("--post-roll-s", type=float, default=1.0, help="Post-step duration (s)")
    parser.add_argument(
        "--reverse-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run an equal-magnitude reverse step after the forward step",
    )
    parser.add_argument("--rest-s", type=float, default=0.5, help="Rest between repeats (s)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat count")
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path (default: logs/gimbal_step_<axis>_<ts>.csv)",
    )
    parser.add_argument(
        "--emit-plant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print control.mpc.plant YAML snippet after the run",
    )
    parser.add_argument(
        "--min-fit-samples",
        type=int,
        default=20,
        help="Minimum samples required for plant fit",
    )
    parser.add_argument(
        "--warn-no-encoder-s",
        type=float,
        default=1.0,
        help="Warn when encoder replies stall longer than this (s)",
    )
    parser.add_argument(
        "--assume-exclusive",
        action="store_true",
        help="Skip warning about exclusive gimbal control",
    )
    parser.add_argument(
        "--start-serial-io",
        action="store_true",
        help="Start serial_io_service as a child process",
    )
    parser.add_argument(
        "--serial-io-wait-s",
        type=float,
        default=0.5,
        help="Wait time after starting serial_io_service (s)",
    )
    return parser.parse_args()


def _start_serial_io_service(
    args: argparse.Namespace,
    *,
    gimbal_cfg: Mapping[str, Any],
    net_cfg: Mapping[str, Any],
    serial_update_ep: str,
    serial_reply_ep: str,
) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "tools.serial_io_service"]

    # serial_io configuration now lives in configs/control.yaml.
    control_cfg_path = Path(__file__).resolve().parents[2] / "configs" / "control.yaml"
    if control_cfg_path.exists():
        cmd.extend(["--config", str(control_cfg_path)])

    port = gimbal_cfg.get("serial_port")
    baud = gimbal_cfg.get("baudrate")
    timeout = gimbal_cfg.get("timeout")
    retries = gimbal_cfg.get("retries")
    if port:
        cmd.extend(["--port", str(port)])
    if baud is not None:
        cmd.extend(["--baud", str(int(baud))])
    if timeout is not None:
        cmd.extend(["--timeout", str(float(timeout))])
    if retries is not None:
        cmd.extend(["--retries", str(int(retries))])

    serial_cmd_ep = net_cfg.get("zmq_serial_cmd")
    if serial_cmd_ep:
        cmd.extend(["--command-endpoint", str(serial_cmd_ep)])
    if serial_update_ep:
        cmd.extend(["--update-endpoint", str(serial_update_ep)])
    if serial_reply_ep:
        cmd.extend(["--reply-endpoint", str(serial_reply_ep)])

    _LOG.info("starting serial_io_service")
    _LOG.debug("serial_io_service cmd: %s", " ".join(cmd))
    return subprocess.Popen(cmd)


def _stop_serial_io_service(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _LOG.info("stopping serial_io_service (pid=%d)", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        _LOG.warning("serial_io_service did not exit; killing")
        proc.kill()
        proc.wait(timeout=2.0)


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _accel_byte_from_limit(value: Any, *, key: str) -> int:
    raw = _DEFAULT_GIMBAL_ACCEL_LIMIT_RAD_S2 if value is None else value
    try:
        accel = float(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"gimbal.{key} must be a positive finite number") from exc
    if not math.isfinite(accel) or accel <= 0.0:
        raise SystemExit(f"gimbal.{key} must be a positive finite number")
    return _mks_accel_byte_from_physical(accel)


def _build_axis_config(cfg: Mapping[str, Any], axis: str, accel_override: Optional[int]) -> AxisConfig:
    gimbal_cfg = cfg.get("gimbal")
    if not isinstance(gimbal_cfg, Mapping):
        raise SystemExit("config missing 'gimbal' section")

    counts_per_rev = int(gimbal_cfg.get("counts_per_rev", 0x4000))
    yaw_ratio = float(gimbal_cfg.get("yaw_gear_ratio", 1.0))
    pitch_ratio = float(gimbal_cfg.get("pitch_gear_ratio", 1.0))

    yaw_addr = int(gimbal_cfg.get("yaw_addr", 1))
    yaw_motor_sign = float(gimbal_cfg.get("yaw_motor_sign", 1.0))
    camstate_yaw_sign = float(gimbal_cfg.get("camstate_yaw_sign", 1.0))
    yaw_accel_byte = _accel_byte_from_limit(
        gimbal_cfg.get("yaw_accel_limit_rad_s2"),
        key="yaw_accel_limit_rad_s2",
    )
    yaw_rate_limit = float(gimbal_cfg.get("yaw_rate_limit_rad_s", 10.0))
    yaw_min_rad = _maybe_float(gimbal_cfg.get("yaw_min_rad"))
    yaw_max_rad = _maybe_float(gimbal_cfg.get("yaw_max_rad"))

    try:
        pitch_a_addr = int(gimbal_cfg["pitch_motor_a_addr"])
        pitch_b_addr = int(gimbal_cfg["pitch_motor_b_addr"])
    except KeyError as exc:
        raise SystemExit("gimbal.pitch_motor_a_addr and pitch_motor_b_addr are required") from exc

    pitch_a_sign = float(gimbal_cfg.get("pitch_motor_a_sign", 1.0))
    pitch_b_sign = float(gimbal_cfg.get("pitch_motor_b_sign", -1.0))
    camstate_pitch_sign = float(gimbal_cfg.get("camstate_pitch_sign", 1.0))
    pitch_accel_byte = _accel_byte_from_limit(
        gimbal_cfg.get("pitch_accel_limit_rad_s2"),
        key="pitch_accel_limit_rad_s2",
    )
    pitch_rate_limit = float(gimbal_cfg.get("pitch_rate_limit_rad_s", 10.0))
    pitch_min_rad = _maybe_float(gimbal_cfg.get("pitch_min_rad"))
    pitch_max_rad = _maybe_float(gimbal_cfg.get("pitch_max_rad"))
    pitch_authority = str(gimbal_cfg.get("pitch_encoder_authority", "a")).lower()
    respond_on_writes = bool(gimbal_cfg.get("respond_on_writes", True))

    if yaw_motor_sign == 0.0 or pitch_a_sign == 0.0 or pitch_b_sign == 0.0:
        raise SystemExit("gimbal motor signs must be non-zero")

    if axis == "yaw":
        accel_byte = yaw_accel_byte if accel_override is None else accel_override
        return AxisConfig(
            axis="yaw",
            counts_per_rev=counts_per_rev,
            gear_ratio=yaw_ratio,
            rate_limit=yaw_rate_limit,
            accel_byte=accel_byte,
            command_addrs=[yaw_addr],
            command_signs=[yaw_motor_sign],
            command_labels=["yaw"],
            encoder_addr=yaw_addr,
            encoder_sign=camstate_yaw_sign,
            angle_min_rad=yaw_min_rad,
            angle_max_rad=yaw_max_rad,
            respond_on_writes=respond_on_writes,
        )

    if pitch_authority not in {"a", "b"}:
        raise SystemExit("gimbal.pitch_encoder_authority must be 'a' or 'b'")
    encoder_addr = pitch_a_addr if pitch_authority == "a" else pitch_b_addr
    accel_byte = pitch_accel_byte if accel_override is None else accel_override
    return AxisConfig(
        axis="pitch",
        counts_per_rev=counts_per_rev,
        gear_ratio=pitch_ratio,
        rate_limit=pitch_rate_limit,
        accel_byte=accel_byte,
        command_addrs=[pitch_a_addr, pitch_b_addr],
        command_signs=[pitch_a_sign, pitch_b_sign],
        command_labels=["pitch_a", "pitch_b"],
        encoder_addr=encoder_addr,
        encoder_sign=camstate_pitch_sign,
        angle_min_rad=pitch_min_rad,
        angle_max_rad=pitch_max_rad,
        respond_on_writes=respond_on_writes,
    )


def _extract_counts(reply: Mapping[str, Any]) -> Optional[int]:
    parsed = reply.get("reply", {}).get("parsed")
    if isinstance(parsed, Mapping):
        counts = parsed.get("counts")
        if isinstance(counts, int):
            return counts
    raw = reply.get("reply", {}).get("bytes")
    if isinstance(raw, list) and len(raw) == 6:
        try:
            return int.from_bytes(bytes(raw), byteorder="big", signed=True)
        except Exception:
            return None
    return None


def _fit_mpc_plant(samples: Sequence[Mapping[str, Any]], min_samples: int) -> Optional[dict]:
    if len(samples) < 2:
        return None

    rows = []
    targets = []
    dt_values = []
    for idx in range(len(samples) - 1):
        s0 = samples[idx]
        s1 = samples[idx + 1]
        omega0 = s0.get("omega_rad_s")
        omega1 = s1.get("omega_rad_s")
        u0 = s0.get("cmd_rate_applied")
        t0 = s0.get("t_s")
        t1 = s1.get("t_s")
        if omega0 is None or omega1 is None or u0 is None or t0 is None or t1 is None:
            continue
        if not all(math.isfinite(float(val)) for val in (omega0, omega1, u0, t0, t1)):
            continue
        dt = float(t1) - float(t0)
        if dt <= 0.0:
            continue
        y = (float(omega1) - float(omega0)) / dt
        rows.append([float(omega0), float(u0), 1.0])
        targets.append(y)
        dt_values.append(dt)

    if len(targets) < min_samples:
        return None

    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat
    sse = float(np.sum(residuals * residuals))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    rmse = math.sqrt(sse / max(len(y), 1))
    r2 = 0.0 if sst <= 0.0 else max(0.0, 1.0 - sse / sst)

    a_f = -float(beta[0])
    a_u = float(beta[1])
    d_hat = float(beta[2])

    return {
        "a_u": a_u,
        "a_f": a_f,
        "d": d_hat,
        "rmse": rmse,
        "r2": r2,
        "n": len(y),
        "dt_mean": float(np.mean(dt_values)) if dt_values else 0.0,
    }


def _motor_rpm_from_cmd(u_rad_s: float, gear_ratio: float) -> float:
    """Convert axis command (rad/s) to motor RPM (signed).

    Uses the same conversion as the encoder/payload packing so the fitted
    regressor works in the actuator's native units.
    """
    return float(u_rad_s) * 60.0 / (2.0 * math.pi) * float(gear_ratio)


def _fit_mpc_plant_with_delay(
    samples: Sequence[Mapping[str, Any]],
    min_samples: int,
    *,
    gear_ratio: float,
    max_delay_s: float = 0.2,
    delay_step_s: Optional[float] = None,
) -> Optional[dict]:
    """Fit MPC plant with a simple discrete delay search.

    The tuner converts the controller rate command into motor RPM units for
    regression, searches over integer sample shifts up to `max_delay_s` to
    compensate transport/actuation lag, and returns the best-fit plant where
    `a_u` is converted back to controller units (per rad/s).
    """
    if len(samples) < 2:
        return None

    # Derive an intrinsic sample step from consecutive timestamps.
    times = [s.get("t_s") for s in samples if s.get("t_s") is not None]
    if len(times) < 2:
        return None
    dt_list = []
    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]
        if t0 is None or t1 is None:
            continue
        try:
            d = float(t1) - float(t0)
        except Exception:
            continue
        if d > 0.0:
            dt_list.append(d)
    if not dt_list:
        return None
    median_dt = float(np.median(np.asarray(dt_list)))
    step_s = delay_step_s if (delay_step_s is not None and delay_step_s > 0.0) else median_dt
    max_steps = max(0, int(math.ceil(max_delay_s / step_s)))

    # Build base arrays
    base_rows = []
    base_targets = []
    base_t = []
    base_u_motor = []
    base_omega = []
    for s in samples:
        omega = s.get("omega_rad_s")
        u_ctrl = s.get("cmd_rate_applied")
        t0 = s.get("t_s")
        if omega is None or u_ctrl is None or t0 is None:
            base_rows.append(None)
            base_targets.append(None)
            base_t.append(None)
            base_u_motor.append(None)
            base_omega.append(None)
            continue
        base_omega.append(float(omega))
        base_u_motor.append(_motor_rpm_from_cmd(float(u_ctrl), gear_ratio))
        base_t.append(float(t0))

    best = None
    best_rmse = float("inf")
    best_step = 0
    best_fit = None

    # Try shifts from 0..max_steps (shift = how many samples u lags behind omega)
    for shift in range(0, max_steps + 1):
        rows = []
        targets = []
        dt_values = []
        # Build pairs (i -> i+1) but use u at index i-shift (earlier command)
        for i in range(len(samples) - 1):
            omega0 = base_omega[i]
            omega1 = base_omega[i + 1]
            t0 = base_t[i]
            t1 = base_t[i + 1]
            u_idx = i - shift
            if omega0 is None or omega1 is None or t0 is None or t1 is None:
                continue
            if u_idx < 0 or u_idx >= len(base_u_motor):
                continue
            u_motor = base_u_motor[u_idx]
            if u_motor is None:
                continue
            dt = float(t1) - float(t0)
            if dt <= 0.0:
                continue
            y = (float(omega1) - float(omega0)) / dt
            rows.append([float(omega0), float(u_motor), 1.0])
            targets.append(y)
            dt_values.append(dt)

        if len(targets) < min_samples:
            continue

        X = np.asarray(rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        residuals = y - y_hat
        sse = float(np.sum(residuals * residuals))
        rmse = math.sqrt(sse / max(len(y), 1))
        sst = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 0.0 if sst <= 0.0 else max(0.0, 1.0 - sse / sst)

        a_f = -float(beta[0])
        a_u_motor = float(beta[1])
        d_hat = float(beta[2])

        if not all(math.isfinite(val) for val in (a_f, a_u_motor, d_hat, rmse, r2)):
            continue
        # Reject fits that imply a wildly unstable discrete pole.
        # For the simple omega dynamics, 0 < a_f < ~2/Ts keeps 1 - Ts*a_f
        # from flipping sign with huge magnitude.
        if a_f <= 0.0 or a_f >= 1.9 / max(step_s, 1e-9):
            continue

        if rmse < best_rmse:
            best_rmse = rmse
            best_step = shift
            best_fit = {
                "a_u_motor": a_u_motor,
                "a_f": a_f,
                "d": d_hat,
                "rmse": rmse,
                "r2": r2,
                "n": len(y),
                "dt_mean": float(np.mean(dt_values)) if dt_values else 0.0,
            }

    if best_fit is None:
        fallback = _fit_mpc_plant(samples, min_samples)
        if fallback is None:
            return None
        if not all(math.isfinite(float(fallback.get(key, 0.0))) for key in ("a_u", "a_f", "rmse", "r2")):
            return None
        if float(fallback["a_f"]) <= 0.0:
            return None
        if float(fallback["a_f"]) >= 1.9 / max(step_s, 1e-9):
            return None
        fallback["delay_s"] = 0.0
        return fallback

    # Convert a_u from motor-RPM units back to controller units (per rad/s).
    motor_per_ctrl = 60.0 / (2.0 * math.pi) * float(gear_ratio)
    a_u_controller = float(best_fit["a_u_motor"]) * motor_per_ctrl

    return {
        "a_u": a_u_controller,
        "a_f": best_fit["a_f"],
        "d": best_fit["d"],
        "rmse": best_fit["rmse"],
        "r2": best_fit["r2"],
        "n": best_fit["n"],
        "dt_mean": best_fit["dt_mean"],
        "delay_s": best_step * step_s,
    }


def _emit_plant_snippet(fit: Mapping[str, Any], *, axis: str) -> None:
    print(
        f"# MPC plant fit axis={axis} n={fit['n']} rmse={fit['rmse']:.6f} r2={fit['r2']:.3f} dt_mean={fit['dt_mean']:.4f}s"
    )
    print(f"# disturbance d={fit['d']:.6f} (not used in config)")
    if "delay_s" in fit:
        try:
            print(f"# estimated actuator delay={float(fit['delay_s']):.4f}s")
        except Exception:
            pass
    print("control:")
    print("  mpc:")
    print("    plant:")
    print(f"      a_u: {fit['a_u']:.6f}")
    print(f"      a_f: {fit['a_f']:.6f}")


def _resolve_axes(axis: str) -> list[str]:
    if axis == "both":
        return ["yaw", "pitch"]
    return [axis]


def _output_path_for_axis(
    base: Optional[str], *, axis: str, ts_label: int, multi_axis: bool
) -> Path:
    if base is None:
        return Path(f"logs/gimbal_step_{axis}_{ts_label}.csv")
    if not multi_axis:
        return Path(base)
    path = Path(base)
    if path.suffix:
        return path.with_name(f"{path.stem}_{axis}{path.suffix}")
    return Path(f"{path}_{axis}")


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if args.sample_hz <= 0.0:
        _LOG.error("--sample-hz must be > 0")
        return 2
    if args.repeat <= 0:
        _LOG.error("--repeat must be >= 1")
        return 2

    if not args.assume_exclusive:
        _LOG.warning(
            "Ensure gimbal_bridge/controller are stopped to avoid conflicting F6 commands. "
            "Use --assume-exclusive to suppress this warning."
        )

    config_paths = expand_config_paths(args.config, args.config_extra)
    cfg = load_merged_config(config_paths)

    net_raw = cfg.get("net")
    net_cfg: Mapping[str, Any] = net_raw if isinstance(net_raw, Mapping) else {}
    gimbal_raw = cfg.get("gimbal")
    gimbal_cfg: Mapping[str, Any] = gimbal_raw if isinstance(gimbal_raw, Mapping) else {}

    serial_target = str(gimbal_cfg.get("serial_target", "gimbal"))
    serial_update_ep = gimbal_cfg.get("serial_update_endpoint") or net_cfg.get(
        "zmq_serial_update"
    )
    if not serial_update_ep:
        serial_update_ep = "tcp://127.0.0.1:5571"
    serial_reply_ep = gimbal_cfg.get("serial_reply_endpoint") or net_cfg.get(
        "zmq_serial_reply"
    )
    if not serial_reply_ep:
        serial_reply_ep = "tcp://127.0.0.1:5572"

    axes = _resolve_axes(args.axis)
    multi_axis = len(axes) > 1
    ts_label = int(time.time())

    period_s = 1.0 / float(args.sample_hz)
    stop_event = install_signal_handlers()

    serial_proc: Optional[subprocess.Popen] = None
    ctx: Optional[zmq.Context] = None
    update_pub: Optional[SerialUpdatePublisher] = None
    reply_sub: Optional[SerialReplySubscriber] = None

    try:
        if args.start_serial_io:
            serial_proc = _start_serial_io_service(
                args,
                gimbal_cfg=gimbal_cfg,
                net_cfg=net_cfg,
                serial_update_ep=serial_update_ep,
                serial_reply_ep=serial_reply_ep,
            )
            if args.serial_io_wait_s > 0.0:
                time.sleep(args.serial_io_wait_s)
            if serial_proc.poll() is not None:
                _LOG.error("serial_io_service exited early; check logs")
                return 2

        ctx = zmq.Context()
        update_pub = SerialUpdatePublisher(serial_update_ep, ctx=ctx)
        reply_sub = SerialReplySubscriber(
            serial_reply_ep,
            topics=[f"serial.reply.{serial_target}"],
            ctx=ctx,
        )

        assert update_pub is not None
        assert reply_sub is not None

        for axis_name in axes:
            axis_cfg = _build_axis_config(cfg, axis_name, args.accel_byte)
            output_file = _output_path_for_axis(
                args.output,
                axis=axis_cfg.axis,
                ts_label=ts_label,
                multi_axis=multi_axis,
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)

            encoder_state = EncoderState()
            samples: list[dict] = []
            sample_idx = 0

            def send_speed_update(
                cmd_rate: float, *, phase: str, trial_idx: int, trial_time: float
            ) -> None:
                nonlocal sample_idx

                cmd_applied = _apply_hard_angle_limit(
                    cmd_rate,
                    encoder_state.last_angle,
                    axis_cfg.angle_min_rad,
                    axis_cfg.angle_max_rad,
                    axis_cfg.axis,
                )

                commands = []
                for addr, sign, label in zip(
                    axis_cfg.command_addrs,
                    axis_cfg.command_signs,
                    axis_cfg.command_labels,
                    strict=True,
                ):
                    payload = _encode_speed_cmd(
                        sign * cmd_applied,
                        acc=axis_cfg.accel_byte,
                        gear_ratio=axis_cfg.gear_ratio,
                        max_rate=axis_cfg.rate_limit,
                    )
                    cmd_id = f"speed:{label}:{time.time_ns()}"
                    commands.append(
                        _build_command(
                            cmd_id=cmd_id,
                            func="F6",
                            addr=addr,
                            payload=payload,
                            expect_reply=axis_cfg.respond_on_writes,
                            expected_len=1 if axis_cfg.respond_on_writes else None,
                            priority="high",
                            target=serial_target,
                        )
                    )

                commands.append(
                    _build_command(
                        cmd_id=f"enc:{axis_cfg.axis}:{time.time_ns()}",
                        func="0x31",
                        addr=axis_cfg.encoder_addr,
                        payload=[],
                        expect_reply=True,
                        expected_len=6,
                        priority="high",
                        target=serial_target,
                    )
                )

                update_sent = update_pub.send_update(
                    _build_update(
                        source="jetson.gimbal_step_tuning",
                        target=serial_target,
                        commands=commands,
                    )
                )
                if not update_sent:
                    _LOG.warning("serial update publish dropped")

                for reply in reply_sub.recv_nowait():
                    if _reply_func_byte(reply) != 0x31:
                        continue
                    if reply.get("addr") != axis_cfg.encoder_addr:
                        continue
                    counts = _extract_counts(reply)
                    if counts is None:
                        continue

                    now_mono = time.monotonic()
                    angle = axis_cfg.encoder_sign * _counts_to_rad(
                        counts,
                        counts_per_rev=axis_cfg.counts_per_rev,
                        gear_ratio=axis_cfg.gear_ratio,
                    )

                    omega = None
                    dt_s = None
                    if encoder_state.last_angle is not None and encoder_state.last_timestamp is not None:
                        dt_s = now_mono - encoder_state.last_timestamp
                        if dt_s > 0.0:
                            omega = _wrapped_delta(angle, encoder_state.last_angle) / dt_s

                    encoder_state.last_angle = angle
                    encoder_state.last_timestamp = now_mono
                    encoder_state.last_counts = counts
                    encoder_state.last_omega = omega
                    encoder_state.last_reply_mono = now_mono

                    row = {
                        "sample_idx": sample_idx,
                        "trial": trial_idx,
                        "phase": phase,
                        "t_s": now_mono - t0,
                        "trial_time_s": trial_time,
                        "cmd_rate": cmd_rate,
                        "cmd_rate_applied": cmd_applied,
                        "counts": counts,
                        "angle_rad": angle,
                        "omega_rad_s": omega,
                        "dt_s": dt_s,
                    }
                    writer.writerow(row)
                    csv_handle.flush()
                    samples.append(row)
                    sample_idx += 1

            with output_file.open("w", newline="", encoding="utf-8") as csv_handle:
                fieldnames = [
                    "sample_idx",
                    "trial",
                    "phase",
                    "t_s",
                    "trial_time_s",
                    "cmd_rate",
                    "cmd_rate_applied",
                    "counts",
                    "angle_rad",
                    "omega_rad_s",
                    "dt_s",
                ]
                writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
                writer.writeheader()

                t0 = time.monotonic()
                _LOG.info(
                    "starting step test axis=%s rate=%.3f rad/s repeats=%d output=%s",
                    axis_cfg.axis,
                    args.rate,
                    args.repeat,
                    output_file,
                )

                try:
                    for trial in range(args.repeat):
                        trial_start = time.monotonic()
                        _LOG.info("trial %d/%d: pre-roll", trial + 1, args.repeat)
                        _run_phase(
                            duration_s=args.pre_roll_s,
                            phase="pre",
                            cmd_rate=0.0,
                            trial_idx=trial,
                            trial_start=trial_start,
                            period_s=period_s,
                            send_fn=send_speed_update,
                            stop_event=stop_event,
                        )

                        _LOG.info("trial %d/%d: step (+)", trial + 1, args.repeat)
                        _run_phase(
                            duration_s=args.step_s,
                            phase="step_pos",
                            cmd_rate=args.rate,
                            trial_idx=trial,
                            trial_start=trial_start,
                            period_s=period_s,
                            send_fn=send_speed_update,
                            stop_event=stop_event,
                        )

                        _LOG.info("trial %d/%d: post-roll", trial + 1, args.repeat)
                        _run_phase(
                            duration_s=args.post_roll_s,
                            phase="post",
                            cmd_rate=0.0,
                            trial_idx=trial,
                            trial_start=trial_start,
                            period_s=period_s,
                            send_fn=send_speed_update,
                            stop_event=stop_event,
                        )

                        if stop_event.is_set():
                            break

                        if args.reverse_after:
                            _LOG.info("trial %d/%d: step (-)", trial + 1, args.repeat)
                            _run_phase(
                                duration_s=args.step_s,
                                phase="step_neg",
                                cmd_rate=-args.rate,
                                trial_idx=trial,
                                trial_start=trial_start,
                                period_s=period_s,
                                send_fn=send_speed_update,
                                stop_event=stop_event,
                            )

                            _LOG.info("trial %d/%d: post-roll after reverse", trial + 1, args.repeat)
                            _run_phase(
                                duration_s=args.post_roll_s,
                                phase="post_rev",
                                cmd_rate=0.0,
                                trial_idx=trial,
                                trial_start=trial_start,
                                period_s=period_s,
                                send_fn=send_speed_update,
                                stop_event=stop_event,
                            )

                            if stop_event.is_set():
                                break

                        if args.rest_s > 0.0 and trial < args.repeat - 1:
                            _LOG.info("trial %d/%d: rest", trial + 1, args.repeat)
                            _run_phase(
                                duration_s=args.rest_s,
                                phase="rest",
                                cmd_rate=0.0,
                                trial_idx=trial,
                                trial_start=trial_start,
                                period_s=period_s,
                                send_fn=send_speed_update,
                                stop_event=stop_event,
                            )

                        if stop_event.is_set():
                            break

                        if (
                            encoder_state.last_reply_mono is not None
                            and args.warn_no_encoder_s > 0.0
                            and (time.monotonic() - encoder_state.last_reply_mono)
                            > args.warn_no_encoder_s
                        ):
                            _LOG.warning(
                                "encoder replies have stalled for %.2f s",
                                time.monotonic() - encoder_state.last_reply_mono,
                            )

                except KeyboardInterrupt:
                    _LOG.info("interrupted; stopping")

            _send_zero_speed(update_pub, axis_cfg, serial_target)
            time.sleep(0.05)
            _send_zero_speed(update_pub, axis_cfg, serial_target)

            if args.emit_plant:
                fit = _fit_mpc_plant_with_delay(
                    samples,
                    args.min_fit_samples,
                    gear_ratio=axis_cfg.gear_ratio,
                    max_delay_s=0.2,
                )
                if fit is None:
                    _LOG.warning(
                        "not enough samples to fit MPC plant for axis=%s (need %d); skipping snippet",
                        axis_cfg.axis,
                        args.min_fit_samples,
                    )
                else:
                    if fit["a_u"] <= 0.0:
                        _LOG.warning("fit a_u=%.4f is not positive", fit["a_u"])
                    if fit["a_f"] < 0.0:
                        _LOG.warning("fit a_f=%.4f is negative", fit["a_f"])
                    _LOG.info(
                        "MPC plant fit axis=%s n=%d rmse=%.6f r2=%.3f delay_s=%.4f",
                        axis_cfg.axis,
                        fit.get("n", 0),
                        fit.get("rmse", float("nan")),
                        fit.get("r2", float("nan")),
                        float(fit.get("delay_s", 0.0)),
                    )
                    _emit_plant_snippet(fit, axis=axis_cfg.axis)

            if stop_event.is_set():
                break

        return 0
    finally:
        if update_pub is not None:
            update_pub.close()
        if reply_sub is not None:
            reply_sub.close()
        if ctx is not None:
            ctx.term()
        if serial_proc is not None:
            _stop_serial_io_service(serial_proc)


def _send_zero_speed(
    update_pub: SerialUpdatePublisher, axis_cfg: AxisConfig, serial_target: str
) -> None:
    commands = []
    for addr, sign, label in zip(
        axis_cfg.command_addrs,
        axis_cfg.command_signs,
        axis_cfg.command_labels,
        strict=True,
    ):
        payload = _encode_speed_cmd(
            0.0,
            acc=axis_cfg.accel_byte,
            gear_ratio=axis_cfg.gear_ratio,
            max_rate=axis_cfg.rate_limit,
        )
        cmd_id = f"speed:{label}:{time.time_ns()}"
        commands.append(
            _build_command(
                cmd_id=cmd_id,
                func="F6",
                addr=addr,
                payload=payload,
                expect_reply=axis_cfg.respond_on_writes,
                expected_len=1 if axis_cfg.respond_on_writes else None,
                priority="high",
                target=serial_target,
            )
        )

    update_pub.send_update(
        _build_update(
            source="jetson.gimbal_step_tuning",
            target=serial_target,
            commands=commands,
        )
    )


def _run_phase(
    *,
    duration_s: float,
    phase: str,
    cmd_rate: float,
    trial_idx: int,
    trial_start: float,
    period_s: float,
    send_fn,
    stop_event,
) -> None:
    if duration_s <= 0.0:
        return
    end_time = time.monotonic() + duration_s
    next_tick = time.monotonic()
    while not stop_event.is_set() and time.monotonic() < end_time:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.01, next_tick - now))
            continue
        trial_time = now - trial_start
        send_fn(cmd_rate, phase=phase, trial_idx=trial_idx, trial_time=trial_time)
        next_tick += period_s


if __name__ == "__main__":
    sys.exit(main())
