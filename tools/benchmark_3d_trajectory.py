#!/usr/bin/env python3
"""Benchmark the offline PID gimbal model along a 3D spline trajectory.

The input describes target positions relative to the gimbal origin in metres,
using ``x=forward, y=right, z=up``.  The tool passes a cubic Hermite spline
through those points, converts each sampled point to yaw/pitch references, and
simulates the configured PID rate loop against the plant in a gimbal-response
fit report.  It never opens ZMQ, serial, or a camera device.

Examples:
    python -m tools.benchmark_3d_trajectory \
      examples/gimbal_trajectory_3d.json \
      --fit-report artifacts/gimbal_fit/codex_20260718_230016/fit_report.json \
      --output-dir artifacts/trajectory_benchmark/demo --plot

JSON input is either a list of ``[x_m, y_m, z_m]`` rows or an object with a
``points`` list.  A point may also be an object with ``x_m``, ``y_m``, ``z_m``
and optional ``t_s``.  CSV input needs x/y/z columns (``x_m`` etc. preferred)
and may include ``t_s``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from common.config_sync import expand_config_paths, load_merged_config
from common.control import ControlConfig
from common.gimbal.mks_servo42_rs485 import SpeedCommandDither


_EPS = 1e-9
_FRAME = "x_forward_y_right_z_up"


@dataclass(frozen=True)
class AxisPlant:
    """Continuous first-order equivalent reconstructed from a fit report."""

    a_u_positive: float
    a_u_negative: float
    a_f: float
    bias: float
    selected_model: str


@dataclass
class PidAxisState:
    integral: float = 0.0
    prev_error: Optional[float] = None
    prev_command: float = 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points", help="JSON or CSV 3D coordinate file")
    parser.add_argument("--fit-report", required=True, help="fit_report.json from tools.fit_gimbal_response")
    parser.add_argument("--config", default="configs/network.yaml", help="Base YAML configuration")
    parser.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
        help="Comma-separated YAML configs merged over --config",
    )
    parser.add_argument("--output-dir", default="artifacts/trajectory_benchmark", help="Output directory")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Duration for points without t_s; defaults to one second per segment",
    )
    parser.add_argument("--sample-hz", type=float, default=None, help="Simulation rate; defaults to control.loop_hz")
    parser.add_argument("--command-delay-s", type=float, default=0.0, help="Additional held command delay")
    parser.add_argument("--measurement-delay-s", type=float, default=0.0, help="Additional held angle measurement delay")
    parser.add_argument("--plot", action="store_true", help="Write trajectory_benchmark.png (requires matplotlib)")
    return parser.parse_args()


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


def _point_from_mapping(point: Mapping[str, Any], index: int) -> Tuple[float, float, float, Optional[float]]:
    def get_component(primary: str, alternate: str) -> float:
        if primary in point:
            return _finite_float(point[primary], f"point {index}.{primary}")
        if alternate in point:
            return _finite_float(point[alternate], f"point {index}.{alternate}")
        raise ValueError(f"point {index} needs {primary}, {alternate}, or both")

    x = get_component("x_m", "x")
    y = get_component("y_m", "y")
    z = get_component("z_m", "z")
    raw_t = point.get("t_s", point.get("t"))
    return x, y, z, None if raw_t in (None, "") else _finite_float(raw_t, f"point {index}.t_s")


def load_points(path: Path, duration_s: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Load control points and return knot times plus an N-by-3 coordinate array."""

    suffix = path.suffix.lower()
    raw_points: list[Any]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_points = payload.get("points", []) if isinstance(payload, Mapping) else payload
        if not isinstance(raw_points, list):
            raise ValueError("JSON input must be a point list or an object containing points")
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_points = list(csv.DictReader(handle))
    else:
        raise ValueError("points must use .json or .csv")

    if len(raw_points) < 2:
        raise ValueError("at least two 3D points are required")

    parsed: list[Tuple[float, float, float, Optional[float]]] = []
    for index, point in enumerate(raw_points):
        if isinstance(point, Mapping):
            parsed.append(_point_from_mapping(point, index))
        elif isinstance(point, (list, tuple)) and len(point) in (3, 4):
            x = _finite_float(point[0], f"point {index}[0]")
            y = _finite_float(point[1], f"point {index}[1]")
            z = _finite_float(point[2], f"point {index}[2]")
            t = None if len(point) == 3 else _finite_float(point[3], f"point {index}[3]")
            parsed.append((x, y, z, t))
        else:
            raise ValueError(f"point {index} must be [x, y, z], [x, y, z, t], or an object")

    coordinates = np.asarray([[x, y, z] for x, y, z, _t in parsed], dtype=float)
    distances = np.linalg.norm(coordinates, axis=1)
    if np.any(distances <= _EPS):
        raise ValueError("a target coordinate cannot be at the gimbal origin")

    explicit_times = [t for _x, _y, _z, t in parsed]
    if any(t is None for t in explicit_times) and any(t is not None for t in explicit_times):
        raise ValueError("either provide t_s for every point or for none")
    if all(t is not None for t in explicit_times):
        times = np.asarray([float(t) for t in explicit_times], dtype=float)
        if abs(times[0]) > _EPS:
            raise ValueError("the first t_s must be 0")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("point t_s values must be strictly increasing")
    else:
        total = duration_s if duration_s is not None else float(len(parsed) - 1)
        if total <= 0.0:
            raise ValueError("duration_s must be positive")
        times = np.linspace(0.0, float(total), len(parsed))
    return times, coordinates


def cubic_hermite_spline(knots_s: np.ndarray, points: np.ndarray, sample_s: np.ndarray) -> np.ndarray:
    """Evaluate an interpolating cubic Hermite spline without SciPy."""

    if points.ndim != 2 or points.shape[1] != 3 or len(knots_s) != len(points):
        raise ValueError("knots and points must have matching N-by-3 shapes")
    tangents = np.empty_like(points)
    tangents[0] = (points[1] - points[0]) / (knots_s[1] - knots_s[0])
    tangents[-1] = (points[-1] - points[-2]) / (knots_s[-1] - knots_s[-2])
    for index in range(1, len(points) - 1):
        tangents[index] = (points[index + 1] - points[index - 1]) / (knots_s[index + 1] - knots_s[index - 1])

    result = np.empty((len(sample_s), 3), dtype=float)
    for row, now_s in enumerate(sample_s):
        segment = min(max(int(np.searchsorted(knots_s, now_s, side="right") - 1), 0), len(points) - 2)
        t0, t1 = knots_s[segment], knots_s[segment + 1]
        width = t1 - t0
        q = min(max((now_s - t0) / width, 0.0), 1.0)
        q2, q3 = q * q, q * q * q
        h00 = 2.0 * q3 - 3.0 * q2 + 1.0
        h10 = q3 - 2.0 * q2 + q
        h01 = -2.0 * q3 + 3.0 * q2
        h11 = q3 - q2
        result[row] = h00 * points[segment] + h10 * width * tangents[segment] + h01 * points[segment + 1] + h11 * width * tangents[segment + 1]
    return result


def coordinates_to_axes(coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert x-forward/y-right/z-up positions into continuous yaw/pitch radians."""

    horizontal = np.hypot(coordinates[:, 0], coordinates[:, 1])
    if np.any(np.hypot(horizontal, coordinates[:, 2]) <= _EPS):
        raise ValueError("spline passes through the gimbal origin")
    yaw = np.unwrap(np.arctan2(coordinates[:, 1], coordinates[:, 0]))
    pitch = np.arctan2(coordinates[:, 2], horizontal)
    return yaw, pitch


def _load_axis_plant(report: Mapping[str, Any], axis: str) -> AxisPlant:
    try:
        entry = report["axes"][axis]
        params = entry["parameters"]
        a_u = _finite_float(params["a_u"], f"{axis}.a_u")
        a_f = _finite_float(params["a_f"], f"{axis}.a_f")
        bias = _finite_float(params.get("bias", 0.0), f"{axis}.bias")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"fit report has no usable {axis} plant") from exc
    if a_f <= 0.0 or a_u <= 0.0:
        raise ValueError(f"{axis} plant needs positive a_u and a_f")

    a_u_positive = a_u
    a_u_negative = a_u
    for candidate in entry.get("model_comparison", []):
        if not isinstance(candidate, Mapping) or candidate.get("model") != entry.get("selected_model"):
            continue
        coeffs = candidate.get("coefficients")
        if not isinstance(coeffs, Mapping):
            continue
        c_omega = _finite_float(coeffs.get("c_omega", 0.0), f"{axis}.c_omega")
        c_pos = coeffs.get("c_u_pos")
        c_neg = coeffs.get("c_u_neg")
        if 0.0 <= c_omega < 1.0 and c_pos is not None and c_neg is not None:
            scale = a_f / max(1.0 - c_omega, _EPS)
            a_u_positive = _finite_float(c_pos, f"{axis}.c_u_pos") * scale
            a_u_negative = -_finite_float(c_neg, f"{axis}.c_u_neg") * scale
        break
    return AxisPlant(a_u_positive, a_u_negative, a_f, bias, str(entry.get("selected_model", "unknown")))


def _held_value(times: Sequence[float], values: Sequence[float], query_s: float, default: float) -> float:
    index = int(np.searchsorted(np.asarray(times), query_s, side="right") - 1)
    return default if index < 0 else float(values[index])


def _pid_command(
    state: PidAxisState,
    *,
    error: float,
    dt_s: float,
    kp: float,
    ki: float,
    kd: float,
    rate_limit: float,
    accel_limit: float,
) -> Tuple[float, float, bool, bool]:
    derivative = 0.0 if state.prev_error is None else (error - state.prev_error) / dt_s
    if ki <= 0.0:
        state.integral = 0.0
    else:
        state.integral += error * dt_s
        max_integral = rate_limit / max(ki, 1e-6)
        state.integral = min(max(state.integral, -max_integral), max_integral)
    raw = kp * error + kd * derivative + ki * state.integral
    desired = min(max(raw, -rate_limit), rate_limit)
    if accel_limit > 0.0:
        delta = accel_limit * dt_s
        command = min(max(desired, state.prev_command - delta), state.prev_command + delta)
    else:
        command = desired
    rate_limited = abs(raw - desired) > 1e-12
    slew_limited = abs(desired - command) > 1e-12
    state.prev_error = error
    state.prev_command = command
    return command, raw, rate_limited, slew_limited


def _step_plant(theta: float, omega: float, command: float, plant: AxisPlant, dt_s: float) -> Tuple[float, float]:
    """Exact zero-order-hold step for theta_dot=omega, omega_dot=a_u*u-a_f*omega+bias."""

    a_u = plant.a_u_positive if command >= 0.0 else plant.a_u_negative
    steady_rate = (a_u * command + plant.bias) / plant.a_f
    decay = math.exp(-plant.a_f * dt_s)
    rate_integral = (1.0 - decay) / plant.a_f
    next_theta = theta + omega * rate_integral + steady_rate * (dt_s - rate_integral)
    next_omega = decay * omega + (1.0 - decay) * steady_rate
    return next_theta, next_omega


def _axis_metrics(error: np.ndarray, command: np.ndarray, rate_limited: np.ndarray, slew_limited: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(error)
    return {
        "rms_error_rad": float(math.sqrt(float(np.mean(np.square(error))))),
        "mae_rad": float(np.mean(absolute)),
        "p95_abs_error_rad": float(np.percentile(absolute, 95)),
        "max_abs_error_rad": float(np.max(absolute)),
        "final_error_rad": float(error[-1]),
        "max_abs_command_rad_s": float(np.max(np.abs(command))),
        "rate_saturation_fraction": float(np.mean(rate_limited)),
        "slew_saturation_fraction": float(np.mean(slew_limited)),
    }


def simulate_pid_benchmark(
    times_s: np.ndarray,
    reference_yaw: np.ndarray,
    reference_pitch: np.ndarray,
    plants: Mapping[str, AxisPlant],
    control_cfg: ControlConfig,
    command_delay_s: float,
    measurement_delay_s: float,
    gear_ratios: Optional[Mapping[str, float]] = None,
) -> dict[str, np.ndarray]:
    if command_delay_s < 0.0 or measurement_delay_s < 0.0:
        raise ValueError("delays must be non-negative")
    if len(times_s) < 2:
        raise ValueError("at least two simulation samples are required")
    dt_s = float(times_s[1] - times_s[0])
    if dt_s <= 0.0 or not np.allclose(np.diff(times_s), dt_s):
        raise ValueError("simulation times must use a fixed positive interval")

    result: dict[str, np.ndarray] = {"t_s": times_s.copy(), "ref_yaw_rad": reference_yaw.copy(), "ref_pitch_rad": reference_pitch.copy()}
    for axis, reference in (("yaw", reference_yaw), ("pitch", reference_pitch)):
        pid = control_cfg.pid
        kp = pid.kp.yaw if axis == "yaw" else pid.kp.pitch
        ki = pid.ki.yaw if axis == "yaw" else pid.ki.pitch
        kd = pid.kd.yaw if axis == "yaw" else pid.kd.pitch
        rate_limit = pid.rate_limits.yaw if axis == "yaw" else pid.rate_limits.pitch
        configured_accel = pid.accel_limits.yaw if axis == "yaw" else pid.accel_limits.pitch
        gimbal_accel = control_cfg.gimbal_accel_limits.yaw if axis == "yaw" else control_cfg.gimbal_accel_limits.pitch
        accel_limit = min(float(configured_accel), float(gimbal_accel))

        theta = np.zeros(len(times_s), dtype=float)
        omega = np.zeros(len(times_s), dtype=float)
        command = np.zeros(len(times_s), dtype=float)
        encoded_command = np.zeros(len(times_s), dtype=float)
        raw_command = np.zeros(len(times_s), dtype=float)
        error = np.zeros(len(times_s), dtype=float)
        rate_limited = np.zeros(len(times_s), dtype=bool)
        slew_limited = np.zeros(len(times_s), dtype=bool)
        state = PidAxisState()
        dither = SpeedCommandDither(float((gear_ratios or {}).get(axis, 1.0)))
        state_times: list[float] = [float(times_s[0])]
        state_values: list[float] = [0.0]
        command_times: list[float] = []
        encoded_command_values: list[float] = []
        for index, now_s in enumerate(times_s[:-1]):
            measurement = _held_value(state_times, state_values, float(now_s - measurement_delay_s), 0.0)
            error[index] = float(reference[index] - measurement)
            cmd, raw, at_rate_limit, at_slew_limit = _pid_command(
                state,
                error=error[index],
                dt_s=dt_s,
                kp=float(kp),
                ki=float(ki),
                kd=float(kd),
                rate_limit=float(rate_limit),
                accel_limit=accel_limit,
            )
            command[index] = cmd
            encoded_command[index] = dither.quantize(cmd)
            raw_command[index] = raw
            rate_limited[index] = at_rate_limit
            slew_limited[index] = at_slew_limit
            command_times.append(float(now_s))
            encoded_command_values.append(float(encoded_command[index]))
            applied = _held_value(command_times, encoded_command_values, float(now_s - command_delay_s), 0.0)
            theta[index + 1], omega[index + 1] = _step_plant(theta[index], omega[index], applied, plants[axis], dt_s)
            state_times.append(float(times_s[index + 1]))
            state_values.append(float(theta[index + 1]))
        error[-1] = reference[-1] - _held_value(state_times, state_values, float(times_s[-1] - measurement_delay_s), 0.0)
        command[-1] = command[-2]
        encoded_command[-1] = encoded_command[-2]
        raw_command[-1] = raw_command[-2]
        rate_limited[-1] = rate_limited[-2]
        slew_limited[-1] = slew_limited[-2]
        result.update(
            {
                f"theta_{axis}_rad": theta,
                f"omega_{axis}_rad_s": omega,
                f"command_{axis}_rad_s": command,
                f"encoded_command_{axis}_rad_s": encoded_command,
                f"raw_command_{axis}_rad_s": raw_command,
                f"error_{axis}_rad": error,
                f"rate_limited_{axis}": rate_limited,
                f"slew_limited_{axis}": slew_limited,
            }
        )
    return result


def _direction(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)))


def build_summary(sim: Mapping[str, np.ndarray], plants: Mapping[str, AxisPlant], command_delay_s: float, measurement_delay_s: float) -> dict[str, Any]:
    target = _direction(sim["ref_yaw_rad"], sim["ref_pitch_rad"])
    actual = _direction(sim["theta_yaw_rad"], sim["theta_pitch_rad"])
    dot = np.clip(np.sum(target * actual, axis=1), -1.0, 1.0)
    pointing_error = np.arccos(dot)
    return {
        "format": "idcs.trajectory_closed_loop_benchmark",
        "version": 1,
        "coordinate_frame": _FRAME,
        "offline_only": True,
        "simulation": {
            "samples": int(len(sim["t_s"])),
            "duration_s": float(sim["t_s"][-1]),
            "sample_hz": float(1.0 / (sim["t_s"][1] - sim["t_s"][0])),
            "command_delay_s": command_delay_s,
            "measurement_delay_s": measurement_delay_s,
        },
        "plant": {
            axis: {
                "selected_model": plant.selected_model,
                "a_u_positive": plant.a_u_positive,
                "a_u_negative": plant.a_u_negative,
                "a_f": plant.a_f,
                "bias": plant.bias,
            }
            for axis, plant in plants.items()
        },
        "axes": {
            axis: _axis_metrics(sim[f"error_{axis}_rad"], sim[f"command_{axis}_rad_s"], sim[f"rate_limited_{axis}"], sim[f"slew_limited_{axis}"])
            for axis in ("yaw", "pitch")
        },
        "pointing_error": {
            "rms_rad": float(math.sqrt(float(np.mean(np.square(pointing_error))))),
            "mae_rad": float(np.mean(pointing_error)),
            "p95_rad": float(np.percentile(pointing_error, 95)),
            "max_rad": float(np.max(pointing_error)),
            "final_rad": float(pointing_error[-1]),
        },
    }


def _write_csv(path: Path, coordinates: np.ndarray, sim: Mapping[str, np.ndarray]) -> None:
    fields = [
        "t_s", "target_x_m", "target_y_m", "target_z_m", "ref_yaw_rad", "ref_pitch_rad",
        "theta_yaw_rad", "theta_pitch_rad", "omega_yaw_rad_s", "omega_pitch_rad_s",
        "command_yaw_rad_s", "command_pitch_rad_s", "encoded_command_yaw_rad_s", "encoded_command_pitch_rad_s", "raw_command_yaw_rad_s", "raw_command_pitch_rad_s",
        "error_yaw_rad", "error_pitch_rad", "rate_limited_yaw", "rate_limited_pitch", "slew_limited_yaw", "slew_limited_pitch",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(sim["t_s"])):
            writer.writerow({
                "t_s": f"{sim['t_s'][index]:.9f}",
                "target_x_m": f"{coordinates[index, 0]:.9f}", "target_y_m": f"{coordinates[index, 1]:.9f}", "target_z_m": f"{coordinates[index, 2]:.9f}",
                **{field: (int(sim[field][index]) if sim[field].dtype == bool else f"{sim[field][index]:.9f}") for field in fields[4:]},
            })


def _write_plot(path: Path, sim: Mapping[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires matplotlib; install with pip install -e .[sysid]") from exc
    t = sim["t_s"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for axis, color in (("yaw", "tab:blue"), ("pitch", "tab:orange")):
        axes[0].plot(t, sim[f"ref_{axis}_rad"], "--", color=color, label=f"{axis} reference")
        axes[0].plot(t, sim[f"theta_{axis}_rad"], color=color, label=f"{axis} actual")
        axes[1].plot(t, sim[f"error_{axis}_rad"], color=color, label=f"{axis} error")
        axes[2].plot(t, sim[f"command_{axis}_rad_s"], color=color, label=f"{axis} command")
    axes[0].set_ylabel("angle (rad)")
    axes[1].set_ylabel("error (rad)")
    axes[2].set_ylabel("rate command (rad/s)")
    axes[2].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    point_path = Path(args.points)
    fit_path = Path(args.fit_report)
    report = json.loads(fit_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("fit report must be a JSON object")
    knot_s, control_points = load_points(point_path, args.duration_s)

    merged = load_merged_config(expand_config_paths(args.config, args.config_extra))
    control_cfg = ControlConfig.from_raw_config(merged, (1280, 720))
    gimbal_cfg = merged.get("gimbal")
    gimbal_cfg = gimbal_cfg if isinstance(gimbal_cfg, Mapping) else {}
    gear_ratios = {
        "yaw": float(gimbal_cfg.get("yaw_gear_ratio", 1.0)),
        "pitch": float(gimbal_cfg.get("pitch_gear_ratio", 1.0)),
    }
    sample_hz = float(args.sample_hz if args.sample_hz is not None else (control_cfg.loop_hz or 50.0))
    if sample_hz <= 0.0:
        raise ValueError("sample_hz must be positive")
    dt_s = 1.0 / sample_hz
    sample_count = max(2, int(math.ceil(float(knot_s[-1]) / dt_s)) + 1)
    sample_s = np.arange(sample_count, dtype=float) * dt_s
    coordinates = cubic_hermite_spline(knot_s, control_points, sample_s)
    reference_yaw, reference_pitch = coordinates_to_axes(coordinates)
    plants = {axis: _load_axis_plant(report, axis) for axis in ("yaw", "pitch")}
    sim = simulate_pid_benchmark(sample_s, reference_yaw, reference_pitch, plants, control_cfg, float(args.command_delay_s), float(args.measurement_delay_s), gear_ratios)
    summary = build_summary(sim, plants, float(args.command_delay_s), float(args.measurement_delay_s))
    summary["controller"] = {
        "type": "pid_rate",
        "runtime_configured_controller": control_cfg.controller,
        "note": "This offline benchmark always evaluates control.pid; it does not simulate MPC.",
        "kp": {"yaw": control_cfg.pid.kp.yaw, "pitch": control_cfg.pid.kp.pitch},
        "ki": {"yaw": control_cfg.pid.ki.yaw, "pitch": control_cfg.pid.ki.pitch},
        "kd": {"yaw": control_cfg.pid.kd.yaw, "pitch": control_cfg.pid.kd.pitch},
        "rate_limits_rad_s": {"yaw": control_cfg.pid.rate_limits.yaw, "pitch": control_cfg.pid.rate_limits.pitch},
        "effective_accel_limits_rad_s2": {
            "yaw": min(control_cfg.pid.accel_limits.yaw, control_cfg.gimbal_accel_limits.yaw),
            "pitch": min(control_cfg.pid.accel_limits.pitch, control_cfg.gimbal_accel_limits.pitch),
        },
    }
    summary["inputs"] = {"points": str(point_path), "fit_report": str(fit_path), "config": args.config, "config_extra": args.config_extra}
    summary["trajectory"] = {"control_points": int(len(control_points)), "knot_duration_s": float(knot_s[-1]), "reference_max_abs_yaw_rad": float(np.max(np.abs(reference_yaw))), "reference_max_abs_pitch_rad": float(np.max(np.abs(reference_pitch)))}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trajectory_closed_loop.csv", coordinates, sim)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.plot:
        _write_plot(output_dir / "trajectory_benchmark.png", sim)
    print(f"Wrote benchmark: {output_dir / 'summary.json'}")
    print(f"Pointing error: rms={summary['pointing_error']['rms_rad']:.5f} rad, p95={summary['pointing_error']['p95_rad']:.5f} rad")
    return 0


def main() -> int:
    try:
        return run(_parse_args())
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
