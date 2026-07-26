"""Run a 3D spline tracking benchmark against the physical RS485 gimbal.

Coordinates are target positions relative to the gimbal origin in metres using
``x=forward, y=right, z=up``.  The first point defines the current physical
aim; later points become yaw/pitch offsets from that aim.  The tool closes a
PID rate loop on the live authoritative encoders and writes a CSV/JSON record.

This utility opens the configured serial port directly.  Stop gimbal_bridge
and serial_io_service first.  Motion requires both --execute and
--assume-exclusive; without --execute it only writes a reference-plan CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml
import zmq

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from common.config_sync import expand_config_paths, load_merged_config
from common.control import ControlConfig
from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus, SpeedCommandDither
from common.serial_io import SerialReplySubscriber, SerialUpdatePublisher
from common.shutdown import install_signal_handlers
from jetson.tools.gimbal_response_sweep import (
    AxisConfig,
    _apply_hard_angle_limit,
    _build_axis_config,
    _build_command,
    _build_update,
    _counts_to_rad,
    _extract_counts,
    _reply_func_byte,
    _send_zero_speed,
)
from tools.benchmark_3d_trajectory import (
    PidAxisState,
    _pid_command,
    coordinates_to_axes,
    cubic_hermite_spline,
    load_points,
)


CSV_FIELDS = [
    "sample_idx", "elapsed_s", "phase", "target_x_m", "target_y_m", "target_z_m",
    "ref_yaw_rad", "ref_pitch_rad", "measured_yaw_rad", "measured_pitch_rad",
    "error_yaw_rad", "error_pitch_rad", "command_yaw_rad_s", "command_pitch_rad_s",
    "encoded_command_yaw_rad_s", "encoded_command_pitch_rad_s",
    "raw_command_yaw_rad_s", "raw_command_pitch_rad_s", "rate_limited_yaw",
    "rate_limited_pitch", "slew_limited_yaw", "slew_limited_pitch", "limit_blocked_yaw",
    "limit_blocked_pitch", "loop_dt_s",
]


@dataclass
class HardwareAxis:
    config: AxisConfig
    motors: list[tuple[MksServo42Axis, float]]
    pid_state: PidAxisState
    initial_angle: float = 0.0
    measured_angle: float = 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points", help="JSON or CSV 3D coordinate file")
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
    )
    parser.add_argument("--duration-s", type=float, default=None, help="Required only when points omit t_s")
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument("--hold-s", type=float, default=1.0, help="Continue tracking the final point for this duration")
    parser.add_argument("--max-command-rate-rad-s", type=float, default=0.30)
    parser.add_argument("--max-reference-offset-rad", type=float, default=0.35)
    parser.add_argument("--port", default=None, help="Override gimbal.serial_port")
    parser.add_argument("--baud", type=int, default=None, help="Override gimbal.baudrate")
    parser.add_argument("--timeout-s", type=float, default=None, help="Override gimbal.timeout")
    parser.add_argument("--retries", type=int, default=None, help="Override gimbal.retries")
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--manifest", default=None, help="JSON manifest path")
    parser.add_argument("--execute", action="store_true", help="Allow physical motion")
    parser.add_argument("--assume-exclusive", action="store_true", help="Confirm no bridge, serial_io_service, or other motor publisher is running")
    parser.add_argument(
        "--transport",
        choices=["serial-io", "direct"],
        default="serial-io",
        help="RS485 path; serial-io is the tested transport for this rig",
    )
    parser.add_argument("--operator-note", default="")
    return parser.parse_args()


def _axis_value(pair: Any, axis: str) -> float:
    return float(pair.yaw if axis == "yaw" else pair.pitch)


def _make_axis(bus: RS485Bus, config: AxisConfig) -> HardwareAxis:
    motors = [
        (
            MksServo42Axis(
                bus,
                addr,
                counts_per_rev=config.counts_per_rev,
                gear_ratio=config.gear_ratio,
                use_group_writes=False,
                # The deployed MKS controllers reply reliably to encoder reads but
                # not consistently to individual F6 writes.  Waiting for these
                # acknowledgements turns a valid speed command into a timeout.
                respond_on_writes=False,
            ),
            sign,
        )
        for addr, sign in zip(config.command_addrs, config.command_signs)
    ]
    return HardwareAxis(config=config, motors=motors, pid_state=PidAxisState())


def _read_axis(axis: HardwareAxis) -> float:
    authority = next(motor for motor, _sign in axis.motors if motor.addr == axis.config.encoder_addr)
    counts = authority.read_axis_counts()
    angle = axis.config.encoder_sign * _counts_to_rad(
        counts,
        counts_per_rev=axis.config.counts_per_rev,
        gear_ratio=axis.config.gear_ratio,
    )
    axis.measured_angle = angle
    return angle


def _send_axis_speed(axis: HardwareAxis, command_rad_s: float) -> None:
    for motor, sign in axis.motors:
        motor.command_speed(sign * command_rad_s, acc=axis.config.accel_byte, use_group=False)


def _enable_axes(axes: Sequence[HardwareAxis]) -> None:
    """Mirror the serial_io startup enable sequence before commanding speed."""

    for axis in axes:
        for motor, _sign in axis.motors:
            motor.enable(True, use_group=False)
    time.sleep(0.05)
    for axis in axes:
        for motor, _sign in axis.motors:
            if motor.status() == 0:
                raise RuntimeError(f"motor address {motor.addr} did not enable")


def _stop_axes(axes: Sequence[HardwareAxis]) -> None:
    for _ in range(3):
        for axis in axes:
            for motor, sign in axis.motors:
                with suppress(Exception):
                    motor.command_speed(0.0 * sign, acc=axis.config.accel_byte, use_group=False)
        time.sleep(0.04)


def _reference_at(
    knots_s: np.ndarray,
    points: np.ndarray,
    elapsed_s: float,
    yaw_zero: float,
    pitch_zero: float,
) -> tuple[np.ndarray, float, float]:
    now_s = min(max(0.0, elapsed_s), float(knots_s[-1]))
    coordinate = cubic_hermite_spline(knots_s, points, np.asarray([now_s]))[0]
    yaw, pitch = coordinates_to_axes(np.asarray([coordinate]))
    return coordinate, yaw_zero + float(yaw[0]), pitch_zero + float(pitch[0])


def _summary(rows: Sequence[Mapping[str, Any]], *, args: argparse.Namespace, axis_cfgs: Sequence[AxisConfig], status: str, error: Optional[str]) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=float) if rows else np.asarray([], dtype=float)

    def axis_metrics(axis: str) -> dict[str, Optional[float]]:
        err = values(f"error_{axis}_rad")
        if not len(err):
            return {"rms_error_rad": None, "mae_rad": None, "p95_abs_error_rad": None, "max_abs_error_rad": None}
        absolute = np.abs(err)
        return {
            "rms_error_rad": float(math.sqrt(float(np.mean(np.square(err))))),
            "mae_rad": float(np.mean(absolute)),
            "p95_abs_error_rad": float(np.percentile(absolute, 95)),
            "max_abs_error_rad": float(np.max(absolute)),
        }

    if rows:
        ref_yaw, ref_pitch = values("ref_yaw_rad"), values("ref_pitch_rad")
        yaw, pitch = values("measured_yaw_rad"), values("measured_pitch_rad")
        dots = np.clip(
            np.cos(ref_pitch) * np.cos(ref_yaw) * np.cos(pitch) * np.cos(yaw)
            + np.cos(ref_pitch) * np.sin(ref_yaw) * np.cos(pitch) * np.sin(yaw)
            + np.sin(ref_pitch) * np.sin(pitch),
            -1.0,
            1.0,
        )
        pointing = np.arccos(dots)
        pointing_metrics: dict[str, Optional[float]] = {
            "rms_rad": float(math.sqrt(float(np.mean(np.square(pointing))))),
            "p95_rad": float(np.percentile(pointing, 95)),
            "max_rad": float(np.max(pointing)),
            "final_rad": float(pointing[-1]),
        }
    else:
        pointing_metrics = {"rms_rad": None, "p95_rad": None, "max_rad": None, "final_rad": None}
    return {
        "format": "idcs.hardware_trajectory_benchmark",
        "version": 1,
        "status": status,
        "error": error,
        "offline_only": False,
        "coordinate_frame": "x_forward_y_right_z_up",
        "operator_note": args.operator_note,
        "safety": {
            "execute": bool(args.execute),
            "assume_exclusive": bool(args.assume_exclusive),
            "max_command_rate_rad_s": args.max_command_rate_rad_s,
            "max_reference_offset_rad": args.max_reference_offset_rad,
        },
        "samples": len(rows),
        "axes": {axis: axis_metrics(axis) for axis in ("yaw", "pitch")},
        "pointing_error": pointing_metrics,
        "axis_configs": [{"axis": cfg.axis, "encoder_addr": cfg.encoder_addr, "rate_limit": cfg.rate_limit, "accel_byte": cfg.accel_byte} for cfg in axis_cfgs],
    }


def _write_plan(path: Path, knots_s: np.ndarray, points: np.ndarray) -> None:
    sample_s = np.linspace(0.0, float(knots_s[-1]), max(2, int(math.ceil(float(knots_s[-1]) * 20.0)) + 1))
    coords = cubic_hermite_spline(knots_s, points, sample_s)
    yaw, pitch = coordinates_to_axes(coords)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t_s", "target_x_m", "target_y_m", "target_z_m", "relative_yaw_rad", "relative_pitch_rad"])
        writer.writeheader()
        for index, now_s in enumerate(sample_s):
            writer.writerow({"t_s": f"{now_s:.9f}", "target_x_m": f"{coords[index, 0]:.9f}", "target_y_m": f"{coords[index, 1]:.9f}", "target_z_m": f"{coords[index, 2]:.9f}", "relative_yaw_rad": f"{yaw[index] - yaw[0]:.9f}", "relative_pitch_rad": f"{pitch[index] - pitch[0]:.9f}"})


def _serial_io_config() -> Path:
    """Create a temporary service config with startup commands but no scheduler."""

    source = _REPO_ROOT / "configs" / "control.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    serial_io = payload.get("serial_io") if isinstance(payload, Mapping) else None
    startup = serial_io.get("startup", []) if isinstance(serial_io, Mapping) else []
    handle = tempfile.NamedTemporaryFile(mode="w", prefix="idcs_trajectory_serial_", suffix=".yaml", delete=False, encoding="utf-8")
    with handle:
        yaml.safe_dump({"serial_io": {"startup": startup, "schedule": []}}, handle, sort_keys=False)
    return Path(handle.name)


def _start_serial_io_service(
    *,
    service_config: Path,
    port: str,
    baud: int,
    timeout_s: float,
    retries: int,
    command_endpoint: str,
    update_endpoint: str,
    reply_endpoint: str,
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable, "-m", "tools.serial_io_service", "--config", str(service_config),
            "--port", port, "--baud", str(baud), "--timeout", str(timeout_s), "--retries", str(retries),
            "--command-endpoint", command_endpoint, "--update-endpoint", update_endpoint, "--reply-endpoint", reply_endpoint,
        ],
        cwd=str(_REPO_ROOT),
    )


def _stop_serial_service(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


def _read_encoder_pair(
    update_pub: SerialUpdatePublisher,
    reply_sub: SerialReplySubscriber,
    *,
    target: str,
    axes: Mapping[str, AxisConfig],
    timeout_s: float,
) -> dict[str, float]:
    pending: dict[str, str] = {}
    commands = []
    for axis_name, axis in axes.items():
        cmd_id = f"trajectory:enc:{axis_name}:{time.time_ns()}"
        pending[cmd_id] = axis_name
        commands.append(_build_command(cmd_id=cmd_id, func="0x31", addr=axis.encoder_addr, payload=[], expect_reply=True, expected_len=6, priority="high", target=target))
    if not update_pub.send_update(_build_update(source="jetson.gimbal_trajectory_benchmark", target=target, commands=commands)):
        raise RuntimeError("serial encoder update publish dropped")
    result: dict[str, float] = {}
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        for reply in reply_sub.recv_nowait():
            cmd_id = str(reply.get("cmd_id", ""))
            axis_name = pending.pop(cmd_id, None)
            if axis_name is None:
                continue
            axis = axes[axis_name]
            if _reply_func_byte(reply) != 0x31 or reply.get("addr") != axis.encoder_addr:
                raise RuntimeError(f"unexpected encoder reply for {axis_name}")
            counts = _extract_counts(reply)
            if counts is None:
                raise RuntimeError(f"malformed encoder reply for {axis_name}")
            result[axis_name] = axis.encoder_sign * _counts_to_rad(counts, counts_per_rev=axis.counts_per_rev, gear_ratio=axis.gear_ratio)
        if pending:
            time.sleep(0.002)
    if pending:
        raise RuntimeError("missing encoder reply for " + ", ".join(sorted(pending.values())))
    return result


def _send_speed_pair(
    update_pub: SerialUpdatePublisher,
    *,
    target: str,
    axes: Mapping[str, AxisConfig],
    commands_by_axis: Mapping[str, float],
) -> None:
    commands = []
    for axis_name, axis in axes.items():
        for addr, sign, label in zip(axis.command_addrs, axis.command_signs, axis.command_labels):
            payload = MksServo42Axis._encode_speed_payload(sign * commands_by_axis[axis_name], axis.accel_byte, axis.gear_ratio)
            commands.append(_build_command(cmd_id=f"trajectory:speed:{label}:{time.time_ns()}", func="F6", addr=addr, payload=payload, expect_reply=axis.respond_on_writes, expected_len=1 if axis.respond_on_writes else None, priority="high", target=target))
    if not update_pub.send_update(_build_update(source="jetson.gimbal_trajectory_benchmark", target=target, commands=commands)):
        raise RuntimeError("serial speed update publish dropped")


def _run_serial_io(
    args: argparse.Namespace,
    *,
    knots_s: np.ndarray,
    points: np.ndarray,
    plan_yaw: np.ndarray,
    plan_pitch: np.ndarray,
    control: ControlConfig,
    gimbal: Mapping[str, Any],
    axis_cfgs: Sequence[AxisConfig],
    output: Path,
    manifest_path: Path,
    port: str,
    baud: int,
    timeout_s: float,
    retries: int,
    merged: Mapping[str, Any],
) -> int:
    net = merged.get("net") if isinstance(merged.get("net"), Mapping) else {}
    target = str(gimbal.get("serial_target", "gimbal"))
    update_endpoint = str(gimbal.get("serial_update_endpoint") or net.get("zmq_serial_update") or "tcp://127.0.0.1:5571")
    reply_endpoint = str(gimbal.get("serial_reply_endpoint") or net.get("zmq_serial_reply") or "tcp://127.0.0.1:5572")
    command_endpoint = str(net.get("zmq_serial_cmd") or "tcp://127.0.0.1:5570")
    service_config = _serial_io_config()
    service: Optional[subprocess.Popen] = None
    ctx: Optional[zmq.Context] = None
    update_pub: Optional[SerialUpdatePublisher] = None
    reply_sub: Optional[SerialReplySubscriber] = None
    rows: list[dict[str, Any]] = []
    status, error = "complete", None
    try:
        service = _start_serial_io_service(service_config=service_config, port=port, baud=baud, timeout_s=timeout_s, retries=retries, command_endpoint=command_endpoint, update_endpoint=update_endpoint, reply_endpoint=reply_endpoint)
        time.sleep(0.8)
        if service.poll() is not None:
            raise RuntimeError("serial_io_service exited during startup")
        ctx = zmq.Context()
        update_pub = SerialUpdatePublisher(update_endpoint, ctx=ctx)
        reply_sub = SerialReplySubscriber(reply_endpoint, topics=[f"serial.reply.{target}"], ctx=ctx)
        time.sleep(0.2)
        axes = {cfg.axis: cfg for cfg in axis_cfgs}
        measured = _read_encoder_pair(update_pub, reply_sub, target=target, axes=axes, timeout_s=max(0.25, 4.0 * timeout_s))
        initial = dict(measured)
        yaw_zero = initial["yaw"] - float(plan_yaw[0])
        pitch_zero = initial["pitch"] - float(plan_pitch[0])
        for axis_name, relative in (("yaw", plan_yaw - plan_yaw[0]), ("pitch", plan_pitch - plan_pitch[0])):
            axis = axes[axis_name]
            target_angles = initial[axis_name] + relative
            if axis.angle_min_rad is not None and float(np.min(target_angles)) < axis.angle_min_rad:
                raise RuntimeError(f"{axis_name} reference crosses configured minimum angle")
            if axis.angle_max_rad is not None and float(np.max(target_angles)) > axis.angle_max_rad:
                raise RuntimeError(f"{axis_name} reference crosses configured maximum angle")
        pid_state = {"yaw": PidAxisState(), "pitch": PidAxisState()}
        speed_dither = {axis: SpeedCommandDither(axes[axis].gear_ratio) for axis in axes}
        stop_event = install_signal_handlers()
        period_s = 1.0 / args.sample_hz
        start_mono = time.monotonic()
        previous_mono = start_mono
        next_tick = start_mono
        while not stop_event.is_set():
            now = time.monotonic()
            elapsed = now - start_mono
            if elapsed > float(knots_s[-1]) + args.hold_s:
                break
            if now < next_tick:
                time.sleep(min(0.005, next_tick - now))
                continue
            dt_s = min(max(now - previous_mono, 0.001), 0.2)
            previous_mono = now
            coordinate, ref_yaw, ref_pitch = _reference_at(knots_s, points, elapsed, yaw_zero, pitch_zero)
            measured = _read_encoder_pair(update_pub, reply_sub, target=target, axes=axes, timeout_s=max(0.25, 4.0 * timeout_s))
            computed: dict[str, tuple[float, float, bool, bool, bool]] = {}
            for axis_name, reference in (("yaw", ref_yaw), ("pitch", ref_pitch)):
                axis = axes[axis_name]
                pid = control.pid
                rate_limit = min(_axis_value(pid.rate_limits, axis_name), axis.rate_limit, args.max_command_rate_rad_s)
                accel_limit = min(_axis_value(pid.accel_limits, axis_name), _axis_value(control.gimbal_accel_limits, axis_name))
                command, raw, rate_limited, slew_limited = _pid_command(pid_state[axis_name], error=reference - measured[axis_name], dt_s=dt_s, kp=_axis_value(pid.kp, axis_name), ki=_axis_value(pid.ki, axis_name), kd=_axis_value(pid.kd, axis_name), rate_limit=rate_limit, accel_limit=accel_limit)
                applied = _apply_hard_angle_limit(command, measured[axis_name], axis.angle_min_rad, axis.angle_max_rad, axis_name)
                limit_blocked = command != 0.0 and applied == 0.0
                if limit_blocked:
                    pid_state[axis_name].prev_command = 0.0
                computed[axis_name] = (applied, raw, rate_limited, slew_limited, limit_blocked)
            encoded_commands = {axis: speed_dither[axis].quantize(values[0]) for axis, values in computed.items()}
            _send_speed_pair(update_pub, target=target, axes=axes, commands_by_axis=encoded_commands)
            rows.append({"sample_idx": len(rows), "elapsed_s": elapsed, "phase": "track" if elapsed <= float(knots_s[-1]) else "hold", "target_x_m": coordinate[0], "target_y_m": coordinate[1], "target_z_m": coordinate[2], "ref_yaw_rad": ref_yaw, "ref_pitch_rad": ref_pitch, "measured_yaw_rad": measured["yaw"], "measured_pitch_rad": measured["pitch"], "error_yaw_rad": ref_yaw - measured["yaw"], "error_pitch_rad": ref_pitch - measured["pitch"], "command_yaw_rad_s": computed["yaw"][0], "command_pitch_rad_s": computed["pitch"][0], "encoded_command_yaw_rad_s": encoded_commands["yaw"], "encoded_command_pitch_rad_s": encoded_commands["pitch"], "raw_command_yaw_rad_s": computed["yaw"][1], "raw_command_pitch_rad_s": computed["pitch"][1], "rate_limited_yaw": int(computed["yaw"][2]), "rate_limited_pitch": int(computed["pitch"][2]), "slew_limited_yaw": int(computed["yaw"][3]), "slew_limited_pitch": int(computed["pitch"][3]), "limit_blocked_yaw": int(computed["yaw"][4]), "limit_blocked_pitch": int(computed["pitch"][4]), "loop_dt_s": dt_s})
            next_tick += period_s
        if stop_event.is_set():
            status, error = "interrupted", "signal received"
    except Exception as exc:  # noqa: BLE001
        status, error = "error", str(exc)
    finally:
        if update_pub is not None:
            for cfg in axis_cfgs:
                with suppress(Exception):
                    _send_zero_speed(update_pub, cfg, target)
            time.sleep(0.12)
            update_pub.close()
        if reply_sub is not None:
            reply_sub.close()
        if ctx is not None:
            ctx.term()
        _stop_serial_service(service)
        with suppress(FileNotFoundError):
            service_config.unlink()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = _summary(rows, args=args, axis_cfgs=axis_cfgs, status=status, error=error)
    manifest.update({"output_csv": str(output), "port": port, "baudrate": baud, "transport": "serial-io"})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote hardware benchmark: {output}")
    if status != "complete":
        print(f"hardware benchmark failed: {error}", file=sys.stderr)
        return 2
    print(f"Pointing RMS error: {manifest['pointing_error']['rms_rad']:.5f} rad")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.sample_hz <= 0.0 or args.hold_s < 0.0 or args.max_command_rate_rad_s <= 0.0 or args.max_reference_offset_rad <= 0.0:
        raise ValueError("sample rate and safety limits must be positive; hold_s must be non-negative")
    if args.execute and not args.assume_exclusive:
        raise ValueError("--execute requires --assume-exclusive")
    knots_s, points = load_points(Path(args.points), args.duration_s)
    plan_samples = cubic_hermite_spline(knots_s, points, np.linspace(0.0, float(knots_s[-1]), 401))
    plan_yaw, plan_pitch = coordinates_to_axes(plan_samples)
    if max(float(np.max(np.abs(plan_yaw - plan_yaw[0]))), float(np.max(np.abs(plan_pitch - plan_pitch[0])))) > args.max_reference_offset_rad:
        raise ValueError("trajectory exceeds --max-reference-offset-rad; use a smaller path or explicitly raise the guarded limit")

    merged = load_merged_config(expand_config_paths(args.config, args.config_extra))
    control = ControlConfig.from_raw_config(merged, (1280, 720))
    gimbal = merged.get("gimbal")
    if not isinstance(gimbal, Mapping):
        raise ValueError("config missing gimbal section")
    axis_cfgs = [_build_axis_config(merged, "yaw", None), _build_axis_config(merged, "pitch", None)]
    timestamp = int(time.time())
    output = Path(args.output or f"logs/gimbal_trajectory_benchmark_{timestamp}.csv")
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        _write_plan(output, knots_s, points)
        manifest_path.write_text(json.dumps({"format": "idcs.hardware_trajectory_benchmark", "version": 1, "status": "dry_run", "offline_only": True, "coordinate_frame": "x_forward_y_right_z_up", "output_csv": str(output), "note": "No serial port was opened. Re-run with --execute --assume-exclusive for hardware."}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote dry-run reference plan: {output}")
        return 0

    port = str(args.port or gimbal.get("serial_port") or "/dev/ttyTHS0")
    baud = int(args.baud or gimbal.get("baudrate") or 256000)
    timeout_s = float(args.timeout_s if args.timeout_s is not None else gimbal.get("timeout", 0.1))
    retries = int(args.retries if args.retries is not None else gimbal.get("retries", 1))
    if args.transport == "serial-io":
        return _run_serial_io(
            args,
            knots_s=knots_s,
            points=points,
            plan_yaw=plan_yaw,
            plan_pitch=plan_pitch,
            control=control,
            gimbal=gimbal,
            axis_cfgs=axis_cfgs,
            output=output,
            manifest_path=manifest_path,
            port=port,
            baud=baud,
            timeout_s=timeout_s,
            retries=retries,
            merged=merged,
        )
    stop_event = install_signal_handlers()
    rows: list[dict[str, Any]] = []
    status, error = "complete", None
    axes: list[HardwareAxis] = []
    try:
        with RS485Bus(port, baud, timeout=timeout_s, max_retries=max(0, retries)) as bus:
            axes = [_make_axis(bus, cfg) for cfg in axis_cfgs]
            by_axis = {axis.config.axis: axis for axis in axes}
            _enable_axes(axes)
            for axis in axes:
                axis.initial_angle = _read_axis(axis)
                axis.measured_angle = axis.initial_angle
            yaw_zero = by_axis["yaw"].initial_angle - float(plan_yaw[0])
            pitch_zero = by_axis["pitch"].initial_angle - float(plan_pitch[0])
            for axis, start, relative in ((by_axis["yaw"], by_axis["yaw"].initial_angle, plan_yaw - plan_yaw[0]), (by_axis["pitch"], by_axis["pitch"].initial_angle, plan_pitch - plan_pitch[0])):
                targets = start + relative
                if axis.config.angle_min_rad is not None and float(np.min(targets)) < axis.config.angle_min_rad:
                    raise RuntimeError(f"{axis.config.axis} reference crosses configured minimum angle")
                if axis.config.angle_max_rad is not None and float(np.max(targets)) > axis.config.angle_max_rad:
                    raise RuntimeError(f"{axis.config.axis} reference crosses configured maximum angle")

            period_s = 1.0 / args.sample_hz
            start_mono = time.monotonic()
            previous_mono = start_mono
            next_tick = start_mono
            sample_idx = 0
            while not stop_event.is_set():
                now = time.monotonic()
                elapsed = now - start_mono
                if elapsed > float(knots_s[-1]) + args.hold_s:
                    break
                if now < next_tick:
                    time.sleep(min(0.005, next_tick - now))
                    continue
                dt_s = min(max(now - previous_mono, 0.001), 0.2)
                previous_mono = now
                coordinate, ref_yaw, ref_pitch = _reference_at(knots_s, points, elapsed, yaw_zero, pitch_zero)
                _read_axis(by_axis["yaw"])
                _read_axis(by_axis["pitch"])
                commands: dict[str, tuple[float, float, bool, bool, bool]] = {}
                for axis_name, reference in (("yaw", ref_yaw), ("pitch", ref_pitch)):
                    axis = by_axis[axis_name]
                    pid = control.pid
                    rate_limit = min(_axis_value(pid.rate_limits, axis_name), axis.config.rate_limit, args.max_command_rate_rad_s)
                    accel_limit = min(_axis_value(pid.accel_limits, axis_name), _axis_value(control.gimbal_accel_limits, axis_name))
                    command, raw, rate_limited, slew_limited = _pid_command(axis.pid_state, error=reference - axis.measured_angle, dt_s=dt_s, kp=_axis_value(pid.kp, axis_name), ki=_axis_value(pid.ki, axis_name), kd=_axis_value(pid.kd, axis_name), rate_limit=rate_limit, accel_limit=accel_limit)
                    applied = _apply_hard_angle_limit(command, axis.measured_angle, axis.config.angle_min_rad, axis.config.angle_max_rad, axis_name)
                    limit_blocked = command != 0.0 and applied == 0.0
                    if limit_blocked:
                        axis.pid_state.prev_command = 0.0
                    commands[axis_name] = (applied, raw, rate_limited, slew_limited, limit_blocked)
                _send_axis_speed(by_axis["yaw"], commands["yaw"][0])
                _send_axis_speed(by_axis["pitch"], commands["pitch"][0])
                row = {
                    "sample_idx": sample_idx, "elapsed_s": elapsed, "phase": "track" if elapsed <= float(knots_s[-1]) else "hold",
                    "target_x_m": coordinate[0], "target_y_m": coordinate[1], "target_z_m": coordinate[2],
                    "ref_yaw_rad": ref_yaw, "ref_pitch_rad": ref_pitch,
                    "measured_yaw_rad": by_axis["yaw"].measured_angle, "measured_pitch_rad": by_axis["pitch"].measured_angle,
                    "error_yaw_rad": ref_yaw - by_axis["yaw"].measured_angle, "error_pitch_rad": ref_pitch - by_axis["pitch"].measured_angle,
                    "command_yaw_rad_s": commands["yaw"][0], "command_pitch_rad_s": commands["pitch"][0],
                    "raw_command_yaw_rad_s": commands["yaw"][1], "raw_command_pitch_rad_s": commands["pitch"][1],
                    "rate_limited_yaw": int(commands["yaw"][2]), "rate_limited_pitch": int(commands["pitch"][2]),
                    "slew_limited_yaw": int(commands["yaw"][3]), "slew_limited_pitch": int(commands["pitch"][3]),
                    "limit_blocked_yaw": int(commands["yaw"][4]), "limit_blocked_pitch": int(commands["pitch"][4]), "loop_dt_s": dt_s,
                }
                rows.append(row)
                sample_idx += 1
                next_tick += period_s
            if stop_event.is_set():
                status, error = "interrupted", "signal received"
            _stop_axes(axes)
    except Exception as exc:  # noqa: BLE001 - cleanup must run for every hardware failure.
        status, error = "error", str(exc)
        if axes:
            _stop_axes(axes)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = _summary(rows, args=args, axis_cfgs=axis_cfgs, status=status, error=error)
    manifest.update({"output_csv": str(output), "port": port, "baudrate": baud, "config_paths": [str(path) for path in expand_config_paths(args.config, args.config_extra)]})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote hardware benchmark: {output}")
    if status != "complete":
        print(f"hardware benchmark failed: {error}", file=sys.stderr)
        return 2
    print(f"Pointing RMS error: {manifest['pointing_error']['rms_rad']:.5f} rad")
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"hardware benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
