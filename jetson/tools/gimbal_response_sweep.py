"""Record raw motor/encoder responses over configurable command profiles.

This utility deliberately performs no system identification or controller
tuning. It owns the serial I/O command stream for a short experiment, sends
open-loop speed commands, and records the raw encoder response to a flat CSV
plus a JSON manifest. The output is intended for offline analysis in MATLAB or
Python.

The gimbal bridge and other motor-command publishers must be stopped while the
sweep owns the serial bus.

Example::

    python -m jetson.tools.gimbal_response_sweep \
        --axis both --rates 0.1,0.25,0.5,1.0 \
        --accel-bytes 1,5,10,20 --repeat 3 --start-serial-io
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import zmq

from common.config_sync import expand_config_paths, load_merged_config
from common.serial_io import SerialReplySubscriber, SerialUpdatePublisher
from common.shutdown import install_signal_handlers
from jetson.tools.gimbal_step_tuning import (
    AxisConfig,
    EncoderState,
    _apply_hard_angle_limit,
    _build_axis_config,
    _build_command,
    _build_update,
    _counts_to_rad,
    _encode_speed_cmd,
    _extract_counts,
    _reply_func_byte,
    _resolve_axes,
    _start_serial_io_service,
    _stop_serial_io_service,
    _wrapped_delta,
)

_LOG = logging.getLogger(__name__)

MANIFEST_VERSION = 2
PROFILE_CHOICES = ("step", "random-step", "prbs", "chirp", "sine")

BASE_CSV_FIELDS = [
    "sample_idx",
    "setting_id",
    "axis",
    "trial",
    "direction",
    "phase",
    "rate_setting_rad_s",
    "accel_byte",
    "cmd_rate_rad_s",
    "cmd_rate_applied_rad_s",
    "limit_blocked",
    "command_addrs",
    "command_payloads_hex",
    "command_tx_monotonic_ns",
    "command_tx_wall_ns",
    "response_rx_monotonic_ns",
    "response_rx_wall_ns",
    "elapsed_s",
    "setting_elapsed_s",
    "phase_elapsed_s",
    "encoder_addr",
    "counts",
    "angle_rad",
    "omega_rad_s",
    "encoder_dt_s",
]

V2_CSV_FIELDS = [
    "profile",
    "segment_id",
    "profile_step_idx",
    "profile_elapsed_s",
    "requested_rate_source",
    "command_cmd_ids",
    "reply_cmd_id",
    "reply_bytes_hex",
    "reply_parsed_json",
    "send_ok",
    "valid_encoder",
    "settle_phase",
    "settled",
    "settle_timeout",
    "missing_reply",
    "send_dropped",
    "reply_latency_ms",
    "pending_query_count",
    "dropped_query_count",
    "stale_reply_count",
    "encoder_sample_monotonic_ns",
    "omega_valid",
    "omega_invalid_reason",
]

CSV_FIELDS = BASE_CSV_FIELDS + V2_CSV_FIELDS


@dataclass(frozen=True)
class ProfileSegment:
    """One constant-rate command segment in a generated experiment profile."""

    segment_id: int
    profile_step_idx: int
    phase: str
    rate_rad_s: float
    duration_s: float
    profile_elapsed_s: float
    requested_rate_source: str
    settle_phase: bool = False


def _csv_numbers(raw: str, *, name: str, integer: bool = False) -> list[float] | list[int]:
    values: list[float] | list[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item, 0) if integer else float(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} contains invalid value {item!r}") from exc
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


def _parse_rates(raw: str) -> list[float]:
    values = [float(v) for v in _csv_numbers(raw, name="--rates")]
    if any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise argparse.ArgumentTypeError("--rates values must be positive and finite")
    return values


def _parse_accel_bytes(raw: str) -> list[int]:
    values = [int(v) for v in _csv_numbers(raw, name="--accel-bytes", integer=True)]
    if any(v < 0 or v > 255 for v in values):
        raise argparse.ArgumentTypeError("--accel-bytes values must be within 0..255")
    return values


def _parse_sine_freqs(raw: str) -> list[float]:
    values = [float(v) for v in _csv_numbers(raw, name="--sine-freqs")]
    if any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise argparse.ArgumentTypeError("--sine-freqs values must be positive and finite")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml", help="Base YAML config")
    parser.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
        help="Comma-separated YAML configs merged over --config",
    )
    parser.add_argument("--axis", choices=["yaw", "pitch", "both"], default="both")
    parser.add_argument(
        "--rates",
        type=_parse_rates,
        default=_parse_rates("0.1,0.25,0.5,1.0"),
        help="Comma-separated positive rate magnitudes in rad/s",
    )
    parser.add_argument(
        "--accel-bytes",
        type=_parse_accel_bytes,
        default=None,
        help="Comma-separated MKS acceleration bytes; default uses each axis config",
    )
    parser.add_argument(
        "--directions",
        choices=["both", "positive", "negative"],
        default="both",
        help="Command directions tested for every setting",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Repetitions per setting/direction")
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument(
        "--command-refresh-s",
        type=float,
        default=0.0,
        help=(
            "Minimum period for resending unchanged speed commands. "
            "Default 0 sends only when the segment command or limit-applied command changes."
        ),
    )
    parser.add_argument("--pre-roll-s", type=float, default=0.5)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--post-roll-s", type=float, default=1.0)
    parser.add_argument("--rest-s", type=float, default=0.5)
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="step",
        help="Open-loop command profile to execute for each rate/axis setting.",
    )
    parser.add_argument(
        "--profile-duration-s",
        type=float,
        default=10.0,
        help="Duration of generated profiles other than step.",
    )
    parser.add_argument(
        "--profile-segment-s",
        type=float,
        default=0.25,
        help="Duration of random/PRBS segments and chirp/sine discretization.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base seed for deterministic profiles.")
    parser.add_argument(
        "--zero-hold-s",
        type=float,
        default=0.5,
        help="Zero-command hold after generated profiles before post settling.",
    )
    parser.add_argument("--chirp-start-hz", type=float, default=0.05)
    parser.add_argument("--chirp-end-hz", type=float, default=1.0)
    parser.add_argument(
        "--sine-freqs",
        type=_parse_sine_freqs,
        default=_parse_sine_freqs("0.1,0.25,0.5,1.0"),
        help="Comma-separated sine frequencies in Hz.",
    )
    parser.add_argument(
        "--operator-note",
        default="",
        help="Optional free-form note copied into the manifest.",
    )
    parser.add_argument(
        "--settle-rate-rad-s",
        type=float,
        default=0.03,
        help="Absolute encoder rate considered stationary",
    )
    parser.add_argument(
        "--settle-hold-s",
        type=float,
        default=0.25,
        help="Continuous time below the rate threshold required for settling",
    )
    parser.add_argument(
        "--settle-timeout-s",
        type=float,
        default=5.0,
        help="Maximum extra time allowed for pre/post settling",
    )
    parser.add_argument(
        "--reply-drain-s",
        type=float,
        default=0.5,
        help="Maximum time to drain outstanding encoder replies after each phase",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV path (default: logs/gimbal_response_sweep_<timestamp>.csv)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="JSON manifest path (default: beside CSV with .json suffix)",
    )
    parser.add_argument("--warn-no-encoder-s", type=float, default=1.0)
    parser.add_argument(
        "--assume-exclusive",
        action="store_true",
        help="Confirm no other process is publishing motor commands",
    )
    parser.add_argument(
        "--fail-if-no-exclusive",
        action="store_true",
        help="Abort unless --assume-exclusive is provided.",
    )
    parser.add_argument("--start-serial-io", action="store_true")
    parser.add_argument("--serial-io-wait-s", type=float, default=0.5)
    return parser.parse_args()


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _directions(selection: str) -> list[int]:
    if selection == "positive":
        return [1]
    if selection == "negative":
        return [-1]
    return [1, -1]


def _payload_text(payloads: Sequence[tuple[int, Sequence[int]]]) -> str:
    return ";".join(f"{addr}:" + "".join(f"{byte:02X}" for byte in payload) for addr, payload in payloads)


def _bytes_hex(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ""
    try:
        return "".join(f"{int(byte) & 0xFF:02X}" for byte in value)
    except (TypeError, ValueError):
        return ""


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)
    except TypeError:
        return json.dumps(str(value))


def _profile_seed(
    base_seed: int,
    *,
    profile: str,
    axis: str,
    accel_byte: int,
    rate: float,
    trial: int,
    direction: int,
) -> int:
    text = f"{base_seed}|{profile}|{axis}|{accel_byte}|{rate:.12g}|{trial}|{direction}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _append_segment(
    segments: list[ProfileSegment],
    *,
    phase: str,
    rate_rad_s: float,
    duration_s: float,
    profile_elapsed_s: float,
    requested_rate_source: str,
    settle_phase: bool = False,
) -> None:
    if duration_s <= 0.0 and not settle_phase:
        return
    step_idx = sum(1 for segment in segments if not segment.settle_phase)
    segments.append(
        ProfileSegment(
            segment_id=len(segments),
            profile_step_idx=step_idx,
            phase=phase,
            rate_rad_s=float(rate_rad_s),
            duration_s=float(duration_s),
            profile_elapsed_s=float(profile_elapsed_s),
            requested_rate_source=requested_rate_source,
            settle_phase=settle_phase,
        )
    )


def _build_command_profile(
    args: argparse.Namespace,
    *,
    rate: float,
    direction: int,
    seed: int,
) -> list[ProfileSegment]:
    """Return command segments for one axis/rate/trial setting."""

    profile = str(_arg(args, "profile", "step"))
    pre_roll_s = float(_arg(args, "pre_roll_s", 0.5))
    post_roll_s = float(_arg(args, "post_roll_s", 1.0))
    rest_s = float(_arg(args, "rest_s", 0.5))
    segments: list[ProfileSegment] = []

    _append_segment(
        segments,
        phase="pre",
        rate_rad_s=0.0,
        duration_s=pre_roll_s,
        profile_elapsed_s=0.0,
        requested_rate_source="settle",
        settle_phase=True,
    )

    if profile == "step":
        _append_segment(
            segments,
            phase="step",
            rate_rad_s=direction * float(rate),
            duration_s=float(_arg(args, "step_s", 1.0)),
            profile_elapsed_s=0.0,
            requested_rate_source="step",
        )
    elif profile == "random-step":
        _append_random_step_segments(args, segments, rate=rate, direction=direction, seed=seed)
    elif profile == "prbs":
        _append_prbs_segments(args, segments, rate=rate, direction=direction, seed=seed)
    elif profile == "chirp":
        _append_chirp_segments(args, segments, rate=rate, direction=direction)
    elif profile == "sine":
        _append_sine_segments(args, segments, rate=rate, direction=direction)
    else:  # pragma: no cover - argparse normally prevents this
        raise ValueError(f"unsupported profile: {profile}")

    if profile != "step":
        _append_segment(
            segments,
            phase="zero_hold",
            rate_rad_s=0.0,
            duration_s=float(_arg(args, "zero_hold_s", 0.5)),
            profile_elapsed_s=float(_arg(args, "profile_duration_s", 10.0)),
            requested_rate_source="zero_hold",
        )

    _append_segment(
        segments,
        phase="post",
        rate_rad_s=0.0,
        duration_s=post_roll_s,
        profile_elapsed_s=0.0,
        requested_rate_source="settle",
        settle_phase=True,
    )
    _append_segment(
        segments,
        phase="rest",
        rate_rad_s=0.0,
        duration_s=rest_s,
        profile_elapsed_s=0.0,
        requested_rate_source="rest",
    )
    return segments


def _profile_segment_count(args: argparse.Namespace) -> tuple[int, float]:
    duration_s = float(_arg(args, "profile_duration_s", 10.0))
    segment_s = float(_arg(args, "profile_segment_s", 0.25))
    count = max(1, int(math.ceil(duration_s / segment_s)))
    return count, duration_s / count


def _append_random_step_segments(
    args: argparse.Namespace,
    segments: list[ProfileSegment],
    *,
    rate: float,
    direction: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    count, segment_s = _profile_segment_count(args)
    elapsed = 0.0
    for _ in range(count):
        magnitude = rng.choice([0.0, 0.5 * float(rate), float(rate)])
        sign = direction if magnitude == 0.0 else direction * rng.choice([1, -1])
        _append_segment(
            segments,
            phase="random_step",
            rate_rad_s=sign * magnitude,
            duration_s=segment_s,
            profile_elapsed_s=elapsed,
            requested_rate_source="random-step",
        )
        elapsed += segment_s


def _append_prbs_segments(
    args: argparse.Namespace,
    segments: list[ProfileSegment],
    *,
    rate: float,
    direction: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    count, segment_s = _profile_segment_count(args)
    sign = direction if rng.random() >= 0.5 else -direction
    elapsed = 0.0
    for _ in range(count):
        if rng.random() >= 0.5:
            sign *= -1
        _append_segment(
            segments,
            phase="prbs",
            rate_rad_s=sign * float(rate),
            duration_s=segment_s,
            profile_elapsed_s=elapsed,
            requested_rate_source="prbs",
        )
        elapsed += segment_s


def _append_chirp_segments(
    args: argparse.Namespace,
    segments: list[ProfileSegment],
    *,
    rate: float,
    direction: int,
) -> None:
    count, segment_s = _profile_segment_count(args)
    duration_s = float(_arg(args, "profile_duration_s", 10.0))
    f0 = float(_arg(args, "chirp_start_hz", 0.05))
    f1 = float(_arg(args, "chirp_end_hz", 1.0))
    elapsed = 0.0
    for idx in range(count):
        t_mid = min(duration_s, elapsed + 0.5 * segment_s)
        ratio = 0.0 if duration_s <= 0.0 else t_mid / duration_s
        freq = f0 + (f1 - f0) * ratio
        phase = 2.0 * math.pi * (f0 * t_mid + 0.5 * (f1 - f0) * ratio * t_mid)
        _append_segment(
            segments,
            phase="chirp",
            rate_rad_s=direction * float(rate) * math.sin(phase),
            duration_s=segment_s,
            profile_elapsed_s=elapsed,
            requested_rate_source=f"chirp:{freq:.6g}Hz",
        )
        elapsed += segment_s


def _append_sine_segments(
    args: argparse.Namespace,
    segments: list[ProfileSegment],
    *,
    rate: float,
    direction: int,
) -> None:
    freqs = list(_arg(args, "sine_freqs", [0.1, 0.25, 0.5, 1.0]))
    count, segment_s = _profile_segment_count(args)
    duration_s = float(_arg(args, "profile_duration_s", 10.0))
    block_s = duration_s / max(1, len(freqs))
    elapsed = 0.0
    for idx in range(count):
        t_mid = min(duration_s, elapsed + 0.5 * segment_s)
        freq_idx = min(int(t_mid / max(block_s, 1e-9)), len(freqs) - 1)
        freq = float(freqs[freq_idx])
        local_t = t_mid - freq_idx * block_s
        phase = 2.0 * math.pi * freq * local_t
        _append_segment(
            segments,
            phase="sine",
            rate_rad_s=direction * float(rate) * math.sin(phase),
            duration_s=segment_s,
            profile_elapsed_s=elapsed,
            requested_rate_source=f"sine:{freq:.6g}Hz",
        )
        elapsed += segment_s


def _send_zero_speed(
    update_pub: SerialUpdatePublisher,
    axis_cfg: AxisConfig,
    serial_target: str,
) -> None:
    commands = []
    for addr, label in zip(axis_cfg.command_addrs, axis_cfg.command_labels, strict=True):
        payload = _encode_speed_cmd(
            0.0,
            acc=axis_cfg.accel_byte,
            gear_ratio=axis_cfg.gear_ratio,
            max_rate=axis_cfg.rate_limit,
        )
        commands.append(
            _build_command(
                cmd_id=f"stop:{label}:{time.time_ns()}",
                func="F6",
                addr=addr,
                payload=payload,
                expect_reply=axis_cfg.respond_on_writes,
                expected_len=1 if axis_cfg.respond_on_writes else None,
                priority="critical",
                target=serial_target,
            )
        )
    update_pub.send_update(
        _build_update(source="jetson.gimbal_response_sweep", target=serial_target, commands=commands)
    )


def _file_sha256(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _config_metadata(paths: Sequence[Path]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for path in paths:
        stat = None
        try:
            stat = path.stat()
        except OSError:
            pass
        metadata.append(
            {
                "path": str(path),
                "exists": stat is not None,
                "size": None if stat is None else int(stat.st_size),
                "mtime_ns": None if stat is None else int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
                "sha256": _file_sha256(path),
            }
        )
    return metadata


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(args: Sequence[str]) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(repo_root),
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return proc.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status.strip()),
        "status_short": status,
    }


def _summarize_numbers(values: Sequence[float]) -> dict[str, Optional[float] | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _empty_quality() -> dict[str, Any]:
    return {
        "samples": 0,
        "valid_encoder": 0,
        "send_dropped": 0,
        "dropped_queries": 0,
        "stale_replies": 0,
        "non_encoder_replies": 0,
        "malformed_replies": 0,
        "missing_replies": 0,
        "limit_blocked": 0,
        "settle_timeouts": 0,
        "preflight_replies": 0,
        "reply_latency_ms_values": [],
        "encoder_dt_s_values": [],
        "encoder_dt_invalid": 0,
        "omega_invalid": 0,
        "omega_invalid_reasons": {},
        "samples_by_axis": {},
        "samples_by_phase": {},
    }


def _quality_manifest(quality: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(quality)
    latency = data.pop("reply_latency_ms_values", [])
    encoder_dt = data.pop("encoder_dt_s_values", [])
    data["reply_latency_ms"] = _summarize_numbers(latency)
    data["encoder_dt_s"] = _summarize_numbers(encoder_dt)
    return data


def _manifest_data(
    *,
    args: argparse.Namespace,
    config_paths: Sequence[Path],
    cfg: Mapping[str, Any],
    axis_configs: Sequence[AxisConfig],
    output_csv: Path,
    quality: Optional[Mapping[str, Any]] = None,
    status: str = "running",
    error: Optional[str] = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    profile = str(_arg(args, "profile", "step"))
    data = {
        "format": "idcs.gimbal_response_sweep",
        "version": MANIFEST_VERSION,
        "status": status,
        "error": error,
        "created_wall_ns": time.time_ns(),
        "output_csv": str(output_csv),
        "config_paths": [str(p) for p in config_paths],
        "config_files": _config_metadata(list(config_paths)),
        "git": _git_metadata(repo_root),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "operator_note": str(_arg(args, "operator_note", "")),
        "csv_fields": CSV_FIELDS,
        "sweep": {
            "axes": [axis.axis for axis in axis_configs],
            "rates_rad_s": list(_arg(args, "rates", [])),
            "accel_bytes": list(args.accel_bytes) if getattr(args, "accel_bytes", None) is not None else None,
            "directions": _arg(args, "directions", "both"),
            "repeat": _arg(args, "repeat", 1),
            "sample_hz": _arg(args, "sample_hz", 50.0),
            "command_refresh_s": _arg(args, "command_refresh_s", 0.0),
            "pre_roll_s": _arg(args, "pre_roll_s", 0.5),
            "step_s": _arg(args, "step_s", 1.0),
            "post_roll_s": _arg(args, "post_roll_s", 1.0),
            "rest_s": _arg(args, "rest_s", 0.5),
            "settle_rate_rad_s": _arg(args, "settle_rate_rad_s", 0.03),
            "settle_hold_s": _arg(args, "settle_hold_s", 0.25),
            "settle_timeout_s": _arg(args, "settle_timeout_s", 5.0),
            "reply_drain_s": _arg(args, "reply_drain_s", 0.5),
        },
        "profile": {
            "name": profile,
            "duration_s": _arg(args, "profile_duration_s", 10.0),
            "segment_s": _arg(args, "profile_segment_s", 0.25),
            "seed": _arg(args, "seed", 1),
            "zero_hold_s": _arg(args, "zero_hold_s", 0.5),
            "chirp_start_hz": _arg(args, "chirp_start_hz", 0.05),
            "chirp_end_hz": _arg(args, "chirp_end_hz", 1.0),
            "sine_freqs": list(_arg(args, "sine_freqs", [])),
        },
        "axis_configs": [asdict(axis) for axis in axis_configs],
        "merged_config": cfg,
    }
    if quality is not None:
        data["quality"] = _quality_manifest(quality)
    return data


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    config_paths: Sequence[Path],
    cfg: Mapping[str, Any],
    axis_configs: Sequence[AxisConfig],
    output_csv: Path,
    quality: Optional[Mapping[str, Any]] = None,
    status: str = "running",
    error: Optional[str] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _manifest_data(
        args=args,
        config_paths=config_paths,
        cfg=cfg,
        axis_configs=axis_configs,
        output_csv=output_csv,
        quality=quality,
        status=status,
        error=error,
    )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.repeat < 1:
        return "--repeat must be >= 1"
    if not math.isfinite(args.sample_hz) or args.sample_hz <= 0.0:
        return "--sample-hz must be positive and finite"
    for name in (
        "pre_roll_s",
        "step_s",
        "post_roll_s",
        "rest_s",
        "settle_rate_rad_s",
        "settle_hold_s",
        "settle_timeout_s",
        "reply_drain_s",
        "command_refresh_s",
        "profile_duration_s",
        "profile_segment_s",
        "zero_hold_s",
    ):
        value = float(_arg(args, name, 0.0))
        if not math.isfinite(value) or value < 0.0:
            return f"--{name.replace('_', '-')} must be non-negative and finite"
    if args.step_s <= 0.0:
        return "--step-s must be > 0"
    profile = str(_arg(args, "profile", "step"))
    if profile not in PROFILE_CHOICES:
        return f"--profile must be one of: {', '.join(PROFILE_CHOICES)}"
    if profile != "step":
        if float(_arg(args, "profile_duration_s", 0.0)) <= 0.0:
            return "--profile-duration-s must be > 0 for generated profiles"
        if float(_arg(args, "profile_segment_s", 0.0)) <= 0.0:
            return "--profile-segment-s must be > 0 for generated profiles"
    if profile == "chirp":
        if float(_arg(args, "chirp_start_hz", 0.0)) <= 0.0:
            return "--chirp-start-hz must be > 0"
        if float(_arg(args, "chirp_end_hz", 0.0)) <= 0.0:
            return "--chirp-end-hz must be > 0"
    if profile == "sine" and not list(_arg(args, "sine_freqs", [])):
        return "--sine-freqs must include at least one frequency"
    if bool(_arg(args, "fail_if_no_exclusive", False)) and not bool(_arg(args, "assume_exclusive", False)):
        return "--fail-if-no-exclusive requires --assume-exclusive"
    return None


def _confirm_exclusive(args: argparse.Namespace) -> bool:
    if args.assume_exclusive:
        return True
    message = (
        "Stop gimbal_bridge, manual_control, and other F6 publishers before testing. "
        "Use --assume-exclusive once the serial bus is exclusively owned."
    )
    if args.fail_if_no_exclusive:
        _LOG.error("%s Refusing to run without --assume-exclusive.", message)
        return False
    _LOG.warning(message)
    if sys.stdin.isatty():
        response = input("Type 'yes' to continue with open-loop motor commands: ").strip().lower()
        if response != "yes":
            _LOG.error("exclusive-control confirmation declined")
            return False
    return True


def _preflight_reply_activity(reply_sub: SerialReplySubscriber, timeout_s: float = 0.25) -> int:
    deadline = time.monotonic() + max(0.0, timeout_s)
    count = 0
    while time.monotonic() < deadline:
        replies = reply_sub.recv_nowait()
        count += len(replies)
        if not replies:
            time.sleep(0.02)
    return count


def _bump_counter(container: dict[str, Any], key: str, item: str) -> None:
    block = container.setdefault(key, {})
    block[item] = int(block.get(item, 0)) + 1


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    validation_error = _validate_args(args)
    if validation_error:
        _LOG.error(validation_error)
        return 2
    if not _confirm_exclusive(args):
        return 2

    config_paths = expand_config_paths(args.config, args.config_extra)
    cfg = load_merged_config(config_paths)
    net_raw = cfg.get("net")
    net_cfg: Mapping[str, Any] = net_raw if isinstance(net_raw, Mapping) else {}
    gimbal_raw = cfg.get("gimbal")
    gimbal_cfg: Mapping[str, Any] = gimbal_raw if isinstance(gimbal_raw, Mapping) else {}
    serial_target = str(gimbal_cfg.get("serial_target", "gimbal"))
    serial_update_ep = str(
        gimbal_cfg.get("serial_update_endpoint")
        or net_cfg.get("zmq_serial_update")
        or "tcp://127.0.0.1:5571"
    )
    serial_reply_ep = str(
        gimbal_cfg.get("serial_reply_endpoint")
        or net_cfg.get("zmq_serial_reply")
        or "tcp://127.0.0.1:5572"
    )

    axis_configs = [_build_axis_config(cfg, axis, None) for axis in _resolve_axes(args.axis)]
    timestamp = int(time.time())
    output_csv = Path(args.output or f"logs/gimbal_response_sweep_{timestamp}.csv")
    manifest_path = Path(args.manifest) if args.manifest else output_csv.with_suffix(".json")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    quality = _empty_quality()
    _write_manifest(
        manifest_path,
        args=args,
        config_paths=config_paths,
        cfg=cfg,
        axis_configs=axis_configs,
        output_csv=output_csv,
        quality=quality,
    )

    stop_event = install_signal_handlers()
    period_s = 1.0 / args.sample_hz
    serial_proc: Optional[subprocess.Popen] = None
    ctx: Optional[zmq.Context] = None
    update_pub: Optional[SerialUpdatePublisher] = None
    reply_sub: Optional[SerialReplySubscriber] = None
    last_axis_cfg: Optional[AxisConfig] = None
    sample_idx = 0
    setting_id = 0
    run_start_mono = time.monotonic()
    final_status = "complete"
    final_error: Optional[str] = None

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
                _LOG.error("serial_io_service exited early; check its logs")
                final_status = "error"
                final_error = "serial_io_service exited early"
                return 2

        ctx = zmq.Context()
        update_pub = SerialUpdatePublisher(serial_update_ep, ctx=ctx)
        reply_sub = SerialReplySubscriber(
            serial_reply_ep,
            topics=[f"serial.reply.{serial_target}"],
            ctx=ctx,
        )
        preflight_count = _preflight_reply_activity(reply_sub)
        quality["preflight_replies"] = preflight_count
        if preflight_count and not args.assume_exclusive:
            _LOG.warning(
                "observed %d serial replies before sweep start; another publisher may be active",
                preflight_count,
            )

        with output_csv.open("w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for base_axis_cfg in axis_configs:
                accel_values = args.accel_bytes or [base_axis_cfg.accel_byte]
                encoder_state = EncoderState()
                pending_encoder_queries: dict[str, dict[str, Any]] = {}
                last_speed_payloads: Optional[tuple[tuple[int, tuple[int, ...]], ...]] = None
                last_speed_command_mono = -float("inf")

                def record_available_replies(axis_cfg: AxisConfig) -> int:
                    nonlocal sample_idx
                    recorded = 0
                    for reply in reply_sub.recv_nowait():
                        reply_cmd_id = str(reply.get("cmd_id", ""))
                        if _reply_func_byte(reply) != 0x31 or reply.get("addr") != axis_cfg.encoder_addr:
                            quality["non_encoder_replies"] += 1
                            continue
                        query = pending_encoder_queries.pop(reply_cmd_id, None)
                        if query is None:
                            quality["stale_replies"] += 1
                            continue
                        counts = _extract_counts(reply)
                        if counts is None:
                            quality["malformed_replies"] += 1
                            continue

                        rx_mono_ns = time.monotonic_ns()
                        rx_wall_ns = time.time_ns()
                        rx_mono = rx_mono_ns / 1e9
                        sample_mono_ns = int(query["tx_mono_ns"])
                        sample_mono = sample_mono_ns / 1e9
                        angle = axis_cfg.encoder_sign * _counts_to_rad(
                            counts,
                            counts_per_rev=axis_cfg.counts_per_rev,
                            gear_ratio=axis_cfg.gear_ratio,
                        )
                        omega = None
                        encoder_dt = None
                        omega_valid = 0
                        omega_invalid_reason = "first_sample"
                        if (
                            encoder_state.last_angle is not None
                            and encoder_state.last_timestamp is not None
                        ):
                            encoder_dt = sample_mono - encoder_state.last_timestamp
                            min_encoder_dt_s = max(0.001, 0.5 * period_s)
                            if encoder_dt <= 0.0:
                                omega_invalid_reason = "nonpositive_encoder_dt"
                                quality["encoder_dt_invalid"] += 1
                            elif encoder_dt < min_encoder_dt_s:
                                omega_invalid_reason = "short_encoder_dt"
                                quality["encoder_dt_invalid"] += 1
                            else:
                                omega = _wrapped_delta(angle, encoder_state.last_angle) / encoder_dt
                                omega_valid = int(math.isfinite(float(omega)))
                                omega_invalid_reason = "" if omega_valid else "nonfinite_omega"
                                quality["encoder_dt_s_values"].append(float(encoder_dt))
                        if not omega_valid:
                            quality["omega_invalid"] += 1
                            _bump_counter(quality, "omega_invalid_reasons", omega_invalid_reason)
                        encoder_state.last_angle = angle
                        encoder_state.last_timestamp = sample_mono
                        encoder_state.last_counts = counts
                        encoder_state.last_omega = omega if omega_valid else None
                        encoder_state.last_reply_mono = rx_mono

                        reply_block = reply.get("reply", {})
                        reply_bytes = reply_block.get("bytes") if isinstance(reply_block, Mapping) else None
                        reply_parsed = reply_block.get("parsed") if isinstance(reply_block, Mapping) else None
                        reply_latency_ms = (rx_mono_ns - int(query["tx_mono_ns"])) / 1e6
                        quality["reply_latency_ms_values"].append(float(reply_latency_ms))

                        settled = 0
                        if query["settle_phase"] and omega is not None and math.isfinite(float(omega)):
                            settled = int(abs(float(omega)) <= float(args.settle_rate_rad_s))

                        quality["samples"] += 1
                        quality["valid_encoder"] += 1
                        _bump_counter(quality, "samples_by_axis", axis_cfg.axis)
                        _bump_counter(quality, "samples_by_phase", str(query["phase"]))

                        writer.writerow(
                            {
                                "sample_idx": sample_idx,
                                "setting_id": query["setting_id"],
                                "axis": axis_cfg.axis,
                                "trial": query["trial"],
                                "direction": query["direction"],
                                "phase": query["phase"],
                                "rate_setting_rad_s": query["rate"],
                                "accel_byte": query["accel_byte"],
                                "cmd_rate_rad_s": query["requested_rate"],
                                "cmd_rate_applied_rad_s": query["applied_rate"],
                                "limit_blocked": query["limit_blocked"],
                                "command_addrs": ";".join(str(a) for a in axis_cfg.command_addrs),
                                "command_payloads_hex": _payload_text(query["payloads"]),
                                "command_tx_monotonic_ns": query["tx_mono_ns"],
                                "command_tx_wall_ns": query["tx_wall_ns"],
                                "response_rx_monotonic_ns": rx_mono_ns,
                                "response_rx_wall_ns": rx_wall_ns,
                                "elapsed_s": rx_mono - run_start_mono,
                                "setting_elapsed_s": rx_mono - query["setting_start"],
                                "phase_elapsed_s": rx_mono - query["phase_start"],
                                "encoder_addr": axis_cfg.encoder_addr,
                                "counts": counts,
                                "angle_rad": angle,
                                "omega_rad_s": omega,
                                "encoder_dt_s": encoder_dt,
                                "profile": args.profile,
                                "segment_id": query["segment_id"],
                                "profile_step_idx": query["profile_step_idx"],
                                "profile_elapsed_s": query["profile_elapsed_s"],
                                "requested_rate_source": query["requested_rate_source"],
                                "command_cmd_ids": ";".join(query["command_cmd_ids"]),
                                "reply_cmd_id": reply_cmd_id,
                                "reply_bytes_hex": _bytes_hex(reply_bytes),
                                "reply_parsed_json": _json_cell(reply_parsed),
                                "send_ok": int(bool(query["send_ok"])),
                                "valid_encoder": 1,
                                "settle_phase": int(bool(query["settle_phase"])),
                                "settled": settled,
                                "settle_timeout": 0,
                                "missing_reply": 0,
                                "send_dropped": int(not bool(query["send_ok"])),
                                "reply_latency_ms": reply_latency_ms,
                                "pending_query_count": len(pending_encoder_queries),
                                "dropped_query_count": quality["dropped_queries"],
                                "stale_reply_count": quality["stale_replies"],
                                "encoder_sample_monotonic_ns": sample_mono_ns,
                                "omega_valid": omega_valid,
                                "omega_invalid_reason": omega_invalid_reason,
                            }
                        )
                        csv_handle.flush()
                        sample_idx += 1
                        recorded += 1
                    return recorded

                for accel_byte in accel_values:
                    axis_cfg = replace(base_axis_cfg, accel_byte=int(accel_byte))
                    last_axis_cfg = axis_cfg
                    for rate in args.rates:
                        setting_id += 1
                        for trial in range(1, args.repeat + 1):
                            for direction in _directions(args.directions):
                                if stop_event.is_set():
                                    break
                                profile_seed = _profile_seed(
                                    int(args.seed),
                                    profile=args.profile,
                                    axis=axis_cfg.axis,
                                    accel_byte=axis_cfg.accel_byte,
                                    rate=float(rate),
                                    trial=trial,
                                    direction=direction,
                                )
                                segments = _build_command_profile(
                                    args,
                                    rate=float(rate),
                                    direction=direction,
                                    seed=profile_seed,
                                )
                                setting_start = time.monotonic()
                                _LOG.info(
                                    "setting=%d axis=%s profile=%s rate=%+.4f accel=%d trial=%d/%d",
                                    setting_id,
                                    axis_cfg.axis,
                                    args.profile,
                                    direction * rate,
                                    axis_cfg.accel_byte,
                                    trial,
                                    args.repeat,
                                )
                                for segment in segments:
                                    if stop_event.is_set():
                                        break
                                    phase_start = time.monotonic()
                                    next_tick = phase_start
                                    settled_since: Optional[float] = None
                                    while not stop_event.is_set():
                                        now = time.monotonic()
                                        phase_elapsed = now - phase_start
                                        if not segment.settle_phase and phase_elapsed >= segment.duration_s:
                                            break
                                        if segment.settle_phase and phase_elapsed >= segment.duration_s:
                                            if (
                                                settled_since is not None
                                                and now - settled_since >= args.settle_hold_s
                                            ):
                                                _LOG.info(
                                                    "%s settled: |omega| <= %.4f rad/s for %.3f s",
                                                    axis_cfg.axis,
                                                    args.settle_rate_rad_s,
                                                    now - settled_since,
                                                )
                                                break
                                            if phase_elapsed >= segment.duration_s + args.settle_timeout_s:
                                                quality["settle_timeouts"] += 1
                                                raise RuntimeError(
                                                    f"{axis_cfg.axis} failed to settle during {segment.phase} "
                                                    f"within {args.settle_timeout_s:.3f} s"
                                                )
                                        if now < next_tick:
                                            time.sleep(min(0.01, next_tick - now))
                                            continue

                                        requested_rate = float(segment.rate_rad_s)
                                        applied_rate = _apply_hard_angle_limit(
                                            requested_rate,
                                            encoder_state.last_angle,
                                            axis_cfg.angle_min_rad,
                                            axis_cfg.angle_max_rad,
                                            axis_cfg.axis,
                                        )
                                        limit_blocked = int(requested_rate != 0.0 and applied_rate == 0.0)
                                        if limit_blocked:
                                            quality["limit_blocked"] += 1
                                        payloads: list[tuple[int, Sequence[int]]] = []
                                        speed_commands = []
                                        command_cmd_ids: list[str] = []
                                        for addr, sign, label in zip(
                                            axis_cfg.command_addrs,
                                            axis_cfg.command_signs,
                                            axis_cfg.command_labels,
                                            strict=True,
                                        ):
                                            payload = _encode_speed_cmd(
                                                sign * applied_rate,
                                                acc=axis_cfg.accel_byte,
                                                gear_ratio=axis_cfg.gear_ratio,
                                                max_rate=axis_cfg.rate_limit,
                                            )
                                            payloads.append((addr, payload))
                                            speed_commands.append(
                                                _build_command(
                                                    cmd_id=f"sweep:{label}:{time.time_ns()}",
                                                    func="F6",
                                                    addr=addr,
                                                    payload=payload,
                                                    expect_reply=axis_cfg.respond_on_writes,
                                                    expected_len=1 if axis_cfg.respond_on_writes else None,
                                                    priority="high",
                                                    target=serial_target,
                                                )
                                            )
                                        payload_key = tuple(
                                            (int(addr), tuple(int(byte) & 0xFF for byte in payload))
                                            for addr, payload in payloads
                                        )
                                        refresh_due = (
                                            args.command_refresh_s > 0.0
                                            and now - last_speed_command_mono >= args.command_refresh_s
                                        )
                                        send_speed = payload_key != last_speed_payloads or refresh_due
                                        commands = []
                                        if send_speed:
                                            commands.extend(speed_commands)
                                            command_cmd_ids.extend(str(command["cmd_id"]) for command in speed_commands)
                                            last_speed_payloads = payload_key
                                            last_speed_command_mono = now
                                        encoder_cmd_id = f"sweep:enc:{axis_cfg.axis}:{time.time_ns()}"
                                        command_cmd_ids.append(encoder_cmd_id)
                                        commands.append(
                                            _build_command(
                                                cmd_id=encoder_cmd_id,
                                                func="0x31",
                                                addr=axis_cfg.encoder_addr,
                                                payload=[],
                                                expect_reply=True,
                                                expected_len=6,
                                                priority="high",
                                                target=serial_target,
                                            )
                                        )
                                        tx_mono_ns = time.monotonic_ns()
                                        tx_wall_ns = time.time_ns()
                                        query = {
                                            "setting_id": setting_id,
                                            "trial": trial,
                                            "direction": direction,
                                            "phase": segment.phase,
                                            "rate": rate,
                                            "accel_byte": axis_cfg.accel_byte,
                                            "requested_rate": requested_rate,
                                            "applied_rate": applied_rate,
                                            "limit_blocked": limit_blocked,
                                            "payloads": payloads,
                                            "command_cmd_ids": command_cmd_ids,
                                            "tx_mono_ns": tx_mono_ns,
                                            "tx_wall_ns": tx_wall_ns,
                                            "setting_start": setting_start,
                                            "phase_start": phase_start,
                                            "segment_id": segment.segment_id,
                                            "profile_step_idx": segment.profile_step_idx,
                                            "profile_elapsed_s": segment.profile_elapsed_s + phase_elapsed,
                                            "requested_rate_source": segment.requested_rate_source,
                                            "settle_phase": segment.settle_phase,
                                            "send_ok": False,
                                        }
                                        pending_encoder_queries[encoder_cmd_id] = query
                                        if len(pending_encoder_queries) > 1000:
                                            oldest_id = next(iter(pending_encoder_queries))
                                            pending_encoder_queries.pop(oldest_id, None)
                                            quality["dropped_queries"] += 1
                                            _LOG.warning("discarding stale unmatched encoder query %s", oldest_id)
                                        send_ok = update_pub.send_update(
                                            _build_update(
                                                source="jetson.gimbal_response_sweep",
                                                target=serial_target,
                                                commands=commands,
                                            )
                                        )
                                        query["send_ok"] = send_ok
                                        if not send_ok:
                                            quality["send_dropped"] += 1
                                            _LOG.warning("serial update publish dropped")

                                        recorded = record_available_replies(axis_cfg)
                                        if segment.settle_phase and recorded > 0:
                                            omega = encoder_state.last_omega
                                            if (
                                                omega is not None
                                                and math.isfinite(omega)
                                                and abs(omega) <= args.settle_rate_rad_s
                                            ):
                                                if settled_since is None:
                                                    settled_since = encoder_state.last_reply_mono
                                            else:
                                                settled_since = None
                                        next_tick += period_s

                                    drain_deadline = time.monotonic() + args.reply_drain_s
                                    while pending_encoder_queries and time.monotonic() < drain_deadline:
                                        record_available_replies(axis_cfg)
                                        if pending_encoder_queries:
                                            time.sleep(min(0.005, period_s))
                                    if pending_encoder_queries:
                                        missing = len(pending_encoder_queries)
                                        quality["missing_replies"] += missing
                                        pending_encoder_queries.clear()
                                        raise RuntimeError(
                                            f"{axis_cfg.axis} phase {segment.phase} ended with "
                                            f"{missing} missing encoder response(s)"
                                        )

                                if args.warn_no_encoder_s > 0.0 and (
                                    encoder_state.last_reply_mono is None
                                    or time.monotonic() - encoder_state.last_reply_mono > args.warn_no_encoder_s
                                ):
                                    _LOG.warning("no recent encoder response for axis=%s", axis_cfg.axis)
                            if stop_event.is_set():
                                break
                        if stop_event.is_set():
                            break
                    if stop_event.is_set():
                        break
                if update_pub is not None:
                    _send_zero_speed(update_pub, replace(base_axis_cfg, accel_byte=base_axis_cfg.accel_byte), serial_target)
                    time.sleep(0.05)
                if stop_event.is_set():
                    break

        _LOG.info("sweep complete: samples=%d csv=%s manifest=%s", sample_idx, output_csv, manifest_path)
        return 0
    except KeyboardInterrupt:
        _LOG.info("interrupted; stopping motors")
        final_status = "interrupted"
        final_error = "KeyboardInterrupt"
        return 130
    except RuntimeError as exc:
        _LOG.error("sweep aborted: %s", exc)
        final_status = "error"
        final_error = str(exc)
        return 1
    finally:
        if update_pub is not None and last_axis_cfg is not None:
            try:
                _send_zero_speed(update_pub, last_axis_cfg, serial_target)
                time.sleep(0.05)
                _send_zero_speed(update_pub, last_axis_cfg, serial_target)
            except Exception as exc:  # noqa: BLE001
                _LOG.error("failed to send final zero-speed command: %s", exc)
        try:
            _write_manifest(
                manifest_path,
                args=args,
                config_paths=config_paths,
                cfg=cfg,
                axis_configs=axis_configs,
                output_csv=output_csv,
                quality=quality,
                status=final_status,
                error=final_error,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.error("failed to write final manifest: %s", exc)
        if update_pub is not None:
            update_pub.close()
        if reply_sub is not None:
            reply_sub.close()
        if ctx is not None:
            ctx.term()
        if serial_proc is not None:
            _stop_serial_io_service(serial_proc)


if __name__ == "__main__":
    sys.exit(main())
