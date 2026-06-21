"""Record raw motor/encoder responses over a configurable command sweep.

This utility deliberately performs no system identification or controller
tuning.  It exercises yaw and/or pitch over a matrix of speed commands and MKS
acceleration-byte settings, then writes a flat CSV plus a JSON manifest for
offline MATLAB/Simulink work.

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
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import asdict, replace
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

CSV_FIELDS = [
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
    parser.add_argument("--pre-roll-s", type=float, default=0.5)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--post-roll-s", type=float, default=1.0)
    parser.add_argument("--rest-s", type=float, default=0.5)
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
    parser.add_argument("--start-serial-io", action="store_true")
    parser.add_argument("--serial-io-wait-s", type=float, default=0.5)
    return parser.parse_args()


def _directions(selection: str) -> list[int]:
    if selection == "positive":
        return [1]
    if selection == "negative":
        return [-1]
    return [1, -1]


def _payload_text(payloads: Sequence[tuple[int, Sequence[int]]]) -> str:
    return ";".join(f"{addr}:" + "".join(f"{byte:02X}" for byte in payload) for addr, payload in payloads)


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


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    config_paths: Sequence[str],
    cfg: Mapping[str, Any],
    axis_configs: Sequence[AxisConfig],
    output_csv: Path,
) -> None:
    data = {
        "format": "idcs.gimbal_response_sweep",
        "version": 1,
        "created_wall_ns": time.time_ns(),
        "output_csv": str(output_csv),
        "config_paths": [str(p) for p in config_paths],
        "sweep": {
            "axes": [axis.axis for axis in axis_configs],
            "rates_rad_s": list(args.rates),
            "accel_bytes": list(args.accel_bytes) if args.accel_bytes is not None else None,
            "directions": args.directions,
            "repeat": args.repeat,
            "sample_hz": args.sample_hz,
            "pre_roll_s": args.pre_roll_s,
            "step_s": args.step_s,
            "post_roll_s": args.post_roll_s,
            "rest_s": args.rest_s,
            "settle_rate_rad_s": args.settle_rate_rad_s,
            "settle_hold_s": args.settle_hold_s,
            "settle_timeout_s": args.settle_timeout_s,
            "reply_drain_s": args.reply_drain_s,
        },
        "axis_configs": [asdict(axis) for axis in axis_configs],
        "merged_config": cfg,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
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
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            return f"--{name.replace('_', '-')} must be non-negative and finite"
    if args.step_s <= 0.0:
        return "--step-s must be > 0"
    return None


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
    if not args.assume_exclusive:
        _LOG.warning(
            "Stop gimbal_bridge, manual_control, and other F6 publishers before testing. "
            "Use --assume-exclusive once the serial bus is exclusively owned."
        )

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
    _write_manifest(
        manifest_path,
        args=args,
        config_paths=config_paths,
        cfg=cfg,
        axis_configs=axis_configs,
        output_csv=output_csv,
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
                return 2

        ctx = zmq.Context()
        update_pub = SerialUpdatePublisher(serial_update_ep, ctx=ctx)
        reply_sub = SerialReplySubscriber(
            serial_reply_ep,
            topics=[f"serial.reply.{serial_target}"],
            ctx=ctx,
        )

        with output_csv.open("w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for base_axis_cfg in axis_configs:
                accel_values = args.accel_bytes or [base_axis_cfg.accel_byte]
                encoder_state = EncoderState()
                pending_encoder_queries: dict[str, dict[str, Any]] = {}

                def record_available_replies(axis_cfg: AxisConfig) -> int:
                    nonlocal sample_idx
                    recorded = 0
                    for reply in reply_sub.recv_nowait():
                        if (
                            _reply_func_byte(reply) != 0x31
                            or reply.get("addr") != axis_cfg.encoder_addr
                        ):
                            continue
                        query = pending_encoder_queries.pop(str(reply.get("cmd_id", "")), None)
                        if query is None:
                            continue
                        counts = _extract_counts(reply)
                        if counts is None:
                            continue
                        rx_mono_ns = time.monotonic_ns()
                        rx_wall_ns = time.time_ns()
                        rx_mono = rx_mono_ns / 1e9
                        angle = axis_cfg.encoder_sign * _counts_to_rad(
                            counts,
                            counts_per_rev=axis_cfg.counts_per_rev,
                            gear_ratio=axis_cfg.gear_ratio,
                        )
                        omega = None
                        encoder_dt = None
                        if (
                            encoder_state.last_angle is not None
                            and encoder_state.last_timestamp is not None
                        ):
                            encoder_dt = rx_mono - encoder_state.last_timestamp
                            if encoder_dt > 0.0:
                                omega = (
                                    _wrapped_delta(angle, encoder_state.last_angle) / encoder_dt
                                )
                        encoder_state.last_angle = angle
                        encoder_state.last_timestamp = rx_mono
                        encoder_state.last_counts = counts
                        encoder_state.last_omega = omega
                        encoder_state.last_reply_mono = rx_mono

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
                                "limit_blocked": int(
                                    query["requested_rate"] != 0.0
                                    and query["applied_rate"] == 0.0
                                ),
                                "command_addrs": ";".join(
                                    str(a) for a in axis_cfg.command_addrs
                                ),
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
                                setting_start = time.monotonic()
                                _LOG.info(
                                    "setting=%d axis=%s rate=%+.4f accel=%d trial=%d/%d",
                                    setting_id,
                                    axis_cfg.axis,
                                    direction * rate,
                                    axis_cfg.accel_byte,
                                    trial,
                                    args.repeat,
                                )
                                phases = [
                                    ("pre", 0.0, args.pre_roll_s),
                                    ("step", direction * rate, args.step_s),
                                    ("post", 0.0, args.post_roll_s),
                                    ("rest", 0.0, args.rest_s),
                                ]
                                for phase, requested_rate, duration_s in phases:
                                    settle_phase = phase in {"pre", "post"}
                                    if (duration_s <= 0.0 and not settle_phase) or stop_event.is_set():
                                        continue
                                    phase_start = time.monotonic()
                                    next_tick = phase_start
                                    settled_since: Optional[float] = None
                                    while not stop_event.is_set():
                                        now = time.monotonic()
                                        phase_elapsed = now - phase_start
                                        if not settle_phase and phase_elapsed >= duration_s:
                                            break
                                        if settle_phase and phase_elapsed >= duration_s:
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
                                            if phase_elapsed >= duration_s + args.settle_timeout_s:
                                                raise RuntimeError(
                                                    f"{axis_cfg.axis} failed to settle during {phase} "
                                                    f"within {args.settle_timeout_s:.3f} s"
                                                )
                                        if now < next_tick:
                                            time.sleep(min(0.01, next_tick - now))
                                            continue

                                        applied_rate = _apply_hard_angle_limit(
                                            requested_rate,
                                            encoder_state.last_angle,
                                            axis_cfg.angle_min_rad,
                                            axis_cfg.angle_max_rad,
                                            axis_cfg.axis,
                                        )
                                        payloads: list[tuple[int, Sequence[int]]] = []
                                        commands = []
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
                                            commands.append(
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
                                        encoder_cmd_id = f"sweep:enc:{axis_cfg.axis}:{time.time_ns()}"
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
                                        pending_encoder_queries[encoder_cmd_id] = {
                                            "setting_id": setting_id,
                                            "trial": trial,
                                            "direction": direction,
                                            "phase": phase,
                                            "rate": rate,
                                            "accel_byte": axis_cfg.accel_byte,
                                            "requested_rate": requested_rate,
                                            "applied_rate": applied_rate,
                                            "payloads": payloads,
                                            "tx_mono_ns": tx_mono_ns,
                                            "tx_wall_ns": tx_wall_ns,
                                            "setting_start": setting_start,
                                            "phase_start": phase_start,
                                        }
                                        if len(pending_encoder_queries) > 1000:
                                            oldest_id = next(iter(pending_encoder_queries))
                                            pending_encoder_queries.pop(oldest_id, None)
                                            _LOG.warning("discarding stale unmatched encoder query %s", oldest_id)
                                        if not update_pub.send_update(
                                            _build_update(
                                                source="jetson.gimbal_response_sweep",
                                                target=serial_target,
                                                commands=commands,
                                            )
                                        ):
                                            _LOG.warning("serial update publish dropped")

                                        recorded = record_available_replies(axis_cfg)
                                        if settle_phase and recorded > 0:
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
                                        pending_encoder_queries.clear()
                                        raise RuntimeError(
                                            f"{axis_cfg.axis} phase {phase} ended with "
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
        return 130
    except RuntimeError as exc:
        _LOG.error("sweep aborted: %s", exc)
        return 1
    finally:
        if update_pub is not None and last_axis_cfg is not None:
            try:
                _send_zero_speed(update_pub, last_axis_cfg, serial_target)
                time.sleep(0.05)
                _send_zero_speed(update_pub, last_axis_cfg, serial_target)
            except Exception as exc:  # noqa: BLE001
                _LOG.error("failed to send final zero-speed command: %s", exc)
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
