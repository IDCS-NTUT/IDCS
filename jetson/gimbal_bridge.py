"""Bridge ControlCmd messages to MKS serial (TTL-to-RS485) gimbal motion commands.

This Jetson-side process subscribes to the ControlCmd PUB socket, translates
pan/tilt rate commands into MKS SR_CLOSE speed mode writes, and periodically
publishes encoder-derived :class:`CamState` telemetry. Dual-pitch rigs use a
shared group address for commands with opposing motor "Dir" settings so a
single speed command moves both actuators in mirrored directions (default
pitch motor A CCW, motor B CW).
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

import zmq
import yaml

from common.config_sync import merge_config_maps, parse_config_text, read_snapshot
from common.schemas import CamState, control_cmd_from_json
from common.serial_io import SerialReplySubscriber, SerialUpdatePublisher
from common.shutdown import install_signal_handlers
from common.gimbal.mks_servo42_rs485 import MksServo42Axis

_LOG = logging.getLogger(__name__)


def _parse_tcp_port(endpoint: str, name: str) -> int:
    try:
        port = int(endpoint.rsplit(":", 1)[1])
    except Exception as exc:  # noqa: BLE001 - defensive parsing
        raise SystemExit(f"invalid {name} endpoint: {endpoint!r}") from exc
    if port <= 0 or port > 65535:
        raise SystemExit(f"{name} port must be in 1..65535 (got {port})")
    return port


def _load_config(paths: Iterable[Path]) -> Mapping[str, Any]:
    configs = []
    for path in paths:
        snapshot = read_snapshot(path)
        configs.append(parse_config_text(snapshot.text, str(path)))
    return merge_config_maps(*configs)


def _auto_control_enabled(cfg: Mapping[str, Any]) -> bool:
    try:
        gimbal_cfg = cfg.get("gimbal") or {}
        return bool(gimbal_cfg.get("auto_control_enabled", False))
    except Exception:  # noqa: BLE001 - defensive config parsing
        return False


def _build_serial_targets(cfg: Mapping[str, Any]) -> Tuple[Mapping[str, Any], float]:
    gimbal_cfg = cfg.get("gimbal")
    if not isinstance(gimbal_cfg, Mapping):
        raise SystemExit("config missing 'gimbal' section")

    counts_per_rev = int(gimbal_cfg.get("counts_per_rev", 0x4000))
    yaw_ratio = float(gimbal_cfg.get("yaw_gear_ratio", 1.0))
    pitch_ratio = float(gimbal_cfg.get("pitch_gear_ratio", 1.0))

    yaw_addr = int(gimbal_cfg.get("yaw_addr", 1))
    yaw_group_addr = gimbal_cfg.get("yaw_group_addr")
    yaw_group_addr = int(yaw_group_addr) if yaw_group_addr is not None else None

    pitch_group_addr = gimbal_cfg.get("pitch_group_addr")
    if pitch_group_addr is None:
        raise SystemExit("gimbal.pitch_group_addr is required for dual-pitch setup")
    pitch_group_addr = int(pitch_group_addr)

    respond_on_writes = bool(gimbal_cfg.get("respond_on_writes", False))

    try:
        pitch_motor_a_addr = int(gimbal_cfg["pitch_motor_a_addr"])
        pitch_motor_b_addr = int(gimbal_cfg["pitch_motor_b_addr"])
    except KeyError as exc:
        raise SystemExit("gimbal.pitch_motor_a_addr and pitch_motor_b_addr are required") from exc

    authority = gimbal_cfg.get("pitch_encoder_authority", "a")
    if authority not in {"a", "b"}:
        raise SystemExit("gimbal.pitch_encoder_authority must be 'a' or 'b'")

    yaw_accel_byte = int(gimbal_cfg.get("yaw_accel_byte", 10))
    pitch_accel_byte = int(gimbal_cfg.get("pitch_accel_byte", 10))
    yaw_rate_limit = float(gimbal_cfg.get("yaw_rate_limit_rad_s", 10.0))
    pitch_rate_limit = float(gimbal_cfg.get("pitch_rate_limit_rad_s", 10.0))
    pitch_div_thresh = float(gimbal_cfg.get("pitch_divergence_thresh_rad", 0.0873))

    serial_targets = {
        "counts_per_rev": counts_per_rev,
        "yaw_ratio": yaw_ratio,
        "pitch_ratio": pitch_ratio,
        "yaw_addr": yaw_addr,
        "yaw_group_addr": yaw_group_addr,
        "pitch_group_addr": pitch_group_addr,
        "pitch_motor_a_addr": pitch_motor_a_addr,
        "pitch_motor_b_addr": pitch_motor_b_addr,
        "pitch_authority": authority,
        "respond_on_writes": respond_on_writes,
        "yaw_accel_byte": yaw_accel_byte,
        "pitch_accel_byte": pitch_accel_byte,
        "yaw_rate_limit": yaw_rate_limit,
        "pitch_rate_limit": pitch_rate_limit,
    }
    return serial_targets, pitch_div_thresh


def _load_parameter_map(path: Path) -> Mapping[int, Tuple[int, ...]]:
    snapshot = read_snapshot(path)
    data = yaml.safe_load(snapshot.text) or {}
    motors = data.get("motors") if isinstance(data, Mapping) else {}
    if not isinstance(motors, Mapping):
        raise SystemExit(f"parameter file {path} must contain a 'motors' mapping")

    parameter_map: dict[int, Tuple[int, ...]] = {}
    for addr_str, entry in motors.items():
        try:
            addr = int(addr_str)
        except Exception as exc:  # noqa: BLE001 - defensive config parsing
            raise SystemExit(f"invalid motor address key {addr_str!r} in {path}") from exc

        params = entry.get("parameters") if isinstance(entry, Mapping) else entry
        if params is None:
            _LOG.info("no parameters listed for motor %s; skipping", addr_str)
            continue
        try:
            payload = tuple(int(b) & 0xFF for b in params)
        except Exception as exc:  # noqa: BLE001 - defensive config parsing
            raise SystemExit(f"invalid parameter payload for motor {addr_str!r}: {entry}") from exc
        if len(payload) != 34:
            raise SystemExit(
                f"motor {addr} parameters must have 34 bytes (Byte4-Byte37); got {len(payload)}"
            )
        parameter_map[addr] = payload

    return parameter_map


def _build_param_command(
    addr: int,
    payload: Tuple[int, ...],
    *,
    expect_reply: bool,
    target: str,
) -> Mapping[str, Any]:
    return {
        "cmd_id": f"params:{addr}",
        "func": "0x46",
        "addr": addr,
        "payload": list(payload),
        "expect_reply": expect_reply,
        "expected_len": 1 if expect_reply else None,
        "priority": "high",
        "target": target,
    }


def _publish_cam_state(
    pub: zmq.Socket,
    sample,
    *,
    frame_id: int,
    src_ts_ms: int,
) -> None:
    cam_state = CamState(
        frame_id=frame_id,
        src_ts_ms=src_ts_ms,
        pan=float(sample.pan_rad),
        tilt=float(sample.tilt_rad),
        pan_rate=sample.pan_rate_rad_s,
        tilt_rate=sample.tilt_rate_rad_s,
    )
    pub.send_string(cam_state.model_dump_json(exclude_none=True))


def _make_control_sub(ctx: zmq.Context, endpoint: str) -> zmq.Socket:
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(endpoint)
    return sub


def _make_state_pub(ctx: zmq.Context, endpoint: Optional[str]) -> Optional[zmq.Socket]:
    if not endpoint:
        _LOG.warning("gimbal_state endpoint not configured; telemetry will not be published")
        return None
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)
    port = _parse_tcp_port(endpoint, "net.zmq_gimbal_state")
    pub.bind(f"tcp://0.0.0.0:{port}")
    _LOG.info("publishing CamState on tcp://0.0.0.0:%d", port)
    return pub


def _counts_to_rad(counts: int, *, counts_per_rev: int, gear_ratio: float) -> float:
    motor_revs = counts / float(counts_per_rev)
    axis_revs = motor_revs / gear_ratio
    return axis_revs * 2.0 * math.pi


def _encode_speed_cmd(
    omega_rad_s: float,
    *,
    acc: int,
    gear_ratio: float,
    max_rate: float,
) -> Tuple[int, int, int]:
    omega = max(min(omega_rad_s, max_rate), -max_rate)
    return MksServo42Axis._encode_speed_payload(omega, acc, gear_ratio)


def _wait_for_status(
    reply_sub: SerialReplySubscriber,
    expected_addrs: Iterable[int],
    *,
    timeout_s: float = 2.0,
) -> None:
    expected = set(expected_addrs)
    deadline = time.monotonic() + timeout_s
    while expected and time.monotonic() < deadline:
        for reply in reply_sub.recv_nowait():
            if reply.get("func") != "F1":
                continue
            addr = reply.get("addr")
            if addr in expected:
                status = reply.get("reply", {}).get("parsed", {}).get("status")
                if status in (None, 0):
                    raise SystemExit(f"status query failed for addr={addr}")
                _LOG.info("axis addr=%s status=%s", addr, status)
                expected.remove(addr)
        time.sleep(0.01)
    if expected:
        raise SystemExit(f"status query timed out for addr(s): {sorted(expected)}")


def _build_update(
    *,
    source: str,
    target: str,
    commands: Iterable[Mapping[str, Any]],
    fields: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    return {
        "type": "SerialUpdate",
        "source": source,
        "target": target,
        "fields": dict(fields or {}),
        "commands": list(commands),
        "update_ts_ms": int(time.time() * 1000),
    }


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
) -> Mapping[str, Any]:
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


@dataclass
class _AngleSample:
    timestamp: float
    pan_rad: float
    tilt_rad: float
    pan_rate_rad_s: Optional[float]
    tilt_rate_rad_s: Optional[float]
    secondary_pitch_rad: Optional[float] = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dev.yaml", help="Path to YAML config")
    ap.add_argument(
        "--config-extra",
        default="configs/dev_extra.yaml",
        help="Optional second YAML config merged over --config.",
    )
    ap.add_argument(
        "--feedback-hz",
        type=float,
        default=None,
        help="Override telemetry publish rate (Hz); defaults to gimbal.feedback_hz",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    config_paths = [Path(args.config)]
    if args.config_extra:
        config_paths.append(Path(args.config_extra))
    cfg = _load_config(config_paths)
    runtime_control_enabled = _auto_control_enabled(cfg)
    if not runtime_control_enabled:
        _LOG.warning(
            "Auto control disabled by config (gimbal.auto_control_enabled=false); applying ControlCmd setpoints will be skipped, but startup/zeroing/stop/disable sequences still run"
        )

    net_cfg = cfg.get("net") or {}
    ctrl_ep = net_cfg.get("zmq_control")
    if not ctrl_ep:
        raise SystemExit("config missing net.zmq_control endpoint")

    state_ep = net_cfg.get("zmq_gimbal_state")

    serial_targets, pitch_div_thresh = _build_serial_targets(cfg)
    parameter_map: Mapping[int, Tuple[int, ...]] = {}
    gimbal_cfg = cfg.get("gimbal") or {}
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

    param_path = gimbal_cfg.get("parameter_file")
    if param_path:
        parameter_map = _load_parameter_map(Path(str(param_path)))
        _LOG.info("loaded parameter sets for %d motors from %s", len(parameter_map), param_path)

    _LOG.info(
        "configured serial gimbal: yaw addr=%d group=%s, pitch group=%d authority=%s, divergence_thresh=%.4f rad",
        serial_targets["yaw_addr"],
        serial_targets["yaw_group_addr"],
        serial_targets["pitch_group_addr"],
        serial_targets["pitch_authority"],
        pitch_div_thresh,
    )

    feedback_hz = args.feedback_hz
    if feedback_hz is None:
        try:
            feedback_hz = float(cfg.get("gimbal", {}).get("feedback_hz", 20.0))
        except Exception:  # noqa: BLE001 - config parsing guard
            feedback_hz = 20.0
    feedback_hz = max(0.1, feedback_hz)
    feedback_period = 1.0 / feedback_hz

    stop_event = install_signal_handlers()

    ctx = zmq.Context()
    sub = _make_control_sub(ctx, ctrl_ep)
    pub = _make_state_pub(ctx, state_ep)
    update_pub = SerialUpdatePublisher(serial_update_ep, ctx=ctx)
    reply_sub = SerialReplySubscriber(
        serial_reply_ep,
        topics=[f"serial.reply.{serial_target}"],
        ctx=ctx,
    )
    _LOG.info("subscribing to ControlCmd on %s (feedback %.1f Hz)", ctrl_ep, feedback_hz)
    _LOG.info("publishing SerialUpdate to %s (target=%s)", serial_update_ep, serial_target)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    last_cmd = None
    last_pub_time = 0.0
    last_stats_log = 0.0
    last_sample: Optional[_AngleSample] = None
    last_divergence_log = 0.0
    local_frame_id = 0
    yaw_addr = int(serial_targets["yaw_addr"])
    pitch_group_addr = int(serial_targets["pitch_group_addr"])
    pitch_a_addr = int(serial_targets["pitch_motor_a_addr"])
    pitch_b_addr = int(serial_targets["pitch_motor_b_addr"])
    pitch_authority = serial_targets["pitch_authority"]
    yaw_ratio = float(serial_targets["yaw_ratio"])
    pitch_ratio = float(serial_targets["pitch_ratio"])
    yaw_accel = int(serial_targets["yaw_accel_byte"])
    pitch_accel = int(serial_targets["pitch_accel_byte"])
    yaw_rate_limit = float(serial_targets["yaw_rate_limit"])
    pitch_rate_limit = float(serial_targets["pitch_rate_limit"])
    counts_per_rev = int(serial_targets["counts_per_rev"])
    pitch_authority_addr = pitch_a_addr if pitch_authority == "a" else pitch_b_addr

    startup_timeout_s = float(gimbal_cfg.get("timeout", 0.1)) * 20.0
    startup_timeout_s = max(2.0, startup_timeout_s)
    startup_retry_s = float(gimbal_cfg.get("startup_retry_s", 1.0))

    startup_start = time.monotonic()
    if parameter_map:
        param_cmds = [
            _build_param_command(
                addr,
                payload,
                expect_reply=bool(serial_targets["respond_on_writes"]),
                target=serial_target,
            )
            for addr, payload in parameter_map.items()
        ]
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=param_cmds,
            )
        )

    startup_attempt = 0
    while True:
        startup_attempt += 1
        _send_stop_commands("startup")
        enable_cmds = [
            _build_command(
                cmd_id="enable:yaw",
                func="F3",
                addr=yaw_addr,
                payload=[0x01],
                expect_reply=True,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
            _build_command(
                cmd_id="enable:pitch",
                func="F3",
                addr=pitch_group_addr,
                payload=[0x01],
                expect_reply=True,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
        ]
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=enable_cmds,
            )
        )
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=[
                    _build_command(
                        cmd_id="zero:yaw",
                        func="0x92",
                        addr=yaw_addr,
                        payload=[],
                        expect_reply=True,
                        expected_len=None,
                        priority="high",
                        target=serial_target,
                    ),
                    _build_command(
                        cmd_id="zero:pitch_a",
                        func="0x92",
                        addr=pitch_a_addr,
                        payload=[],
                        expect_reply=True,
                        expected_len=None,
                        priority="high",
                        target=serial_target,
                    ),
                    _build_command(
                        cmd_id="zero:pitch_b",
                        func="0x92",
                        addr=pitch_b_addr,
                        payload=[],
                        expect_reply=True,
                        expected_len=None,
                        priority="high",
                        target=serial_target,
                    ),
                ],
            )
        )
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=[
                    _build_command(
                        cmd_id="status:yaw",
                        func="F1",
                        addr=yaw_addr,
                        payload=[],
                        expect_reply=True,
                        expected_len=1,
                        priority="high",
                        target=serial_target,
                    ),
                    _build_command(
                        cmd_id="status:pitch_a",
                        func="F1",
                        addr=pitch_a_addr,
                        payload=[],
                        expect_reply=True,
                        expected_len=1,
                        priority="high",
                        target=serial_target,
                    ),
                    _build_command(
                        cmd_id="status:pitch_b",
                        func="F1",
                        addr=pitch_b_addr,
                        payload=[],
                        expect_reply=True,
                        expected_len=1,
                        priority="high",
                        target=serial_target,
                    ),
                ],
            )
        )
        try:
            _wait_for_status(
                reply_sub,
                [yaw_addr, pitch_a_addr, pitch_b_addr],
                timeout_s=startup_timeout_s,
            )
        except SystemExit as exc:
            _LOG.warning("startup status failed (attempt %d): %s", startup_attempt, exc)
            time.sleep(startup_retry_s)
            continue
        break
    startup_elapsed = time.monotonic() - startup_start
    _LOG.info("serial startup sequence completed in %.3f s", startup_elapsed)
    if not runtime_control_enabled:
        _LOG.info(
            "Runtime ControlCmd motion will be ignored; startup, zeroing, and shutdown commands remain active"
        )

    yaw_counts: Optional[int] = None
    pitch_counts: dict[int, int] = {}
    def _send_stop_commands(reason: str) -> None:
        _LOG.info("sending gimbal stop commands (%s)", reason)
        stop_cmds = [
            _build_command(
                cmd_id="stop:yaw",
                func="F6",
                addr=yaw_addr,
                payload=_encode_speed_cmd(
                    0.0, acc=yaw_accel, gear_ratio=yaw_ratio, max_rate=yaw_rate_limit
                ),
                expect_reply=False,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
            _build_command(
                cmd_id="stop:pitch",
                func="F6",
                addr=pitch_group_addr,
                payload=_encode_speed_cmd(
                    0.0,
                    acc=pitch_accel,
                    gear_ratio=pitch_ratio,
                    max_rate=pitch_rate_limit,
                ),
                expect_reply=False,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
            _build_command(
                cmd_id="disable:yaw",
                func="F3",
                addr=yaw_addr,
                payload=[0x00],
                expect_reply=False,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
            _build_command(
                cmd_id="disable:pitch",
                func="F3",
                addr=pitch_group_addr,
                payload=[0x00],
                expect_reply=False,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
        ]
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=stop_cmds,
            )
        )

    stop_commands_sent = False
    try:
        while not stop_event.is_set():
            timeout_ms = int(math.ceil(feedback_period * 1000))
            events = dict(poller.poll(timeout=timeout_ms))
            if events.get(sub) == zmq.POLLIN:
                payload = sub.recv()
                try:
                    last_cmd = control_cmd_from_json(payload)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("failed to decode ControlCmd: %s", exc)
                else:
                    if runtime_control_enabled:
                        yaw_payload = _encode_speed_cmd(
                            float(last_cmd.pan_rate_cmd),
                            acc=yaw_accel,
                            gear_ratio=yaw_ratio,
                            max_rate=yaw_rate_limit,
                        )
                        pitch_payload = _encode_speed_cmd(
                            float(last_cmd.tilt_rate_cmd),
                            acc=pitch_accel,
                            gear_ratio=pitch_ratio,
                            max_rate=pitch_rate_limit,
                        )
                        update_pub.send_update(
                            _build_update(
                                source="jetson.gimbal_bridge",
                                target=serial_target,
                                commands=[
                                    _build_command(
                                        cmd_id=f"speed:yaw:{time.time_ns()}",
                                        func="F6",
                                        addr=yaw_addr,
                                        payload=yaw_payload,
                                        expect_reply=False,
                                        expected_len=None,
                                        priority="high",
                                        target=serial_target,
                                    ),
                                    _build_command(
                                        cmd_id=f"speed:pitch:{time.time_ns()}",
                                        func="F6",
                                        addr=pitch_group_addr,
                                        payload=pitch_payload,
                                        expect_reply=False,
                                        expected_len=None,
                                        priority="high",
                                        target=serial_target,
                                    ),
                                ],
                                fields={
                                    "pan_rate_cmd": float(last_cmd.pan_rate_cmd),
                                    "tilt_rate_cmd": float(last_cmd.tilt_rate_cmd),
                                    "yaw_accel_byte": yaw_accel,
                                    "pitch_accel_byte": pitch_accel,
                                },
                            )
                        )
                    else:
                        _LOG.debug(
                            "Received ControlCmd while serial commands disabled; pan_rate_cmd=%.3f tilt_rate_cmd=%.3f ignored",
                            float(last_cmd.pan_rate_cmd),
                            float(last_cmd.tilt_rate_cmd),
                        )

            for reply in reply_sub.recv_nowait():
                func = _reply_func_byte(reply)
                addr = reply.get("addr")
                if func == 0x31 and isinstance(addr, int):
                    parsed = reply.get("reply", {}).get("parsed", {})
                    if "counts" in parsed:
                        if addr == yaw_addr:
                            yaw_counts = int(parsed["counts"])
                        else:
                            pitch_counts[addr] = int(parsed["counts"])

            now = time.monotonic()
            if pub is None:
                continue
            if (now - last_pub_time) < feedback_period:
                continue
            last_pub_time = now
            if yaw_counts is None or pitch_authority_addr not in pitch_counts:
                continue

            pan_rad = _counts_to_rad(
                yaw_counts, counts_per_rev=counts_per_rev, gear_ratio=yaw_ratio
            )
            tilt_rad = _counts_to_rad(
                pitch_counts[pitch_authority_addr],
                counts_per_rev=counts_per_rev,
                gear_ratio=pitch_ratio,
            )
            secondary_pitch_rad = None
            for addr, counts in pitch_counts.items():
                if addr != pitch_authority_addr:
                    secondary_pitch_rad = _counts_to_rad(
                        counts, counts_per_rev=counts_per_rev, gear_ratio=pitch_ratio
                    )
                    break

            pan_rate = tilt_rate = None
            if last_sample is not None:
                dt = now - last_sample.timestamp
                if dt > 0:
                    pan_rate = (pan_rad - last_sample.pan_rad) / dt
                    tilt_rate = (tilt_rad - last_sample.tilt_rad) / dt

            sample = _AngleSample(
                timestamp=now,
                pan_rad=pan_rad,
                tilt_rad=tilt_rad,
                pan_rate_rad_s=pan_rate,
                tilt_rate_rad_s=tilt_rate,
                secondary_pitch_rad=secondary_pitch_rad,
            )
            last_sample = sample
            if sample.secondary_pitch_rad is not None:
                divergence = abs(sample.secondary_pitch_rad - sample.tilt_rad)
                if divergence >= pitch_div_thresh and (now - last_divergence_log) >= 2.0:
                    last_divergence_log = now
                    _LOG.warning(
                        "pitch encoder divergence %.4f rad exceeds threshold %.4f (primary=%.4f secondary=%.4f)",
                        divergence,
                        pitch_div_thresh,
                        float(sample.tilt_rad),
                        float(sample.secondary_pitch_rad),
                    )
            if last_cmd is not None:
                frame_id = int(last_cmd.frame_id)
                src_ts_ms = int(last_cmd.src_ts_ms)
            else:
                frame_id = local_frame_id
                local_frame_id += 1
                src_ts_ms = int(time.monotonic_ns() / 1e6)
            try:
                _publish_cam_state(pub, sample, frame_id=frame_id, src_ts_ms=src_ts_ms)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("failed to publish CamState: %s", exc)
            if (now - last_stats_log) >= 5.0 and last_sample is not None:
                last_stats_log = now
                pan_rate = (
                    last_sample.pan_rate_rad_s
                    if last_sample.pan_rate_rad_s is not None
                    else float("nan")
                )
                tilt_rate = (
                    last_sample.tilt_rate_rad_s
                    if last_sample.tilt_rate_rad_s is not None
                    else float("nan")
                )
                _LOG.info(
                    "gimbal heartbeat pan=%.3f tilt=%.3f pan_rate=%.3f tilt_rate=%.3f frame_id=%s",
                    float(last_sample.pan_rad),
                    float(last_sample.tilt_rad),
                    float(pan_rate),
                    float(tilt_rate),
                    getattr(last_cmd, "frame_id", "n/a"),
                )
    except KeyboardInterrupt:
        _send_stop_commands("keyboard interrupt")
        stop_commands_sent = True
        raise
    finally:
        if not stop_commands_sent:
            _send_stop_commands("shutdown")
        try:
            poller.unregister(sub)
        except Exception:  # noqa: BLE001
            pass
        try:
            sub.close(linger=0)
        except Exception:  # noqa: BLE001
            pass
        if pub is not None:
            try:
                poller.unregister(pub)
            except Exception:  # noqa: BLE001
                pass
            try:
                pub.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
        update_pub.close()
        reply_sub.close()
        try:
            ctx.destroy(linger=0)
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
