"""Standalone serial I/O service core.

Opens a serial port exclusively, runs a blocking command loop, applies
timeouts/retries, and publishes reply data immediately.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
import zmq

from common.config_sync import expand_config_paths, merge_config_maps, parse_config_text, read_snapshot
from common.gimbal.mks_servo42_rs485 import RS485Bus


_LOG = logging.getLogger(__name__)

_PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "normal": 2,
    "low": 3,
}

_STATUS_LABELS = {
    0x00: "Query failed",
    0x01: "Motor stopped",
    0x02: "Speeding up",
    0x03: "Slowing down",
    0x04: "Full speed",
    0x05: "Homing",
}

_F6_FUNC_BYTE = 0xF6
_F7_FUNC_BYTE = 0xF7
_MULTI_FRAME_MAX_COMMANDS = 5
_DEFAULT_SINGLE_BYTE_REPLY_FUNCS = {0xF3, 0xF6, 0xF7, 0x92, 0x46}
_reply_sequence = 0


@dataclass
class CommandSpec:
    name: str
    func: str
    addr: int
    payload: Tuple[int, ...]
    expect_reply: bool
    expected_len: Optional[int]
    interval_ms: int
    priority: str
    target: str


@dataclass
class ScheduledCommand:
    spec: CommandSpec
    next_due_ts_ms: int


@dataclass
class SerialCommand:
    cmd_id: str
    func: str
    addr: int
    payload: Tuple[int, ...]
    expect_reply: bool
    expected_len: Optional[int]
    priority: str
    target: str
    timeout_ms: Optional[int]
    retry: Optional[int]
    sent_ts_ms: Optional[int] = None
    enqueued_monotonic_ns: Optional[int] = None


@dataclass
class AckResponse:
    accepted: bool
    queued: bool
    queue_position: Optional[int]
    reason: Optional[str]


class StopFlag:
    def __init__(self) -> None:
        self._stop = False

    def set(self) -> None:
        self._stop = True

    def is_set(self) -> bool:
        return self._stop


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument(
        "--config-extra",
        default=None,
        help="Optional second YAML config merged over --config.",
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial device path")
    parser.add_argument("--baud", type=int, default=256000, help="Serial baudrate")
    parser.add_argument("--timeout", type=float, default=0.1, help="Serial timeout (s)")
    parser.add_argument("--retries", type=int, default=1, help="Serial retry count")
    parser.add_argument(
        "--command-endpoint",
        default="tcp://127.0.0.1:5570",
        help="ZMQ REP endpoint for SerialCommandRequest",
    )
    parser.add_argument(
        "--update-endpoint",
        default="tcp://127.0.0.1:5571",
        help="ZMQ SUB endpoint for SerialUpdate messages",
    )
    parser.add_argument(
        "--reply-endpoint",
        default="tcp://127.0.0.1:5572",
        help="ZMQ PUB endpoint for data-bearing replies",
    )
    parser.add_argument(
        "--idle-sleep-ms",
        type=int,
        default=5,
        help="Idle sleep time between rounds (ms)",
    )
    return parser.parse_args()


def _load_config(paths: Sequence[Optional[str]]) -> Mapping[str, Any]:
    configs = []
    for path in paths:
        if not path:
            continue
        cfg_path = Path(path)
        snapshot = read_snapshot(cfg_path)
        configs.append(parse_config_text(snapshot.text, str(cfg_path)))
    return merge_config_maps(*configs)


def _parse_schedule(cfg: Mapping[str, Any]) -> List[ScheduledCommand]:
    serial_cfg = cfg.get("serial_io") if isinstance(cfg, Mapping) else None
    if not isinstance(serial_cfg, Mapping):
        return []
    schedule = serial_cfg.get("schedule")
    if not isinstance(schedule, list):
        return []
    scheduled: List[ScheduledCommand] = []
    now_ms = int(time.time() * 1000)
    for entry in schedule:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "unnamed"))
        func = str(entry.get("func"))
        addr = int(entry.get("addr", 1))
        payload = tuple(int(b) & 0xFF for b in entry.get("payload", []))
        expect_reply = bool(entry.get("expect_reply", True))
        expected_len = entry.get("expected_len")
        expected_len = int(expected_len) if expected_len is not None else None
        interval_ms = int(entry.get("interval_ms", 1000))
        priority = str(entry.get("priority", "normal"))
        target = str(entry.get("target", "gimbal"))
        spec = CommandSpec(
            name=name,
            func=func,
            addr=addr,
            payload=payload,
            expect_reply=expect_reply,
            expected_len=expected_len,
            interval_ms=interval_ms,
            priority=priority,
            target=target,
        )
        scheduled.append(ScheduledCommand(spec=spec, next_due_ts_ms=now_ms))
    return scheduled


def _priority_key(priority: str) -> int:
    return _PRIORITY_ORDER.get(priority, _PRIORITY_ORDER["normal"])


def _is_f6_command(cmd: SerialCommand) -> bool:
    try:
        return _func_to_byte(cmd.func) == _F6_FUNC_BYTE
    except Exception:  # noqa: BLE001
        return False


def _is_runtime_speed_command(cmd: SerialCommand) -> bool:
    if not _is_f6_command(cmd):
        return False
    if cmd.expect_reply:
        return False
    if _is_critical_command(cmd):
        return False
    cmd_id = str(cmd.cmd_id)
    return cmd_id.startswith("speed:yaw:") or cmd_id.startswith("speed:pitch_a:") or cmd_id.startswith("speed:pitch_b:")


def _can_use_multi_frame(cmd: SerialCommand) -> bool:
    if not _is_runtime_speed_command(cmd):
        return False
    return len(cmd.payload) <= 8


def _send_multi_frame_batch(bus: RS485Bus, batch: Sequence[SerialCommand]) -> None:
    if not batch:
        return
    slots = [
        (cmd.addr, _func_to_byte(cmd.func), cmd.payload)
        for cmd in batch
    ]
    bus.send_multi_command_frame(slots)
    _LOG.debug(
        "sent multi-command frame with %d command(s): %s",
        len(batch),
        [f"{cmd.cmd_id}@{cmd.addr}:{cmd.func}" for cmd in batch],
    )


def _is_critical_command(cmd: SerialCommand) -> bool:
    try:
        func = _func_to_byte(cmd.func)
    except Exception:  # noqa: BLE001
        return cmd.priority == "critical"
    if func == _F7_FUNC_BYTE:
        return True
    if func == 0xF3 and cmd.payload and cmd.payload[0] == 0x00:
        return True
    return cmd.priority == "critical"


def _effective_priority_key(cmd: SerialCommand) -> int:
    if _is_critical_command(cmd):
        return _PRIORITY_ORDER["critical"]
    return _priority_key(cmd.priority)


def _coalesce_key(cmd: SerialCommand) -> Optional[Tuple[str, int, int]]:
    if not _is_f6_command(cmd):
        return None
    return (cmd.target, cmd.addr, _func_to_byte(cmd.func))


def _get_stale_threshold_ms(cfg: Mapping[str, Any]) -> int:
    serial_cfg = cfg.get("serial_io") if isinstance(cfg, Mapping) else None
    if not isinstance(serial_cfg, Mapping):
        return 120
    raw_value = serial_cfg.get("f6_stale_threshold_ms", 120)
    try:
        return max(int(raw_value), 0)
    except Exception:  # noqa: BLE001
        _LOG.warning("invalid f6_stale_threshold_ms=%r, using default 120", raw_value)
        return 120


def _parse_startup(cfg: Mapping[str, Any]) -> List[SerialCommand]:
    serial_cfg = cfg.get("serial_io") if isinstance(cfg, Mapping) else None
    if not isinstance(serial_cfg, Mapping):
        return []
    startup = serial_cfg.get("startup")
    if not isinstance(startup, list):
        return []
    commands: List[SerialCommand] = []
    for idx, entry in enumerate(startup):
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", f"{idx}"))
        try:
            cmd = SerialCommand(
                cmd_id=f"startup:{name}:{idx}",
                func=str(entry["func"]),
                addr=int(entry.get("addr", 1)),
                payload=tuple(int(b) & 0xFF for b in entry.get("payload", [])),
                expect_reply=bool(entry.get("expect_reply", True)),
                expected_len=(
                    int(entry["expected_len"]) if entry.get("expected_len") is not None else None
                ),
                priority=str(entry.get("priority", "high")),
                target=str(entry.get("target", "gimbal")),
                timeout_ms=(
                    int(entry["timeout_ms"]) if entry.get("timeout_ms") is not None else None
                ),
                retry=int(entry["retry"]) if entry.get("retry") is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("invalid startup command entry: %s", exc)
            continue
        errors = _validate_command(cmd)
        if errors:
            _LOG.warning("invalid startup command: %s", "; ".join(errors))
            continue
        commands.append(cmd)
    return commands


def _decode_cmd(data: bytes) -> Tuple[Optional[SerialCommand], AckResponse]:
    enqueued_monotonic_ns = time.monotonic_ns()
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, AckResponse(False, False, None, f"invalid json: {exc}")

    if payload.get("type") != "SerialCommandRequest":
        return None, AckResponse(False, False, None, "unexpected message type")

    try:
        cmd = SerialCommand(
            cmd_id=str(payload["cmd_id"]),
            func=str(payload["func"]),
            addr=int(payload["addr"]),
            payload=tuple(int(b) & 0xFF for b in payload.get("payload", [])),
            expect_reply=bool(payload.get("expect_reply", True)),
            expected_len=(
                int(payload["expected_len"]) if payload.get("expected_len") is not None else None
            ),
            priority=str(payload.get("priority", "normal")),
            target=str(payload.get("target", "gimbal")),
            timeout_ms=(
                int(payload["timeout_ms"]) if payload.get("timeout_ms") is not None else None
            ),
            retry=int(payload["retry"]) if payload.get("retry") is not None else None,
            sent_ts_ms=(
                int(payload["sent_ts_ms"]) if payload.get("sent_ts_ms") is not None else None
            ),
            enqueued_monotonic_ns=enqueued_monotonic_ns,
        )
    except Exception as exc:  # noqa: BLE001
        return None, AckResponse(False, False, None, f"invalid command payload: {exc}")

    errors = _validate_command(cmd)
    if errors:
        return None, AckResponse(False, False, None, "; ".join(errors))

    return cmd, AckResponse(True, True, None, None)


def _ack_message(cmd_id: Optional[str], ack: AckResponse) -> str:
    msg = {
        "type": "SerialCommandAck",
        "cmd_id": cmd_id,
        "accepted": ack.accepted,
        "queued": ack.queued,
        "queue_position": ack.queue_position,
        "reason": ack.reason,
        "ack_ts_ms": int(time.time() * 1000),
    }
    return json.dumps(msg)


def _func_to_byte(func: str) -> int:
    if func.lower().startswith("0x"):
        return int(func, 16)
    if func.upper().startswith("F"):
        return int(func, 16)
    return int(func)


def _validate_command(cmd: SerialCommand) -> List[str]:
    errors: List[str] = []
    if not cmd.cmd_id:
        errors.append("cmd_id is required")
    if not (0 <= cmd.addr <= 0xFF):
        errors.append(f"addr out of range: {cmd.addr}")
    try:
        _func_to_byte(cmd.func)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid func: {cmd.func} ({exc})")
    for b in cmd.payload:
        if not (0 <= b <= 0xFF):
            errors.append(f"payload byte out of range: {b}")
            break
    if cmd.expected_len is not None and cmd.expected_len < 0:
        errors.append("expected_len must be >= 0")
    if cmd.timeout_ms is not None and cmd.timeout_ms < 0:
        errors.append("timeout_ms must be >= 0")
    if cmd.retry is not None and cmd.retry < 0:
        errors.append("retry must be >= 0")
    return errors


def _should_publish(func: str, data: bytes) -> bool:
    if not data:
        return False
    func_hex = _func_to_byte(func)
    if func_hex in {0xF3, 0xF6, 0xF7, 0x92, 0x46}:
        return False
    return True


def _parse_reply(func: str, data: bytes) -> Dict[str, Any]:
    func_hex = _func_to_byte(func)
    if func_hex == 0xF1 and data:
        status = data[0]
        if status not in _STATUS_LABELS:
            _LOG.warning("Unexpected F1 status byte: 0x%02X", status)
        return {"status": int(status), "status_label": _STATUS_LABELS.get(status)}
    if func_hex == 0x31 and len(data) == 6:
        counts = int.from_bytes(data, byteorder="big", signed=True)
        return {"counts": counts}
    if func_hex == 0x47:
        if len(data) != 34:
            _LOG.warning("Unexpected 0x47 payload length: %d", len(data))
        return {"parameters": list(data)}
    return {}


def _validate_reply(cmd: SerialCommand, reply: bytes) -> bool:
    if not cmd.expect_reply:
        return True
    if cmd.expected_len is not None and len(reply) != cmd.expected_len:
        _LOG.warning(
            "Reply length mismatch addr=%d func=%s expected_len=%s got=%d",
            cmd.addr,
            cmd.func,
            cmd.expected_len,
            len(reply),
        )
        return False
    func_hex = _func_to_byte(cmd.func)
    if func_hex == 0xF1 and len(reply) < 1:
        _LOG.warning("Missing status byte for F1 reply (addr=%d)", cmd.addr)
        return False
    if func_hex == 0x31 and len(reply) != 6:
        _LOG.warning("Malformed encoder reply length=%d addr=%d", len(reply), cmd.addr)
        return False
    if func_hex == 0x46 and cmd.expect_reply and len(reply) != 1:
        _LOG.warning("Malformed 0x46 reply length=%d addr=%d", len(reply), cmd.addr)
        return False
    if func_hex == 0x47 and len(reply) != 34:
        _LOG.warning("Malformed 0x47 reply length=%d addr=%d", len(reply), cmd.addr)
        return False
    return True


def _publish_reply(
    pub: zmq.Socket,
    topic: str,
    cmd: SerialCommand,
    reply: bytes,
    sent_ts_ms: int,
    reply_ts_ms: int,
    execute_start_monotonic_ns: int,
    reply_monotonic_ns: int,
) -> None:
    global _reply_sequence
    _reply_sequence += 1
    enqueued_monotonic_ns = cmd.enqueued_monotonic_ns or execute_start_monotonic_ns
    queue_age_ms = max(0.0, (execute_start_monotonic_ns - enqueued_monotonic_ns) / 1e6)
    duration_ms = max(0.0, (reply_monotonic_ns - execute_start_monotonic_ns) / 1e6)
    msg = {
        "type": "SerialReplyData",
        "cmd_id": cmd.cmd_id,
        "sequence": _reply_sequence,
        "source": "serial_io_service",
        "target": cmd.target,
        "addr": cmd.addr,
        "func": cmd.func,
        "reply": {
            "bytes": list(reply),
            "parsed": _parse_reply(cmd.func, reply) or None,
        },
        "timing": {
            "sent_ts_ms": sent_ts_ms,
            "reply_ts_ms": reply_ts_ms,
            "duration_ms": reply_ts_ms - sent_ts_ms,
            "enqueued_monotonic_ns": enqueued_monotonic_ns,
            "execute_start_monotonic_ns": execute_start_monotonic_ns,
            "reply_monotonic_ns": reply_monotonic_ns,
            "queue_age_ms": queue_age_ms,
            "bus_duration_ms": duration_ms,
        },
    }
    payload = f"{topic} {json.dumps(msg)}"
    _LOG.debug(
        "publish reply topic=%s cmd_id=%s addr=%d func=%s bytes=%s duration_ms=%d",
        topic,
        cmd.cmd_id,
        cmd.addr,
        cmd.func,
        list(reply),
        reply_ts_ms - sent_ts_ms,
    )
    pub.send_string(payload)


def _apply_command_timeout(
    bus: RS485Bus,
    timeout_ms: Optional[int],
) -> Tuple[Optional[float], Optional[float]]:
    old_timeout = bus._serial.timeout
    old_write_timeout = bus._serial.write_timeout
    if timeout_ms is not None:
        new_timeout = max(timeout_ms / 1000.0, 0.0)
        bus._serial.timeout = new_timeout
        bus._serial.write_timeout = new_timeout
    return old_timeout, old_write_timeout


def _restore_command_timeout(
    bus: RS485Bus,
    old_timeout: Optional[float],
    old_write_timeout: Optional[float],
) -> None:
    bus._serial.timeout = old_timeout
    bus._serial.write_timeout = old_write_timeout


def _process_command(
    bus: RS485Bus,
    cmd: SerialCommand,
    pub: Optional[zmq.Socket],
) -> None:
    sent_ts_ms = cmd.sent_ts_ms or int(time.time() * 1000)
    cmd.sent_ts_ms = sent_ts_ms
    execute_start_monotonic_ns = time.monotonic_ns()
    if cmd.enqueued_monotonic_ns is None:
        cmd.enqueued_monotonic_ns = execute_start_monotonic_ns
    _LOG.debug(
        "process cmd cmd_id=%s target=%s priority=%s addr=%d func=%s payload=%s expect_reply=%s expected_len=%s timeout_ms=%s retry=%s",
        cmd.cmd_id,
        cmd.target,
        cmd.priority,
        cmd.addr,
        cmd.func,
        list(cmd.payload),
        cmd.expect_reply,
        cmd.expected_len,
        cmd.timeout_ms,
        cmd.retry,
    )
    old_timeout, old_write_timeout = _apply_command_timeout(bus, cmd.timeout_ms)
    resolved_expected_len = cmd.expected_len
    if cmd.expect_reply and resolved_expected_len is None:
        try:
            func_byte = _func_to_byte(cmd.func)
        except Exception:  # noqa: BLE001
            func_byte = None
        if func_byte in _DEFAULT_SINGLE_BYTE_REPLY_FUNCS:
            resolved_expected_len = 1
    try:
        reply = bus.send_command(
            cmd.addr,
            _func_to_byte(cmd.func),
            cmd.payload,
            response_expected=cmd.expect_reply,
            expected_response_len=resolved_expected_len,
            retries=cmd.retry,
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.debug(
            "Serial command failed (already logged by transport) addr=%d func=%s payload=%s: %s",
            cmd.addr,
            cmd.func,
            list(cmd.payload),
            exc,
        )
        return
    finally:
        _restore_command_timeout(bus, old_timeout, old_write_timeout)

    _LOG.debug(
        "command reply cmd_id=%s addr=%d func=%s len=%d bytes=%s",
        cmd.cmd_id,
        cmd.addr,
        cmd.func,
        len(reply),
        list(reply),
    )

    if not _validate_reply(cmd, reply):
        return
    if not pub:
        return
    if not _should_publish(cmd.func, reply):
        return

    reply_monotonic_ns = time.monotonic_ns()
    reply_ts_ms = int(time.time() * 1000)
    topic = f"serial.reply.{cmd.target}"
    _publish_reply(
        pub,
        topic,
        cmd,
        reply,
        sent_ts_ms,
        reply_ts_ms,
        execute_start_monotonic_ns,
        reply_monotonic_ns,
    )


def _install_stop_handlers(stop_flag: StopFlag) -> None:
    def _handler(_signum, _frame):
        stop_flag.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _decode_update(data: bytes) -> List[SerialCommand]:
    enqueued_monotonic_ns = time.monotonic_ns()
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("invalid update json: %s", exc)
        return []
    if payload.get("type") != "SerialUpdate":
        return []

    ingest_ts_ms = int(time.time() * 1000)

    def _resolve_sent_ts_ms(raw_value: Any, fallback: int, context: str) -> int:
        if raw_value is None:
            return fallback
        if isinstance(raw_value, bool):
            _LOG.warning(
                "invalid %s sent_ts_ms=%r; using fallback=%d",
                context,
                raw_value,
                fallback,
            )
            return fallback
        try:
            return int(raw_value)
        except Exception:  # noqa: BLE001
            _LOG.warning(
                "invalid %s sent_ts_ms=%r; using fallback=%d",
                context,
                raw_value,
                fallback,
            )
            return fallback

    payload_sent_ts_ms = _resolve_sent_ts_ms(
        payload.get("sent_ts_ms"),
        ingest_ts_ms,
        "update",
    )

    commands: List[SerialCommand] = []
    for entry in payload.get("commands", []):
        if not isinstance(entry, dict):
            continue
        cmd_id = entry.get("cmd_id") or f"update:{ingest_ts_ms}"
        entry_sent_ts_ms = _resolve_sent_ts_ms(
            entry.get("sent_ts_ms"),
            payload_sent_ts_ms,
            f"command cmd_id={cmd_id}",
        )
        try:
            cmd = SerialCommand(
                cmd_id=str(cmd_id),
                func=str(entry["func"]),
                addr=int(entry["addr"]),
                payload=tuple(int(b) & 0xFF for b in entry.get("payload", [])),
                expect_reply=bool(entry.get("expect_reply", True)),
                expected_len=(
                    int(entry["expected_len"]) if entry.get("expected_len") is not None else None
                ),
                priority=str(entry.get("priority", "normal")),
                target=str(entry.get("target", payload.get("target", "gimbal"))),
                timeout_ms=(
                    int(entry["timeout_ms"]) if entry.get("timeout_ms") is not None else None
                ),
                retry=int(entry["retry"]) if entry.get("retry") is not None else None,
                sent_ts_ms=entry_sent_ts_ms,
                enqueued_monotonic_ns=enqueued_monotonic_ns,
            )
            errors = _validate_command(cmd)
            if errors:
                _LOG.warning("invalid update command: %s", "; ".join(errors))
                continue
            commands.append(cmd)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("invalid update command entry: %s", exc)
            continue
    return commands


def _drain_updates(
    socket: zmq.Socket,
    queue: Deque[SerialCommand],
    stats: Dict[str, int],
) -> None:
    drained = 0
    coalesce_map: Dict[Tuple[str, int, int], int] = {}
    for idx, queued_cmd in enumerate(queue):
        key = _coalesce_key(queued_cmd)
        if key is not None and not _is_critical_command(queued_cmd):
            coalesce_map[key] = idx

    while True:
        try:
            payload = socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        for cmd in _decode_update(payload):
            key = _coalesce_key(cmd)
            if key is None or _is_critical_command(cmd):
                queue.append(cmd)
                drained += 1
                continue
            existing_idx = coalesce_map.get(key)
            if existing_idx is not None and not _is_critical_command(queue[existing_idx]):
                queue[existing_idx] = cmd
                stats["coalesced_count"] += 1
            else:
                queue.append(cmd)
                coalesce_map[key] = len(queue) - 1
            drained += 1
    if drained:
        _LOG.debug(
            "drained %d update command(s) into next-round queue (coalesced_count=%d)",
            drained,
            stats["coalesced_count"],
        )


def _collect_due_schedule(
    schedule: List[ScheduledCommand],
    now_ms: int,
) -> List[SerialCommand]:
    due: List[SerialCommand] = []
    enqueued_monotonic_ns = time.monotonic_ns()
    for entry in schedule:
        if now_ms < entry.next_due_ts_ms:
            continue
        spec = entry.spec
        due.append(
            SerialCommand(
                cmd_id=f"schedule:{spec.name}:{now_ms}",
                func=spec.func,
                addr=spec.addr,
                payload=spec.payload,
                expect_reply=spec.expect_reply,
                expected_len=spec.expected_len,
                priority=spec.priority,
                target=spec.target,
                timeout_ms=None,
                retry=None,
                enqueued_monotonic_ns=enqueued_monotonic_ns,
            )
        )
        entry.next_due_ts_ms = now_ms + spec.interval_ms
    return due


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    config = _load_config([str(path) for path in expand_config_paths(args.config, args.config_extra)])
    schedule = _parse_schedule(config)
    startup_commands = _parse_startup(config)
    f6_stale_threshold_ms = _get_stale_threshold_ms(config)

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(args.command_endpoint)

    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.bind(args.update_endpoint)

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(args.reply_endpoint)

    command_queue: Deque[SerialCommand] = deque()
    next_round_queue: Deque[SerialCommand] = deque()
    stats = {
        "coalesced_count": 0,
        "dropped_stale_count": 0,
    }
    stop_flag = StopFlag()
    _install_stop_handlers(stop_flag)

    with RS485Bus(
        port=args.port,
        baudrate=args.baud,
        timeout=args.timeout,
        max_retries=max(args.retries, 0),
    ) as bus:
        _LOG.info("Serial I/O service started on %s @ %d", args.port, args.baud)
        if startup_commands:
            _LOG.info("Running %d serial startup command(s)", len(startup_commands))
            for cmd in startup_commands:
                if stop_flag.is_set():
                    break
                _process_command(bus, cmd, pub)
        while not stop_flag.is_set():
            now_ms = int(time.time() * 1000)

            try:
                payload = rep.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                payload = None

            if payload:
                cmd, ack = _decode_cmd(payload)
                if cmd is not None:
                    command_queue.append(cmd)
                    ack.queue_position = len(command_queue)
                    _LOG.debug(
                        "enqueued REQ cmd_id=%s queue_position=%s addr=%d func=%s",
                        cmd.cmd_id,
                        ack.queue_position,
                        cmd.addr,
                        cmd.func,
                    )
                rep.send_string(_ack_message(cmd.cmd_id if cmd else None, ack))

            if next_round_queue:
                command_queue.extend(next_round_queue)
                _LOG.debug(
                    "moved %d next-round command(s) into active queue",
                    len(next_round_queue),
                )
                next_round_queue.clear()

            _drain_updates(sub, next_round_queue, stats)

            due_commands = _collect_due_schedule(schedule, now_ms)
            if due_commands:
                command_queue.extend(due_commands)
                _LOG.debug("scheduled %d periodic command(s)", len(due_commands))

            if not command_queue:
                time.sleep(max(args.idle_sleep_ms, 0) / 1000.0)
                continue

            current_round: List[SerialCommand] = list(command_queue)
            command_queue.clear()
            current_round.sort(key=_effective_priority_key)
            _LOG.debug(
                "processing round with %d command(s) (coalesced_count=%d dropped_stale_count=%d)",
                len(current_round),
                stats["coalesced_count"],
                stats["dropped_stale_count"],
            )

            idx = 0
            while idx < len(current_round):
                cmd = current_round[idx]
                if (
                    _is_f6_command(cmd)
                    and not _is_critical_command(cmd)
                    and cmd.sent_ts_ms is not None
                ):
                    cmd_check_ts_ms = int(time.time() * 1000)
                    age_ms = cmd_check_ts_ms - cmd.sent_ts_ms
                    if age_ms > f6_stale_threshold_ms:
                        stats["dropped_stale_count"] += 1
                        _LOG.debug(
                            "drop stale non-critical F6 cmd_id=%s age_ms=%d threshold_ms=%d dropped_stale_count=%d",
                            cmd.cmd_id,
                            age_ms,
                            f6_stale_threshold_ms,
                            stats["dropped_stale_count"],
                        )
                        idx += 1
                        continue

                if _can_use_multi_frame(cmd):
                    batch: List[SerialCommand] = [cmd]
                    lookahead = idx + 1
                    while (
                        lookahead < len(current_round)
                        and len(batch) < _MULTI_FRAME_MAX_COMMANDS
                        and _can_use_multi_frame(current_round[lookahead])
                    ):
                        batch.append(current_round[lookahead])
                        lookahead += 1
                    try:
                        _send_multi_frame_batch(bus, batch)
                    except Exception as exc:  # noqa: BLE001
                        _LOG.warning(
                            "multi-command frame send failed for batch size=%d: %s; falling back to single-command sends",
                            len(batch),
                            exc,
                        )
                        for fallback_cmd in batch:
                            _process_command(bus, fallback_cmd, pub)
                    idx += len(batch)
                    continue

                _process_command(bus, cmd, pub)
                idx += 1

    rep.close(linger=0)
    sub.close(linger=0)
    pub.close(linger=0)
    ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
