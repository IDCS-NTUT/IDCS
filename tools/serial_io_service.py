"""Standalone serial I/O service core.

Opens a serial port exclusively, runs a blocking command loop, applies
timeouts/retries, and publishes reply data immediately.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

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

_F5_FUNC_BYTE = 0xF5
_F6_FUNC_BYTE = 0xF6
_F7_FUNC_BYTE = 0xF7
_MULTI_FRAME_MAX_COMMANDS = 5
_DEFAULT_SINGLE_BYTE_REPLY_FUNCS = {0xF3, 0xF5, 0xF6, 0xF7, 0x92, 0x46, 0x98}
_MAX_NON_EMERGENCY_BLOCK_MS = 20.0
_CRITICAL_LATENCY_BUDGET_MS = 25.0
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
    request_monotonic_ns: Optional[int] = None
    request_host: Optional[str] = None
    update_id: Optional[str] = None


@dataclass
class AckResponse:
    accepted: bool
    queued: bool
    queue_position: Optional[int]
    reason: Optional[str]


@dataclass(frozen=True)
class ExecutionFeedbackConfig:
    publish_command_events: bool = False
    publish_actuation_state: bool = False
    actuation_state_heartbeat_ms: int = 50


class SerialExecutionPublisher:
    """Publish terminal command outcomes and recoverable F6 wire state."""

    _TERMINAL_EVENTS = {
        "wire_sent",
        "superseded",
        "preempted",
        "stale",
        "write_failed",
        "wire_uncertain",
        "cancelled",
    }

    def __init__(
        self,
        pub: zmq.Socket,
        config: ExecutionFeedbackConfig,
        *,
        service_epoch: Optional[str] = None,
    ) -> None:
        self._pub = pub
        self._config = config
        self.service_epoch = service_epoch or uuid.uuid4().hex
        self.sequence = 0
        self.snapshot_sequence = 0
        self.admitted_count = 0
        self.event_send_failures = 0
        self.snapshot_send_failures = 0
        self.counters: Dict[str, int] = {
            event: 0 for event in self._TERMINAL_EVENTS
        }
        self._actuation_by_target: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._last_snapshot_ns: Dict[str, int] = {}

    def admit(self, _cmd: SerialCommand) -> None:
        self.admitted_count += 1

    @staticmethod
    def _timed_f6_runtime_ms(cmd: SerialCommand) -> Optional[int]:
        if not _is_f6_command(cmd) or len(cmd.payload) < 7:
            return None
        units = int.from_bytes(bytes(cmd.payload[3:7]), byteorder="big")
        return units * 10 if units > 0 else None

    @staticmethod
    def _f6_speed_rpm(cmd: SerialCommand) -> Optional[int]:
        if not _is_f6_command(cmd) or len(cmd.payload) < 2:
            return None
        return ((int(cmd.payload[0]) & 0x0F) << 8) | int(cmd.payload[1])

    def _send(self, topic: str, message: Mapping[str, Any], *, snapshot: bool) -> bool:
        payload = f"{topic} {json.dumps(message, separators=(',', ':'))}"
        try:
            self._pub.send_string(payload, flags=zmq.NOBLOCK)
        except TypeError:
            # Lightweight test doubles may not expose the optional flags arg.
            self._pub.send_string(payload)
        except zmq.Again:
            if snapshot:
                self.snapshot_send_failures += 1
            else:
                self.event_send_failures += 1
            return False
        return True

    def terminal(
        self,
        cmd: SerialCommand,
        event: str,
        *,
        reason: Optional[str] = None,
        related_cmd_id: Optional[str] = None,
        execute_start_monotonic_ns: Optional[int] = None,
        wire_monotonic_ns: Optional[int] = None,
        reply_confirmed: Optional[bool] = None,
    ) -> Mapping[str, Any]:
        if event not in self._TERMINAL_EVENTS:
            raise ValueError(f"unsupported terminal serial event: {event}")
        self.sequence += 1
        self.counters[event] += 1
        event_ns = time.monotonic_ns()
        message: Dict[str, Any] = {
            "type": "SerialCommandEventV1",
            "version": 1,
            "service_epoch": self.service_epoch,
            "service_host": socket.gethostname(),
            "sequence": self.sequence,
            "source": "serial_io_service",
            "target": cmd.target,
            "update_id": cmd.update_id,
            "cmd_id": cmd.cmd_id,
            "addr": cmd.addr,
            "func": cmd.func,
            "payload": list(cmd.payload),
            "event": event,
            "terminal": True,
            "reason": reason,
            "related_cmd_id": related_cmd_id,
            "reply_expected": bool(cmd.expect_reply),
            "reply_confirmed": reply_confirmed,
            "timing": {
                "ingest_monotonic_ns": cmd.enqueued_monotonic_ns,
                "execute_start_monotonic_ns": execute_start_monotonic_ns,
                "wire_monotonic_ns": wire_monotonic_ns,
                "event_monotonic_ns": event_ns,
            },
            "accounting": {
                "admitted": self.admitted_count,
                "terminal": self.sequence,
                "pending": max(0, self.admitted_count - self.sequence),
            },
        }
        if self._config.publish_command_events:
            self._send(
                f"serial.command.{cmd.target}", message, snapshot=False
            )
        if (
            event in {"wire_sent", "wire_uncertain"}
            and _is_f6_command(cmd)
            and wire_monotonic_ns is not None
        ):
            runtime_ms = self._timed_f6_runtime_ms(cmd)
            expires_ns = (
                wire_monotonic_ns + runtime_ms * 1_000_000
                if runtime_ms is not None
                else None
            )
            target_state = self._actuation_by_target.setdefault(cmd.target, {})
            target_state[cmd.addr] = {
                "cmd_id": cmd.cmd_id,
                "update_id": cmd.update_id,
                "addr": cmd.addr,
                "func": cmd.func,
                "payload": list(cmd.payload),
                "speed_rpm": self._f6_speed_rpm(cmd),
                "wire_monotonic_ns": wire_monotonic_ns,
                "runtime_ms": runtime_ms,
                "expires_monotonic_ns": expires_ns,
                "reply_confirmed": reply_confirmed,
                "wire_outcome": event,
                "event_sequence": self.sequence,
            }
            self.publish_snapshot(cmd.target, now_ns=event_ns, force=True)
        return message

    def publish_snapshot(
        self, target: str, *, now_ns: Optional[int] = None, force: bool = False
    ) -> Optional[Mapping[str, Any]]:
        if not self._config.publish_actuation_state:
            return None
        states = self._actuation_by_target.get(target)
        if not states:
            return None
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        interval_ns = self._config.actuation_state_heartbeat_ms * 1_000_000
        if not force and current_ns - self._last_snapshot_ns.get(target, 0) < interval_ns:
            return None
        self.snapshot_sequence += 1
        axes: Dict[str, Dict[str, Any]] = {}
        for addr, state in states.items():
            axis_state = dict(state)
            expires_ns = axis_state.get("expires_monotonic_ns")
            speed_rpm = axis_state.get("speed_rpm")
            axis_state["active"] = bool(speed_rpm) and (
                expires_ns is None or current_ns < int(expires_ns)
            )
            axes[str(addr)] = axis_state
        message: Dict[str, Any] = {
            "type": "SerialActuationStateV1",
            "version": 1,
            "service_epoch": self.service_epoch,
            "service_host": socket.gethostname(),
            "snapshot_sequence": self.snapshot_sequence,
            "event_sequence": self.sequence,
            "source": "serial_io_service",
            "target": target,
            "event_monotonic_ns": current_ns,
            "axes": axes,
            "accounting": {
                "admitted": self.admitted_count,
                "terminal": self.sequence,
                "pending": max(0, self.admitted_count - self.sequence),
            },
        }
        self._send(f"serial.actuation.{target}", message, snapshot=True)
        self._last_snapshot_ns[target] = current_ns
        return message

    def heartbeat(self, *, now_ns: Optional[int] = None) -> None:
        current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        for target in tuple(self._actuation_by_target):
            self.publish_snapshot(target, now_ns=current_ns)


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
    parser.add_argument("--port", default="/dev/ttyCH341USB0", help="Serial device path")
    parser.add_argument("--baud", type=int, default=38400, help="Serial baudrate")
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
    parser.add_argument("--check", action="store_true",
                        help="validate config and schedule, then exit without opening the bus")
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


def _is_f5_command(cmd: SerialCommand) -> bool:
    try:
        return _func_to_byte(cmd.func) == _F5_FUNC_BYTE
    except Exception:  # noqa: BLE001
        return False


def _is_latest_wins_motion_command(cmd: SerialCommand) -> bool:
    """F6 speeds and F5 absolute targets: a newer one replaces an older one."""

    return _is_f6_command(cmd) or _is_f5_command(cmd)


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


def _is_zero_speed_command(cmd: SerialCommand) -> bool:
    if len(cmd.payload) < 2:
        return False
    if _is_f6_command(cmd):
        speed_rpm = ((cmd.payload[0] & 0x0F) << 8) | cmd.payload[1]
    elif _is_f5_command(cmd):
        # F5 speed is an unsigned 16-bit field; speed 0 is the F5 stop command.
        speed_rpm = (cmd.payload[0] << 8) | cmd.payload[1]
    else:
        return False
    return speed_rpm == 0


def _is_emergency_command(cmd: SerialCommand) -> bool:
    """Return whether *cmd* must bypass normal reply waits and queue ordering."""

    try:
        func = _func_to_byte(cmd.func)
    except Exception:  # noqa: BLE001
        return False
    if func == _F7_FUNC_BYTE:
        return True
    if func in {_F6_FUNC_BYTE, _F5_FUNC_BYTE}:
        return cmd.priority == "critical" and _is_zero_speed_command(cmd)
    return func == 0xF3 and bool(cmd.payload) and cmd.payload[0] == 0x00


def _is_discardable_motion_command(cmd: SerialCommand) -> bool:
    if _is_emergency_command(cmd):
        return False
    try:
        func = _func_to_byte(cmd.func)
    except Exception:  # noqa: BLE001
        return False
    if func in {_F6_FUNC_BYTE, _F5_FUNC_BYTE, 0xFD}:
        return True
    return func == 0xF3 and bool(cmd.payload) and cmd.payload[0] != 0x00


def _discard_motion_for_pending_emergency(
    queue: Deque[SerialCommand],
    *,
    on_terminal: Optional[
        Callable[[SerialCommand, str, Optional[str]], None]
    ] = None,
) -> int:
    """Discard queued motion/enable writes whenever an emergency is pending."""

    emergency = next((cmd for cmd in queue if _is_emergency_command(cmd)), None)
    if emergency is None:
        return 0
    dropped_commands = [
        cmd for cmd in queue if _is_discardable_motion_command(cmd)
    ]
    retained = deque(cmd for cmd in queue if cmd not in dropped_commands)
    queue.clear()
    queue.extend(retained)
    if on_terminal is not None:
        for cmd in dropped_commands:
            on_terminal(cmd, "preempted", emergency.cmd_id)
    return len(dropped_commands)


def _pop_next_command(queue: Deque[SerialCommand]) -> SerialCommand:
    """Pop the first emergency, otherwise preserve FIFO/startup ordering."""

    for index, cmd in enumerate(queue):
        if not _is_emergency_command(cmd):
            continue
        queue.rotate(-index)
        selected = queue.popleft()
        queue.rotate(index)
        return selected
    return queue.popleft()


def _effective_priority_key(cmd: SerialCommand) -> int:
    if _is_emergency_command(cmd):
        return -1
    if _is_critical_command(cmd):
        return _PRIORITY_ORDER["critical"]
    return _priority_key(cmd.priority)


def _coalesce_key(cmd: SerialCommand) -> Optional[Tuple[str, int, int]]:
    if not _is_latest_wins_motion_command(cmd):
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


def _get_execution_feedback_config(cfg: Mapping[str, Any]) -> ExecutionFeedbackConfig:
    serial_cfg = cfg.get("serial_io") if isinstance(cfg, Mapping) else None
    if not isinstance(serial_cfg, Mapping):
        return ExecutionFeedbackConfig()
    try:
        heartbeat_ms = max(
            10, int(serial_cfg.get("actuation_state_heartbeat_ms", 50))
        )
    except (TypeError, ValueError):
        heartbeat_ms = 50
    return ExecutionFeedbackConfig(
        publish_command_events=bool(
            serial_cfg.get("publish_command_events", False)
        ),
        publish_actuation_state=bool(
            serial_cfg.get("publish_actuation_state", False)
        ),
        actuation_state_heartbeat_ms=heartbeat_ms,
    )


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
            request_monotonic_ns=(
                int(payload["request_monotonic_ns"])
                if payload.get("request_monotonic_ns") is not None
                else None
            ),
            request_host=(
                str(payload["request_host"])
                if payload.get("request_host") is not None
                else None
            ),
            update_id=(
                str(payload["update_id"])
                if payload.get("update_id") is not None
                else None
            ),
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
    # High-rate write ACKs stay on the bus side. F3 (enable/disable) ACKs are
    # rare and are how the bridge confirms each axis is energized.
    if func_hex in {0xF5, 0xF6, 0xF7, 0x92, 0x46, 0x98}:
        return False
    return True


def _should_publish_for_command(cmd: SerialCommand, data: bytes) -> bool:
    """Expose F6 wire timing only for explicitly instrumented sweep commands."""

    if _should_publish(cmd.func, data):
        return True
    return bool(data) and cmd.cmd_id.startswith("sweep:") and _func_to_byte(cmd.func) == 0xF6


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
    if func_hex == 0x33 and len(data) == 4:
        # Microstep count ("pulses received"): the motor's own step position.
        return {"steps": int.from_bytes(data, byteorder="big", signed=True)}
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
    if func_hex == 0x33 and len(reply) != 4:
        _LOG.warning("Malformed step-count reply length=%d addr=%d", len(reply), cmd.addr)
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
    wire_monotonic_ns: int,
    reply_monotonic_ns: int,
) -> None:
    global _reply_sequence
    _reply_sequence += 1
    enqueued_monotonic_ns = cmd.enqueued_monotonic_ns or execute_start_monotonic_ns
    queue_age_ms = max(
        0.0,
        (execute_start_monotonic_ns - enqueued_monotonic_ns) / 1e6,
    )
    bus_duration_ms = max(
        0.0,
        (reply_monotonic_ns - execute_start_monotonic_ns) / 1e6,
    )
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
            "wire_monotonic_ns": wire_monotonic_ns,
            "reply_monotonic_ns": reply_monotonic_ns,
            "queue_age_ms": queue_age_ms,
            "bus_duration_ms": bus_duration_ms,
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


def _publish_emergency_timing(
    pub: zmq.Socket,
    cmd: SerialCommand,
    wire_monotonic_ns: int,
    critical_latency_budget_ms: float,
) -> None:
    service_host = socket.gethostname()
    same_host_request = (
        cmd.request_monotonic_ns is not None and cmd.request_host == service_host
    )
    origin_ns = (
        cmd.request_monotonic_ns if same_host_request else cmd.enqueued_monotonic_ns
    )
    request_to_wire_ms = (
        max(0.0, (wire_monotonic_ns - origin_ns) / 1e6)
        if origin_ns is not None
        else None
    )
    budget_missed = (
        request_to_wire_ms is not None
        and request_to_wire_ms > critical_latency_budget_ms
    )
    message = {
        "type": "SerialEmergencyTiming",
        "status": "complete",
        "cmd_id": cmd.cmd_id,
        "source": "serial_io_service",
        "target": cmd.target,
        "addr": cmd.addr,
        "func": cmd.func,
        "timing": {
            "request_monotonic_ns": cmd.request_monotonic_ns,
            "enqueued_monotonic_ns": cmd.enqueued_monotonic_ns,
            "wire_monotonic_ns": wire_monotonic_ns,
            "request_to_wire_ms": request_to_wire_ms,
            "budget_ms": critical_latency_budget_ms,
            "budget_scope": (
                "same_host_request_to_wire"
                if same_host_request
                else "service_enqueue_to_wire"
            ),
            "budget_missed": budget_missed,
        },
    }
    pub.send_string(f"serial.telemetry.{cmd.target} {json.dumps(message)}")


def _process_command(
    bus: RS485Bus,
    cmd: SerialCommand,
    pub: Optional[zmq.Socket],
    *,
    execution: Optional[SerialExecutionPublisher] = None,
    critical_latency_budget_ms: float = 25.0,
    max_non_emergency_block_ms: float = 20.0,
) -> None:
    sent_ts_ms = cmd.sent_ts_ms or int(time.time() * 1000)
    cmd.sent_ts_ms = sent_ts_ms
    execute_start_monotonic_ns = time.monotonic_ns()
    if cmd.enqueued_monotonic_ns is None:
        cmd.enqueued_monotonic_ns = execute_start_monotonic_ns
    emergency = _is_emergency_command(cmd)
    bus_retry_limit = max(int(getattr(bus, "max_retries", 0)), 0)
    requested_retries = (
        bus_retry_limit if cmd.retry is None else max(int(cmd.retry), 0)
    )
    resolved_retries = 0 if emergency else min(requested_retries, bus_retry_limit)
    current_timeout_s = max(float(bus._serial.timeout or 0.0), 0.0)
    requested_timeout_s = (
        max(float(cmd.timeout_ms) / 1000.0, 0.0)
        if cmd.timeout_ms is not None
        else current_timeout_s
    )
    per_attempt_budget_s = (
        max(float(max_non_emergency_block_ms), 0.0)
        / 1000.0
        / max(resolved_retries + 1, 1)
    )
    if emergency:
        resolved_timeout_s = min(
            requested_timeout_s,
            max(float(critical_latency_budget_ms), 0.0) / 1000.0,
        )
    else:
        resolved_timeout_s = min(requested_timeout_s, per_attempt_budget_s)
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
    old_timeout, old_write_timeout = _apply_command_timeout(
        bus, resolved_timeout_s * 1000.0
    )
    resolved_expected_len = cmd.expected_len
    response_expected = cmd.expect_reply and not emergency
    if response_expected and resolved_expected_len is None:
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
            response_expected=response_expected,
            expected_response_len=resolved_expected_len if response_expected else None,
            retries=resolved_retries,
        )
    except Exception as exc:  # noqa: BLE001
        wire_complete_ns = getattr(bus, "last_tx_complete_monotonic_ns", None)
        if execution is not None:
            if (
                isinstance(wire_complete_ns, int)
                and wire_complete_ns >= execute_start_monotonic_ns
            ):
                execution.terminal(
                    cmd,
                    "wire_uncertain",
                    reason=type(exc).__name__,
                    execute_start_monotonic_ns=execute_start_monotonic_ns,
                    wire_monotonic_ns=wire_complete_ns,
                    reply_confirmed=False if response_expected else None,
                )
            else:
                execution.terminal(
                    cmd,
                    "write_failed",
                    reason=type(exc).__name__,
                    execute_start_monotonic_ns=execute_start_monotonic_ns,
                )
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

    wire_monotonic_ns = getattr(bus, "last_tx_monotonic_ns", None)
    if wire_monotonic_ns is None:
        wire_monotonic_ns = time.monotonic_ns()
    wire_complete_monotonic_ns = getattr(
        bus, "last_tx_complete_monotonic_ns", None
    )
    if wire_complete_monotonic_ns is None:
        wire_complete_monotonic_ns = wire_monotonic_ns
    reply_valid = _validate_reply(cmd, reply)
    if execution is not None:
        execution.terminal(
            cmd,
            "wire_sent",
            execute_start_monotonic_ns=execute_start_monotonic_ns,
            wire_monotonic_ns=wire_complete_monotonic_ns,
            reply_confirmed=(reply_valid if response_expected else None),
        )
    if emergency and pub is not None:
        _publish_emergency_timing(
            pub,
            cmd,
            wire_monotonic_ns,
            critical_latency_budget_ms,
        )

    _LOG.debug(
        "command reply cmd_id=%s addr=%d func=%s len=%d bytes=%s",
        cmd.cmd_id,
        cmd.addr,
        cmd.func,
        len(reply),
        list(reply),
    )

    if not reply_valid:
        return
    if not pub:
        return
    if not _should_publish_for_command(cmd, reply):
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
        wire_monotonic_ns,
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

    update_id_raw = payload.get("update_id")
    update_id = (
        str(update_id_raw)
        if update_id_raw is not None
        else f"serial-update:{enqueued_monotonic_ns}"
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
                request_monotonic_ns=(
                    int(entry["request_monotonic_ns"])
                    if entry.get("request_monotonic_ns") is not None
                    else None
                ),
                request_host=(
                    str(entry["request_host"])
                    if entry.get("request_host") is not None
                    else None
                ),
                update_id=(
                    str(entry["update_id"])
                    if entry.get("update_id") is not None
                    else update_id
                ),
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
    execution: Optional[SerialExecutionPublisher] = None,
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
            if execution is not None:
                execution.admit(cmd)
            key = _coalesce_key(cmd)
            if key is None or _is_critical_command(cmd):
                queue.append(cmd)
                drained += 1
                continue
            existing_idx = coalesce_map.get(key)
            if existing_idx is not None and not _is_critical_command(queue[existing_idx]):
                replaced = queue[existing_idx]
                queue[existing_idx] = cmd
                stats["coalesced_count"] += 1
                if execution is not None:
                    execution.terminal(
                        replaced,
                        "superseded",
                        reason="latest_wins_f5" if _is_f5_command(cmd) else "latest_wins_f6",
                        related_cmd_id=cmd.cmd_id,
                    )
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
    execution_config = _get_execution_feedback_config(config)
    if args.check:
        print(json.dumps({
            "check_only": True, "port": args.port, "baud": args.baud,
            "startup_commands": len(startup_commands),
            "schedule": [{"name": item.spec.name, "func": item.spec.func, "addr": item.spec.addr,
                          "interval_ms": item.spec.interval_ms} for item in schedule],
        }, sort_keys=True))
        return 0

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
    execution = SerialExecutionPublisher(pub, execution_config)

    command_queue: Deque[SerialCommand] = deque(startup_commands)
    stats = {
        "coalesced_count": 0,
        "dropped_stale_count": 0,
        "emergency_dropped_motion_count": 0,
    }
    stop_flag = StopFlag()
    _install_stop_handlers(stop_flag)
    for startup_command in startup_commands:
        execution.admit(startup_command)

    with RS485Bus(
        port=args.port,
        baudrate=args.baud,
        timeout=args.timeout,
        max_retries=max(args.retries, 0),
    ) as bus:
        _LOG.info("Serial I/O service started on %s @ %d", args.port, args.baud)
        _LOG.info(
            "serial execution feedback events=%s actuation_state=%s heartbeat_ms=%d epoch=%s",
            execution_config.publish_command_events,
            execution_config.publish_actuation_state,
            execution_config.actuation_state_heartbeat_ms,
            execution.service_epoch,
        )
        if startup_commands:
            _LOG.info("Queued %d serial startup command(s)", len(startup_commands))
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
                    execution.admit(cmd)
                    ack.queue_position = len(command_queue)
                    _LOG.debug(
                        "enqueued REQ cmd_id=%s queue_position=%s addr=%d func=%s",
                        cmd.cmd_id,
                        ack.queue_position,
                        cmd.addr,
                        cmd.func,
                    )
                rep.send_string(_ack_message(cmd.cmd_id if cmd else None, ack))

            _drain_updates(sub, command_queue, stats, execution)

            due_commands = _collect_due_schedule(schedule, now_ms)
            if due_commands:
                for due_command in due_commands:
                    execution.admit(due_command)
                command_queue.extend(due_commands)
                _LOG.debug("scheduled %d periodic command(s)", len(due_commands))

            execution.heartbeat()

            if not command_queue:
                time.sleep(max(args.idle_sleep_ms, 0) / 1000.0)
                continue

            dropped = _discard_motion_for_pending_emergency(
                command_queue,
                on_terminal=lambda dropped_cmd, event, related: execution.terminal(
                    dropped_cmd,
                    event,
                    reason="emergency_pending",
                    related_cmd_id=related,
                ),
            )
            if dropped:
                stats["emergency_dropped_motion_count"] += dropped
                _LOG.warning(
                    "emergency pending: discarded %d queued motion/enable command(s)",
                    dropped,
                )

            cmd = _pop_next_command(command_queue)
            if (
                _is_latest_wins_motion_command(cmd)
                and not _is_emergency_command(cmd)
                and cmd.sent_ts_ms is not None
            ):
                cmd_check_ts_ms = int(time.time() * 1000)
                age_ms = cmd_check_ts_ms - cmd.sent_ts_ms
                if age_ms > f6_stale_threshold_ms:
                    stats["dropped_stale_count"] += 1
                    execution.terminal(
                        cmd,
                        "stale",
                        reason=f"age_ms={age_ms} threshold_ms={f6_stale_threshold_ms}",
                    )
                    _LOG.debug(
                        "drop stale non-emergency motion cmd_id=%s age_ms=%d threshold_ms=%d dropped_stale_count=%d",
                        cmd.cmd_id,
                        age_ms,
                        f6_stale_threshold_ms,
                        stats["dropped_stale_count"],
                    )
                    continue

            _process_command(bus, cmd, pub, execution=execution)

    while command_queue:
        execution.terminal(
            command_queue.popleft(),
            "cancelled",
            reason="service_shutdown",
        )
    _LOG.info(
        "serial execution feedback summary admitted=%d terminal=%d pending=%d events=%s event_send_failures=%d snapshot_send_failures=%d coalesced=%d stale=%d emergency_dropped=%d",
        execution.admitted_count,
        execution.sequence,
        max(0, execution.admitted_count - execution.sequence),
        execution.counters,
        execution.event_send_failures,
        execution.snapshot_send_failures,
        stats["coalesced_count"],
        stats["dropped_stale_count"],
        stats["emergency_dropped_motion_count"],
    )

    rep.close(linger=0)
    sub.close(linger=0)
    pub.close(linger=0)
    ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
