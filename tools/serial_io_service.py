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
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml
import zmq

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
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--port", default="/dev/ttyTHS0", help="Serial device path")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baudrate")
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
        help="ZMQ PULL endpoint for SerialUpdate messages",
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


def _load_config(path: Optional[str]) -> Mapping[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file {cfg_path} not found")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"config file {cfg_path} must be a mapping")
    return data


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


def _decode_cmd(data: bytes) -> Tuple[Optional[SerialCommand], AckResponse]:
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
        )
    except Exception as exc:  # noqa: BLE001
        return None, AckResponse(False, False, None, f"invalid command payload: {exc}")

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
        return int(func[1:], 16)
    return int(func)


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
        return {"status": int(status), "status_label": _STATUS_LABELS.get(status)}
    if func_hex == 0x31 and len(data) == 6:
        counts = int.from_bytes(data, byteorder="big", signed=True)
        return {"counts": counts}
    if func_hex == 0x47:
        return {"parameters": list(data)}
    return {}


def _publish_reply(
    pub: zmq.Socket,
    topic: str,
    cmd: SerialCommand,
    reply: bytes,
    sent_ts_ms: int,
    reply_ts_ms: int,
) -> None:
    msg = {
        "type": "SerialReplyData",
        "cmd_id": cmd.cmd_id,
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
        },
    }
    payload = f"{topic} {json.dumps(msg)}"
    pub.send_string(payload)


def _apply_command_timeout(bus: RS485Bus, timeout_ms: Optional[int]) -> Tuple[float, float]:
    old_timeout = bus._serial.timeout
    old_write_timeout = bus._serial.write_timeout
    if timeout_ms is not None:
        new_timeout = max(timeout_ms / 1000.0, 0.0)
        bus._serial.timeout = new_timeout
        bus._serial.write_timeout = new_timeout
    return old_timeout, old_write_timeout


def _restore_command_timeout(bus: RS485Bus, old_timeout: float, old_write_timeout: float) -> None:
    bus._serial.timeout = old_timeout
    bus._serial.write_timeout = old_write_timeout


def _process_command(
    bus: RS485Bus,
    cmd: SerialCommand,
    pub: Optional[zmq.Socket],
) -> None:
    sent_ts_ms = cmd.sent_ts_ms or int(time.time() * 1000)
    cmd.sent_ts_ms = sent_ts_ms
    old_timeout, old_write_timeout = _apply_command_timeout(bus, cmd.timeout_ms)
    try:
        reply = bus.send_command(
            cmd.addr,
            _func_to_byte(cmd.func),
            cmd.payload,
            response_expected=cmd.expect_reply,
            expected_response_len=cmd.expected_len,
            retries=cmd.retry,
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "Serial command failed addr=%d func=%s payload=%s: %s",
            cmd.addr,
            cmd.func,
            list(cmd.payload),
            exc,
        )
        return
    finally:
        _restore_command_timeout(bus, old_timeout, old_write_timeout)

    if not pub:
        return
    if not _should_publish(cmd.func, reply):
        return

    reply_ts_ms = int(time.time() * 1000)
    topic = f"serial.reply.{cmd.target}"
    _publish_reply(pub, topic, cmd, reply, sent_ts_ms, reply_ts_ms)


def _install_stop_handlers(stop_flag: StopFlag) -> None:
    def _handler(_signum, _frame):
        stop_flag.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _drain_updates(socket: zmq.Socket) -> None:
    while True:
        try:
            socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break


def _collect_due_schedule(
    schedule: List[ScheduledCommand],
    now_ms: int,
) -> List[SerialCommand]:
    due: List[SerialCommand] = []
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
            )
        )
        entry.next_due_ts_ms = now_ms + spec.interval_ms
    return due


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    config = _load_config(args.config)
    schedule = _parse_schedule(config)

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(args.command_endpoint)

    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.LINGER, 0)
    pull.bind(args.update_endpoint)

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(args.reply_endpoint)

    command_queue: Deque[SerialCommand] = deque()
    stop_flag = StopFlag()
    _install_stop_handlers(stop_flag)

    with RS485Bus(
        port=args.port,
        baudrate=args.baud,
        timeout=args.timeout,
        max_retries=max(args.retries, 0),
    ) as bus:
        _LOG.info("Serial I/O service started on %s @ %d", args.port, args.baud)
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
                rep.send_string(_ack_message(cmd.cmd_id if cmd else None, ack))

            _drain_updates(pull)

            due_commands = _collect_due_schedule(schedule, now_ms)
            if due_commands:
                command_queue.extend(due_commands)

            if not command_queue:
                time.sleep(max(args.idle_sleep_ms, 0) / 1000.0)
                continue

            current_round: List[SerialCommand] = list(command_queue)
            command_queue.clear()
            current_round.sort(key=lambda cmd: _priority_key(cmd.priority))

            for cmd in current_round:
                _process_command(bus, cmd, pub)

    rep.close(linger=0)
    pull.close(linger=0)
    pub.close(linger=0)
    ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
