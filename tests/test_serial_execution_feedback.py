from __future__ import annotations

import json
import time
from collections import deque

import zmq

from tools import serial_io_service


class _FakePub:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send_string(self, payload: str, flags: int = 0) -> None:
        del flags
        self.sent.append(payload)


class _FakeSub:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = deque(payloads)

    def recv(self, flags: int = 0) -> bytes:
        del flags
        if not self.payloads:
            raise zmq.Again()
        return self.payloads.popleft()


class _FakeSerial:
    timeout = 0.01
    write_timeout = 0.01


class _SuccessfulBus:
    def __init__(self) -> None:
        self._serial = _FakeSerial()
        self.max_retries = 0
        self.last_tx_monotonic_ns: int | None = None
        self.last_tx_complete_monotonic_ns: int | None = None

    def send_command(self, *_args, **_kwargs) -> bytes:
        self.last_tx_monotonic_ns = time.monotonic_ns()
        self.last_tx_complete_monotonic_ns = self.last_tx_monotonic_ns + 1000
        return b""


class _FailingBus(_SuccessfulBus):
    def __init__(self, *, after_write: bool) -> None:
        super().__init__()
        self.after_write = after_write

    def send_command(self, *_args, **_kwargs) -> bytes:
        self.last_tx_monotonic_ns = time.monotonic_ns()
        self.last_tx_complete_monotonic_ns = (
            self.last_tx_monotonic_ns + 1000 if self.after_write else None
        )
        raise TimeoutError("synthetic failure")


def _command(
    cmd_id: str,
    *,
    payload: tuple[int, ...] = (0, 2, 10, 0, 0, 0, 10),
    priority: str = "high",
    expect_reply: bool = False,
) -> serial_io_service.SerialCommand:
    return serial_io_service.SerialCommand(
        cmd_id=cmd_id,
        update_id="intent:1",
        func="F6",
        addr=1,
        payload=payload,
        expect_reply=expect_reply,
        expected_len=1 if expect_reply else None,
        priority=priority,
        target="gimbal",
        timeout_ms=None,
        retry=None,
        sent_ts_ms=int(time.time() * 1000),
        enqueued_monotonic_ns=time.monotonic_ns(),
    )


def _execution(pub: _FakePub) -> serial_io_service.SerialExecutionPublisher:
    return serial_io_service.SerialExecutionPublisher(
        pub,  # type: ignore[arg-type]
        serial_io_service.ExecutionFeedbackConfig(
            publish_command_events=True,
            publish_actuation_state=True,
            actuation_state_heartbeat_ms=50,
        ),
        service_epoch="test-epoch",
    )


def _messages(pub: _FakePub, topic: str) -> list[dict[str, object]]:
    return [
        json.loads(payload.split(" ", 1)[1])
        for payload in pub.sent
        if payload.startswith(topic + " ")
    ]


def test_decode_update_retains_update_id_on_every_command() -> None:
    commands = serial_io_service._decode_update(
        json.dumps(
            {
                "type": "SerialUpdate",
                "target": "gimbal",
                "update_id": "intent:42",
                "commands": [
                    {
                        "cmd_id": "intent:yaw:42",
                        "func": "F6",
                        "addr": 1,
                        "payload": [0, 2, 10],
                        "expect_reply": False,
                    }
                ],
            }
        ).encode()
    )

    assert len(commands) == 1
    assert commands[0].update_id == "intent:42"


def test_latest_wins_coalescing_emits_superseded_terminal_outcome() -> None:
    old = _command("intent:yaw:old")
    queue = deque([old])
    update = {
        "type": "SerialUpdate",
        "target": "gimbal",
        "update_id": "intent:new",
        "commands": [
            {
                "cmd_id": "intent:yaw:new",
                "func": "F6",
                "addr": 1,
                "payload": [0, 3, 10, 0, 0, 0, 10],
                "expect_reply": False,
                "priority": "high",
            }
        ],
    }
    pub = _FakePub()
    execution = _execution(pub)
    stats = {"coalesced_count": 0}

    serial_io_service._drain_updates(
        _FakeSub([json.dumps(update).encode()]),  # type: ignore[arg-type]
        queue,
        stats,
        execution,
    )

    assert stats["coalesced_count"] == 1
    assert queue[0].cmd_id == "intent:yaw:new"
    event = _messages(pub, "serial.command.gimbal")[0]
    assert event["event"] == "superseded"
    assert event["cmd_id"] == "intent:yaw:old"
    assert event["related_cmd_id"] == "intent:yaw:new"


def test_emergency_discard_emits_preempted_for_each_removed_command() -> None:
    motion = _command("intent:yaw:motion")
    stop = _command(
        "intent:yaw:stop", payload=(0, 0, 10), priority="critical"
    )
    queue = deque([motion, stop])
    pub = _FakePub()
    execution = _execution(pub)

    dropped = serial_io_service._discard_motion_for_pending_emergency(
        queue,
        on_terminal=lambda cmd, event, related: execution.terminal(
            cmd, event, related_cmd_id=related
        ),
    )

    assert dropped == 1
    event = _messages(pub, "serial.command.gimbal")[0]
    assert event["event"] == "preempted"
    assert event["related_cmd_id"] == stop.cmd_id


def test_successful_timed_f6_emits_wire_event_and_recoverable_snapshot() -> None:
    pub = _FakePub()
    execution = _execution(pub)
    command = _command("intent:yaw:sent")

    serial_io_service._process_command(
        _SuccessfulBus(), command, pub, execution=execution  # type: ignore[arg-type]
    )

    event = _messages(pub, "serial.command.gimbal")[0]
    snapshot = _messages(pub, "serial.actuation.gimbal")[0]
    assert event["event"] == "wire_sent"
    assert event["timing"]["wire_monotonic_ns"] is not None  # type: ignore[index]
    axis = snapshot["axes"]["1"]  # type: ignore[index]
    assert axis["cmd_id"] == command.cmd_id
    assert axis["runtime_ms"] == 100
    assert axis["active"] is True


def test_failure_after_complete_write_is_uncertain_not_write_failed() -> None:
    pub = _FakePub()
    execution = _execution(pub)

    serial_io_service._process_command(
        _FailingBus(after_write=True),  # type: ignore[arg-type]
        _command("intent:yaw:uncertain", expect_reply=True),
        pub,  # type: ignore[arg-type]
        execution=execution,
    )

    event = _messages(pub, "serial.command.gimbal")[0]
    assert event["event"] == "wire_uncertain"
    snapshot = _messages(pub, "serial.actuation.gimbal")[0]
    assert snapshot["axes"]["1"]["wire_outcome"] == "wire_uncertain"  # type: ignore[index]


def test_failure_before_complete_write_is_write_failed() -> None:
    pub = _FakePub()
    execution = _execution(pub)

    serial_io_service._process_command(
        _FailingBus(after_write=False),  # type: ignore[arg-type]
        _command("intent:yaw:failed", expect_reply=True),
        pub,  # type: ignore[arg-type]
        execution=execution,
    )

    event = _messages(pub, "serial.command.gimbal")[0]
    assert event["event"] == "write_failed"


def test_actuation_heartbeat_marks_timed_command_expired() -> None:
    pub = _FakePub()
    execution = _execution(pub)
    command = _command("intent:yaw:expires")
    wire_ns = time.monotonic_ns()
    execution.terminal(command, "wire_sent", wire_monotonic_ns=wire_ns)
    pub.sent.clear()

    execution.heartbeat(now_ns=wire_ns + 200_000_000)

    snapshot = _messages(pub, "serial.actuation.gimbal")[0]
    assert snapshot["axes"]["1"]["active"] is False  # type: ignore[index]


def test_alternating_motion_and_stop_accounts_every_command_without_phantom_motion() -> None:
    pub = _FakePub()
    execution = _execution(pub)
    bus = _SuccessfulBus()

    for index in range(100):
        motion = _command(f"intent:yaw:motion:{index}")
        stop = _command(
            f"intent:yaw:stop:{index}",
            payload=(0, 0, 10),
            priority="critical",
        )
        execution.admit(motion)
        execution.admit(stop)
        queue = deque([motion, stop])
        serial_io_service._discard_motion_for_pending_emergency(
            queue,
            on_terminal=lambda cmd, event, related: execution.terminal(
                cmd, event, related_cmd_id=related
            ),
        )
        serial_io_service._process_command(
            bus, queue.popleft(), pub, execution=execution  # type: ignore[arg-type]
        )

    assert execution.admitted_count == 200
    assert execution.sequence == 200
    assert execution.counters["preempted"] == 100
    assert execution.counters["wire_sent"] == 100
    snapshots = _messages(pub, "serial.actuation.gimbal")
    assert all("motion" not in str(snapshot["axes"]) for snapshot in snapshots)
