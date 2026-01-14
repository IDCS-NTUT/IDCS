"""Dedicated RS485 service that owns the serial port and drains incoming frames."""

from __future__ import annotations

import logging
import threading
from queue import Empty, Queue
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.gimbal.mks_servo42_rs485 import RS485Bus, RS485CRCError, RS485Error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RS485ServiceConfig:
    """Configuration for the RS485 service."""

    port: str
    baudrate: int = 115200
    timeout: float = 0.1
    max_retries: int = 1
    history_size: int = 256
    stats_log_interval_s: float = 5.0
    schedule_interval_s: float = 0.1


@dataclass(frozen=True)
class RS485Frame:
    """Parsed metadata for an RS485 frame."""

    raw: bytes
    received_ts: float
    addr: Optional[int]
    func: Optional[int]


@dataclass(frozen=True)
class CommandRequest:
    addr: int
    func: int
    data: Optional[Tuple[int, ...]]
    response_expected: bool
    expected_response_len: Optional[int]
    response_queue: "Queue[bytes]"
    error_queue: "Queue[Exception]"


class RS485Service:
    """Owns the serial port and continuously drains incoming RS485 frames."""

    _WRITE_FUNCS = {0xF3, 0xF6, 0xF7, 0x92, 0x46}

    def __init__(self, config: RS485ServiceConfig) -> None:
        self._config = config
        from common.gimbal.mks_servo42_rs485 import RS485Bus, RS485CRCError, RS485Error

        self._rs485_crc_error = RS485CRCError
        self._rs485_error = RS485Error
        self._bus = RS485Bus(
            config.port,
            baudrate=config.baudrate,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames: Deque[RS485Frame] = deque(maxlen=config.history_size)
        self._latest: Dict[Tuple[int, int], RS485Frame] = {}
        self._history: Dict[Tuple[int, int], Deque[RS485Frame]] = defaultdict(
            lambda: deque(maxlen=config.history_size)
        )
        self._error_counts = {"timeout": 0, "crc": 0, "framing": 0, "other": 0}
        self._command_counts = {"ok": 0, "timeout": 0, "error": 0}
        self._last_command_error: Optional[str] = None
        self._rx_count = 0
        self._last_rx_ts: Optional[float] = None
        self._lock = threading.Lock()
        self._command_queue: Queue[CommandRequest] = Queue()
        self._schedule: Dict[str, List[Mapping[str, object]]] = {}
        self._external_data: Dict[str, object] = {}
        self._publish_latest: Dict[str, RS485Frame] = {}
        self._schedule_thread: Optional[threading.Thread] = None

    def load_schedule(self, schedule: Mapping[str, List[Mapping[str, object]]]) -> None:
        """Load a command schedule into the service."""

        self._schedule = {key: list(value) for key, value in schedule.items()}

    def update_external_data(self, key: str, value: object) -> None:
        """Update external data needed by scheduled commands."""

        self._external_data[key] = value

    def latest_by_key(self, key: str) -> Optional[RS485Frame]:
        """Return the latest frame stored under a publish key."""

        return self._publish_latest.get(key)

    def enqueue_command(self, command: "CommandRequest") -> None:
        """Queue a command for the drain loop to execute."""

        self._command_queue.put(command)

    def start(self) -> None:
        """Start the drain loop in a background thread."""

        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain_loop, name="rs485-drain", daemon=True)
        self._thread.start()

        if self._schedule.get("startup"):
            self._thread = threading.Thread(
                target=self._run_startup_sequence, name="rs485-startup", daemon=True
            )
            self._thread.start()
        if self._schedule.get("poll") or self._schedule.get("control"):
            self._schedule_thread = threading.Thread(
                target=self._run_schedule_loop, name="rs485-schedule", daemon=True
            )
            self._schedule_thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop the drain loop and close the serial port."""

        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._schedule_thread:
            self._schedule_thread.join(timeout=timeout)
        self._bus.close()

    def snapshot(self) -> list[RS485Frame]:
        """Return a snapshot of the buffered frames."""

        with self._lock:
            return list(self._frames)

    def latest_snapshot(self) -> Dict[Tuple[int, int], RS485Frame]:
        """Return the latest frame per (addr, func)."""

        with self._lock:
            return dict(self._latest)

    def history_snapshot(self) -> Dict[Tuple[int, int], list[RS485Frame]]:
        """Return history buffers keyed by (addr, func)."""

        with self._lock:
            return {key: list(buffer) for key, buffer in self._history.items()}

    def error_counts(self) -> Dict[str, int]:
        """Return cumulative error counters for the drain loop."""

        with self._lock:
            return dict(self._error_counts)

    def stats_snapshot(self) -> Dict[str, Optional[object]]:
        """Return basic service stats for observability."""

        with self._lock:
            return {
                "rx_count": float(self._rx_count),
                "last_rx_ts": self._last_rx_ts,
                "timeout_errors": float(self._error_counts["timeout"]),
                "crc_errors": float(self._error_counts["crc"]),
                "framing_errors": float(self._error_counts["framing"]),
                "other_errors": float(self._error_counts["other"]),
                "cmd_ok": float(self._command_counts["ok"]),
                "cmd_timeout": float(self._command_counts["timeout"]),
                "cmd_error": float(self._command_counts["error"]),
                "last_cmd_error": self._last_command_error,
            }

    def send_command(
        self,
        addr: int,
        func: int,
        data: Optional[Iterable[int]] = None,
        *,
        response_expected: bool = True,
        expected_response_len: Optional[int] = None,
    ) -> bytes:
        """Send a command to the RS485 bus."""

        response_queue: Queue[bytes] = Queue(maxsize=1)
        error_queue: Queue[Exception] = Queue(maxsize=1)
        command = CommandRequest(
            addr=addr,
            func=func,
            data=tuple(int(b) & 0xFF for b in data) if data else None,
            response_expected=response_expected,
            expected_response_len=expected_response_len,
            response_queue=response_queue,
            error_queue=error_queue,
        )
        self.enqueue_command(command)

        try:
            return response_queue.get(timeout=self._config.timeout * 2)
        except Empty:
            with self._lock:
                self._command_counts["timeout"] += 1
                self._last_command_error = "Timed out waiting for RS485 command response"
            if not error_queue.empty():
                raise error_queue.get()
            raise TimeoutError("Timed out waiting for RS485 command response") from None

    def _drain_loop(self) -> None:
        last_stats_log = time.monotonic()
        last_rx_count = 0
        while not self._stop.is_set():
            try:
                while True:
                    try:
                        command = self._command_queue.get_nowait()
                    except Empty:
                        command = None

                    if command is None:
                        break

                    try:
                        resp = self._bus.send_command(
                            command.addr,
                            command.func,
                            command.data,
                            response_expected=command.response_expected,
                            expected_response_len=command.expected_response_len,
                        )
                        with self._lock:
                            self._command_counts["ok"] += 1
                            self._last_command_error = None
                        command.response_queue.put(resp)
                    except Exception as exc:  # noqa: BLE001
                        with self._lock:
                            self._command_counts["error"] += 1
                            self._last_command_error = str(exc)
                        command.error_queue.put(exc)
                        command.response_queue.put(b"")

                if not self._command_queue.empty():
                    continue

                frame = self._bus._read_frame(expected_data_len=None)
                if len(frame) < 4:
                    continue
                if self._bus._crc8(frame[:-1]) != frame[-1]:
                    raise RS485CRCError("CRC mismatch on drained RS485 frame")
                addr = frame[1]
                func = frame[2]
                record = RS485Frame(
                    raw=frame,
                    received_ts=time.time(),
                    addr=addr,
                    func=func,
                )
                with self._lock:
                    self._frames.append(record)
                    key = (addr, func)
                    self._latest[key] = record
                    self._history[key].append(record)
                    self._rx_count += 1
                    self._last_rx_ts = record.received_ts
                if self._schedule.get("poll"):
                    self._run_poll_sequence()
            except TimeoutError:
                with self._lock:
                    self._error_counts["timeout"] += 1
                continue
            except self._rs485_crc_error as exc:
                with self._lock:
                    self._error_counts["crc"] += 1
                logger.warning("RS485 drain error: %s", exc)
            except self._rs485_error as exc:
                with self._lock:
                    self._error_counts["framing"] += 1
                logger.warning("RS485 drain error: %s", exc)
            except Exception as exc:  # noqa: BLE001 - keep drain loop alive
                with self._lock:
                    self._error_counts["other"] += 1
                logger.exception("Unexpected RS485 drain failure: %s", exc)
            finally:
                now = time.monotonic()
                if now - last_stats_log >= self._config.stats_log_interval_s:
                    with self._lock:
                        rx_count = self._rx_count
                        errors = dict(self._error_counts)
                        last_rx = self._last_rx_ts
                    delta = rx_count - last_rx_count
                    rate = delta / max(now - last_stats_log, 1e-6)
                    logger.info(
                        "RS485 stats rx_total=%d rx_rate=%.1f/s last_rx=%.3f errors=%s",
                        rx_count,
                        rate,
                        last_rx or 0.0,
                        errors,
                    )
                    last_stats_log = now
                    last_rx_count = rx_count

    def _run_startup_sequence(self) -> None:
        for command in self._schedule.get("startup", []):
            self._execute_scheduled_command(command)

    def _run_poll_sequence(self) -> None:
        for command in self._schedule.get("poll", []):
            self._execute_scheduled_command(command)

    def _run_control_sequence(self) -> None:
        for command in self._schedule.get("control", []):
            self._execute_scheduled_command(command)

    def _run_schedule_loop(self) -> None:
        while not self._stop.is_set():
            if self._schedule.get("poll"):
                self._run_poll_sequence()
            if self._schedule.get("control"):
                self._run_control_sequence()
            time.sleep(self._config.schedule_interval_s)

    def _execute_scheduled_command(self, command: Mapping[str, object]) -> None:
        addr = int(command["addr"])
        func = int(command["func"])
        data = command.get("data")
        data_source = command.get("data_source")
        expect_reply_value = command.get("expect_reply")
        if expect_reply_value is None:
            expect_reply = func not in self._WRITE_FUNCS
        else:
            expect_reply = bool(expect_reply_value)
        expected_len = command.get("expected_len")

        if data_source:
            data = self._external_data.get(str(data_source))
            if data is None:
                return

        payload = None
        if data:
            payload = tuple(int(b) & 0xFF for b in data)
        resp = self.send_command(
            addr,
            func,
            payload,
            response_expected=expect_reply,
            expected_response_len=int(expected_len) if expected_len is not None else None,
        )

        publish_key = command.get("publish_key")
        if publish_key and expect_reply:
            record = RS485Frame(
                raw=bytes(resp),
                received_ts=time.time(),
                addr=addr,
                func=func,
            )
            with self._lock:
                self._latest[(addr, func)] = record
                self._publish_latest[str(publish_key)] = record
