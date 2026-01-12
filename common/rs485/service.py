"""Dedicated RS485 service that owns the serial port and drains incoming frames."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

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


@dataclass(frozen=True)
class RS485Frame:
    """Parsed metadata for an RS485 frame."""

    raw: bytes
    received_ts: float
    addr: Optional[int]
    func: Optional[int]


class RS485Service:
    """Owns the serial port and continuously drains incoming RS485 frames."""

    def __init__(self, config: RS485ServiceConfig) -> None:
        self._config = config
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
        self._rx_count = 0
        self._last_rx_ts: Optional[float] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the drain loop in a background thread."""

        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain_loop, name="rs485-drain", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop the drain loop and close the serial port."""

        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
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

    def stats_snapshot(self) -> Dict[str, Optional[float]]:
        """Return basic service stats for observability."""

        with self._lock:
            return {
                "rx_count": float(self._rx_count),
                "last_rx_ts": self._last_rx_ts,
                "timeout_errors": float(self._error_counts["timeout"]),
                "crc_errors": float(self._error_counts["crc"]),
                "framing_errors": float(self._error_counts["framing"]),
                "other_errors": float(self._error_counts["other"]),
            }

    def _drain_loop(self) -> None:
        last_stats_log = time.monotonic()
        last_rx_count = 0
        while not self._stop.is_set():
            try:
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
            except TimeoutError:
                with self._lock:
                    self._error_counts["timeout"] += 1
                continue
            except RS485CRCError as exc:
                with self._lock:
                    self._error_counts["crc"] += 1
                logger.warning("RS485 drain error: %s", exc)
            except RS485Error as exc:
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
