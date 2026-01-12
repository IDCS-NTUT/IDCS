"""Dedicated RS485 service that owns the serial port and drains incoming frames."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

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

    def _drain_loop(self) -> None:
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
            except TimeoutError:
                continue
            except RS485Error as exc:
                logger.warning("RS485 drain error: %s", exc)
            except Exception as exc:  # noqa: BLE001 - keep drain loop alive
                logger.exception("Unexpected RS485 drain failure: %s", exc)
