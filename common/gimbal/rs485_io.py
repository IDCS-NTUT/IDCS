"""Dedicated RS485 serial I/O module with background draining and caching."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import serial


logger = logging.getLogger(__name__)

SLAVE_START = 0xFB


@dataclass
class RS485IO:
    """Own the serial port, drain incoming data, and cache responses."""

    port: str
    baudrate: int
    timeout: float = 0.1
    raw_buffer_size: int = 4096
    _serial: serial.Serial = field(init=False)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _condition: threading.Condition = field(init=False, default_factory=threading.Condition)
    _latest_frames: Dict[Tuple[int, int], bytes] = field(init=False, default_factory=dict)
    _seq: Dict[Tuple[int, int], int] = field(init=False, default_factory=dict)
    _raw_buffer: Deque[int] = field(init=False)
    _thread: Optional[threading.Thread] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._serial = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )
        self._raw_buffer = deque(maxlen=self.raw_buffer_size)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.timeout * 2)
        if self._serial and self._serial.is_open:
            self._serial.close()

    def write(self, frame: bytes) -> None:
        self._serial.write(frame)
        self._serial.flush()

    def reset_buffers(self) -> None:
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        with self._condition:
            self._raw_buffer.clear()

    def clear_frame_cache(self, addr: int, func: int) -> None:
        key = (addr, func)
        with self._condition:
            self._latest_frames.pop(key, None)
            self._seq.pop(key, None)

    def get_seq(self, addr: int, func: int) -> int:
        with self._condition:
            return self._seq.get((addr, func), 0)

    def get_latest_frame(self, addr: int, func: int) -> Optional[bytes]:
        with self._condition:
            return self._latest_frames.get((addr, func))

    def wait_for_frame(
        self,
        addr: int,
        func: int,
        *,
        expected_data_len: Optional[int] = None,
        since_seq: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        key = (addr, func)
        deadline = None if timeout is None else time.monotonic() + timeout
        if since_seq is None:
            since_seq = self.get_seq(addr, func)

        expected_len = None
        if expected_data_len is not None:
            expected_len = 3 + expected_data_len + 1

        with self._condition:
            while True:
                seq = self._seq.get(key, 0)
                frame = self._latest_frames.get(key)
                if frame is not None and seq > since_seq:
                    if expected_len is None or len(frame) == expected_len:
                        return frame
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Timeout waiting for RS485 response frame")
                else:
                    remaining = None
                self._condition.wait(timeout=remaining)

    def read_raw(self, size: int, *, timeout: Optional[float] = None) -> bytes:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while len(self._raw_buffer) < size:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Timeout waiting for {size} raw RS485 bytes"
                        )
                else:
                    remaining = None
                self._condition.wait(timeout=remaining)
            return bytes(self._raw_buffer.popleft() for _ in range(size))

    def _record_bytes(self, data: bytes) -> None:
        with self._condition:
            self._raw_buffer.extend(data)
            self._condition.notify_all()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            start = self._serial.read(1)
            if not start:
                continue
            self._record_bytes(start)
            if start[0] != SLAVE_START:
                continue

            addr = self._serial.read(1)
            if not addr:
                continue
            func = self._serial.read(1)
            if not func:
                continue
            self._record_bytes(addr + func)

            addr_val = addr[0]
            func_val = func[0]
            while addr_val == SLAVE_START:
                addr_val = func_val
                next_func = self._serial.read(1)
                if not next_func:
                    break
                func_val = next_func[0]
                self._record_bytes(next_func)
            else:
                payload = bytearray([SLAVE_START, addr_val, func_val])
                while True:
                    chunk = self._serial.read(1)
                    if not chunk:
                        break
                    self._record_bytes(chunk)
                    payload.extend(chunk)
                if len(payload) < 4:
                    logger.debug("Discarding short RS485 frame: %s", payload)
                    continue
                key = (addr_val, func_val)
                with self._condition:
                    self._latest_frames[key] = bytes(payload)
                    self._seq[key] = self._seq.get(key, 0) + 1
                    self._condition.notify_all()
