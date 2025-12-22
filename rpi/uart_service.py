"""Raspberry Pi UART service handling framed link messages from the Jetson."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import serial

from common.shutdown import install_signal_handlers
from common.uart_protocol import (
    Frame,
    FrameParser,
    LinkMode,
    MessageType,
    build_error_payload,
    build_handshake_ack_payload,
    build_handshake_payload,
    build_heartbeat_payload,
    build_mode_ack_payload,
    encode_frame,
    parse_handshake_ack_payload,
    parse_handshake_payload,
    parse_heartbeat_payload,
    parse_mode_request_payload,
)

_LOG = logging.getLogger("rpi.uart")


class LinkState(str):
    IDLE = "idle"
    SYNC = "sync"
    ACTIVE = "active"
    FAULT = "fault"


@dataclass
class LinkMetrics:
    frames_sent: int = 0
    frames_received: int = 0
    crc_failures: int = 0
    frames_rate_limited: int = 0
    last_heard_ts: Optional[float] = None
    last_sent_ts: Optional[float] = None


class RaspberryPiLink:
    """Bidirectional framed UART service running on the Raspberry Pi."""

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        heartbeat_interval_s: float,
        heartbeat_timeout_s: float,
        rate_limit_hz: float = 50.0,
        role: str = "rpi",
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._heartbeat_interval = max(heartbeat_interval_s, 0.1)
        self._heartbeat_timeout = max(heartbeat_timeout_s, self._heartbeat_interval * 2)
        self._min_interval = 1.0 / max(rate_limit_hz, 1.0)
        self._role = role

        self._state = LinkState.IDLE
        self._metrics = LinkMetrics()
        self._seq = 0
        self._stop = threading.Event()
        self._serial = serial.Serial(
            self._port, self._baudrate, timeout=0, write_timeout=0
        )
        self._parser = FrameParser(on_crc_error=self._on_crc_error)
        self._tx_lock = threading.Lock()

        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._last_heartbeat_tx = 0.0
        self._last_handshake = 0.0

    @property
    def metrics(self) -> LinkMetrics:
        return self._metrics

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, new_state: str) -> None:
        if new_state != self._state:
            _LOG.info("link state %s -> %s", self._state, new_state)
            self._state = new_state

    def start(self) -> None:
        self._stop.clear()
        self._rx_thread.start()
        self._tx_thread.start()
        self._set_state(LinkState.SYNC)
        self._send_handshake(force=True)

    def stop(self) -> None:
        self._stop.set()
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        if self._tx_thread.is_alive():
            self._tx_thread.join(timeout=1.0)
        if self._serial and self._serial.is_open:
            self._serial.close()

    def _on_crc_error(self, frame_bytes: bytes, expected: int, actual: int) -> None:
        self._metrics.crc_failures += 1
        _LOG.warning(
            "CRC mismatch (read=%s calc=%s) len=%d", hex(expected), hex(actual), len(frame_bytes)
        )

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            data = self._serial.read(self._serial.in_waiting or 64)
            if data:
                for frame in self._parser.feed(data):
                    self._metrics.frames_received += 1
                    self._metrics.last_heard_ts = time.monotonic()
                    self._handle_frame(frame)
            else:
                self._check_timeout()
                time.sleep(0.01)

    def _tx_loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            if self._state in {LinkState.IDLE, LinkState.FAULT}:
                if now - self._last_handshake > self._heartbeat_interval:
                    self._set_state(LinkState.SYNC)
                    self._send_handshake(force=True)
            elif self._state == LinkState.SYNC and now - self._last_handshake > self._heartbeat_interval:
                self._send_handshake(force=True)
            elif self._state == LinkState.ACTIVE and now - self._last_heartbeat_tx > self._heartbeat_interval:
                self._send_heartbeat()
            self._check_timeout()
            time.sleep(0.02)

    def _check_timeout(self) -> None:
        if self._metrics.last_heard_ts is None:
            return
        if time.monotonic() - self._metrics.last_heard_ts > self._heartbeat_timeout:
            _LOG.error("heartbeat timeout; transitioning to FAULT")
            self._set_state(LinkState.FAULT)

    def _send_frame(self, msg_type: MessageType, payload: bytes, *, force: bool = False) -> None:
        with self._tx_lock:
            now = time.monotonic()
            if not force and (now - (self._metrics.last_sent_ts or 0.0) < self._min_interval):
                self._metrics.frames_rate_limited += 1
                _LOG.debug("rate limiting outbound %s frame", msg_type.name)
                return
            frame = encode_frame(msg_type, self._seq, payload)
            self._seq = (self._seq + 1) & 0xFFFF
            try:
                self._serial.write(frame)
                self._serial.flush()
                self._metrics.frames_sent += 1
                self._metrics.last_sent_ts = time.monotonic()
            except serial.SerialTimeoutException:
                _LOG.warning("write timeout sending %s", msg_type.name)

    def _send_handshake(self, *, force: bool = False) -> None:
        payload = build_handshake_payload(
            role=self._role, heartbeat_ms=int(self._heartbeat_interval * 1000), capabilities="rpi-link"
        )
        self._last_handshake = time.monotonic()
        self._send_frame(MessageType.HANDSHAKE, payload, force=force)

    def _send_heartbeat(self) -> None:
        payload = build_heartbeat_payload(uptime_ms=int(time.monotonic() * 1000))
        self._last_heartbeat_tx = time.monotonic()
        self._send_frame(MessageType.HEARTBEAT, payload)

    def _handle_frame(self, frame: Frame) -> None:
        if frame.msg_type == MessageType.HANDSHAKE:
            info = parse_handshake_payload(frame.payload)
            _LOG.info("handshake from %s capabilities=%s", info["role"], info["capabilities"])
            ack = build_handshake_ack_payload(role=self._role, ok=True, info="ready")
            self._send_frame(MessageType.HANDSHAKE_ACK, ack, force=True)
            self._set_state(LinkState.ACTIVE)
            return

        if frame.msg_type == MessageType.HANDSHAKE_ACK:
            info = parse_handshake_ack_payload(frame.payload)
            if info["ok"]:
                self._set_state(LinkState.ACTIVE)
                _LOG.info("peer acknowledged handshake role=%s", info["role"])
            else:
                self._set_state(LinkState.FAULT)
                _LOG.error("peer rejected handshake info=%s", info["info"])
            return

        if frame.msg_type == MessageType.HEARTBEAT:
            hb = parse_heartbeat_payload(frame.payload)
            _LOG.debug("heartbeat from peer uptime_ms=%d", hb["uptime_ms"])
            if self._state == LinkState.SYNC:
                self._set_state(LinkState.ACTIVE)
            return

        if frame.msg_type == MessageType.MODE_REQUEST:
            req = parse_mode_request_payload(frame.payload)
            _LOG.info("mode request %s reason=%s", req["mode"].name, req["reason"])
            ack = build_mode_ack_payload(req["mode"], accepted=True, info="mode applied")
            self._send_frame(MessageType.MODE_ACK, ack, force=True)
            return

        _LOG.debug("unhandled frame type %s", frame.msg_type.name)

    def send_error(self, code: int, info: str) -> None:
        payload = build_error_payload(code, info)
        self._send_frame(MessageType.ERROR_REPORT, payload, force=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi UART link service")
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="UART device path (default aligns with manual_control.py)",
    )
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument(
        "--heartbeat-interval", type=float, default=1.0, help="Heartbeat interval (seconds)"
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=3.0,
        help="Heartbeat timeout before marking the link faulted",
    )
    parser.add_argument(
        "--rate-limit-hz",
        type=float,
        default=50.0,
        help="Maximum outbound frame rate to avoid bus contention",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    service = RaspberryPiLink(
        port=args.port,
        baudrate=args.baud,
        heartbeat_interval_s=args.heartbeat_interval,
        heartbeat_timeout_s=args.heartbeat_timeout,
        rate_limit_hz=args.rate_limit_hz,
        role="rpi",
    )

    stop_event = install_signal_handlers()

    service.start()
    _LOG.info("UART service started on %s @ %d baud", args.port, args.baud)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
            metrics = service.metrics
            _LOG.debug(
                "state=%s rx=%d tx=%d crc=%d last_heard=%.3f",
                service.state,
                metrics.frames_received,
                metrics.frames_sent,
                metrics.crc_failures,
                metrics.last_heard_ts or -1.0,
            )
    finally:
        service.stop()


if __name__ == "__main__":
    main()
