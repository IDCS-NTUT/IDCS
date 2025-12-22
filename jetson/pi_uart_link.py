"""Jetson-side UART link service for Raspberry Pi coordination."""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import serial

from common.config_sync import parse_config_text, read_snapshot
from common.shutdown import install_signal_handlers
from common.uart_protocol import (
    Frame,
    FrameParser,
    LinkMode,
    MessageType,
    build_handshake_ack_payload,
    build_handshake_payload,
    build_heartbeat_payload,
    build_mode_request_payload,
    encode_frame,
    parse_error_payload,
    parse_handshake_ack_payload,
    parse_handshake_payload,
    parse_heartbeat_payload,
    parse_mode_ack_payload,
)

_LOG = logging.getLogger("jetson.pi_uart")


class LinkState(str):
    IDLE = "idle"
    SYNC = "sync"
    ACTIVE = "active"
    FAULT = "fault"


@dataclass
class LinkMetrics:
    frames_sent: int = 0
    frames_received: int = 0
    frames_dropped: int = 0
    crc_failures: int = 0
    heartbeat_missed: int = 0
    last_heard_ts: Optional[float] = None
    last_sent_ts: Optional[float] = None


def _load_config(path: Path) -> Mapping[str, object]:
    snapshot = read_snapshot(path)
    return parse_config_text(snapshot.text, str(path))


class PiLinkService:
    """Maintains the dedicated UART channel to the Raspberry Pi."""

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        heartbeat_interval_s: float,
        heartbeat_timeout_s: float,
        handshake_interval_s: float,
        requested_mode: LinkMode,
        role: str = "jetson",
        capabilities: str = "telemetry",
        read_timeout_s: float = 0.0,
        write_timeout_s: float = 0.0,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._heartbeat_interval = max(heartbeat_interval_s, 0.1)
        self._heartbeat_timeout = max(heartbeat_timeout_s, self._heartbeat_interval * 2)
        self._handshake_interval = max(handshake_interval_s, 0.5)
        self._requested_mode = requested_mode
        self._role = role
        self._capabilities = capabilities
        self._read_timeout_s = read_timeout_s
        self._write_timeout_s = write_timeout_s

        self._state = LinkState.IDLE
        self._metrics = LinkMetrics()
        self._seq = 0
        self._peer_interval_s: Optional[float] = None

        self._stop = threading.Event()
        self._parser = FrameParser(on_crc_error=self._on_crc_error)
        self._tx_queue: queue.Queue[bytes] = queue.Queue(maxsize=128)

        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._serial: Optional[serial.Serial] = None

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
        self._serial = serial.Serial(
            self._port,
            self._baudrate,
            timeout=self._read_timeout_s,
            write_timeout=self._write_timeout_s,
        )
        self._stop.clear()
        self._rx_thread.start()
        self._tx_thread.start()
        self._set_state(LinkState.SYNC)
        self._enqueue_handshake()

    def stop(self) -> None:
        self._stop.set()
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        if self._tx_thread.is_alive():
            self._tx_thread.join(timeout=1.0)
        if self._serial and self._serial.is_open:
            self._serial.close()

    def send_mode_request(self, mode: LinkMode, reason: str = "") -> None:
        payload = build_mode_request_payload(mode, reason)
        self._enqueue(MessageType.MODE_REQUEST, payload)

    def _enqueue_handshake(self) -> None:
        payload = build_handshake_payload(
            role=self._role,
            heartbeat_ms=int(self._heartbeat_interval * 1000),
            capabilities=self._capabilities,
        )
        self._enqueue(MessageType.HANDSHAKE, payload)

    def _enqueue_heartbeat(self) -> None:
        uptime_ms = int(time.monotonic() * 1000)
        payload = build_heartbeat_payload(uptime_ms=uptime_ms)
        self._enqueue(MessageType.HEARTBEAT, payload)

    def _enqueue(self, msg_type: MessageType, payload: bytes) -> None:
        frame = encode_frame(msg_type, self._seq, payload)
        self._seq = (self._seq + 1) & 0xFFFF
        try:
            self._tx_queue.put_nowait(frame)
        except queue.Full:
            self._metrics.frames_dropped += 1
            _LOG.warning("dropping outbound frame %s (queue full)", msg_type.name)

    def _on_crc_error(self, frame_bytes: bytes, expected: int, actual: int) -> None:
        self._metrics.crc_failures += 1
        _LOG.warning(
            "CRC failure (read=%s calc=%s) on %d-byte frame", hex(expected), hex(actual), len(frame_bytes)
        )

    def _rx_loop(self) -> None:
        assert self._serial is not None
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
        assert self._serial is not None
        last_handshake = 0.0
        last_heartbeat = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if self._state in {LinkState.IDLE, LinkState.FAULT}:
                if now - last_handshake >= self._handshake_interval:
                    self._set_state(LinkState.SYNC)
                    self._enqueue_handshake()
                    last_handshake = now

            if self._state == LinkState.SYNC and now - last_handshake >= self._handshake_interval:
                self._enqueue_handshake()
                last_handshake = now

            if self._state == LinkState.ACTIVE and now - last_heartbeat >= self._heartbeat_interval:
                self._enqueue_heartbeat()
                last_heartbeat = now

            try:
                frame = self._tx_queue.get(timeout=0.1)
            except queue.Empty:
                self._check_timeout()
                continue

            try:
                self._serial.write(frame)
                self._serial.flush()
                self._metrics.frames_sent += 1
                self._metrics.last_sent_ts = time.monotonic()
            except serial.SerialTimeoutException:
                self._metrics.frames_dropped += 1
                _LOG.warning("write timeout sending frame")
            finally:
                self._tx_queue.task_done()
            self._check_timeout()

    def _handle_frame(self, frame: Frame) -> None:
        if frame.msg_type == MessageType.HANDSHAKE:
            info = parse_handshake_payload(frame.payload)
            self._peer_interval_s = max(0.1, float(info["heartbeat_ms"]) / 1000.0)
            ack_payload = build_handshake_ack_payload(role=self._role, ok=True, info="sync")
            self._enqueue(MessageType.HANDSHAKE_ACK, ack_payload)
            self._set_state(LinkState.ACTIVE)
            _LOG.info(
                "handshake from %s capabilities=%s heartbeat=%.3fs",
                info["role"],
                info["capabilities"],
                self._peer_interval_s,
            )
            return

        if frame.msg_type == MessageType.HANDSHAKE_ACK:
            info = parse_handshake_ack_payload(frame.payload)
            if info["ok"]:
                self._set_state(LinkState.ACTIVE)
                _LOG.info("handshake acknowledged by peer role=%s info=%s", info["role"], info["info"])
            else:
                self._set_state(LinkState.FAULT)
                _LOG.error("handshake nack: %s", info["info"])
            return

        if frame.msg_type == MessageType.HEARTBEAT:
            details = parse_heartbeat_payload(frame.payload)
            if self._state == LinkState.SYNC:
                self._set_state(LinkState.ACTIVE)
            _LOG.debug("heartbeat rx uptime_ms=%d last_error=%d", details["uptime_ms"], details["last_error"])
            return

        if frame.msg_type == MessageType.MODE_ACK:
            ack = parse_mode_ack_payload(frame.payload)
            _LOG.info(
                "mode ack: mode=%s accepted=%s info=%s", ack["mode"].name, ack["accepted"], ack["info"]
            )
            return

        if frame.msg_type == MessageType.ERROR_REPORT:
            err = parse_error_payload(frame.payload)
            self._set_state(LinkState.FAULT)
            _LOG.error("peer error code=%d info=%s", err["code"], err["info"])
            return

        _LOG.debug("unhandled frame type %s", frame.msg_type.name)

    def _check_timeout(self) -> None:
        if self._metrics.last_heard_ts is None:
            return
        now = time.monotonic()
        timeout_s = self._peer_interval_s or self._heartbeat_timeout
        if now - self._metrics.last_heard_ts > max(timeout_s, self._heartbeat_timeout):
            self._metrics.heartbeat_missed += 1
            self._set_state(LinkState.FAULT)


def _resolve_mode(value: str) -> LinkMode:
    normalized = value.strip().lower()
    if normalized == "standby":
        return LinkMode.STANDBY
    if normalized == "diagnostic":
        return LinkMode.DIAGNOSTIC
    return LinkMode.ACTIVE


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson UART link to Raspberry Pi service")
    parser.add_argument("--config", type=Path, default=Path("configs/dev.yaml"), help="Path to config file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    cfg = _load_config(args.config)
    pi_cfg = cfg.get("pi_uart") if isinstance(cfg, Mapping) else {}
    gimbal_cfg = cfg.get("gimbal") if isinstance(cfg, Mapping) else {}
    if not isinstance(pi_cfg, Mapping):
        raise SystemExit("pi_uart config section is required")

    default_port = gimbal_cfg.get("serial_port") if isinstance(gimbal_cfg, Mapping) else "/dev/ttyTHS0"
    default_baud = gimbal_cfg.get("baudrate") if isinstance(gimbal_cfg, Mapping) else 38400
    port = str(pi_cfg.get("port", default_port or "/dev/ttyTHS0"))
    baudrate = int(pi_cfg.get("baudrate", default_baud or 38400))
    heartbeat_interval_s = float(pi_cfg.get("heartbeat_interval_s", 1.0))
    heartbeat_timeout_s = float(pi_cfg.get("heartbeat_timeout_s", 3.0))
    handshake_interval_s = float(pi_cfg.get("handshake_interval_s", 2.0))
    read_timeout_s = float(pi_cfg.get("read_timeout_s", 0.0))
    write_timeout_s = float(pi_cfg.get("write_timeout_s", 0.0))
    requested_mode = _resolve_mode(str(pi_cfg.get("requested_mode", "active")))
    role = str(pi_cfg.get("role", "jetson"))
    capabilities = str(pi_cfg.get("capabilities", "link"))

    service = PiLinkService(
        port=port,
        baudrate=baudrate,
        heartbeat_interval_s=heartbeat_interval_s,
        heartbeat_timeout_s=heartbeat_timeout_s,
        handshake_interval_s=handshake_interval_s,
        requested_mode=requested_mode,
        role=role,
        capabilities=capabilities,
        read_timeout_s=read_timeout_s,
        write_timeout_s=write_timeout_s,
    )

    stop_event = threading.Event()

    def _stop_handler() -> None:
        stop_event.set()

    install_signal_handlers(_stop_handler)

    service.start()
    service.send_mode_request(requested_mode, "startup")
    _LOG.info("pi UART link started on %s @ %d baud", port, baudrate)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
            metrics = service.metrics
            _LOG.debug(
                "state=%s last_heard=%.3f crc=%d rx=%d tx=%d dropped=%d missed=%d",
                service.state,
                metrics.last_heard_ts or -1.0,
                metrics.crc_failures,
                metrics.frames_received,
                metrics.frames_sent,
                metrics.frames_dropped,
                metrics.heartbeat_missed,
            )
    finally:
        service.stop()


if __name__ == "__main__":
    main()
