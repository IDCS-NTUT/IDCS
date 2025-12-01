"""RS485 driver for MKS SERVO42D/57D_RS485 controllers in SR_CLOSE mode.

This module implements the minimal serial protocol pieces needed to talk to the
MKS SERVO42D/57D_RS485 boards over a USB-to-RS485 adapter as described in the
controller manual included with the project. It covers the following commands:

- F1: status query
- F3: enable/disable motor
- F6: speed mode control
- F7: emergency stop
- 0x31: read encoder "addition" (multi-turn) value

Frames are encoded as ``FA [addr] [func] [data...] [crc]`` for writes and
responses are expected as ``FB [addr] [func] [data...] [crc]``. The CRC is the
8-bit sum of all prior bytes in the frame (masked with ``0xFF``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import serial


class RS485Error(Exception):
    """Base exception for RS485 communication failures."""


class RS485CRCError(RS485Error):
    """Raised when a CRC mismatch is detected."""


class RS485FramingError(RS485Error):
    """Raised when framing or addressing is incorrect."""


@dataclass
class RS485Bus:
    """Thin wrapper around a ``pyserial.Serial`` port for MKS controllers."""

    port: str
    baudrate: int = 115200
    timeout: float = 0.1

    def __post_init__(self) -> None:
        self._serial = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )

    def __enter__(self) -> "RS485Bus":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying serial port."""

        if self._serial and self._serial.is_open:
            self._serial.close()

    @staticmethod
    def _crc8(frame_bytes: Iterable[int]) -> int:
        return sum(frame_bytes) & 0xFF

    def _read_exact(self, size: int) -> bytes:
        data = self._serial.read(size)
        if len(data) != size:
            raise TimeoutError(f"Timeout while reading {size} bytes from RS485 port")
        return data

    def _read_frame(self, expected_data_len: Optional[int] = None) -> bytes:
        # Wait for start byte 0xFB
        while True:
            start = self._serial.read(1)
            if not start:
                raise TimeoutError("Timeout waiting for RS485 response start byte")
            if start[0] == 0xFB:
                break
        header = self._read_exact(2)  # addr, func
        payload = bytearray(start + header)

        # Either read a known payload length (+CRC) or drain until timeout.
        if expected_data_len is None:
            while True:
                chunk = self._serial.read(1)
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) < 4:
                raise RS485FramingError("Incomplete RS485 frame received")
            return bytes(payload)

        remaining = expected_data_len + 1  # payload + CRC
        payload.extend(self._read_exact(remaining))
        return bytes(payload)

    def send_command(
        self,
        addr: int,
        func: int,
        data: Optional[Iterable[int]] = None,
        *,
        response_expected: bool = True,
        expected_response_len: Optional[int] = None,
    ) -> bytes:
        """Send a command frame and optionally wait for a response.

        Args:
            addr: Slave address (0-255).
            func: Function code (e.g., 0xF3, 0xF6).
            data: Optional data payload.
            response_expected: If False, skip waiting for a reply (broadcast).
            expected_response_len: Expected number of response data bytes; if
                provided, the response will be read to that length before CRC.

        Returns:
            The data bytes from the response (without framing/CRC).
        """

        payload = bytearray()
        if data:
            payload.extend(int(b) & 0xFF for b in data)
        frame_wo_crc = bytearray([0xFA, addr & 0xFF, func & 0xFF]) + payload
        crc = self._crc8(frame_wo_crc)
        frame = frame_wo_crc + bytes([crc])
        self._serial.write(frame)
        self._serial.flush()

        if not response_expected:
            return b""

        resp = self._read_frame(expected_response_len)
        if resp[0] != 0xFB:
            raise RS485FramingError(f"Unexpected start byte: {resp[0]:#x}")
        if resp[1] != (addr & 0xFF):
            raise RS485FramingError(
                f"Address mismatch: expected {addr & 0xFF:#x}, got {resp[1]:#x}"
            )
        if resp[2] != (func & 0xFF):
            raise RS485FramingError(
                f"Function mismatch: expected {func & 0xFF:#x}, got {resp[2]:#x}"
            )
        if self._crc8(resp[:-1]) != resp[-1]:
            raise RS485CRCError("CRC mismatch on RS485 response")

        return bytes(resp[3:-1])


@dataclass
class MksServo42Axis:
    """High-level abstraction for a single MKS SERVO42D/57D_RS485 axis."""

    bus: RS485Bus
    addr: int
    counts_per_rev: int = 0x4000
    gear_ratio: float = 1.0

    def enable(self, on: bool) -> None:
        """Enable or disable the motor using command F3."""

        self.bus.send_command(self.addr, 0xF3, [0x01 if on else 0x00])

    def status(self) -> int:
        """Query status using command F1 and return the status byte."""

        data = self.bus.send_command(self.addr, 0xF1, expected_response_len=1)
        if not data:
            raise RS485FramingError("No status data returned")
        return data[0]

    def emergency_stop(self) -> None:
        """Send emergency stop (command F7)."""

        self.bus.send_command(self.addr, 0xF7, [])

    def read_axis_counts(self) -> int:
        """Read the 48-bit encoder addition value (command 0x31)."""

        data = self.bus.send_command(self.addr, 0x31, expected_response_len=6)
        if len(data) != 6:
            raise RS485FramingError(
                f"Expected 6 bytes for encoder value, received {len(data)}"
            )
        return int.from_bytes(data, byteorder="big", signed=True)

    def read_angle_rad(self) -> float:
        """Convert encoder counts to axis angle in radians."""

        counts = self.read_axis_counts()
        motor_revs = counts / float(self.counts_per_rev)
        axis_revs = motor_revs / self.gear_ratio
        return axis_revs * 2.0 * math.pi

    def command_speed(self, omega_rad_s: float, acc: int = 10) -> None:
        """Command the motor in speed mode (F6) using a rad/s setpoint."""

        motor_rpm = omega_rad_s * 60.0 / (2.0 * math.pi) * self.gear_ratio
        direction = 0x01 if motor_rpm < 0 else 0x00
        speed_value = int(min(max(abs(motor_rpm), 0), 3000))
        acc_byte = int(min(max(acc, 0), 255))
        data = [direction, speed_value & 0xFF, (speed_value >> 8) & 0xFF, acc_byte]
        self.bus.send_command(self.addr, 0xF6, data)


class GimbalInterface:
    """Skeleton hardware interface for a two-axis MKS-driven gimbal."""

    def __init__(
        self,
        yaw_axis: MksServo42Axis,
        pitch_axis: MksServo42Axis,
        *,
        max_rate_rad_s: float = 10.0,
    ) -> None:
        self.yaw_axis = yaw_axis
        self.pitch_axis = pitch_axis
        self.max_rate_rad_s = max_rate_rad_s

    def apply_rate_commands(self, pan_rate_cmd: float, tilt_rate_cmd: float) -> None:
        """Map pan/tilt rate commands (rad/s) to motor speed commands."""

        pan_rate = max(min(pan_rate_cmd, self.max_rate_rad_s), -self.max_rate_rad_s)
        tilt_rate = max(min(tilt_rate_cmd, self.max_rate_rad_s), -self.max_rate_rad_s)

        self.yaw_axis.command_speed(pan_rate)
        self.pitch_axis.command_speed(tilt_rate)

    def stop(self) -> None:
        """Issue a soft stop to both axes."""

        self.yaw_axis.command_speed(0.0)
        self.pitch_axis.command_speed(0.0)

    def emergency_stop(self) -> None:
        """Emergency stop both axes."""

        self.yaw_axis.emergency_stop()
        self.pitch_axis.emergency_stop()
