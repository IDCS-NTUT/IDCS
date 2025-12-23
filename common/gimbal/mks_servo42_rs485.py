"""RS485 driver for MKS SERVO42D/57D_RS485 controllers in SR_CLOSE mode.

This module implements the minimal serial protocol pieces needed to talk to the
MKS SERVO42D/57D_RS485 boards over the RS485 bus described in the controller
manual included with the project. It covers the following commands:

- F1: status query
- F3: enable/disable motor
- F6: speed mode control
- F7: emergency stop
- 0x31: read encoder "addition" (multi-turn) value

Frames are encoded as ``0xFA [addr] [func] [data...] [crc]`` for writes and
responses are expected as ``0xFB [addr] [func] [data...] [crc]``. The CRC is the
8-bit sum of all prior bytes in the frame (masked with ``0xFF``). The default
baudrate is set to 115200 to match the updated controller configuration when
used on the Jetson's ``/dev/ttyTHS0`` UART. The Jetson speaks 3.3 V TTL; any
RS485-level conversion happens in external hardware, so no pyserial RS485 mode
configuration is required here.
"""

from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import serial


logger = logging.getLogger(__name__)


MASTER_START = 0xFA
SLAVE_START = 0xFB
DEFAULT_BAUDRATE = 115200


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
    baudrate: int = DEFAULT_BAUDRATE
    timeout: float = 0.1
    max_retries: int = 1

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
            if start[0] == SLAVE_START:
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
        retries: Optional[int] = None,
    ) -> bytes:
        """Send a command frame and optionally wait for a response.

        Args:
            addr: Slave address (0-255).
            func: Function code (e.g., 0xF3, 0xF6).
            data: Optional data payload.
            response_expected: If False, skip waiting for a reply (broadcast or
                group writes).
            expected_response_len: Expected number of response data bytes; if
                provided, the response will be read to that length before CRC.
            retries: Optional override for retry attempts on timeout/CRC/framing
                errors. If None, uses ``self.max_retries``.

        Returns:
            The data bytes from the response (without framing/CRC).
        """

        attempts = (retries if retries is not None else self.max_retries) + 1
        for attempt in range(1, attempts + 1):
            try:
                payload = bytearray()
                if data:
                    payload.extend(int(b) & 0xFF for b in data)
                frame_wo_crc = (
                    bytearray([MASTER_START, addr & 0xFF, func & 0xFF]) + payload
                )
                crc = self._crc8(frame_wo_crc)
                frame = frame_wo_crc + bytes([crc])
                self._serial.write(frame)
                self._serial.flush()

                if not response_expected:
                    return b""

                resp = self._read_frame(expected_response_len)
                if resp[0] != SLAVE_START:
                    raise RS485FramingError(
                        f"Unexpected start byte: {resp[0]:#x}"
                    )
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
            except (TimeoutError, RS485Error) as exc:
                logger.warning(
                    "RS485 command failed (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    exc,
                )
                if attempt == attempts:
                    raise
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()


@dataclass
class MksServo42Axis:
    """High-level abstraction for a single MKS SERVO42D/57D_RS485 axis.

    Write commands default to using ``group_addr`` when provided (no response
    expected per manual), while all reads and diagnostics use ``addr`` so a
    reply is returned. ``use_group_writes`` can be toggled per instance or per
    call to force individual addressing during bring-up. With the motor
    "Respond" parameter set to ``0``, ``respond_on_writes`` should remain False
    so write commands return immediately without waiting for acknowledgements.
    """

    bus: RS485Bus
    addr: int
    group_addr: Optional[int] = None
    counts_per_rev: int = 0x4000
    gear_ratio: float = 1.0
    use_group_writes: bool = True
    respond_on_writes: bool = False

    def _select_write_addr(self, use_group: Optional[bool]) -> Tuple[int, bool]:
        """Choose the address for write commands and whether to expect a reply."""

        if use_group is None:
            use_group = self.use_group_writes
        if use_group and self.group_addr is not None:
            return self.group_addr, False
        return self.addr, self.respond_on_writes

    def enable(self, on: bool, *, use_group: Optional[bool] = None) -> None:
        """Enable or disable the motor using command F3."""

        addr, expect_reply = self._select_write_addr(use_group)
        self.bus.send_command(
            addr, 0xF3, [0x01 if on else 0x00], response_expected=expect_reply
        )

    def status(self) -> int:
        """Query status using command F1 and return the status byte."""

        data = self.bus.send_command(self.addr, 0xF1, expected_response_len=1)
        if not data:
            raise RS485FramingError("No status data returned")
        return data[0]

    def emergency_stop(self, *, use_group: Optional[bool] = None) -> None:
        """Send emergency stop (command F7)."""

        addr, expect_reply = self._select_write_addr(use_group)
        self.bus.send_command(addr, 0xF7, [], response_expected=expect_reply)

    def zero_axis(self, *, use_group: Optional[bool] = None) -> None:
        """Set the current axis position to zero (manual function 0x92, page 26)."""

        addr, expect_reply = self._select_write_addr(use_group)
        self.bus.send_command(addr, 0x92, [], response_expected=expect_reply)

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

    def write_all_parameters(self, params: Iterable[int]) -> int:
        """Write all configuration parameters (manual §5.9 command ``0x46``).

        The controller expects a contiguous block of parameter bytes covering
        Byte4–Byte37. The response is a single status byte where ``1`` denotes
        success and ``0`` denotes failure.

        Args:
            params: Iterable of exactly 34 parameter bytes corresponding to the
                "Write all configuration parameters" payload.

        Returns:
            Status byte returned by the controller (``1`` on success). When the
            motor ``Respond`` parameter is set to ``0`` (no replies for write
            commands), ``1`` is returned optimistically after the command is
            sent.
        """

        payload = [int(b) & 0xFF for b in params]
        if len(payload) != 34:
            raise ValueError(
                "write_all_parameters expects 34 bytes (Byte4 through Byte37)"
            )
        expect_reply = self.respond_on_writes
        data = self.bus.send_command(
            self.addr,
            0x46,
            payload,
            response_expected=expect_reply,
            expected_response_len=1 if expect_reply else None,
        )
        if not expect_reply:
            return 1
        if not data:
            raise RS485FramingError("No status byte returned for write_all_parameters")
        return data[0]

    def read_all_parameters(self) -> bytes:
        """Read all configuration parameters (manual §5.9 command ``0x47``)."""

        data = self.bus.send_command(
            self.addr, 0x47, expected_response_len=34
        )
        if len(data) != 34:
            raise RS485FramingError(
                f"Expected 34 parameter bytes, received {len(data)}"
            )
        return data

    @staticmethod
    def _encode_speed_payload(
        omega_rad_s: float, acc: int, gear_ratio: float
    ) -> Tuple[int, int, int]:
        """Pack the F6 payload per manual (dir in BYTE4 MSB, 12-bit speed)."""

        motor_rpm = omega_rad_s * 60.0 / (2.0 * math.pi) * gear_ratio
        direction_bit = 0x01 if motor_rpm < 0 else 0x00
        speed_value = int(min(max(abs(motor_rpm), 0), 3000))
        acc_byte = int(min(max(acc, 0), 255))
        byte4 = (direction_bit << 7) | ((speed_value >> 8) & 0x0F)
        byte5 = speed_value & 0xFF
        return byte4, byte5, acc_byte

    def command_speed(
        self, omega_rad_s: float, acc: int = 10, *, use_group: Optional[bool] = None
    ) -> None:
        """Command the motor in speed mode (F6) using a rad/s setpoint."""

        byte4, byte5, acc_byte = self._encode_speed_payload(
            omega_rad_s, acc, self.gear_ratio
        )
        addr, expect_reply = self._select_write_addr(use_group)
        self.bus.send_command(
            addr, 0xF6, [byte4, byte5, acc_byte], response_expected=expect_reply
        )


@dataclass
class PitchAxisGroup:
    """Group-based dual-motor pitch axis using a shared group address.

    Commands are issued once via ``group_addr`` so both pitch motors move in
    tandem. Mechanical mirroring is achieved by configuring motor A as CCW and
    motor B as CW in their driver menus; the single direction bit in the group
    command then results in opposite physical motion. Encoder feedback is read
    from one designated authority motor via its Slave addr so replies are
    received.
    """

    bus: RS485Bus
    group_addr: int
    motor_a: MksServo42Axis
    motor_b: MksServo42Axis
    authority: str = "a"

    def __post_init__(self) -> None:
        if self.authority not in {"a", "b"}:
            raise ValueError("authority must be 'a' or 'b'")
        if self.motor_a.group_addr != self.group_addr:
            logger.debug(
                "Pitch motor A group addr overridden to %s", hex(self.group_addr)
            )
            self.motor_a.group_addr = self.group_addr
        if self.motor_b.group_addr != self.group_addr:
            logger.debug(
                "Pitch motor B group addr overridden to %s", hex(self.group_addr)
            )
            self.motor_b.group_addr = self.group_addr

    @property
    def authority_axis(self) -> MksServo42Axis:
        return self.motor_a if self.authority == "a" else self.motor_b

    def _command_group_speed(self, omega_rad_s: float, acc: int) -> None:
        byte4, byte5, acc_byte = MksServo42Axis._encode_speed_payload(
            omega_rad_s, acc, self.authority_axis.gear_ratio
        )
        self.bus.send_command(
            self.group_addr,
            0xF6,
            [byte4, byte5, acc_byte],
            response_expected=False,
            retries=0,
        )

    def command_speed(self, omega_rad_s: float, acc: int = 10) -> None:
        """Command both pitch motors via a single group F6 write."""

        self._command_group_speed(omega_rad_s, acc)

    def enable(self, on: bool) -> None:
        """Enable/disable both motors via the shared group address without fallback."""

        self.bus.send_command(
            self.group_addr,
            0xF3,
            [0x01 if on else 0x00],
            response_expected=False,
            retries=0,
        )

    def emergency_stop(self) -> None:
        """Estop both motors via the shared group address without fallback."""

        self.bus.send_command(
            self.group_addr, 0xF7, [], response_expected=False, retries=0
        )

    def zero_axis(self) -> None:
        """Zero both pitch motors at their current position (function 0x92)."""

        self.bus.send_command(
            self.group_addr, 0x92, [], response_expected=False, retries=0
        )

    def read_angle_rad(self) -> float:
        """Return the authoritative pitch angle in radians."""

        return self.authority_axis.read_angle_rad()

    def read_secondary_angle_rad(self) -> Optional[float]:
        """Optionally return the non-authority encoder for diagnostics."""

        secondary = self.motor_b if self.authority == "a" else self.motor_a
        try:
            return secondary.read_angle_rad()
        except Exception:  # noqa: BLE001
            logger.debug("Secondary pitch encoder read failed", exc_info=True)
            return None


class GimbalInterface:
    """Hardware interface for a two-axis MKS-driven gimbal.

    Wraps a yaw axis (single motor) and a pitch axis (single or grouped) and
    translates ControlCmd pan/tilt rate requests into speed-mode F6 commands.
    Provides encoder-derived pan/tilt angles and simple rate estimates by
    differencing successive samples, with optional secondary pitch encoder
    reads exposed for diagnostics.
    """

    def __init__(
        self,
        yaw_axis: MksServo42Axis,
        pitch_axis: MksServo42Axis | PitchAxisGroup,
        *,
        max_rate_rad_s: float = 10.0,
        yaw_accel_byte: int = 10,
        pitch_accel_byte: int = 10,
    ) -> None:
        self.yaw_axis = yaw_axis
        self.pitch_axis = pitch_axis
        self.max_rate_rad_s = max_rate_rad_s
        self.yaw_accel_byte = int(min(max(yaw_accel_byte, 0), 255))
        self.pitch_accel_byte = int(min(max(pitch_accel_byte, 0), 255))
        self._last_sample: Optional[GimbalSample] = None

    def _clamp_rate(self, rate: float) -> float:
        return max(min(rate, self.max_rate_rad_s), -self.max_rate_rad_s)

    def apply_rate_commands(
        self,
        pan_rate_cmd: float,
        tilt_rate_cmd: float,
        *,
        yaw_accel_byte: Optional[int] = None,
        pitch_accel_byte: Optional[int] = None,
    ) -> None:
        """Map pan/tilt rate commands (rad/s) to motor speed commands."""

        pan_rate = self._clamp_rate(pan_rate_cmd)
        tilt_rate = self._clamp_rate(tilt_rate_cmd)

        yaw_acc = int(
            min(max(yaw_accel_byte if yaw_accel_byte is not None else self.yaw_accel_byte, 0), 255)
        )
        pitch_acc = int(
            min(
                max(pitch_accel_byte if pitch_accel_byte is not None else self.pitch_accel_byte, 0),
                255,
            )
        )

        self.yaw_axis.command_speed(pan_rate, acc=yaw_acc)
        self.pitch_axis.command_speed(tilt_rate, acc=pitch_acc)

    def stop(self) -> None:
        """Issue a soft stop to both axes."""

        self.yaw_axis.command_speed(0.0)
        self.pitch_axis.command_speed(0.0)

    def zero_axes(self) -> None:
        """Set the current position of both axes to zero (function 0x92)."""

        if hasattr(self.yaw_axis, "zero_axis"):
            self.yaw_axis.zero_axis()
        if hasattr(self.pitch_axis, "zero_axis"):
            self.pitch_axis.zero_axis()

    def emergency_stop(self) -> None:
        """Emergency stop both axes."""

        self.yaw_axis.emergency_stop()
        self.pitch_axis.emergency_stop()

    def read_angles_rad(self) -> Tuple[float, float]:
        """Return the current pan and tilt angles in radians."""

        pan = self.yaw_axis.read_angle_rad()
        tilt = self.pitch_axis.read_angle_rad()
        return pan, tilt

    def read_pitch_secondary_angle_rad(self) -> Optional[float]:
        """Return the non-authority pitch encoder if available."""

        if hasattr(self.pitch_axis, "read_secondary_angle_rad"):
            return self.pitch_axis.read_secondary_angle_rad()
        return None

    def sample_state(self) -> "GimbalSample":
        """Capture a telemetry sample including optional rate estimates."""

        now = time.monotonic()
        pan, tilt = self.read_angles_rad()
        secondary = self.read_pitch_secondary_angle_rad()

        pan_rate = tilt_rate = None
        if self._last_sample is not None:
            dt = now - self._last_sample.timestamp
            if dt > 0:
                pan_rate = (pan - self._last_sample.pan_rad) / dt
                tilt_rate = (tilt - self._last_sample.tilt_rad) / dt

        sample = GimbalSample(
            timestamp=now,
            pan_rad=pan,
            tilt_rad=tilt,
            pan_rate_rad_s=pan_rate,
            tilt_rate_rad_s=tilt_rate,
            secondary_pitch_rad=secondary,
        )
        self._last_sample = sample
        return sample


@dataclass
class GimbalSample:
    """Telemetry snapshot of gimbal angles and estimated rates."""

    timestamp: float = field(default_factory=time.monotonic)
    pan_rad: float = 0.0
    tilt_rad: float = 0.0
    pan_rate_rad_s: Optional[float] = None
    tilt_rate_rad_s: Optional[float] = None
    secondary_pitch_rad: Optional[float] = None
