"""Bridge ControlCmd messages to MKS serial (TTL-to-RS485) gimbal motion commands.

This Jetson-side process subscribes to the ControlCmd PUB socket, translates
pan/tilt rate commands into MKS SR_CLOSE speed mode writes, and periodically
publishes encoder-derived :class:`CamState` telemetry. Dual-pitch rigs send
commands to motor A and motor B individually with software-defined signs so
mirroring does not depend on controller-side "Dir" settings.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

import zmq
import yaml

from common.config_sync import expand_config_paths, merge_config_maps, parse_config_text, read_snapshot
from common.schemas import CamState, control_cmd_from_json
from common.serial_io import SerialReplySubscriber, SerialUpdatePublisher
from common.shutdown import install_signal_handlers
from common.gimbal.mks_servo42_rs485 import MksServo42Axis

_LOG = logging.getLogger(__name__)
_MKS_ACCEL_RAD_S2_PER_BYTE = 0.35
_DEFAULT_GIMBAL_ACCEL_LIMIT_RAD_S2 = 3.5

try:
    from smbus2 import SMBus  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    try:
        from smbus import SMBus  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        SMBus = None  # type: ignore[assignment]


def _wrapped_delta(angle_now: float, angle_prev: float) -> float:
    return math.atan2(math.sin(angle_now - angle_prev), math.cos(angle_now - angle_prev))


def _parse_tcp_port(endpoint: str, name: str) -> int:
    try:
        port = int(endpoint.rsplit(":", 1)[1])
    except Exception as exc:  # noqa: BLE001 - defensive parsing
        raise SystemExit(f"invalid {name} endpoint: {endpoint!r}") from exc
    if port <= 0 or port > 65535:
        raise SystemExit(f"{name} port must be in 1..65535 (got {port})")
    return port


def _load_config(paths: Iterable[Path]) -> Mapping[str, Any]:
    configs = []
    for path in paths:
        snapshot = read_snapshot(path)
        configs.append(parse_config_text(snapshot.text, str(path)))
    return merge_config_maps(*configs)


def _clamp_accel_byte(value: int) -> int:
    return int(min(max(int(value), 0), 255))


def _physical_accel_limit_from_cfg(
    gimbal_cfg: Mapping[str, Any],
    *,
    axis: str,
) -> float:
    key = f"{axis}_accel_limit_rad_s2"
    raw = gimbal_cfg.get(key, _DEFAULT_GIMBAL_ACCEL_LIMIT_RAD_S2)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"gimbal.{key} must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"gimbal.{key} must be a positive finite number")
    return value


def _build_serial_targets(cfg: Mapping[str, Any]) -> Tuple[Mapping[str, Any], float]:
    gimbal_cfg = cfg.get("gimbal")
    if not isinstance(gimbal_cfg, Mapping):
        raise SystemExit("config missing 'gimbal' section")

    counts_per_rev = int(gimbal_cfg.get("counts_per_rev", 0x4000))
    yaw_ratio = float(gimbal_cfg.get("yaw_gear_ratio", 1.0))
    pitch_ratio = float(gimbal_cfg.get("pitch_gear_ratio", 1.0))

    yaw_addr = int(gimbal_cfg.get("yaw_addr", 1))
    yaw_group_addr = gimbal_cfg.get("yaw_group_addr")
    yaw_group_addr = int(yaw_group_addr) if yaw_group_addr is not None else None

    pitch_group_addr = gimbal_cfg.get("pitch_group_addr")
    pitch_group_addr = int(pitch_group_addr) if pitch_group_addr is not None else None

    respond_on_writes = bool(gimbal_cfg.get("respond_on_writes", False))

    try:
        pitch_motor_a_addr = int(gimbal_cfg["pitch_motor_a_addr"])
        pitch_motor_b_addr = int(gimbal_cfg["pitch_motor_b_addr"])
    except KeyError as exc:
        raise SystemExit("gimbal.pitch_motor_a_addr and pitch_motor_b_addr are required") from exc

    authority = gimbal_cfg.get("pitch_encoder_authority", "a")
    if authority not in {"a", "b"}:
        raise SystemExit("gimbal.pitch_encoder_authority must be 'a' or 'b'")

    pitch_motor_a_sign = float(gimbal_cfg.get("pitch_motor_a_sign", 1.0))
    pitch_motor_b_sign = float(gimbal_cfg.get("pitch_motor_b_sign", -1.0))
    yaw_motor_sign = float(gimbal_cfg.get("yaw_motor_sign", 1.0))
    camstate_yaw_sign = float(gimbal_cfg.get("camstate_yaw_sign", 1.0))
    camstate_pitch_sign = float(gimbal_cfg.get("camstate_pitch_sign", 1.0))
    if yaw_motor_sign == 0.0:
        raise SystemExit("gimbal.yaw_motor_sign must be non-zero")
    if camstate_yaw_sign == 0.0:
        raise SystemExit("gimbal.camstate_yaw_sign must be non-zero")
    if camstate_pitch_sign == 0.0:
        raise SystemExit("gimbal.camstate_pitch_sign must be non-zero")
    if pitch_motor_a_sign == 0.0 or pitch_motor_b_sign == 0.0:
        raise SystemExit("gimbal.pitch_motor_a_sign and pitch_motor_b_sign must be non-zero")

    yaw_accel_limit_rad_s2 = _physical_accel_limit_from_cfg(gimbal_cfg, axis="yaw")
    pitch_accel_limit_rad_s2 = _physical_accel_limit_from_cfg(gimbal_cfg, axis="pitch")
    yaw_rate_limit = float(gimbal_cfg.get("yaw_rate_limit_rad_s", 10.0))
    pitch_rate_limit = float(gimbal_cfg.get("pitch_rate_limit_rad_s", 10.0))
    pitch_div_thresh = float(gimbal_cfg.get("pitch_divergence_thresh_rad", 0.0873))

    _yaw_min = gimbal_cfg.get("yaw_min_rad")
    _yaw_max = gimbal_cfg.get("yaw_max_rad")
    _pitch_min = gimbal_cfg.get("pitch_min_rad")
    _pitch_max = gimbal_cfg.get("pitch_max_rad")
    yaw_min_rad: Optional[float] = float(_yaw_min) if _yaw_min is not None else None
    yaw_max_rad: Optional[float] = float(_yaw_max) if _yaw_max is not None else None
    pitch_min_rad: Optional[float] = float(_pitch_min) if _pitch_min is not None else None
    pitch_max_rad: Optional[float] = float(_pitch_max) if _pitch_max is not None else None
    if yaw_min_rad is not None and yaw_max_rad is not None and yaw_min_rad >= yaw_max_rad:
        raise SystemExit("gimbal.yaw_min_rad must be less than gimbal.yaw_max_rad")
    if pitch_min_rad is not None and pitch_max_rad is not None and pitch_min_rad >= pitch_max_rad:
        raise SystemExit("gimbal.pitch_min_rad must be less than gimbal.pitch_max_rad")

    serial_targets = {
        "counts_per_rev": counts_per_rev,
        "yaw_ratio": yaw_ratio,
        "pitch_ratio": pitch_ratio,
        "yaw_addr": yaw_addr,
        "yaw_group_addr": yaw_group_addr,
        "pitch_group_addr": pitch_group_addr,
        "pitch_motor_a_addr": pitch_motor_a_addr,
        "pitch_motor_b_addr": pitch_motor_b_addr,
        "pitch_authority": authority,
        "pitch_motor_a_sign": pitch_motor_a_sign,
        "pitch_motor_b_sign": pitch_motor_b_sign,
        "yaw_motor_sign": yaw_motor_sign,
        "camstate_yaw_sign": camstate_yaw_sign,
        "camstate_pitch_sign": camstate_pitch_sign,
        "respond_on_writes": respond_on_writes,
        "yaw_accel_limit_rad_s2": yaw_accel_limit_rad_s2,
        "pitch_accel_limit_rad_s2": pitch_accel_limit_rad_s2,
        "yaw_rate_limit": yaw_rate_limit,
        "pitch_rate_limit": pitch_rate_limit,
        "yaw_min_rad": yaw_min_rad,
        "yaw_max_rad": yaw_max_rad,
        "pitch_min_rad": pitch_min_rad,
        "pitch_max_rad": pitch_max_rad,
    }
    return serial_targets, pitch_div_thresh


def _load_parameter_map(path: Path) -> Mapping[int, Tuple[int, ...]]:
    snapshot = read_snapshot(path)
    data = yaml.safe_load(snapshot.text) or {}
    motors = data.get("motors") if isinstance(data, Mapping) else {}
    if not isinstance(motors, Mapping):
        raise SystemExit(f"parameter file {path} must contain a 'motors' mapping")

    parameter_map: dict[int, Tuple[int, ...]] = {}
    for addr_str, entry in motors.items():
        try:
            addr = int(addr_str)
        except Exception as exc:  # noqa: BLE001 - defensive config parsing
            raise SystemExit(f"invalid motor address key {addr_str!r} in {path}") from exc

        params = entry.get("parameters") if isinstance(entry, Mapping) else entry
        if params is None:
            _LOG.info("no parameters listed for motor %s; skipping", addr_str)
            continue
        try:
            payload = tuple(int(b) & 0xFF for b in params)
        except Exception as exc:  # noqa: BLE001 - defensive config parsing
            raise SystemExit(f"invalid parameter payload for motor {addr_str!r}: {entry}") from exc
        if len(payload) != 34:
            raise SystemExit(
                f"motor {addr} parameters must have 34 bytes (Byte4-Byte37); got {len(payload)}"
            )
        parameter_map[addr] = payload

    return parameter_map


def _build_param_command(
    addr: int,
    payload: Tuple[int, ...],
    *,
    expect_reply: bool,
    target: str,
) -> Mapping[str, Any]:
    return {
        "cmd_id": f"params:{addr}",
        "func": "0x46",
        "addr": addr,
        "payload": list(payload),
        "expect_reply": expect_reply,
        "expected_len": 1 if expect_reply else None,
        "priority": "high",
        "target": target,
    }


def _publish_cam_state(
    pub: zmq.Socket,
    sample,
    *,
    frame_id: int,
    src_ts_ms: int,
    home_pan: Optional[float] = None,
    home_tilt: Optional[float] = None,
) -> None:
    cam_state = CamState(
        frame_id=frame_id,
        src_ts_ms=src_ts_ms,
        pan=float(sample.pan_rad),
        tilt=float(sample.tilt_rad),
        pan_rate=sample.pan_rate_rad_s,
        tilt_rate=sample.tilt_rate_rad_s,
        home_pan=home_pan,
        home_tilt=home_tilt,
    )
    pub.send_string(cam_state.model_dump_json(exclude_none=True))


def _make_control_sub(ctx: zmq.Context, endpoint: str) -> zmq.Socket:
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(endpoint)
    return sub


def _make_state_pub(ctx: zmq.Context, endpoint: Optional[str]) -> Optional[zmq.Socket]:
    if not endpoint:
        _LOG.warning("gimbal_state endpoint not configured; telemetry will not be published")
        return None
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)
    port = _parse_tcp_port(endpoint, "net.zmq_gimbal_state")
    pub.bind(f"tcp://0.0.0.0:{port}")
    _LOG.info("publishing CamState on tcp://0.0.0.0:%d", port)
    return pub


def _counts_to_rad(counts: int, *, counts_per_rev: int, gear_ratio: float) -> float:
    motor_revs = counts / float(counts_per_rev)
    axis_revs = motor_revs / gear_ratio
    return axis_revs * 2.0 * math.pi


def _encode_speed_cmd(
    omega_rad_s: float,
    *,
    acc: int,
    gear_ratio: float,
    max_rate: float,
) -> Tuple[int, int, int]:
    omega = max(min(omega_rad_s, max_rate), -max_rate)
    return MksServo42Axis._encode_speed_payload(omega, acc, gear_ratio)


def _finite_positive(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _limit_rate_by_accel(
    desired_rate: float,
    previous_rate: float,
    accel_rad_s2: Optional[float],
    dt_s: float,
) -> float:
    accel = _finite_positive(accel_rad_s2)
    if accel is None:
        return float(desired_rate)
    dt = max(float(dt_s), 0.0)
    max_delta = accel * dt
    delta = float(desired_rate) - float(previous_rate)
    if delta > max_delta:
        return float(previous_rate) + max_delta
    if delta < -max_delta:
        return float(previous_rate) - max_delta
    return float(desired_rate)


def _mks_accel_byte_from_physical(
    accel_rad_s2: Optional[float],
) -> int:
    accel = _finite_positive(accel_rad_s2)
    if accel is None:
        return 0
    return _clamp_accel_byte(max(1, int(round(accel / _MKS_ACCEL_RAD_S2_PER_BYTE))))


def _encode_position_cmd(
    omega_rad_s: float,
    *,
    acc: int,
    gear_ratio: float,
    rel_pulses: int,
) -> Tuple[int, int, int, int, int, int, int]:
    return MksServo42Axis._encode_position_payload(omega_rad_s, acc, gear_ratio, rel_pulses)


def _apply_hard_angle_limit(
    rate_cmd: float,
    current_angle: Optional[float],
    angle_min: Optional[float],
    angle_max: Optional[float],
    axis: str,
) -> float:
    """Zero out a rate command when the axis is at or past a hard angle bound.

    A positive command is blocked when the axis is at or beyond *angle_max*;
    a negative command is blocked when the axis is at or below *angle_min*.
    Commands that drive the axis back within bounds are always passed through.
    Returns the original *rate_cmd* unchanged when *current_angle* is None
    (encoder data not yet available) or when neither limit is configured.
    """
    if current_angle is None:
        return rate_cmd
    if angle_max is not None and current_angle >= angle_max and rate_cmd > 0.0:
        _LOG.debug(
            "hard angle limit: %s at %.4f rad >= max %.4f rad; blocking positive command %.4f rad/s",
            axis, current_angle, angle_max, rate_cmd,
        )
        return 0.0
    if angle_min is not None and current_angle <= angle_min and rate_cmd < 0.0:
        _LOG.debug(
            "hard angle limit: %s at %.4f rad <= min %.4f rad; blocking negative command %.4f rad/s",
            axis, current_angle, angle_min, rate_cmd,
        )
        return 0.0
    return rate_cmd


def _wait_for_status(
    reply_sub: SerialReplySubscriber,
    expected_addrs: Iterable[int],
    *,
    timeout_s: float = 2.0,
) -> None:
    expected = set(expected_addrs)
    deadline = time.monotonic() + timeout_s
    while expected and time.monotonic() < deadline:
        for reply in reply_sub.recv_nowait():
            if reply.get("func") != "F1":
                continue
            addr = reply.get("addr")
            if addr in expected:
                status = reply.get("reply", {}).get("parsed", {}).get("status")
                if status in (None, 0):
                    raise SystemExit(f"status query failed for addr={addr}")
                _LOG.info("axis addr=%s status=%s", addr, status)
                expected.remove(addr)
        time.sleep(0.01)
    if expected:
        raise SystemExit(f"status query timed out for addr(s): {sorted(expected)}")


def _wait_for_func_replies(
    reply_sub: SerialReplySubscriber,
    *,
    func_byte: int,
    expected_addrs: Iterable[int],
    timeout_s: float,
) -> set[int]:
    expected = set(expected_addrs)
    deadline = time.monotonic() + timeout_s
    while expected and time.monotonic() < deadline:
        for reply in reply_sub.recv_nowait():
            if _reply_func_byte(reply) != func_byte:
                continue
            addr = reply.get("addr")
            if addr in expected:
                expected.remove(addr)
        time.sleep(0.01)
    return expected


def _build_update(
    *,
    source: str,
    target: str,
    commands: Iterable[Mapping[str, Any]],
    fields: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    return {
        "type": "SerialUpdate",
        "source": source,
        "target": target,
        "fields": dict(fields or {}),
        "commands": list(commands),
        "update_ts_ms": int(time.time() * 1000),
    }


def _build_command(
    *,
    cmd_id: str,
    func: str,
    addr: int,
    payload: Iterable[int],
    expect_reply: bool,
    expected_len: Optional[int],
    priority: str,
    target: str,
) -> Mapping[str, Any]:
    return {
        "cmd_id": cmd_id,
        "func": func,
        "addr": addr,
        "payload": list(payload),
        "expect_reply": expect_reply,
        "expected_len": expected_len,
        "priority": priority,
        "target": target,
    }


def _reply_func_byte(reply: Mapping[str, Any]) -> Optional[int]:
    func = reply.get("func")
    if func is None:
        return None
    if isinstance(func, int):
        return func
    if isinstance(func, str):
        if func.lower().startswith("0x"):
            return int(func, 16)
        if func.upper().startswith("F"):
            return int(func, 16)
        return int(func)
    return None


@dataclass
class _AngleSample:
    timestamp: float
    pan_rad: float
    tilt_rad: float
    pan_rate_rad_s: Optional[float]
    tilt_rate_rad_s: Optional[float]
    secondary_pitch_rad: Optional[float] = None


@dataclass
class _DeviceSensorConfig:
    mpu_bus: int = 7
    mpu_addr: int = 0x68
    mag_addr: int = 0x0C


_SENSOR_PWR_MGMT_1 = 0x6B
_SENSOR_INT_PIN_CFG = 0x37
_SENSOR_INT_BYPASS_VAL = 0x02
_SENSOR_ACCEL_XOUT = 0x3B
_SENSOR_MAG_ST1 = 0x02
_SENSOR_MAG_DATA = 0x03
_SENSOR_MAG_ST2 = 0x09
_SENSOR_MAG_CNTL1 = 0x0A
_SENSOR_MAG_POWER_DOWN = 0x00
_SENSOR_MAG_CONTINUOUS_100HZ = 0x16
_SENSOR_ACCEL_SCALE = 16384.0

_SUPPORTED_CAMSTATE_DEVICE_KEYS = {
    "mpu_bus",
    "mpu_addr",
    "mag_addr",
    "publish_hz",
}
_REMOVED_CAMSTATE_DEVICE_KEYS = {
    "mag_bus",
    "pwr_mgmt_1_reg",
    "int_pin_cfg_reg",
    "int_pin_cfg_bypass_val",
    "accel_xout_reg",
    "gyro_xout_reg",
    "mag_st1_reg",
    "mag_data_reg",
    "mag_st2_reg",
    "mag_cntl1_reg",
    "mag_mode_val",
    "accel_scale",
    "gyro_scale",
    "alpha",
    "pan_sign",
    "tilt_sign",
    "pan_offset_rad",
    "tilt_offset_rad",
    "pitch_gyro_axis",
    "pitch_gyro_sign",
    "pitch_accel_axis",
    "pitch_accel_sign",
    "home_pan",
    "home_tilt",
}


def _int_from_cfg(cfg: Mapping[str, Any], key: str, default: int) -> int:
    raw = cfg.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"camstate_devices.{key} must be an integer, got {raw!r}") from exc


def _validate_camstate_device_keys(cfg: Mapping[str, Any]) -> None:
    keys = {str(key) for key in cfg.keys()}
    unknown = sorted(keys - _SUPPORTED_CAMSTATE_DEVICE_KEYS)
    if not unknown:
        return
    removed = [key for key in unknown if key in _REMOVED_CAMSTATE_DEVICE_KEYS]
    if removed:
        raise SystemExit(
            "camstate_devices contains removed keys: "
            + ", ".join(removed)
            + ". Keep only: "
            + ", ".join(sorted(_SUPPORTED_CAMSTATE_DEVICE_KEYS))
        )
    raise SystemExit(
        "camstate_devices contains unsupported keys: "
        + ", ".join(unknown)
        + ". Keep only: "
        + ", ".join(sorted(_SUPPORTED_CAMSTATE_DEVICE_KEYS))
    )


def _read_word(bus: Any, addr: int, reg: int) -> int:
    hi = bus.read_byte_data(addr, reg)
    lo = bus.read_byte_data(addr, reg + 1)
    val = (hi << 8) | lo
    if val >= 0x8000:
        val -= 65536
    return val


def _accel_pitch_roll(ax: float, ay: float, az: float) -> tuple[float, float]:
    pitch = math.atan2(float(ax), math.sqrt(float(ay) * float(ay) + float(az) * float(az)))
    roll = math.atan2(-float(ay), float(az))
    return pitch, roll


def _tilt_compensated_heading(
    *,
    pitch: float,
    roll: float,
    mx: float,
    my: float,
    mz: float,
) -> float:
    mx_aligned = float(my)
    my_aligned = float(mx)
    mz_aligned = -float(mz)
    mx2 = mx_aligned * math.cos(pitch) + mz_aligned * math.sin(pitch)
    my2 = (
        mx_aligned * math.sin(roll) * math.sin(pitch)
        + my_aligned * math.cos(roll)
        - mz_aligned * math.sin(roll) * math.cos(pitch)
    )
    return math.atan2(my2, mx2)


def _compute_orientation(ax: float, ay: float, az: float, mx: float, my: float, mz: float) -> tuple[float, float]:
    pitch, roll = _accel_pitch_roll(ax, ay, az)
    heading = _tilt_compensated_heading(pitch=pitch, roll=roll, mx=mx, my=my, mz=mz)
    return pitch, heading


def _apply_encoder_horizon_offset(
    *,
    encoder_tilt_rad: float,
    secondary_tilt_rad: Optional[float],
    imu_pitch_rad: Optional[float],
    horizon_offset_rad: Optional[float],
) -> tuple[float, Optional[float], Optional[float], bool]:
    """Align encoder tilt with IMU-defined horizon using a one-time zero offset."""
    new_offset = horizon_offset_rad
    locked_now = False
    if (
        new_offset is None
        and imu_pitch_rad is not None
        and math.isfinite(float(imu_pitch_rad))
        and math.isfinite(float(encoder_tilt_rad))
    ):
        new_offset = float(imu_pitch_rad) - float(encoder_tilt_rad)
        locked_now = True

    if new_offset is None:
        return float(encoder_tilt_rad), secondary_tilt_rad, None, locked_now

    corrected_tilt = float(encoder_tilt_rad) + float(new_offset)
    corrected_secondary = (
        None if secondary_tilt_rad is None else float(secondary_tilt_rad) + float(new_offset)
    )
    return corrected_tilt, corrected_secondary, float(new_offset), locked_now


class _DeviceSensorReader:
    def __init__(self, cfg: _DeviceSensorConfig) -> None:
        if SMBus is None:
            raise SystemExit("camstate_source=devices requires smbus2 or smbus to be installed")
        self._cfg = cfg
        self._mpu = SMBus(cfg.mpu_bus)
        self._mag = SMBus(cfg.mpu_bus)

    def init(self) -> None:
        self._mpu.write_byte_data(self._cfg.mpu_addr, _SENSOR_PWR_MGMT_1, 0)
        time.sleep(0.1)
        self._mpu.write_byte_data(self._cfg.mpu_addr, _SENSOR_INT_PIN_CFG, _SENSOR_INT_BYPASS_VAL)
        self._mag.write_byte_data(self._cfg.mag_addr, _SENSOR_MAG_CNTL1, _SENSOR_MAG_POWER_DOWN)
        time.sleep(0.01)
        self._mag.write_byte_data(self._cfg.mag_addr, _SENSOR_MAG_CNTL1, _SENSOR_MAG_CONTINUOUS_100HZ)
        time.sleep(0.01)

    def close(self) -> None:
        for bus in (self._mpu, self._mag):
            try:
                bus.close()
            except Exception:
                pass

    def read_accel(self) -> tuple[float, float, float]:
        ax = _read_word(self._mpu, self._cfg.mpu_addr, _SENSOR_ACCEL_XOUT) / _SENSOR_ACCEL_SCALE
        ay = _read_word(self._mpu, self._cfg.mpu_addr, _SENSOR_ACCEL_XOUT + 2) / _SENSOR_ACCEL_SCALE
        az = _read_word(self._mpu, self._cfg.mpu_addr, _SENSOR_ACCEL_XOUT + 4) / _SENSOR_ACCEL_SCALE
        return ax, ay, az

    def read_mag(self) -> Optional[tuple[int, int, int]]:
        st1 = self._mag.read_byte_data(self._cfg.mag_addr, _SENSOR_MAG_ST1)
        if not (st1 & 0x01):
            return None
        data = self._mag.read_i2c_block_data(self._cfg.mag_addr, _SENSOR_MAG_DATA, 6)
        self._mag.read_byte_data(self._cfg.mag_addr, _SENSOR_MAG_ST2)
        x = (data[1] << 8) | data[0]
        y = (data[3] << 8) | data[2]
        z = (data[5] << 8) | data[4]
        if x >= 32768:
            x -= 65536
        if y >= 32768:
            y -= 65536
        if z >= 32768:
            z -= 65536
        return x, y, z


def _build_device_sensor_cfg(cfg: Mapping[str, Any]) -> _DeviceSensorConfig:
    _validate_camstate_device_keys(cfg)
    mpu_bus = _int_from_cfg(cfg, "mpu_bus", 7)
    return _DeviceSensorConfig(
        mpu_bus=mpu_bus,
        mpu_addr=_int_from_cfg(cfg, "mpu_addr", 0x68),
        mag_addr=_int_from_cfg(cfg, "mag_addr", 0x0C),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/network.yaml", help="Path to YAML config")
    ap.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
        help="Comma-separated YAML configs merged over --config.",
    )
    ap.add_argument(
        "--feedback-hz",
        type=float,
        default=None,
        help="Override telemetry publish rate (Hz); defaults to gimbal.feedback_hz",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    config_paths = expand_config_paths(args.config, args.config_extra)
    cfg = _load_config(config_paths)

    net_cfg = cfg.get("net") or {}
    ctrl_ep = net_cfg.get("zmq_control")
    if not ctrl_ep:
        raise SystemExit("config missing net.zmq_control endpoint")

    state_ep = net_cfg.get("zmq_gimbal_state")

    serial_targets, pitch_div_thresh = _build_serial_targets(cfg)
    parameter_map: Mapping[int, Tuple[int, ...]] = {}
    gimbal_cfg = cfg.get("gimbal") or {}
    camstate_devices_top = cfg.get("camstate_devices")
    camstate_devices_cfg: Mapping[str, Any]
    if isinstance(camstate_devices_top, Mapping):
        camstate_devices_cfg = camstate_devices_top
    else:
        camstate_devices_cfg = {}
    if isinstance(gimbal_cfg.get("camstate_devices"), Mapping):
        _LOG.warning(
            "gimbal.camstate_devices is deprecated and ignored; use top-level camstate_devices"
        )

    camstate_source_raw = gimbal_cfg.get("camstate_source", "encoder")
    camstate_source = str(camstate_source_raw).strip().lower()
    if camstate_source in {"device", "imu"}:
        camstate_source = "devices"
    if camstate_source not in {"encoder", "devices"}:
        raise SystemExit("gimbal.camstate_source must be 'encoder' or 'devices'")

    device_sensor_cfg: Optional[_DeviceSensorConfig] = None
    device_sensor_reader: Optional[_DeviceSensorReader] = None
    encoder_imu_sensor_cfg: Optional[_DeviceSensorConfig] = None
    encoder_imu_reader: Optional[_DeviceSensorReader] = None
    if camstate_source == "devices":
        device_sensor_cfg = _build_device_sensor_cfg(camstate_devices_cfg)
        device_sensor_reader = _DeviceSensorReader(device_sensor_cfg)
    else:
        if SMBus is None:
            _LOG.info(
                "encoder CamState IMU horizon alignment disabled: smbus2/smbus is not installed"
            )
        else:
            try:
                encoder_imu_sensor_cfg = _build_device_sensor_cfg(camstate_devices_cfg)
                encoder_imu_reader = _DeviceSensorReader(encoder_imu_sensor_cfg)
            except SystemExit as exc:
                _LOG.warning(
                    "encoder CamState IMU horizon alignment disabled by camstate_devices config: %s",
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.info("encoder CamState IMU horizon alignment unavailable: %s", exc)

    serial_target = str(gimbal_cfg.get("serial_target", "gimbal"))
    serial_update_ep = gimbal_cfg.get("serial_update_endpoint") or net_cfg.get(
        "zmq_serial_update"
    )
    if not serial_update_ep:
        serial_update_ep = "tcp://127.0.0.1:5571"
    serial_reply_ep = gimbal_cfg.get("serial_reply_endpoint") or net_cfg.get(
        "zmq_serial_reply"
    )
    if not serial_reply_ep:
        serial_reply_ep = "tcp://127.0.0.1:5572"

    param_path = gimbal_cfg.get("parameter_file")
    if param_path:
        parameter_map = _load_parameter_map(Path(str(param_path)))
        _LOG.info("loaded parameter sets for %d motors from %s", len(parameter_map), param_path)

    _LOG.info(
        "configured serial gimbal: yaw addr=%d group=%s sign=%.1f, pitch a=%d b=%d signs=(%.1f, %.1f) camstate_signs=(%.1f, %.1f) authority=%s, divergence_thresh=%.4f rad",
        serial_targets["yaw_addr"],
        serial_targets["yaw_group_addr"],
        serial_targets["yaw_motor_sign"],
        serial_targets["pitch_motor_a_addr"],
        serial_targets["pitch_motor_b_addr"],
        serial_targets["pitch_motor_a_sign"],
        serial_targets["pitch_motor_b_sign"],
        serial_targets["camstate_yaw_sign"],
        serial_targets["camstate_pitch_sign"],
        serial_targets["pitch_authority"],
        pitch_div_thresh,
    )
    _LOG.info("CamState source mode: %s", camstate_source)
    if device_sensor_cfg is not None:
        _LOG.info(
            "CamState devices: mpu_bus=%d mpu_addr=0x%02x mag_addr=0x%02x",
            device_sensor_cfg.mpu_bus,
            device_sensor_cfg.mpu_addr,
            device_sensor_cfg.mag_addr,
        )
    elif encoder_imu_sensor_cfg is not None:
        _LOG.info(
            "encoder CamState IMU horizon alignment configured: mpu_bus=%d mpu_addr=0x%02x",
            encoder_imu_sensor_cfg.mpu_bus,
            encoder_imu_sensor_cfg.mpu_addr,
        )

    feedback_hz = args.feedback_hz
    if feedback_hz is None:
        try:
            feedback_hz = float(cfg.get("gimbal", {}).get("feedback_hz", 20.0))
        except Exception:  # noqa: BLE001 - config parsing guard
            feedback_hz = 20.0
    feedback_hz = max(0.1, feedback_hz)
    feedback_period = 1.0 / feedback_hz

    stop_event = install_signal_handlers()

    ctx = zmq.Context()
    sub = _make_control_sub(ctx, ctrl_ep)
    pub = _make_state_pub(ctx, state_ep)
    update_pub = SerialUpdatePublisher(serial_update_ep, ctx=ctx)
    reply_sub = SerialReplySubscriber(
        serial_reply_ep,
        topics=[f"serial.reply.{serial_target}"],
        ctx=ctx,
    )
    _LOG.info("subscribing to ControlCmd on %s (feedback %.1f Hz)", ctrl_ep, feedback_hz)
    _LOG.info("publishing SerialUpdate to %s (target=%s)", serial_update_ep, serial_target)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    last_cmd = None
    last_pub_time = 0.0
    last_stats_log = 0.0
    last_sample: Optional[_AngleSample] = None
    last_divergence_log = 0.0
    local_frame_id = 0
    yaw_addr = int(serial_targets["yaw_addr"])
    pitch_a_addr = int(serial_targets["pitch_motor_a_addr"])
    pitch_b_addr = int(serial_targets["pitch_motor_b_addr"])
    pitch_a_sign = float(serial_targets["pitch_motor_a_sign"])
    pitch_b_sign = float(serial_targets["pitch_motor_b_sign"])
    pitch_authority = serial_targets["pitch_authority"]
    yaw_sign = float(serial_targets["yaw_motor_sign"])
    camstate_yaw_sign = float(serial_targets["camstate_yaw_sign"])
    camstate_pitch_sign = float(serial_targets["camstate_pitch_sign"])
    yaw_ratio = float(serial_targets["yaw_ratio"])
    pitch_ratio = float(serial_targets["pitch_ratio"])
    yaw_accel_limit_rad_s2 = float(serial_targets["yaw_accel_limit_rad_s2"])
    pitch_accel_limit_rad_s2 = float(serial_targets["pitch_accel_limit_rad_s2"])
    yaw_accel = _mks_accel_byte_from_physical(yaw_accel_limit_rad_s2)
    pitch_accel = _mks_accel_byte_from_physical(pitch_accel_limit_rad_s2)
    yaw_rate_limit = float(serial_targets["yaw_rate_limit"])
    pitch_rate_limit = float(serial_targets["pitch_rate_limit"])
    counts_per_rev = int(serial_targets["counts_per_rev"])
    respond_on_writes = bool(serial_targets["respond_on_writes"])
    yaw_min_rad: Optional[float] = serial_targets["yaw_min_rad"]
    yaw_max_rad: Optional[float] = serial_targets["yaw_max_rad"]
    pitch_min_rad: Optional[float] = serial_targets["pitch_min_rad"]
    pitch_max_rad: Optional[float] = serial_targets["pitch_max_rad"]
    if yaw_min_rad is not None or yaw_max_rad is not None:
        _LOG.info("hard yaw angle limits: min=%s max=%s rad", yaw_min_rad, yaw_max_rad)
    if pitch_min_rad is not None or pitch_max_rad is not None:
        _LOG.info("hard pitch angle limits: min=%s max=%s rad", pitch_min_rad, pitch_max_rad)
    encoder_stale_warn_s = max(float(gimbal_cfg.get("encoder_stale_warn_s", 0.6)), 0.1)
    command_watchdog_timeout_s = max(float(gimbal_cfg.get("command_watchdog_timeout_s", 0.75)), 0.1)
    command_watchdog_min_speed = abs(float(gimbal_cfg.get("command_watchdog_min_speed_rad_s", 0.1)))
    command_watchdog_min_delta = max(int(gimbal_cfg.get("command_watchdog_min_delta_counts", 1)), 1)
    pitch_authority_addr = pitch_a_addr if pitch_authority == "a" else pitch_b_addr
    device_pitch: Optional[float] = None
    device_heading: Optional[float] = None
    device_last_err_log = 0.0
    encoder_horizon_offset_rad: Optional[float] = None
    encoder_imu_last_err_log = 0.0
    camstate_home_pan: Optional[float] = None
    camstate_home_tilt: Optional[float] = None
    if device_sensor_reader is not None:
        device_sensor_reader.init()
    if encoder_imu_reader is not None:
        try:
            encoder_imu_reader.init()
            _LOG.info("encoder CamState IMU horizon alignment active")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("encoder CamState IMU horizon alignment init failed: %s", exc)
            encoder_imu_reader.close()
            encoder_imu_reader = None

    def _pitch_speed_commands(
        rate_rad_s: float,
        *,
        accel_byte: int,
        priority: str,
    ) -> list[Mapping[str, Any]]:
        return [
            _build_command(
                cmd_id=f"speed:pitch_a:{time.time_ns()}",
                func="F6",
                addr=pitch_a_addr,
                payload=_encode_speed_cmd(
                    pitch_a_sign * rate_rad_s,
                    acc=accel_byte,
                    gear_ratio=pitch_ratio,
                    max_rate=pitch_rate_limit,
                ),
                expect_reply=respond_on_writes,
                expected_len=None,
                priority=priority,
                target=serial_target,
            ),
            _build_command(
                cmd_id=f"speed:pitch_b:{time.time_ns()}",
                func="F6",
                addr=pitch_b_addr,
                payload=_encode_speed_cmd(
                    pitch_b_sign * rate_rad_s,
                    acc=accel_byte,
                    gear_ratio=pitch_ratio,
                    max_rate=pitch_rate_limit,
                ),
                expect_reply=respond_on_writes,
                expected_len=None,
                priority=priority,
                target=serial_target,
            ),
        ]

    def _pitch_position_commands(rel_axis_pulses: int, *, speed_rad_s: float, priority: str) -> list[Mapping[str, Any]]:
        return [
            _build_command(
                cmd_id=f"position:pitch_a:{time.time_ns()}",
                func="FD",
                addr=pitch_a_addr,
                payload=_encode_position_cmd(
                    pitch_a_sign * speed_rad_s,
                    acc=pitch_accel,
                    gear_ratio=pitch_ratio,
                    rel_pulses=int(pitch_a_sign * rel_axis_pulses),
                ),
                expect_reply=respond_on_writes,
                expected_len=None,
                priority=priority,
                target=serial_target,
            ),
            _build_command(
                cmd_id=f"position:pitch_b:{time.time_ns()}",
                func="FD",
                addr=pitch_b_addr,
                payload=_encode_position_cmd(
                    pitch_b_sign * speed_rad_s,
                    acc=pitch_accel,
                    gear_ratio=pitch_ratio,
                    rel_pulses=int(pitch_b_sign * rel_axis_pulses),
                ),
                expect_reply=respond_on_writes,
                expected_len=None,
                priority=priority,
                target=serial_target,
            ),
        ]

    startup_start = time.monotonic()
    if parameter_map:
        param_cmds = [
            _build_param_command(
                addr,
                payload,
                expect_reply=bool(serial_targets["respond_on_writes"]),
                target=serial_target,
            )
            for addr, payload in parameter_map.items()
        ]
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=param_cmds,
            )
        )

    enable_cmds = [
        _build_command(
            cmd_id="enable:yaw",
            func="F3",
            addr=yaw_addr,
            payload=[0x01],
            expect_reply=False,
            expected_len=None,
            priority="critical",
            target=serial_target,
        ),
        _build_command(
            cmd_id="enable:pitch_a",
            func="F3",
            addr=pitch_a_addr,
            payload=[0x01],
            expect_reply=False,
            expected_len=None,
            priority="critical",
            target=serial_target,
        ),
        _build_command(
            cmd_id="enable:pitch_b",
            func="F3",
            addr=pitch_b_addr,
            payload=[0x01],
            expect_reply=False,
            expected_len=None,
            priority="critical",
            target=serial_target,
        ),
    ]
    update_pub.send_update(
        _build_update(
            source="jetson.gimbal_bridge",
            target=serial_target,
            commands=enable_cmds,
        )
    )

    # Step 1: Check IMU horizontal value and move motors to reach zero
    imu_pitch_value: Optional[float] = None
    calibration_speed_rad_s = float(gimbal_cfg.get("calibration_speed_rad_s", 0.5))
    calibration_timeout_s = float(gimbal_cfg.get("calibration_timeout_s", 5.0))
    calibration_wait_margin_s = float(gimbal_cfg.get("calibration_wait_margin_s", 0.25))

    if encoder_imu_reader is not None:
        try:
            ax, ay, az = encoder_imu_reader.read_accel()
            imu_pitch_value, _ = _accel_pitch_roll(ax, ay, az)
            _LOG.info("IMU horizontal (pitch) value at startup: %.4f rad", imu_pitch_value)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Failed to read IMU during startup calibration: %s", exc)

    # Step 2: Move motors to reach zero (horizontal position)
    if calibration_speed_rad_s > 0 and calibration_timeout_s > 0:
        if imu_pitch_value is not None:
            axis_delta_rad = -float(imu_pitch_value)
            # Convert angle delta to controller-relative pulse counts.
            # Use motor mechanical full steps, microstep subdivision, and gear ratio:
            # pulses = angle_rad / (2π) * motor_full_steps_per_rev * subdivision * gear_ratio
            motor_full_steps = int(gimbal_cfg.get("motor_full_steps_per_rev", 200))
            # Try to obtain subdivision (Byte8) from parameter_map if available; fallback to config or 16
            subdivision = int(gimbal_cfg.get("subdivision", 16))
            try:
                if pitch_a_addr in parameter_map:
                    subdivision = int(parameter_map[pitch_a_addr][4])
                elif pitch_b_addr in parameter_map:
                    subdivision = int(parameter_map[pitch_b_addr][4])
            except Exception:
                pass

            rel_axis_pulses = int(
                round(
                    abs(axis_delta_rad) / (2.0 * math.pi) * motor_full_steps * subdivision * pitch_ratio
                )
            )
            if rel_axis_pulses > 0:
                position_speed_rad_s = abs(calibration_speed_rad_s)
                move_direction = "downward" if axis_delta_rad < 0.0 else "upward"
                _LOG.info(
                    "IMU pitch %.4f rad; moving %s by %d pulses to reach zero",
                    imu_pitch_value,
                    move_direction,
                    rel_axis_pulses,
                )
                _LOG.info(
                    "Starting gimbal calibration: position move at %.4f rad/s, timeout %.1f s",
                    position_speed_rad_s,
                    calibration_timeout_s,
                )
                position_cmds = _pitch_position_commands(
                    rel_axis_pulses if axis_delta_rad >= 0.0 else -rel_axis_pulses,
                    speed_rad_s=position_speed_rad_s,
                    priority="high",
                )
                update_pub.send_update(
                    _build_update(
                        source="jetson.gimbal_bridge",
                        target=serial_target,
                        commands=position_cmds,
                    )
                )
                estimated_move_s = abs(axis_delta_rad) / max(position_speed_rad_s, 1e-6)
                time.sleep(min(calibration_timeout_s, estimated_move_s + calibration_wait_margin_s))
            else:
                _LOG.info("IMU pitch %.4f rad is already at zero; skipping position move", imu_pitch_value)
        else:
            _LOG.warning("IMU pitch unavailable at startup; skipping position move")
    # Step 3: Set encoder zero
    update_pub.send_update(
        _build_update(
            source="jetson.gimbal_bridge",
            target=serial_target,
            commands=[
                _build_command(
                    cmd_id="zero:yaw",
                    func="0x92",
                    addr=yaw_addr,
                    payload=[],
                    expect_reply=False,
                    expected_len=None,
                    priority="high",
                    target=serial_target,
                ),
                _build_command(
                    cmd_id="zero:pitch_a",
                    func="0x92",
                    addr=pitch_a_addr,
                    payload=[],
                    expect_reply=False,
                    expected_len=None,
                    priority="high",
                    target=serial_target,
                ),
                _build_command(
                    cmd_id="zero:pitch_b",
                    func="0x92",
                    addr=pitch_b_addr,
                    payload=[],
                    expect_reply=False,
                    expected_len=None,
                    priority="high",
                    target=serial_target,
                ),
            ],
        )
    )
    update_pub.send_update(
        _build_update(
            source="jetson.gimbal_bridge",
            target=serial_target,
            commands=[
                _build_command(
                    cmd_id="status:yaw",
                    func="F1",
                    addr=yaw_addr,
                    payload=[],
                    expect_reply=True,
                    expected_len=1,
                    priority="high",
                    target=serial_target,
                ),
                _build_command(
                    cmd_id="status:pitch_a",
                    func="F1",
                    addr=pitch_a_addr,
                    payload=[],
                    expect_reply=True,
                    expected_len=1,
                    priority="high",
                    target=serial_target,
                ),
                _build_command(
                    cmd_id="status:pitch_b",
                    func="F1",
                    addr=pitch_b_addr,
                    payload=[],
                    expect_reply=True,
                    expected_len=1,
                    priority="high",
                    target=serial_target,
                ),
            ],
        )
    )
    _wait_for_status(reply_sub, [yaw_addr, pitch_a_addr, pitch_b_addr])
    startup_elapsed = time.monotonic() - startup_start
    _LOG.info("gimbal startup sequence completed in %.3f s (IMU check, motor calibration, encoder zero)", startup_elapsed)

    yaw_counts: Optional[int] = None
    pitch_counts: dict[int, int] = {}
    last_encoder_ts: dict[int, float] = {}
    last_change_ts: dict[int, float] = {}
    last_stale_pair_log = 0.0
    motor_state = {
        yaw_addr: {"name": "yaw", "last_cmd_ts": 0.0, "cmd_rate": 0.0, "expect_motion": False, "baseline_counts": None, "deadline": 0.0, "last_warn_ts": 0.0},
        pitch_a_addr: {"name": "pitch_a", "last_cmd_ts": 0.0, "cmd_rate": 0.0, "expect_motion": False, "baseline_counts": None, "deadline": 0.0, "last_warn_ts": 0.0},
        pitch_b_addr: {"name": "pitch_b", "last_cmd_ts": 0.0, "cmd_rate": 0.0, "expect_motion": False, "baseline_counts": None, "deadline": 0.0, "last_warn_ts": 0.0},
    }
    last_rate_cmd_ts = time.monotonic()
    last_limited_yaw_rate_cmd = 0.0
    last_limited_pitch_rate_cmd = 0.0

    def _record_speed_command(addr: int, rate_rad_s: float, now_ts: float) -> None:
        state = motor_state[addr]
        state["last_cmd_ts"] = now_ts
        state["cmd_rate"] = float(rate_rad_s)

        if abs(rate_rad_s) < command_watchdog_min_speed:
            state["expect_motion"] = False
            state["baseline_counts"] = yaw_counts if addr == yaw_addr else pitch_counts.get(addr)
            state["deadline"] = 0.0
            return

        if state["expect_motion"]:
            return

        baseline = yaw_counts if addr == yaw_addr else pitch_counts.get(addr)
        state["expect_motion"] = True
        state["baseline_counts"] = baseline
        state["deadline"] = now_ts + command_watchdog_timeout_s
    try:
        while not stop_event.is_set():
            timeout_ms = int(math.ceil(feedback_period * 1000))
            events = dict(poller.poll(timeout=timeout_ms))
            if events.get(sub) == zmq.POLLIN:
                payload = sub.recv()
                try:
                    last_cmd = control_cmd_from_json(payload)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("failed to decode ControlCmd: %s", exc)
                else:
                    cmd_now = time.monotonic()
                    yaw_desired_rate_cmd = float(last_cmd.pan_rate_cmd)
                    pitch_desired_rate_cmd = float(last_cmd.tilt_rate_cmd)
                    if not math.isfinite(yaw_desired_rate_cmd) or not math.isfinite(pitch_desired_rate_cmd):
                        _LOG.warning(
                            "received non-finite ControlCmd rates (pan=%r tilt=%r); forcing zero command",
                            yaw_desired_rate_cmd,
                            pitch_desired_rate_cmd,
                        )
                        yaw_desired_rate_cmd = 0.0
                        pitch_desired_rate_cmd = 0.0
                    yaw_requested_accel = (
                        _finite_positive(last_cmd.pan_accel_cmd) or yaw_accel_limit_rad_s2
                    )
                    pitch_requested_accel = (
                        _finite_positive(last_cmd.tilt_accel_cmd) or pitch_accel_limit_rad_s2
                    )
                    # Hard angle limits: compute current axis angles from latest encoder counts
                    # and zero out any command that would drive an axis further past its bound.
                    _cur_yaw_rad = (
                        camstate_yaw_sign
                        * _counts_to_rad(yaw_counts, counts_per_rev=counts_per_rev, gear_ratio=yaw_ratio)
                        if yaw_counts is not None else None
                    )
                    encoder_pitch_rad = (
                        camstate_pitch_sign
                        * _counts_to_rad(
                            pitch_counts[pitch_authority_addr],
                            counts_per_rev=counts_per_rev,
                            gear_ratio=pitch_ratio,
                        )
                        if pitch_authority_addr in pitch_counts
                        else None
                    )
                    if camstate_source == "devices" and last_sample is not None:
                        _cur_pitch_rad = float(last_sample.tilt_rad)
                    else:
                        _cur_pitch_rad = encoder_pitch_rad
                    yaw_desired_rate_cmd = _apply_hard_angle_limit(
                        yaw_desired_rate_cmd, _cur_yaw_rad, yaw_min_rad, yaw_max_rad, "yaw"
                    )
                    pitch_desired_rate_cmd = _apply_hard_angle_limit(
                        pitch_desired_rate_cmd, _cur_pitch_rad, pitch_min_rad, pitch_max_rad, "pitch"
                    )
                    dt_s = max(cmd_now - last_rate_cmd_ts, 0.0)
                    yaw_rate_cmd = _limit_rate_by_accel(
                        yaw_desired_rate_cmd,
                        last_limited_yaw_rate_cmd,
                        yaw_requested_accel,
                        dt_s,
                    )
                    pitch_rate_cmd = _limit_rate_by_accel(
                        pitch_desired_rate_cmd,
                        last_limited_pitch_rate_cmd,
                        pitch_requested_accel,
                        dt_s,
                    )
                    yaw_rate_cmd = _apply_hard_angle_limit(
                        yaw_rate_cmd, _cur_yaw_rad, yaw_min_rad, yaw_max_rad, "yaw"
                    )
                    pitch_rate_cmd = _apply_hard_angle_limit(
                        pitch_rate_cmd, _cur_pitch_rad, pitch_min_rad, pitch_max_rad, "pitch"
                    )
                    yaw_cmd_accel_byte = _mks_accel_byte_from_physical(
                        yaw_requested_accel
                    )
                    pitch_cmd_accel_byte = _mks_accel_byte_from_physical(
                        pitch_requested_accel
                    )
                    yaw_motor_rate_cmd = yaw_sign * yaw_rate_cmd
                    yaw_payload = _encode_speed_cmd(
                        yaw_motor_rate_cmd,
                        acc=yaw_cmd_accel_byte,
                        gear_ratio=yaw_ratio,
                        max_rate=yaw_rate_limit,
                    )
                    update_sent = update_pub.send_update(
                        _build_update(
                            source="jetson.gimbal_bridge",
                            target=serial_target,
                            commands=[
                                _build_command(
                                    cmd_id=f"speed:yaw:{time.time_ns()}",
                                    func="F6",
                                    addr=yaw_addr,
                                    payload=yaw_payload,
                                    expect_reply=respond_on_writes,
                                    expected_len=None,
                                    priority="high",
                                    target=serial_target,
                                ),
                                *_pitch_speed_commands(
                                    pitch_rate_cmd,
                                    accel_byte=pitch_cmd_accel_byte,
                                    priority="high",
                                ),
                            ],
                            fields={
                                "pan_rate_desired_cmd": yaw_desired_rate_cmd,
                                "pan_rate_cmd": yaw_rate_cmd,
                                "yaw_motor_rate_cmd": yaw_motor_rate_cmd,
                                "tilt_rate_desired_cmd": pitch_desired_rate_cmd,
                                "tilt_rate_cmd": pitch_rate_cmd,
                                "pan_accel_cmd": yaw_requested_accel,
                                "tilt_accel_cmd": pitch_requested_accel,
                                "yaw_accel_byte": yaw_cmd_accel_byte,
                                "pitch_accel_byte": pitch_cmd_accel_byte,
                            },
                        )
                    )
                    if update_sent:
                        last_rate_cmd_ts = cmd_now
                        last_limited_yaw_rate_cmd = yaw_rate_cmd
                        last_limited_pitch_rate_cmd = pitch_rate_cmd
                        _record_speed_command(yaw_addr, yaw_motor_rate_cmd, cmd_now)
                        _record_speed_command(pitch_a_addr, pitch_a_sign * pitch_rate_cmd, cmd_now)
                        _record_speed_command(pitch_b_addr, pitch_b_sign * pitch_rate_cmd, cmd_now)
                    else:
                        _LOG.warning(
                            "serial update publish dropped; skipping watchdog command expectation update"
                        )

            for reply in reply_sub.recv_nowait():
                func = _reply_func_byte(reply)
                addr = reply.get("addr")
                if func == 0x31 and isinstance(addr, int):
                    parsed = reply.get("reply", {}).get("parsed", {})
                    if "counts" in parsed:
                        counts = int(parsed["counts"])
                        prev = yaw_counts if addr == yaw_addr else pitch_counts.get(addr)
                        last_encoder_ts[addr] = time.monotonic()
                        if prev is None or counts != prev:
                            last_change_ts[addr] = time.monotonic()
                        if addr == yaw_addr:
                            yaw_counts = counts
                        else:
                            pitch_counts[addr] = counts

            now = time.monotonic()
            for addr, state in motor_state.items():
                if not state["expect_motion"]:
                    continue
                counts_now = yaw_counts if addr == yaw_addr else pitch_counts.get(addr)
                if counts_now is None:
                    continue
                baseline = state["baseline_counts"]
                if baseline is None:
                    state["baseline_counts"] = counts_now
                    state["deadline"] = now + command_watchdog_timeout_s
                    continue
                if abs(int(counts_now) - int(baseline)) >= command_watchdog_min_delta:
                    state["expect_motion"] = False
                    continue
                if now < float(state["deadline"]):
                    continue
                if (now - float(state["last_warn_ts"])) >= command_watchdog_timeout_s:
                    state["last_warn_ts"] = now
                    _LOG.warning(
                        "command-health watchdog: motor=%s addr=%d cmd_rate=%.3f rad/s had no encoder delta >=%d counts in %.2fs",
                        state["name"],
                        addr,
                        float(state["cmd_rate"]),
                        command_watchdog_min_delta,
                        command_watchdog_timeout_s,
                    )
                state["deadline"] = now + command_watchdog_timeout_s

            if pitch_a_addr in last_encoder_ts and pitch_b_addr in last_encoder_ts:
                age_a = now - last_encoder_ts[pitch_a_addr]
                age_b = now - last_encoder_ts[pitch_b_addr]
                a_changing = (now - last_change_ts.get(pitch_a_addr, 0.0)) <= encoder_stale_warn_s
                b_changing = (now - last_change_ts.get(pitch_b_addr, 0.0)) <= encoder_stale_warn_s
                stale_mismatch = (age_a > encoder_stale_warn_s and b_changing) or (
                    age_b > encoder_stale_warn_s and a_changing
                )
                if stale_mismatch and (now - last_stale_pair_log) >= 1.0:
                    last_stale_pair_log = now
                    level = _LOG.error if max(age_a, age_b) > (2.0 * encoder_stale_warn_s) else _LOG.warning
                    level(
                        "pitch encoder stale mismatch: age_a=%.3fs age_b=%.3fs changing_a=%s changing_b=%s counts_a=%s counts_b=%s",
                        age_a,
                        age_b,
                        a_changing,
                        b_changing,
                        pitch_counts.get(pitch_a_addr),
                        pitch_counts.get(pitch_b_addr),
                    )

            if pub is None:
                continue
            if (now - last_pub_time) < feedback_period:
                continue
            last_pub_time = now
            secondary_pitch_rad = None
            if camstate_source == "encoder":
                if yaw_counts is None or pitch_authority_addr not in pitch_counts:
                    continue
                pan_rad = camstate_yaw_sign * _counts_to_rad(
                    yaw_counts, counts_per_rev=counts_per_rev, gear_ratio=yaw_ratio
                )
                tilt_rad = camstate_pitch_sign * _counts_to_rad(
                    pitch_counts[pitch_authority_addr],
                    counts_per_rev=counts_per_rev,
                    gear_ratio=pitch_ratio,
                )
                for addr, counts in pitch_counts.items():
                    if addr != pitch_authority_addr:
                        secondary_pitch_rad = camstate_pitch_sign * _counts_to_rad(
                            counts, counts_per_rev=counts_per_rev, gear_ratio=pitch_ratio
                        )
                        break
                encoder_tilt_rad = float(tilt_rad)
                imu_pitch_rad: Optional[float] = None
                if encoder_imu_reader is not None:
                    try:
                        ax, ay, az = encoder_imu_reader.read_accel()
                        imu_pitch_rad, _ = _accel_pitch_roll(ax, ay, az)
                    except OSError as exc:
                        if (now - encoder_imu_last_err_log) >= 1.0:
                            _LOG.warning("encoder CamState IMU read error: %s", exc)
                            encoder_imu_last_err_log = now
                tilt_rad, secondary_pitch_rad, encoder_horizon_offset_rad, offset_locked = (
                    _apply_encoder_horizon_offset(
                        encoder_tilt_rad=encoder_tilt_rad,
                        secondary_tilt_rad=secondary_pitch_rad,
                        imu_pitch_rad=imu_pitch_rad,
                        horizon_offset_rad=encoder_horizon_offset_rad,
                    )
                )
                if offset_locked and encoder_horizon_offset_rad is not None and imu_pitch_rad is not None:
                    _LOG.info(
                        "encoder CamState horizon offset locked: offset=%.4f rad imu_pitch=%.4f rad encoder_tilt=%.4f rad",
                        float(encoder_horizon_offset_rad),
                        float(imu_pitch_rad),
                        encoder_tilt_rad,
                    )
            else:
                if device_sensor_reader is None or device_sensor_cfg is None:
                    continue
                try:
                    ax, ay, az = device_sensor_reader.read_accel()
                    mag = device_sensor_reader.read_mag()
                except OSError as exc:
                    if (now - device_last_err_log) >= 1.0:
                        _LOG.warning("camstate device read error: %s", exc)
                        device_last_err_log = now
                    continue

                device_pitch, _ = _accel_pitch_roll(ax, ay, az)
                if mag is not None:
                    mx, my, mz = mag
                    _pitch, device_heading = _compute_orientation(ax, ay, az, mx, my, mz)
                    device_pitch = _pitch
                if device_heading is None or device_pitch is None:
                    continue
                pan_rad = device_heading
                tilt_rad = device_pitch

            pan_rate = tilt_rate = None
            if last_sample is not None:
                dt = now - last_sample.timestamp
                if dt > 0:
                    pan_rate = _wrapped_delta(pan_rad, last_sample.pan_rad) / dt
                    tilt_rate = _wrapped_delta(tilt_rad, last_sample.tilt_rad) / dt

            sample = _AngleSample(
                timestamp=now,
                pan_rad=pan_rad,
                tilt_rad=tilt_rad,
                pan_rate_rad_s=pan_rate,
                tilt_rate_rad_s=tilt_rate,
                secondary_pitch_rad=secondary_pitch_rad,
            )
            if camstate_home_pan is None:
                camstate_home_pan = float(sample.pan_rad)
            if camstate_home_tilt is None:
                camstate_home_tilt = float(sample.tilt_rad)
            last_sample = sample
            if camstate_source == "encoder" and sample.secondary_pitch_rad is not None:
                divergence = abs(sample.secondary_pitch_rad - sample.tilt_rad)
                if divergence >= pitch_div_thresh and (now - last_divergence_log) >= 2.0:
                    last_divergence_log = now
                    _LOG.warning(
                        "pitch encoder divergence %.4f rad exceeds threshold %.4f (primary=%.4f secondary=%.4f)",
                        divergence,
                        pitch_div_thresh,
                        float(sample.tilt_rad),
                        float(sample.secondary_pitch_rad),
                    )
            if last_cmd is not None:
                frame_id = int(last_cmd.frame_id)
                src_ts_ms = int(last_cmd.src_ts_ms)
            else:
                frame_id = local_frame_id
                local_frame_id += 1
                src_ts_ms = int(time.monotonic_ns() / 1e6)
            try:
                _publish_cam_state(
                    pub,
                    sample,
                    frame_id=frame_id,
                    src_ts_ms=src_ts_ms,
                    home_pan=camstate_home_pan,
                    home_tilt=camstate_home_tilt,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("failed to publish CamState: %s", exc)
            if (now - last_stats_log) >= 5.0 and last_sample is not None:
                last_stats_log = now
                pan_rate = (
                    last_sample.pan_rate_rad_s
                    if last_sample.pan_rate_rad_s is not None
                    else float("nan")
                )
                tilt_rate = (
                    last_sample.tilt_rate_rad_s
                    if last_sample.tilt_rate_rad_s is not None
                    else float("nan")
                )
                if camstate_source == "encoder":
                    _LOG.info(
                        "gimbal heartbeat source=encoder pan=%.3f tilt=%.3f pan_rate=%.3f tilt_rate=%.3f frame_id=%s pitch_a_counts=%s pitch_b_counts=%s pitch_a_stale_s=%.3f pitch_b_stale_s=%.3f",
                        float(last_sample.pan_rad),
                        float(last_sample.tilt_rad),
                        float(pan_rate),
                        float(tilt_rate),
                        getattr(last_cmd, "frame_id", "n/a"),
                        pitch_counts.get(pitch_a_addr),
                        pitch_counts.get(pitch_b_addr),
                        now - last_encoder_ts[pitch_a_addr] if pitch_a_addr in last_encoder_ts else float("nan"),
                        now - last_encoder_ts[pitch_b_addr] if pitch_b_addr in last_encoder_ts else float("nan"),
                    )
                else:
                    _LOG.info(
                        "gimbal heartbeat source=devices pan=%.3f tilt=%.3f pan_rate=%.3f tilt_rate=%.3f frame_id=%s",
                        float(last_sample.pan_rad),
                        float(last_sample.tilt_rad),
                        float(pan_rate),
                        float(tilt_rate),
                        getattr(last_cmd, "frame_id", "n/a"),
                    )
    finally:
        stop_cmds = [
            _build_command(
                cmd_id="stop:yaw",
                func="F6",
                addr=yaw_addr,
                payload=_encode_speed_cmd(
                    0.0, acc=yaw_accel, gear_ratio=yaw_ratio, max_rate=yaw_rate_limit
                ),
                expect_reply=respond_on_writes,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
            _build_command(
                cmd_id="stop:pitch_a",
                func="F6",
                addr=pitch_a_addr,
                payload=_encode_speed_cmd(
                    0.0,
                    acc=pitch_accel,
                    gear_ratio=pitch_ratio,
                    max_rate=pitch_rate_limit,
                ),
                expect_reply=respond_on_writes,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
            _build_command(
                cmd_id="stop:pitch_b",
                func="F6",
                addr=pitch_b_addr,
                payload=_encode_speed_cmd(
                    0.0,
                    acc=pitch_accel,
                    gear_ratio=pitch_ratio,
                    max_rate=pitch_rate_limit,
                ),
                expect_reply=respond_on_writes,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
            _build_command(
                cmd_id="disable:yaw",
                func="F3",
                addr=yaw_addr,
                payload=[0x00],
                expect_reply=False,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
            _build_command(
                cmd_id="disable:pitch_a",
                func="F3",
                addr=pitch_a_addr,
                payload=[0x00],
                expect_reply=False,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
            _build_command(
                cmd_id="disable:pitch_b",
                func="F3",
                addr=pitch_b_addr,
                payload=[0x00],
                expect_reply=False,
                expected_len=None,
                priority="critical",
                target=serial_target,
            ),
        ]
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=stop_cmds,
            )
        )
        try:
            poller.unregister(sub)
        except Exception:  # noqa: BLE001
            pass
        try:
            sub.close(linger=0)
        except Exception:  # noqa: BLE001
            pass
        if pub is not None:
            try:
                poller.unregister(pub)
            except Exception:  # noqa: BLE001
                pass
            try:
                pub.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
        if device_sensor_reader is not None:
            device_sensor_reader.close()
        if encoder_imu_reader is not None:
            encoder_imu_reader.close()
        update_pub.close()
        reply_sub.close()
        try:
            ctx.destroy(linger=0)
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
