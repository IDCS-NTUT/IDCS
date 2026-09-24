"""Bridge live ControlIntent messages to timed MKS RS485 rate commands.

This Jetson-side process subscribes to the V2 intent PUB socket, validates
authority/freshness/order, translates bounded rates into timed speed writes, and periodically
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
from common.schemas import CamState, ControlIntent, control_intent_from_json
from common.serial_io import SerialReplySubscriber, SerialUpdatePublisher
from common.shutdown import install_signal_handlers
from common.gimbal.mks_servo42_rs485 import MksServo42Axis

_LOG = logging.getLogger(__name__)

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

    yaw_accel_byte = int(gimbal_cfg.get("yaw_accel_byte", 10))
    pitch_accel_byte = int(gimbal_cfg.get("pitch_accel_byte", 10))
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
        "yaw_accel_byte": yaw_accel_byte,
        "pitch_accel_byte": pitch_accel_byte,
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
    render_pose: Optional["RenderPose"] = None,
    encoder_pan_counts: Optional[int] = None,
    encoder_tilt_counts: Optional[int] = None,
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
        render_pan=render_pose.pan_rad if render_pose is not None else None,
        render_tilt=render_pose.tilt_rad if render_pose is not None else None,
        render_pan_rate=render_pose.pan_rate_rad_s if render_pose is not None else None,
        render_tilt_rate=render_pose.tilt_rate_rad_s if render_pose is not None else None,
        render_prediction_age_ms=(
            render_pose.encoder_age_s * 1000.0 if render_pose is not None else None
        ),
        render_pan_correction_rad=(
            render_pose.pan_correction_rad if render_pose is not None else None
        ),
        render_tilt_correction_rad=(
            render_pose.tilt_correction_rad if render_pose is not None else None
        ),
        encoder_pan_counts=encoder_pan_counts,
        encoder_tilt_counts=encoder_tilt_counts,
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


@dataclass(frozen=True)
class RenderPose:
    pan_rad: float
    tilt_rad: float
    pan_rate_rad_s: float
    tilt_rate_rad_s: float
    encoder_age_s: float
    pan_correction_rad: Optional[float]
    tilt_correction_rad: Optional[float]


@dataclass(frozen=True)
class _RenderRateCommand:
    start_s: float
    rate_rad_s: float
    expires_s: Optional[float]


class _EncoderAnchoredAxisPredictor:
    def __init__(self) -> None:
        self.anchor_rad: Optional[float] = None
        self.anchor_s: Optional[float] = None
        self.commands: list[_RenderRateCommand] = []
        self.correction_rad: Optional[float] = None
        self.correction_count = 0
        self.abs_correction_sum = 0.0
        self.abs_correction_max = 0.0

    def set_rate(
        self,
        rate_rad_s: float,
        *,
        at_s: float,
        runtime_s: Optional[float] = None,
    ) -> None:
        rate = float(rate_rad_s)
        start = float(at_s)
        if not math.isfinite(rate) or not math.isfinite(start):
            raise ValueError("render predictor command must be finite")
        if runtime_s is not None:
            runtime = float(runtime_s)
            if not math.isfinite(runtime) or runtime <= 0.0:
                raise ValueError("render predictor runtime must be positive and finite")
            expires = start + runtime
        else:
            expires = None
        self.commands.append(_RenderRateCommand(start, rate, expires))
        self.commands.sort(key=lambda command: command.start_s)
        if len(self.commands) > 512:
            self.commands = self.commands[-512:]

    def rate_at(self, at_s: float) -> float:
        timestamp = float(at_s)
        for command in reversed(self.commands):
            if command.start_s > timestamp:
                continue
            if command.expires_s is None or timestamp < command.expires_s:
                return command.rate_rad_s
            return 0.0
        return 0.0

    def _displacement(self, start_s: float, end_s: float) -> float:
        if end_s <= start_s:
            return 0.0
        boundaries = {float(start_s), float(end_s)}
        for command in self.commands:
            if start_s < command.start_s < end_s:
                boundaries.add(command.start_s)
            if (
                command.expires_s is not None
                and start_s < command.expires_s < end_s
            ):
                boundaries.add(command.expires_s)
        ordered = sorted(boundaries)
        displacement = 0.0
        for left, right in zip(ordered, ordered[1:]):
            midpoint = left + (right - left) * 0.5
            displacement += self.rate_at(midpoint) * (right - left)
        return displacement

    def position_at(self, at_s: float) -> Optional[float]:
        if self.anchor_rad is None or self.anchor_s is None:
            return None
        timestamp = float(at_s)
        if timestamp < self.anchor_s:
            return None
        return self.anchor_rad + self._displacement(self.anchor_s, timestamp)

    def anchor(self, angle_rad: float, *, sample_s: float) -> None:
        angle = float(angle_rad)
        timestamp = float(sample_s)
        if not math.isfinite(angle) or not math.isfinite(timestamp):
            raise ValueError("render predictor encoder anchors must be finite")
        predicted = self.position_at(timestamp)
        if predicted is not None:
            self.correction_rad = angle - predicted
            magnitude = abs(self.correction_rad)
            self.correction_count += 1
            self.abs_correction_sum += magnitude
            self.abs_correction_max = max(self.abs_correction_max, magnitude)
        else:
            self.correction_rad = None
        self.anchor_rad = angle
        self.anchor_s = timestamp
        before = [command for command in self.commands if command.start_s <= timestamp]
        after = [command for command in self.commands if command.start_s > timestamp]
        self.commands = ([before[-1]] if before else []) + after


class EncoderAnchoredRenderPredictor:
    """Integrate timestamped commands for rendering, corrected by encoders."""

    def __init__(self) -> None:
        self._pan = _EncoderAnchoredAxisPredictor()
        self._tilt = _EncoderAnchoredAxisPredictor()

    def set_command_rates(
        self,
        pan_rate_rad_s: float,
        tilt_rate_rad_s: float,
        *,
        at_s: float,
        runtime_s: Optional[float] = None,
    ) -> None:
        self._pan.set_rate(pan_rate_rad_s, at_s=at_s, runtime_s=runtime_s)
        self._tilt.set_rate(tilt_rate_rad_s, at_s=at_s, runtime_s=runtime_s)

    def set_axis_command(
        self,
        axis: str,
        rate_rad_s: float,
        *,
        at_s: float,
        runtime_s: Optional[float] = None,
    ) -> None:
        if axis == "pan":
            self._pan.set_rate(rate_rad_s, at_s=at_s, runtime_s=runtime_s)
        elif axis == "tilt":
            self._tilt.set_rate(rate_rad_s, at_s=at_s, runtime_s=runtime_s)
        else:
            raise ValueError(f"unsupported predictor axis: {axis}")

    def anchor_pan(self, pan_rad: float, *, sample_s: float) -> None:
        self._pan.anchor(pan_rad, sample_s=sample_s)

    def anchor_tilt(self, tilt_rad: float, *, sample_s: float) -> None:
        self._tilt.anchor(tilt_rad, sample_s=sample_s)

    def pose(self, *, at_s: float) -> Optional[RenderPose]:
        timestamp = float(at_s)
        pan_rad = self._pan.position_at(timestamp)
        tilt_rad = self._tilt.position_at(timestamp)
        if pan_rad is None or tilt_rad is None:
            return None
        encoder_age_s = max(
            0.0,
            timestamp - min(float(self._pan.anchor_s), float(self._tilt.anchor_s)),
        )
        return RenderPose(
            pan_rad=pan_rad,
            tilt_rad=tilt_rad,
            pan_rate_rad_s=self._pan.rate_at(timestamp),
            tilt_rate_rad_s=self._tilt.rate_at(timestamp),
            encoder_age_s=encoder_age_s,
            pan_correction_rad=self._pan.correction_rad,
            tilt_correction_rad=self._tilt.correction_rad,
        )

    def correction_stats(self) -> Mapping[str, float | int]:
        return {
            "pan_count": self._pan.correction_count,
            "tilt_count": self._tilt.correction_count,
            "pan_mean_abs_rad": (
                self._pan.abs_correction_sum / self._pan.correction_count
                if self._pan.correction_count
                else 0.0
            ),
            "tilt_mean_abs_rad": (
                self._tilt.abs_correction_sum / self._tilt.correction_count
                if self._tilt.correction_count
                else 0.0
            ),
            "pan_max_abs_rad": self._pan.abs_correction_max,
            "tilt_max_abs_rad": self._tilt.abs_correction_max,
        }


@dataclass(frozen=True)
class PendingWireCommand:
    cmd_id: str
    update_id: str
    addr: int
    motor_rate_rad_s: float
    render_axis: Optional[str]
    render_rate_rad_s: Optional[float]
    runtime_s: Optional[float]


@dataclass(frozen=True)
class AppliedWireCommand:
    cmd_id: str
    addr: int
    motor_rate_rad_s: float
    wire_s: float


class WireExecutionRenderTracker:
    """Apply render motion only after authoritative serial wire outcomes."""

    _DROP_EVENTS = {"superseded", "preempted", "stale", "write_failed", "cancelled"}

    def __init__(
        self,
        predictor: EncoderAnchoredRenderPredictor,
        *,
        target: str,
        stale_s: float,
        require_fresh_encoder_anchors: bool = True,
    ) -> None:
        self.predictor = predictor
        self.target = str(target)
        self.stale_s = max(float(stale_s), 0.02)
        self.require_fresh_encoder_anchors = bool(require_fresh_encoder_anchors)
        self.pending: dict[str, PendingWireCommand] = {}
        self._applied: dict[str, None] = {}
        self.service_epoch: Optional[str] = None
        self.last_event_sequence: Optional[int] = None
        self.last_feedback_s: Optional[float] = None
        self.degraded = False
        self.sequence_gap_count = 0
        self._recovery_snapshot_s: Optional[float] = None
        self._recovery_axes: set[str] = set()

    def register(self, commands: Iterable[PendingWireCommand]) -> None:
        for command in commands:
            self.pending[command.cmd_id] = command
        while len(self.pending) > 512:
            self.pending.pop(next(iter(self.pending)))

    def cancel_update(self, update_id: str) -> None:
        self.pending = {
            cmd_id: command
            for cmd_id, command in self.pending.items()
            if command.update_id != update_id
        }

    def _observe_service(
        self,
        *,
        epoch: str,
        sequence: int,
        received_s: float,
        snapshot_s: Optional[float] = None,
    ) -> bool:
        if self.service_epoch is None:
            self.service_epoch = epoch
            self.last_event_sequence = sequence
            self.last_feedback_s = received_s
            return True
        if epoch != self.service_epoch:
            self.service_epoch = epoch
            self.last_event_sequence = sequence
            self.last_feedback_s = received_s
            self.pending.clear()
            self._applied.clear()
            self.degraded = True
            self._recovery_snapshot_s = snapshot_s
            self._recovery_axes = {"pan", "tilt"}
            return True
        previous = self.last_event_sequence
        if previous is not None and sequence < previous:
            return False
        if previous is not None and sequence > previous + 1:
            self.sequence_gap_count += 1
            self.degraded = True
            self._recovery_snapshot_s = snapshot_s
            self._recovery_axes = {"pan", "tilt"}
        self.last_event_sequence = max(previous or sequence, sequence)
        self.last_feedback_s = received_s
        return True

    def _apply_pending(
        self, cmd_id: str, *, wire_ns: int
    ) -> Optional[AppliedWireCommand]:
        if cmd_id in self._applied:
            return None
        command = self.pending.pop(cmd_id, None)
        if command is None:
            return None
        wire_s = int(wire_ns) / 1e9
        if command.render_axis is not None and command.render_rate_rad_s is not None:
            self.predictor.set_axis_command(
                command.render_axis,
                command.render_rate_rad_s,
                at_s=wire_s,
                runtime_s=command.runtime_s,
            )
        self._applied[cmd_id] = None
        while len(self._applied) > 1024:
            self._applied.pop(next(iter(self._applied)))
        return AppliedWireCommand(
            cmd_id=cmd_id,
            addr=command.addr,
            motor_rate_rad_s=command.motor_rate_rad_s,
            wire_s=wire_s,
        )

    def handle_event(
        self, message: Mapping[str, Any], *, received_s: Optional[float] = None
    ) -> Optional[AppliedWireCommand]:
        if message.get("type") != "SerialCommandEventV1":
            return None
        if str(message.get("target")) != self.target:
            return None
        try:
            epoch = str(message["service_epoch"])
            sequence = int(message["sequence"])
        except (KeyError, TypeError, ValueError):
            self.degraded = True
            return None
        now_s = time.monotonic() if received_s is None else float(received_s)
        if not self._observe_service(
            epoch=epoch, sequence=sequence, received_s=now_s
        ):
            return None
        cmd_id = str(message.get("cmd_id", ""))
        event = str(message.get("event", ""))
        if event in self._DROP_EVENTS:
            self.pending.pop(cmd_id, None)
            return None
        if event == "wire_uncertain":
            self.pending.pop(cmd_id, None)
            self.degraded = True
            self._recovery_snapshot_s = now_s
            self._recovery_axes = {"pan", "tilt"}
            return None
        if event != "wire_sent":
            return None
        timing = message.get("timing")
        if not isinstance(timing, Mapping):
            self.degraded = True
            return None
        wire_ns = timing.get("wire_monotonic_ns")
        try:
            return self._apply_pending(cmd_id, wire_ns=int(wire_ns))
        except (TypeError, ValueError):
            self.degraded = True
            return None

    def handle_snapshot(
        self, message: Mapping[str, Any], *, received_s: Optional[float] = None
    ) -> list[AppliedWireCommand]:
        if message.get("type") != "SerialActuationStateV1":
            return []
        if str(message.get("target")) != self.target:
            return []
        try:
            epoch = str(message["service_epoch"])
            sequence = int(message["event_sequence"])
            snapshot_s = int(message["event_monotonic_ns"]) / 1e9
        except (KeyError, TypeError, ValueError):
            self.degraded = True
            return []
        now_s = time.monotonic() if received_s is None else float(received_s)
        if not self._observe_service(
            epoch=epoch,
            sequence=sequence,
            received_s=now_s,
            snapshot_s=snapshot_s,
        ):
            return []
        applied: list[AppliedWireCommand] = []
        axes = message.get("axes")
        uncertain_active = False
        if isinstance(axes, Mapping):
            for state in axes.values():
                if not isinstance(state, Mapping):
                    continue
                if (
                    state.get("wire_outcome") == "wire_uncertain"
                    and bool(state.get("active"))
                ):
                    uncertain_active = True
                cmd_id = str(state.get("cmd_id", ""))
                try:
                    wire_ns = int(state["wire_monotonic_ns"])
                except (KeyError, TypeError, ValueError):
                    continue
                action = self._apply_pending(cmd_id, wire_ns=wire_ns)
                if action is not None:
                    applied.append(action)
        if self.degraded and uncertain_active:
            self._recovery_snapshot_s = None
            self._recovery_axes = {"pan", "tilt"}
        elif self.degraded:
            self._recovery_snapshot_s = snapshot_s
            self._recovery_axes = (
                {"pan", "tilt"} if self.require_fresh_encoder_anchors else set()
            )
            if not self._recovery_axes:
                self.degraded = False
                self._recovery_snapshot_s = None
        return applied

    def note_encoder_anchor(self, axis: str, *, sample_s: float) -> None:
        if axis not in {"pan", "tilt"}:
            raise ValueError(f"unsupported predictor axis: {axis}")
        if (
            self._recovery_snapshot_s is not None
            and float(sample_s) >= self._recovery_snapshot_s
        ):
            self._recovery_axes.discard(axis)
            if not self._recovery_axes:
                self.degraded = False
                self._recovery_snapshot_s = None

    def healthy(self, *, now_s: float) -> bool:
        return (
            self.service_epoch is not None
            and self.last_feedback_s is not None
            and not self.degraded
            and float(now_s) - self.last_feedback_s <= self.stale_s
        )

    def pose(self, *, at_s: float) -> Optional[RenderPose]:
        if not self.healthy(now_s=at_s):
            return None
        return self.predictor.pose(at_s=at_s)


def _encode_speed_cmd(
    omega_rad_s: float,
    *,
    acc: int,
    gear_ratio: float,
    max_rate: float,
) -> Tuple[int, int, int]:
    omega = max(min(omega_rad_s, max_rate), -max_rate)
    return MksServo42Axis._encode_speed_payload(omega, acc, gear_ratio)


def _encode_timed_speed_cmd(
    omega_rad_s: float,
    *,
    acc: int,
    gear_ratio: float,
    max_rate: float,
    runtime_ms: int,
) -> Tuple[int, int, int, int, int, int, int]:
    """Encode firmware-timed F6 motion; runtime uses big-endian 10 ms units."""

    if runtime_ms <= 0:
        raise ValueError("runtime_ms must be positive")
    units = max(1, min(0xFFFFFFFF, int(math.ceil(runtime_ms / 10.0))))
    speed = _encode_speed_cmd(
        omega_rad_s, acc=acc, gear_ratio=gear_ratio, max_rate=max_rate
    )
    return (
        *speed,
        (units >> 24) & 0xFF,
        (units >> 16) & 0xFF,
        (units >> 8) & 0xFF,
        units & 0xFF,
    )


def _quantized_camera_rate(
    rate_rad_s: float,
    *,
    motor_sign: float,
    gear_ratio: float,
    max_rate: float,
) -> float:
    """Return the camera-axis rate represented by the actual F6 payload."""

    motor_rate = motor_sign * float(rate_rad_s)
    bounded_motor_rate = max(-float(max_rate), min(float(max_rate), motor_rate))
    quantized_motor_rate = MksServo42Axis.quantized_speed_rad_s(
        bounded_motor_rate, gear_ratio
    )
    return motor_sign * quantized_motor_rate


@dataclass(frozen=True)
class IntentGateResult:
    accepted: bool
    reason: str
    stop_required: bool


def _rates_have_motion(yaw_rate_rad_s: float, pitch_rate_rad_s: float) -> bool:
    return any(abs(value) > 1e-12 for value in (yaw_rate_rad_s, pitch_rate_rad_s))


def _intent_command_priority(yaw_rate_rad_s: float, pitch_rate_rad_s: float) -> str:
    """Keep tracking coalescible; reserve critical priority for full stops."""
    return "high" if _rates_have_motion(yaw_rate_rad_s, pitch_rate_rad_s) else "critical"


def _should_forward_intent(intent: ControlIntent, *, was_stopped: bool) -> bool:
    """A confirmed stopped state does not need repeated zero-rate writes."""
    return (
        _rates_have_motion(intent.yaw_rate_rad_s, intent.pitch_rate_rad_s)
        or not was_stopped
    )


class LiveIntentGate:
    """Fail-closed ordering, authority, and local-monotonic freshness gate."""

    def __init__(self, *, watchdog_ns: int, future_tolerance_ns: int = 25_000_000) -> None:
        if watchdog_ns <= 0:
            raise ValueError("watchdog_ns must be positive")
        self._watchdog_ns = watchdog_ns
        self._future_tolerance_ns = future_tolerance_ns
        self._last_sequence = -1
        self._last_observation_sequence = -1
        self._last_received_ns: Optional[int] = None
        self._last_valid_until_ns: Optional[int] = None
        self._stopped = True

    def accept(self, intent: ControlIntent, *, now_ns: int) -> IntentGateResult:
        if intent.mode != "live":
            return IntentGateResult(False, "non_live_intent", not self._stopped)
        if intent.issued_monotonic_ns > now_ns + self._future_tolerance_ns:
            return IntentGateResult(False, "issued_in_future", not self._stopped)
        if intent.valid_until_monotonic_ns < now_ns:
            return IntentGateResult(False, "intent_expired", not self._stopped)
        if intent.sequence <= self._last_sequence:
            return IntentGateResult(False, "intent_out_of_order", not self._stopped)
        if intent.observation_sequence < self._last_observation_sequence:
            return IntentGateResult(False, "observation_out_of_order", not self._stopped)
        rates = (intent.yaw_rate_rad_s, intent.pitch_rate_rad_s)
        if not all(math.isfinite(value) for value in rates):
            return IntentGateResult(False, "non_finite_rate", not self._stopped)
        moving = _rates_have_motion(*rates)
        if moving and intent.reason not in {"tracking", "position_limit_hold"}:
            return IntentGateResult(False, "motion_reason_not_authorized", not self._stopped)
        self._last_sequence = intent.sequence
        self._last_observation_sequence = intent.observation_sequence
        self._last_received_ns = now_ns
        self._last_valid_until_ns = intent.valid_until_monotonic_ns
        self._stopped = False
        return IntentGateResult(True, "accepted", False)

    def watchdog_stop_required(self, *, now_ns: int) -> bool:
        if self._stopped or self._last_received_ns is None or self._last_valid_until_ns is None:
            return False
        deadline = min(
            self._last_valid_until_ns,
            self._last_received_ns + self._watchdog_ns,
        )
        if now_ns <= deadline:
            return False
        return True

    def mark_stopped(self) -> None:
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def mark_command_sent(self, intent: ControlIntent) -> None:
        """Record the physical state only after serial publication succeeds."""
        self._stopped = not _rates_have_motion(
            intent.yaw_rate_rad_s, intent.pitch_rate_rad_s
        )


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
) -> set[int]:
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
    return expected


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
    update_id: Optional[str] = None,
) -> Mapping[str, Any]:
    update: dict[str, Any] = {
        "type": "SerialUpdate",
        "source": source,
        "target": target,
        "fields": dict(fields or {}),
        "commands": list(commands),
        "update_ts_ms": int(time.time() * 1000),
    }
    if update_id is not None:
        update["update_id"] = str(update_id)
    return update


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
    pan_timestamp: Optional[float] = None
    tilt_timestamp: Optional[float] = None
    secondary_pitch_rad: Optional[float] = None


def _reply_timing(reply: Mapping[str, Any], *, fallback_mono: float) -> dict[str, float]:
    timing = reply.get("timing", {})
    if not isinstance(timing, Mapping):
        timing = {}

    def ns_to_s(name: str, default_s: float) -> float:
        value = timing.get(name)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default_s
        return parsed / 1e9

    def ms_value(name: str, default_ms: float = 0.0) -> float:
        value = timing.get(name)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default_ms
        return parsed if math.isfinite(parsed) else default_ms

    execute_s = ns_to_s("execute_start_monotonic_ns", fallback_mono)
    reply_s = ns_to_s("reply_monotonic_ns", fallback_mono)
    return {
        "execute_s": execute_s,
        "reply_s": reply_s,
        "queue_age_ms": ms_value("queue_age_ms"),
        "bus_duration_ms": ms_value("bus_duration_ms", ms_value("duration_ms")),
    }


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
    ap.add_argument(
        "--enable-live-intent-actuation",
        action="store_true",
        help="required before live ControlIntent messages can write motor rates",
    )
    ap.add_argument(
        "--enable-startup-calibration",
        action="store_true",
        help="separate acknowledgement for configured startup calibration motion",
    )
    ap.add_argument(
        "--enable-startup-encoder-zero",
        action="store_true",
        help="separate acknowledgement for configured startup encoder zeroing",
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
    render_prediction_cfg = gimbal_cfg.get("render_prediction") or {}
    if not isinstance(render_prediction_cfg, Mapping):
        raise SystemExit("gimbal.render_prediction must be a mapping")
    render_prediction_source = str(
        render_prediction_cfg.get("source", "publication")
    ).strip().lower()
    if render_prediction_source not in {"disabled", "publication", "wire_execution"}:
        raise SystemExit(
            "gimbal.render_prediction.source must be disabled, publication, or wire_execution"
        )
    shadow_wire_execution = bool(
        render_prediction_cfg.get("shadow_wire_execution", False)
    )
    execution_stale_s = max(
        float(render_prediction_cfg.get("execution_stale_ms", 100.0)) / 1000.0,
        0.02,
    )
    require_fresh_execution_anchors = bool(
        render_prediction_cfg.get("require_fresh_encoder_anchors", True)
    )
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
    encoder_imu_horizon_enabled = gimbal_cfg.get("encoder_imu_horizon_enabled", False)
    if not isinstance(encoder_imu_horizon_enabled, bool):
        raise SystemExit("gimbal.encoder_imu_horizon_enabled must be true or false")

    device_sensor_cfg: Optional[_DeviceSensorConfig] = None
    device_sensor_reader: Optional[_DeviceSensorReader] = None
    encoder_imu_sensor_cfg: Optional[_DeviceSensorConfig] = None
    encoder_imu_reader: Optional[_DeviceSensorReader] = None
    if camstate_source == "devices":
        device_sensor_cfg = _build_device_sensor_cfg(camstate_devices_cfg)
        device_sensor_reader = _DeviceSensorReader(device_sensor_cfg)
    else:
        if not encoder_imu_horizon_enabled:
            _LOG.info("encoder CamState IMU horizon alignment disabled by config")
        elif SMBus is None:
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
        topics=[
            f"serial.reply.{serial_target}",
            f"serial.command.{serial_target}",
            f"serial.actuation.{serial_target}",
        ],
        ctx=ctx,
    )
    _LOG.info(
        "subscribing to live ControlIntent on %s (feedback %.1f Hz, actuation=%s)",
        ctrl_ep,
        feedback_hz,
        args.enable_live_intent_actuation,
    )
    _LOG.info("publishing SerialUpdate to %s (target=%s)", serial_update_ep, serial_target)
    _LOG.info(
        "render prediction source=%s wire_shadow=%s execution_stale_ms=%.1f",
        render_prediction_source,
        shadow_wire_execution,
        execution_stale_s * 1000.0,
    )

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    last_pub_time = 0.0
    last_intent: Optional[ControlIntent] = None
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
    yaw_accel = int(serial_targets["yaw_accel_byte"])
    pitch_accel = int(serial_targets["pitch_accel_byte"])
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
    encoder_rate_min_dt_s = max(float(gimbal_cfg.get("encoder_rate_min_dt_s", 0.5 * feedback_period)), 0.001)
    encoder_max_queue_age_ms = max(float(gimbal_cfg.get("encoder_max_queue_age_ms", 80.0)), 0.0)
    encoder_max_bus_duration_ms = max(float(gimbal_cfg.get("encoder_max_bus_duration_ms", 80.0)), 0.0)
    command_watchdog_timeout_s = max(float(gimbal_cfg.get("command_watchdog_timeout_s", 0.75)), 0.1)
    command_watchdog_min_speed = abs(float(gimbal_cfg.get("command_watchdog_min_speed_rad_s", 0.1)))
    command_watchdog_min_delta = max(int(gimbal_cfg.get("command_watchdog_min_delta_counts", 1)), 1)
    intent_watchdog_ns = int(max(float(gimbal_cfg.get("intent_watchdog_ms", 100.0)), 10.0) * 1_000_000.0)
    intent_runtime_ms = int(gimbal_cfg.get("intent_command_runtime_ms", 100))
    if not 10 <= intent_runtime_ms <= 1000:
        raise SystemExit("gimbal.intent_command_runtime_ms must be in 10..1000")
    startup_calibration_enabled = bool(gimbal_cfg.get("startup_calibration_enabled", False))
    startup_encoder_zero_enabled = bool(gimbal_cfg.get("startup_encoder_zero_enabled", False))
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
        priority: str,
        runtime_ms: Optional[int] = None,
        command_token: Optional[int] = None,
        id_prefix: str = "speed",
    ) -> list[Mapping[str, Any]]:
        def payload(sign: float) -> tuple[int, ...]:
            if runtime_ms is None:
                return _encode_speed_cmd(
                    sign * rate_rad_s,
                    acc=pitch_accel,
                    gear_ratio=pitch_ratio,
                    max_rate=pitch_rate_limit,
                )
            return _encode_timed_speed_cmd(
                sign * rate_rad_s,
                acc=pitch_accel,
                gear_ratio=pitch_ratio,
                max_rate=pitch_rate_limit,
                runtime_ms=runtime_ms,
            )

        token = time.time_ns() if command_token is None else int(command_token)
        return [
            _build_command(
                cmd_id=f"{id_prefix}:pitch_a:{token}",
                func="F6",
                addr=pitch_a_addr,
                payload=payload(pitch_a_sign),
                expect_reply=respond_on_writes,
                expected_len=None,
                priority=priority,
                target=serial_target,
            ),
            _build_command(
                cmd_id=f"{id_prefix}:pitch_b:{token}",
                func="F6",
                addr=pitch_b_addr,
                payload=payload(pitch_b_sign),
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
    if parameter_map and args.enable_live_intent_actuation:
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
    if args.enable_live_intent_actuation:
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=enable_cmds,
            )
        )
    else:
        _LOG.info("read-only bridge startup: motor parameter writes and enable commands suppressed")

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
    calibration_authorized = (
        args.enable_live_intent_actuation
        and args.enable_startup_calibration
        and startup_calibration_enabled
    )
    if calibration_authorized and calibration_speed_rad_s > 0 and calibration_timeout_s > 0:
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
    # Step 3: Encoder zeroing is destructive state mutation and requires both
    # a tracked configuration opt-in and a separate command-line acknowledgement.
    zero_authorized = (
        args.enable_live_intent_actuation
        and args.enable_startup_encoder_zero
        and startup_encoder_zero_enabled
    )
    if zero_authorized:
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=[
                    _build_command(
                        cmd_id="zero:yaw", func="0x92", addr=yaw_addr,
                        payload=[], expect_reply=False, expected_len=None,
                        priority="high", target=serial_target,
                    ),
                    _build_command(
                        cmd_id="zero:pitch_a", func="0x92", addr=pitch_a_addr,
                        payload=[], expect_reply=False, expected_len=None,
                        priority="high", target=serial_target,
                    ),
                    _build_command(
                        cmd_id="zero:pitch_b", func="0x92", addr=pitch_b_addr,
                        payload=[], expect_reply=False, expected_len=None,
                        priority="high", target=serial_target,
                    ),
                ],
            )
        )
    else:
        _LOG.info("startup encoder zero suppressed (default fail-closed behavior)")
    status_names = {
        yaw_addr: "yaw",
        pitch_a_addr: "pitch_a",
        pitch_b_addr: "pitch_b",
    }
    pending_status = set(status_names)
    for status_attempt in range(1, 4):
        update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=[
                    _build_command(
                        cmd_id=f"status:{status_names[addr]}:{status_attempt}",
                        func="F1",
                        addr=addr,
                        payload=[],
                        expect_reply=True,
                        expected_len=1,
                        priority="high",
                        target=serial_target,
                    )
                    for addr in sorted(pending_status)
                ],
            )
        )
        pending_status = _wait_for_status(
            reply_sub,
            pending_status,
            timeout_s=0.75,
        )
        if not pending_status:
            break
        _LOG.warning(
            "status query attempt %d/3 missed addr(s) %s; retrying only missing axes",
            status_attempt,
            sorted(pending_status),
        )
    if pending_status:
        raise SystemExit(f"status query timed out for addr(s): {sorted(pending_status)}")
    startup_elapsed = time.monotonic() - startup_start
    _LOG.info(
        "gimbal startup completed in %.3f s (live=%s calibration=%s encoder_zero=%s)",
        startup_elapsed,
        args.enable_live_intent_actuation,
        calibration_authorized,
        zero_authorized,
    )

    yaw_counts: Optional[int] = None
    pitch_counts: dict[int, int] = {}
    last_encoder_ts: dict[int, float] = {}
    last_change_ts: dict[int, float] = {}
    encoder_timing_ok: dict[int, bool] = {}
    last_encoder_sequence: dict[int, int] = {}
    last_stale_pair_log = 0.0
    motor_state = {
        yaw_addr: {"name": "yaw", "last_cmd_ts": 0.0, "cmd_rate": 0.0, "expect_motion": False, "baseline_counts": None, "deadline": 0.0, "last_warn_ts": 0.0},
        pitch_a_addr: {"name": "pitch_a", "last_cmd_ts": 0.0, "cmd_rate": 0.0, "expect_motion": False, "baseline_counts": None, "deadline": 0.0, "last_warn_ts": 0.0},
        pitch_b_addr: {"name": "pitch_b", "last_cmd_ts": 0.0, "cmd_rate": 0.0, "expect_motion": False, "baseline_counts": None, "deadline": 0.0, "last_warn_ts": 0.0},
    }
    publication_render_predictor = EncoderAnchoredRenderPredictor()
    wire_render_predictor = EncoderAnchoredRenderPredictor()
    wire_render_tracker = WireExecutionRenderTracker(
        wire_render_predictor,
        target=serial_target,
        stale_s=execution_stale_s,
        require_fresh_encoder_anchors=require_fresh_execution_anchors,
    )
    render_pan_anchor_ts: Optional[float] = None
    render_tilt_anchor_ts: Optional[float] = None

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

    intent_gate = LiveIntentGate(watchdog_ns=intent_watchdog_ns)

    def _send_intent_rates(
        yaw_rate_cmd: float, pitch_rate_cmd: float, *, reason: str
    ) -> bool:
        if not math.isfinite(yaw_rate_cmd) or not math.isfinite(pitch_rate_cmd):
            yaw_rate_cmd = pitch_rate_cmd = 0.0
            reason = "non_finite_forced_stop"
        current_yaw_rad = (
            camstate_yaw_sign
            * _counts_to_rad(
                yaw_counts, counts_per_rev=counts_per_rev, gear_ratio=yaw_ratio
            )
            if yaw_counts is not None
            else None
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
        current_pitch_rad = (
            float(last_sample.tilt_rad)
            if camstate_source == "devices" and last_sample is not None
            else encoder_pitch_rad
        )
        yaw_rate_cmd = _apply_hard_angle_limit(
            yaw_rate_cmd, current_yaw_rad, yaw_min_rad, yaw_max_rad, "yaw"
        )
        pitch_rate_cmd = _apply_hard_angle_limit(
            pitch_rate_cmd,
            current_pitch_rad,
            pitch_min_rad,
            pitch_max_rad,
            "pitch",
        )
        command_priority = _intent_command_priority(yaw_rate_cmd, pitch_rate_cmd)
        yaw_motor_rate_cmd = yaw_sign * yaw_rate_cmd
        yaw_payload = _encode_timed_speed_cmd(
            yaw_motor_rate_cmd,
            acc=yaw_accel,
            gear_ratio=yaw_ratio,
            max_rate=yaw_rate_limit,
            runtime_ms=intent_runtime_ms,
        )
        command_token = time.time_ns()
        update_id = f"intent:{command_token}"
        yaw_cmd_id = f"intent:yaw:{command_token}"
        pitch_commands = _pitch_speed_commands(
            pitch_rate_cmd,
            priority=command_priority,
            runtime_ms=intent_runtime_ms,
            command_token=command_token,
            id_prefix="intent",
        )
        commands = [
            _build_command(
                cmd_id=yaw_cmd_id,
                func="F6",
                addr=yaw_addr,
                payload=yaw_payload,
                expect_reply=respond_on_writes,
                expected_len=None,
                priority=command_priority,
                target=serial_target,
            ),
            *pitch_commands,
        ]
        quantized_yaw_rate = _quantized_camera_rate(
            yaw_rate_cmd,
            motor_sign=yaw_sign,
            gear_ratio=yaw_ratio,
            max_rate=yaw_rate_limit,
        )
        quantized_pitch_a_rate = _quantized_camera_rate(
            pitch_rate_cmd,
            motor_sign=pitch_a_sign,
            gear_ratio=pitch_ratio,
            max_rate=pitch_rate_limit,
        )
        quantized_pitch_b_rate = _quantized_camera_rate(
            pitch_rate_cmd,
            motor_sign=pitch_b_sign,
            gear_ratio=pitch_ratio,
            max_rate=pitch_rate_limit,
        )
        runtime_s = intent_runtime_ms / 1000.0
        wire_render_tracker.register(
            [
                PendingWireCommand(
                    cmd_id=yaw_cmd_id,
                    update_id=update_id,
                    addr=yaw_addr,
                    motor_rate_rad_s=yaw_sign * quantized_yaw_rate,
                    render_axis="pan",
                    render_rate_rad_s=quantized_yaw_rate,
                    runtime_s=runtime_s,
                ),
                PendingWireCommand(
                    cmd_id=str(pitch_commands[0]["cmd_id"]),
                    update_id=update_id,
                    addr=pitch_a_addr,
                    motor_rate_rad_s=pitch_a_sign * quantized_pitch_a_rate,
                    render_axis="tilt" if pitch_authority_addr == pitch_a_addr else None,
                    render_rate_rad_s=(
                        quantized_pitch_a_rate
                        if pitch_authority_addr == pitch_a_addr
                        else None
                    ),
                    runtime_s=runtime_s,
                ),
                PendingWireCommand(
                    cmd_id=str(pitch_commands[1]["cmd_id"]),
                    update_id=update_id,
                    addr=pitch_b_addr,
                    motor_rate_rad_s=pitch_b_sign * quantized_pitch_b_rate,
                    render_axis="tilt" if pitch_authority_addr == pitch_b_addr else None,
                    render_rate_rad_s=(
                        quantized_pitch_b_rate
                        if pitch_authority_addr == pitch_b_addr
                        else None
                    ),
                    runtime_s=runtime_s,
                ),
            ]
        )
        sent = update_pub.send_update(
            _build_update(
                source="jetson.gimbal_bridge",
                target=serial_target,
                commands=commands,
                fields={
                    "intent_reason": reason,
                    "intent_runtime_ms": intent_runtime_ms,
                    "pan_rate_cmd": yaw_rate_cmd,
                    "yaw_motor_rate_cmd": yaw_motor_rate_cmd,
                    "tilt_rate_cmd": pitch_rate_cmd,
                },
                update_id=update_id,
            )
        )
        if sent:
            now_s = time.monotonic()
            quantized_pitch_rate = (
                quantized_pitch_a_rate
                if pitch_authority_addr == pitch_a_addr
                else quantized_pitch_b_rate
            )
            publication_render_predictor.set_command_rates(
                quantized_yaw_rate,
                quantized_pitch_rate,
                at_s=now_s,
                runtime_s=runtime_s,
            )
            if render_prediction_source != "wire_execution":
                _record_speed_command(
                    yaw_addr, yaw_sign * quantized_yaw_rate, now_s
                )
                _record_speed_command(
                    pitch_a_addr, pitch_a_sign * quantized_pitch_a_rate, now_s
                )
                _record_speed_command(
                    pitch_b_addr, pitch_b_sign * quantized_pitch_b_rate, now_s
                )
        else:
            wire_render_tracker.cancel_update(update_id)
        return bool(sent)
    try:
        while not stop_event.is_set():
            timeout_ms = int(math.ceil(feedback_period * 1000))
            events = dict(poller.poll(timeout=timeout_ms))
            if events.get(sub) == zmq.POLLIN:
                payload = sub.recv()
                try:
                    intent = control_intent_from_json(payload)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("failed to decode ControlIntent: %s", exc)
                else:
                    was_stopped = intent_gate.stopped
                    gate = intent_gate.accept(intent, now_ns=time.monotonic_ns())
                    if not gate.accepted:
                        _LOG.warning(
                            "rejected ControlIntent: %s sequence=%s observation_sequence=%s",
                            gate.reason,
                            intent.sequence,
                            intent.observation_sequence,
                        )
                        if gate.stop_required and args.enable_live_intent_actuation:
                            if _send_intent_rates(0.0, 0.0, reason=gate.reason):
                                intent_gate.mark_stopped()
                            else:
                                _LOG.error(
                                    "serial zero-rate publication dropped after rejected intent; watchdog remains armed"
                                )
                    elif args.enable_live_intent_actuation:
                        last_intent = intent
                        if not _should_forward_intent(intent, was_stopped=was_stopped):
                            intent_gate.mark_stopped()
                            if intent.reason == "controller_shutdown":
                                _LOG.info(
                                    "suppressed redundant controller shutdown sequence=%s observation_sequence=%s",
                                    intent.sequence,
                                    intent.observation_sequence,
                                )
                        else:
                            sent = _send_intent_rates(
                                float(intent.yaw_rate_rad_s),
                                float(intent.pitch_rate_rad_s),
                                reason=intent.reason,
                            )
                            if sent:
                                intent_gate.mark_command_sent(intent)
                                if intent.reason == "controller_shutdown":
                                    _LOG.info(
                                        "forwarded controller shutdown sequence=%s observation_sequence=%s",
                                        intent.sequence,
                                        intent.observation_sequence,
                                    )
                            else:
                                _LOG.warning(
                                    "serial update publish dropped for accepted ControlIntent sequence=%s; watchdog remains armed",
                                    intent.sequence,
                                )
                    else:
                        _LOG.debug("accepted live intent while actuation acknowledgement is absent")

            if (
                args.enable_live_intent_actuation
                and intent_gate.watchdog_stop_required(now_ns=time.monotonic_ns())
            ):
                _LOG.error("live intent watchdog expired; publishing timed zero rates")
                if _send_intent_rates(0.0, 0.0, reason="intent_watchdog_expired"):
                    intent_gate.mark_stopped()
                else:
                    _LOG.error("watchdog zero-rate publication dropped; watchdog remains armed")

            for reply in reply_sub.recv_nowait():
                fallback_reply_mono = time.monotonic()
                message_type = reply.get("type")
                if message_type == "SerialCommandEventV1":
                    applied = wire_render_tracker.handle_event(
                        reply, received_s=fallback_reply_mono
                    )
                    if applied is not None and render_prediction_source == "wire_execution":
                        _record_speed_command(
                            applied.addr,
                            applied.motor_rate_rad_s,
                            applied.wire_s,
                        )
                    continue
                if message_type == "SerialActuationStateV1":
                    applied_commands = wire_render_tracker.handle_snapshot(
                        reply, received_s=fallback_reply_mono
                    )
                    if render_prediction_source == "wire_execution":
                        for applied in applied_commands:
                            _record_speed_command(
                                applied.addr,
                                applied.motor_rate_rad_s,
                                applied.wire_s,
                            )
                    continue
                func = _reply_func_byte(reply)
                addr = reply.get("addr")
                if func == 0x31 and isinstance(addr, int):
                    parsed = reply.get("reply", {}).get("parsed", {})
                    if "counts" in parsed:
                        counts = int(parsed["counts"])
                        sequence_raw = reply.get("sequence")
                        sequence = None
                        try:
                            sequence = int(sequence_raw)
                        except (TypeError, ValueError):
                            sequence = None
                        if sequence is not None and sequence <= last_encoder_sequence.get(addr, -1):
                            continue
                        if sequence is not None:
                            last_encoder_sequence[addr] = sequence
                        timing = _reply_timing(reply, fallback_mono=fallback_reply_mono)
                        sample_ts = timing["reply_s"]
                        timing_ok = True
                        if encoder_max_queue_age_ms > 0.0 and timing["queue_age_ms"] > encoder_max_queue_age_ms:
                            timing_ok = False
                        if (
                            encoder_max_bus_duration_ms > 0.0
                            and timing["bus_duration_ms"] > encoder_max_bus_duration_ms
                        ):
                            timing_ok = False
                        prev = yaw_counts if addr == yaw_addr else pitch_counts.get(addr)
                        last_encoder_ts[addr] = sample_ts
                        encoder_timing_ok[addr] = timing_ok
                        if prev is None or counts != prev:
                            last_change_ts[addr] = sample_ts
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
            pan_timestamp = now
            tilt_timestamp = now
            pan_timing_ok = True
            tilt_timing_ok = True
            if camstate_source == "encoder":
                if yaw_counts is None or pitch_authority_addr not in pitch_counts:
                    continue
                pan_timestamp = last_encoder_ts.get(yaw_addr, now)
                tilt_timestamp = last_encoder_ts.get(pitch_authority_addr, now)
                pan_timing_ok = bool(encoder_timing_ok.get(yaw_addr, False))
                tilt_timing_ok = bool(encoder_timing_ok.get(pitch_authority_addr, False))
                if now - pan_timestamp > encoder_stale_warn_s:
                    pan_timing_ok = False
                if now - tilt_timestamp > encoder_stale_warn_s:
                    tilt_timing_ok = False
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
                if render_pan_anchor_ts != pan_timestamp:
                    publication_render_predictor.anchor_pan(
                        pan_rad, sample_s=pan_timestamp
                    )
                    wire_render_predictor.anchor_pan(
                        pan_rad, sample_s=pan_timestamp
                    )
                    wire_render_tracker.note_encoder_anchor(
                        "pan", sample_s=pan_timestamp
                    )
                    render_pan_anchor_ts = pan_timestamp
                if render_tilt_anchor_ts != tilt_timestamp:
                    publication_render_predictor.anchor_tilt(
                        tilt_rad, sample_s=tilt_timestamp
                    )
                    wire_render_predictor.anchor_tilt(
                        tilt_rad, sample_s=tilt_timestamp
                    )
                    wire_render_tracker.note_encoder_anchor(
                        "tilt", sample_s=tilt_timestamp
                    )
                    render_tilt_anchor_ts = tilt_timestamp
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
                prev_pan_ts = last_sample.pan_timestamp
                prev_tilt_ts = last_sample.tilt_timestamp
                if pan_timing_ok and prev_pan_ts is not None:
                    pan_dt = pan_timestamp - prev_pan_ts
                    if pan_dt >= encoder_rate_min_dt_s:
                        pan_rate = _wrapped_delta(pan_rad, last_sample.pan_rad) / pan_dt
                if tilt_timing_ok and prev_tilt_ts is not None:
                    tilt_dt = tilt_timestamp - prev_tilt_ts
                    if tilt_dt >= encoder_rate_min_dt_s:
                        tilt_rate = _wrapped_delta(tilt_rad, last_sample.tilt_rad) / tilt_dt

            sample = _AngleSample(
                timestamp=now,
                pan_rad=pan_rad,
                tilt_rad=tilt_rad,
                pan_rate_rad_s=pan_rate,
                tilt_rate_rad_s=tilt_rate,
                pan_timestamp=pan_timestamp,
                tilt_timestamp=tilt_timestamp,
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
            if last_intent is not None:
                frame_id = int(last_intent.observation_sequence)
                src_ts_ms = int(last_intent.issued_monotonic_ns // 1_000_000)
            else:
                frame_id = local_frame_id
                local_frame_id += 1
                src_ts_ms = int(time.monotonic_ns() / 1e6)
            try:
                publication_render_pose = (
                    publication_render_predictor.pose(at_s=now)
                    if camstate_source == "encoder"
                    else None
                )
                wire_render_pose = (
                    wire_render_tracker.pose(at_s=now)
                    if camstate_source == "encoder"
                    else None
                )
                if render_prediction_source == "publication":
                    render_pose = publication_render_pose
                elif render_prediction_source == "wire_execution":
                    render_pose = wire_render_pose
                else:
                    render_pose = None
                _publish_cam_state(
                    pub,
                    sample,
                    frame_id=frame_id,
                    src_ts_ms=src_ts_ms,
                    home_pan=camstate_home_pan,
                    home_tilt=camstate_home_tilt,
                    render_pose=render_pose,
                    encoder_pan_counts=yaw_counts,
                    encoder_tilt_counts=pitch_counts.get(pitch_authority_addr),
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
                    stats_predictor = (
                        wire_render_predictor
                        if render_prediction_source == "wire_execution"
                        else publication_render_predictor
                    )
                    correction_stats = stats_predictor.correction_stats()
                    wire_correction_stats = wire_render_predictor.correction_stats()
                    _LOG.info(
                        "gimbal heartbeat source=encoder pan=%.3f tilt=%.3f pan_rate=%.3f tilt_rate=%.3f frame_id=%s pitch_a_counts=%s pitch_b_counts=%s pitch_a_stale_s=%.3f pitch_b_stale_s=%.3f render_source=%s wire_healthy=%s wire_gaps=%d render_pan_correction=%.5f render_tilt_correction=%.5f render_age_s=%.3f render_pan_mean_abs=%.5f render_pan_max_abs=%.5f render_tilt_mean_abs=%.5f render_tilt_max_abs=%.5f wire_pan_mean_abs=%.5f wire_pan_max_abs=%.5f wire_tilt_mean_abs=%.5f wire_tilt_max_abs=%.5f",
                        float(last_sample.pan_rad),
                        float(last_sample.tilt_rad),
                        float(pan_rate),
                        float(tilt_rate),
                        getattr(last_intent, "observation_sequence", "n/a"),
                        pitch_counts.get(pitch_a_addr),
                        pitch_counts.get(pitch_b_addr),
                        now - last_encoder_ts[pitch_a_addr] if pitch_a_addr in last_encoder_ts else float("nan"),
                        now - last_encoder_ts[pitch_b_addr] if pitch_b_addr in last_encoder_ts else float("nan"),
                        render_prediction_source,
                        wire_render_tracker.healthy(now_s=now),
                        wire_render_tracker.sequence_gap_count,
                        (
                            float(render_pose.pan_correction_rad)
                            if render_pose is not None and render_pose.pan_correction_rad is not None
                            else float("nan")
                        ),
                        (
                            float(render_pose.tilt_correction_rad)
                            if render_pose is not None and render_pose.tilt_correction_rad is not None
                            else float("nan")
                        ),
                        float(render_pose.encoder_age_s) if render_pose is not None else float("nan"),
                        float(correction_stats["pan_mean_abs_rad"]),
                        float(correction_stats["pan_max_abs_rad"]),
                        float(correction_stats["tilt_mean_abs_rad"]),
                        float(correction_stats["tilt_max_abs_rad"]),
                        float(wire_correction_stats["pan_mean_abs_rad"]),
                        float(wire_correction_stats["pan_max_abs_rad"]),
                        float(wire_correction_stats["tilt_mean_abs_rad"]),
                        float(wire_correction_stats["tilt_max_abs_rad"]),
                    )
                else:
                    _LOG.info(
                        "gimbal heartbeat source=devices pan=%.3f tilt=%.3f pan_rate=%.3f tilt_rate=%.3f frame_id=%s",
                        float(last_sample.pan_rad),
                        float(last_sample.tilt_rad),
                        float(pan_rate),
                        float(tilt_rate),
                        getattr(last_intent, "observation_sequence", "n/a"),
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
        if args.enable_live_intent_actuation:
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
