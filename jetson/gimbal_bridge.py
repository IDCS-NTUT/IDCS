"""Bridge ControlCmd messages to MKS serial (TTL-to-RS485) gimbal motion commands.

This Jetson-side process subscribes to the ControlCmd PUB socket, translates
pan/tilt rate commands into MKS SR_CLOSE speed mode writes, and periodically
publishes encoder-derived :class:`CamState` telemetry. Dual-pitch rigs use a
shared group address for commands with opposing motor "Dir" settings so a
single speed command moves both actuators in mirrored directions (default
pitch motor A CCW, motor B CW).
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import zmq
import yaml

from common.config_sync import parse_config_text, read_snapshot
from common.schemas import CamState, control_cmd_from_json
from common.shutdown import install_signal_handlers
from common.gimbal.mks_servo42_rs485 import (
    GimbalInterface,
    MksServo42Axis,
    PitchAxisGroup,
    RS485Bus,
)

_LOG = logging.getLogger(__name__)


def _parse_tcp_port(endpoint: str, name: str) -> int:
    try:
        port = int(endpoint.rsplit(":", 1)[1])
    except Exception as exc:  # noqa: BLE001 - defensive parsing
        raise SystemExit(f"invalid {name} endpoint: {endpoint!r}") from exc
    if port <= 0 or port > 65535:
        raise SystemExit(f"{name} port must be in 1..65535 (got {port})")
    return port


def _load_config(path: Path) -> Mapping[str, Any]:
    snapshot = read_snapshot(path)
    return parse_config_text(snapshot.text, str(path))


def _auto_control_enabled(cfg: Mapping[str, Any]) -> bool:
    try:
        gimbal_cfg = cfg.get("gimbal") or {}
        return bool(gimbal_cfg.get("auto_control_enabled", False))
    except Exception:  # noqa: BLE001 - defensive config parsing
        return False


def _build_axes(cfg: Mapping[str, Any]) -> Tuple[RS485Bus, GimbalInterface, float, Mapping[str, Any]]:
    gimbal_cfg = cfg.get("gimbal")
    if not isinstance(gimbal_cfg, Mapping):
        raise SystemExit("config missing 'gimbal' section")

    backend = gimbal_cfg.get("backend")
    if backend != "mks_rs485":
        raise SystemExit(f"gimbal backend {backend!r} is not supported by this bridge")

    try:
        port = str(gimbal_cfg["serial_port"])
    except KeyError as exc:
        raise SystemExit("gimbal.serial_port is required") from exc
    baudrate = int(gimbal_cfg.get("baudrate", 38400))
    timeout = float(gimbal_cfg.get("timeout", 0.1))
    retries = int(gimbal_cfg.get("retries", 1))

    counts_per_rev = int(gimbal_cfg.get("counts_per_rev", 0x4000))
    yaw_ratio = float(gimbal_cfg.get("yaw_gear_ratio", 1.0))
    pitch_ratio = float(gimbal_cfg.get("pitch_gear_ratio", 1.0))

    yaw_addr = int(gimbal_cfg.get("yaw_addr", 1))
    yaw_group_addr = gimbal_cfg.get("yaw_group_addr")
    yaw_group_addr = int(yaw_group_addr) if yaw_group_addr is not None else None

    pitch_group_addr = gimbal_cfg.get("pitch_group_addr")
    if pitch_group_addr is None:
        raise SystemExit("gimbal.pitch_group_addr is required for dual-pitch setup")
    pitch_group_addr = int(pitch_group_addr)

    use_group_writes = bool(gimbal_cfg.get("use_group_writes", True))

    try:
        pitch_motor_a_addr = int(gimbal_cfg["pitch_motor_a_addr"])
        pitch_motor_b_addr = int(gimbal_cfg["pitch_motor_b_addr"])
    except KeyError as exc:
        raise SystemExit("gimbal.pitch_motor_a_addr and pitch_motor_b_addr are required") from exc

    authority = gimbal_cfg.get("pitch_encoder_authority", "a")
    if authority not in {"a", "b"}:
        raise SystemExit("gimbal.pitch_encoder_authority must be 'a' or 'b'")

    yaw_accel_byte = int(gimbal_cfg.get("yaw_accel_byte", 10))
    pitch_accel_byte = int(gimbal_cfg.get("pitch_accel_byte", 10))
    yaw_rate_limit = float(gimbal_cfg.get("yaw_rate_limit_rad_s", 10.0))
    pitch_rate_limit = float(gimbal_cfg.get("pitch_rate_limit_rad_s", 10.0))
    pitch_div_thresh = float(gimbal_cfg.get("pitch_divergence_thresh_rad", 0.0873))

    max_rate = max(yaw_rate_limit, pitch_rate_limit)

    bus = RS485Bus(
        port,
        baudrate=baudrate,
        timeout=timeout,
        max_retries=max(retries, 0),
    )

    yaw_axis = MksServo42Axis(
        bus,
        yaw_addr,
        group_addr=yaw_group_addr,
        counts_per_rev=counts_per_rev,
        gear_ratio=yaw_ratio,
        use_group_writes=use_group_writes,
    )
    pitch_a = MksServo42Axis(
        bus,
        pitch_motor_a_addr,
        group_addr=pitch_group_addr,
        counts_per_rev=counts_per_rev,
        gear_ratio=pitch_ratio,
        use_group_writes=use_group_writes,
    )
    pitch_b = MksServo42Axis(
        bus,
        pitch_motor_b_addr,
        group_addr=pitch_group_addr,
        counts_per_rev=counts_per_rev,
        gear_ratio=pitch_ratio,
        use_group_writes=use_group_writes,
    )
    pitch_axis = PitchAxisGroup(
        bus,
        pitch_group_addr,
        motor_a=pitch_a,
        motor_b=pitch_b,
        authority=authority,
    )
    gimbal = GimbalInterface(
        yaw_axis,
        pitch_axis,
        max_rate_rad_s=max_rate,
        yaw_accel_byte=yaw_accel_byte,
        pitch_accel_byte=pitch_accel_byte,
    )
    axis_meta = {
        "counts_per_rev": counts_per_rev,
        "yaw_ratio": yaw_ratio,
        "pitch_ratio": pitch_ratio,
        "pitch_authority": authority,
        "pitch_motor_a_addr": pitch_motor_a_addr,
        "pitch_motor_b_addr": pitch_motor_b_addr,
        "yaw_addr": yaw_addr,
    }
    return bus, gimbal, pitch_div_thresh, axis_meta


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


def _apply_axis_parameters(
    axis: MksServo42Axis, param_map: Mapping[int, Tuple[int, ...]], name: str
) -> None:
    payload = param_map.get(axis.addr)
    if payload is None:
        _LOG.info("no parameter set provided for %s (addr=%d)", name, axis.addr)
        return
    _LOG.info(
        "writing %d parameter bytes to %s (addr=%d)",
        len(payload),
        name,
        axis.addr,
    )
    status = axis.write_all_parameters(payload)
    if status != 1:
        raise SystemExit(
            f"{name} returned status={status} when writing all parameters"
        )


def _publish_cam_state(pub: zmq.Socket, sample, *, frame_id: int, src_ts_ms: int) -> None:
    cam_state = CamState(
        frame_id=frame_id,
        src_ts_ms=src_ts_ms,
        pan=float(sample.pan_rad),
        tilt=float(sample.tilt_rad),
        pan_rate=sample.pan_rate_rad_s,
        tilt_rate=sample.tilt_rate_rad_s,
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


def _counts_to_angle_rad(counts: int, *, counts_per_rev: int, gear_ratio: float) -> float:
    motor_revs = counts / float(counts_per_rev)
    axis_revs = motor_revs / gear_ratio
    return axis_revs * 2.0 * math.pi


def _update_axis_cache(
    *,
    cache: dict[str, dict[str, float]],
    name: str,
    counts: int,
    counts_per_rev: int,
    gear_ratio: float,
) -> None:
    now = time.monotonic()
    angle = _counts_to_angle_rad(counts, counts_per_rev=counts_per_rev, gear_ratio=gear_ratio)
    state = cache.setdefault(name, {})
    prev_angle = state.get("angle")
    prev_ts = state.get("timestamp")
    rate = None
    if prev_angle is not None and prev_ts is not None:
        dt = now - prev_ts
        if dt > 0:
            rate = (angle - prev_angle) / dt
    state.update({"angle": angle, "timestamp": now})
    if rate is not None:
        state["rate"] = rate


def _drain_encoder_feedback(
    bus: RS485Bus, *, axis_map: Mapping[int, tuple[str, int, float]], cache: dict[str, dict[str, float]]
) -> None:
    """Drain any encoder replies already present on the bus (listen-only mode).

    axis_map: addr -> (name, counts_per_rev, gear_ratio)
    Updates the per-axis cache in-place when 0x31 responses are seen.
    """

    serial_port = bus._serial  # Access underlying Serial for passive reads
    while serial_port.in_waiting:
        try:
            frame = bus._read_frame(expected_data_len=None)
        except TimeoutError:
            break
        except Exception:  # noqa: BLE001 - best-effort drain
            _LOG.debug("failed to read encoder feedback frame", exc_info=True)
            break
        if len(frame) < 4:
            continue
        addr = frame[1]
        func = frame[2]
        data = frame[3:-1]
        mapping = axis_map.get(addr)
        if mapping is None:
            continue
        if func == 0x31 and len(data) == 6:
            counts = int.from_bytes(data, byteorder="big", signed=True)
            name, counts_per_rev, gear_ratio = mapping
            _update_axis_cache(
                cache=cache,
                name=name,
                counts=counts,
                counts_per_rev=counts_per_rev,
                gear_ratio=gear_ratio,
            )


def _build_cam_state_from_cache(
    cache: Mapping[str, Mapping[str, float]],
    *,
    pitch_primary_key: str,
    frame_id: int,
    src_ts_ms: int,
) -> Optional[CamState]:
    yaw = cache.get("yaw", {})
    pitch_primary = cache.get(pitch_primary_key, {})
    if "angle" not in yaw or "angle" not in pitch_primary:
        return None
    cam_state = CamState(
        frame_id=frame_id,
        src_ts_ms=src_ts_ms,
        pan=float(yaw["angle"]),
        tilt=float(pitch_primary["angle"]),
        pan_rate=yaw.get("rate"),
        tilt_rate=pitch_primary.get("rate"),
    )
    return cam_state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dev.yaml", help="Path to YAML config")
    ap.add_argument(
        "--feedback-hz",
        type=float,
        default=None,
        help="Override telemetry publish rate (Hz); defaults to gimbal.feedback_hz",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    cfg = _load_config(Path(args.config))
    serial_commands_enabled = _auto_control_enabled(cfg)
    if not serial_commands_enabled:
        _LOG.warning(
            "Auto control disabled by config (gimbal.auto_control_enabled=false); running in telemetry-only mode with serial motion commands suppressed"
        )

    net_cfg = cfg.get("net") or {}
    ctrl_ep = net_cfg.get("zmq_control")
    if not ctrl_ep:
        raise SystemExit("config missing net.zmq_control endpoint")

    state_ep = net_cfg.get("zmq_gimbal_state")

    bus, gimbal, pitch_div_thresh, axis_meta = _build_axes(cfg)
    parameter_map: Mapping[int, Tuple[int, ...]] = {}
    axis_cache: dict[str, dict[str, float]] = {}
    axis_map: dict[int, Tuple[str, int, float]] = {
        int(axis_meta["yaw_addr"]): ("yaw", int(axis_meta["counts_per_rev"]), float(axis_meta["yaw_ratio"])),
        int(axis_meta["pitch_motor_a_addr"]): (
            "pitch_a",
            int(axis_meta["counts_per_rev"]),
            float(axis_meta["pitch_ratio"]),
        ),
        int(axis_meta["pitch_motor_b_addr"]): (
            "pitch_b",
            int(axis_meta["counts_per_rev"]),
            float(axis_meta["pitch_ratio"]),
        ),
    }
    pitch_authority = "b" if str(axis_meta["pitch_authority"]).lower() == "b" else "a"
    pitch_primary_key = f"pitch_{pitch_authority}"
    pitch_secondary_key = "pitch_b" if pitch_primary_key == "pitch_a" else "pitch_a"
    gimbal_cfg = cfg.get("gimbal") or {}
    param_path = gimbal_cfg.get("parameter_file")
    if param_path:
        parameter_map = _load_parameter_map(Path(str(param_path)))
        _LOG.info("loaded parameter sets for %d motors from %s", len(parameter_map), param_path)

    _LOG.info(
        "configured serial gimbal: yaw addr=%d group=%s (group writes=%s), pitch group=%d authority=%s (group writes=%s), divergence_thresh=%.4f rad",
        gimbal.yaw_axis.addr,
        getattr(gimbal.yaw_axis, "group_addr", None),
        getattr(gimbal.yaw_axis, "use_group_writes", False),
        gimbal.pitch_axis.group_addr if hasattr(gimbal.pitch_axis, "group_addr") else None,
        getattr(gimbal.pitch_axis, "authority", "unknown"),
        getattr(gimbal.pitch_axis, "motor_a", None).use_group_writes if hasattr(gimbal.pitch_axis, "motor_a") else False,
        pitch_div_thresh,
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
    _LOG.info("subscribing to ControlCmd on %s (feedback %.1f Hz)", ctrl_ep, feedback_hz)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    last_cmd = None
    last_pub_time = 0.0
    last_stats_log = 0.0
    last_divergence_log = 0.0
    local_frame_id = 0

    def _query_required_status(axis: MksServo42Axis, name: str) -> int:
        try:
            status = axis.status()
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"failed to read {name} status: {exc}") from exc
        if status == 0:
            raise SystemExit(f"{name} returned status=0 (query failed)")
        _LOG.info("%s status=%d", name, status)
        return status

    try:
        with bus:
            _LOG.info("Serial bus opened on %s @ %d", bus.port, bus.baudrate)
            _query_required_status(gimbal.yaw_axis, "yaw motor")
            if isinstance(gimbal.pitch_axis, PitchAxisGroup):
                _query_required_status(gimbal.pitch_axis.motor_a, "pitch motor A")
                _query_required_status(gimbal.pitch_axis.motor_b, "pitch motor B")
            else:
                _query_required_status(gimbal.pitch_axis, "pitch motor")

            if parameter_map:
                _apply_axis_parameters(gimbal.yaw_axis, parameter_map, "yaw motor")
                if isinstance(gimbal.pitch_axis, PitchAxisGroup):
                    _apply_axis_parameters(
                        gimbal.pitch_axis.motor_a, parameter_map, "pitch motor A"
                    )
                    _apply_axis_parameters(
                        gimbal.pitch_axis.motor_b, parameter_map, "pitch motor B"
                    )
                else:
                    _apply_axis_parameters(gimbal.pitch_axis, parameter_map, "pitch motor")

            try:
                gimbal.yaw_axis.enable(True)
                if hasattr(gimbal.pitch_axis, "enable"):
                    gimbal.pitch_axis.enable(True)  # type: ignore[union-attr]
                _LOG.info(
                    "zeroing all gimbal axes at their current position (function 0x92)"
                )
                gimbal.zero_axes()
                serial_commands_enabled = False
                _LOG.info(
                    "Serial motion commands disabled after startup; entering passive feedback mode"
                )
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"failed to enable gimbal axes: {exc}") from exc
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
                            _LOG.debug(
                                "Received ControlCmd while serial commands disabled; pan_rate_cmd=%.3f tilt_rate_cmd=%.3f ignored",
                                float(last_cmd.pan_rate_cmd),
                                float(last_cmd.tilt_rate_cmd),
                            )
                    now = time.monotonic()
                    _drain_encoder_feedback(bus, axis_map=axis_map, cache=axis_cache)
                    if pub is None:
                        continue
                    if (now - last_pub_time) < feedback_period:
                        continue
                    cam_state = _build_cam_state_from_cache(
                        axis_cache,
                        pitch_primary_key=pitch_primary_key,
                        frame_id=int(last_cmd.frame_id) if last_cmd is not None else local_frame_id,
                        src_ts_ms=int(last_cmd.src_ts_ms) if last_cmd is not None else int(time.monotonic_ns() / 1e6),
                    )
                    if cam_state is None:
                        continue
                    if last_cmd is None:
                        local_frame_id += 1
                    last_pub_time = now
                    pitch_secondary = None
                    secondary_state = axis_cache.get(pitch_secondary_key)
                    if secondary_state is not None and "angle" in secondary_state:
                        pitch_secondary = float(secondary_state["angle"])
                    if pitch_secondary is not None:
                        divergence = abs(pitch_secondary - float(cam_state.tilt))
                        if divergence >= pitch_div_thresh and (now - last_divergence_log) >= 2.0:
                            last_divergence_log = now
                            _LOG.warning(
                                "pitch encoder divergence %.4f rad exceeds threshold %.4f (primary=%.4f secondary=%.4f)",
                                divergence,
                                pitch_div_thresh,
                                float(cam_state.tilt),
                                pitch_secondary,
                            )
                    try:
                        pub.send_string(cam_state.model_dump_json(exclude_none=True))
                    except Exception as exc:  # noqa: BLE001
                        _LOG.warning("failed to publish CamState: %s", exc)
                    if (now - last_stats_log) >= 5.0:
                        last_stats_log = now
                        _LOG.info(
                            "gimbal heartbeat pan=%.3f tilt=%.3f pan_rate=%.3f tilt_rate=%.3f frame_id=%s",
                            float(cam_state.pan),
                            float(cam_state.tilt),
                            float(cam_state.pan_rate) if cam_state.pan_rate is not None else float("nan"),
                            float(cam_state.tilt_rate) if cam_state.tilt_rate is not None else float("nan"),
                            getattr(last_cmd, "frame_id", "n/a"),
                        )
            finally:
                _LOG.info("Passive mode active; skipping outgoing stop/disable commands on shutdown")
    finally:
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
        try:
            ctx.destroy(linger=0)
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
