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
from jetson.gimbal.mks_servo42_rs485 import (
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


def _build_axes(cfg: Mapping[str, Any]) -> Tuple[RS485Bus, GimbalInterface, float]:
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
    return bus, gimbal, pitch_div_thresh


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


def _make_manual_sub(ctx: zmq.Context, endpoint: Optional[str]) -> Optional[zmq.Socket]:
    if not endpoint:
        _LOG.info("manual control endpoint not provided; ignoring manual source")
        return None
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(endpoint)
    _LOG.info("subscribing to manual ControlCmd on %s", endpoint)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dev.yaml", help="Path to YAML config")
    ap.add_argument(
        "--feedback-hz",
        type=float,
        default=None,
        help="Override telemetry publish rate (Hz); defaults to gimbal.feedback_hz",
    )
    ap.add_argument(
        "--manual-endpoint",
        default=None,
        help="Optional endpoint for manual ControlCmds (overrides net.zmq_manual_control)",
    )
    ap.add_argument(
        "--manual-timeout",
        type=float,
        default=1.0,
        help="Seconds to keep manual ControlCmds active before falling back to autonomous",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    cfg = _load_config(Path(args.config))
    net_cfg = cfg.get("net") or {}
    ctrl_ep = net_cfg.get("zmq_control")
    if not ctrl_ep:
        raise SystemExit("config missing net.zmq_control endpoint")

    state_ep = net_cfg.get("zmq_gimbal_state")
    manual_ep = args.manual_endpoint or net_cfg.get("zmq_manual_control")
    manual_timeout = max(args.manual_timeout, 0.0)

    bus, gimbal, pitch_div_thresh = _build_axes(cfg)
    parameter_map: Mapping[int, Tuple[int, ...]] = {}
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
    manual_sub = _make_manual_sub(ctx, manual_ep)
    pub = _make_state_pub(ctx, state_ep)
    _LOG.info("subscribing to ControlCmd on %s (feedback %.1f Hz)", ctrl_ep, feedback_hz)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    if manual_sub is not None:
        poller.register(manual_sub, zmq.POLLIN)

    last_cmd = None
    last_auto_cmd = None
    last_manual_cmd = None
    last_manual_time = 0.0
    last_pub_time = 0.0
    last_stats_log = 0.0
    last_sample = None
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
                _LOG.info("zeroing all gimbal axes at their current position (function 0x92)")
                gimbal.zero_axes()
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"failed to enable gimbal axes: {exc}") from exc
            try:
                while not stop_event.is_set():
                    timeout_ms = int(math.ceil(feedback_period * 1000))
                    events = dict(poller.poll(timeout=timeout_ms))
                    now = time.monotonic()
                    if events.get(sub) == zmq.POLLIN:
                        payload = sub.recv()
                        try:
                            last_auto_cmd = control_cmd_from_json(payload)
                        except Exception as exc:  # noqa: BLE001
                            _LOG.warning("failed to decode ControlCmd: %s", exc)
                        else:
                            last_cmd = last_auto_cmd
                    if manual_sub is not None and events.get(manual_sub) == zmq.POLLIN:
                        payload = manual_sub.recv()
                        try:
                            last_manual_cmd = control_cmd_from_json(payload)
                            last_manual_time = now
                        except Exception as exc:  # noqa: BLE001
                            _LOG.warning("failed to decode manual ControlCmd: %s", exc)
                        else:
                            last_cmd = last_manual_cmd

                    active_cmd = None
                    if last_manual_cmd is not None and (now - last_manual_time) <= manual_timeout:
                        active_cmd = last_manual_cmd
                    elif last_auto_cmd is not None:
                        active_cmd = last_auto_cmd

                    if active_cmd is not None:
                        gimbal.apply_rate_commands(
                            float(active_cmd.pan_rate_cmd),
                            float(active_cmd.tilt_rate_cmd),
                        )
                    now = time.monotonic()
                    if pub is None:
                        continue
                    if (now - last_pub_time) < feedback_period:
                        continue
                    last_pub_time = now
                    sample = gimbal.sample_state()
                    last_sample = sample
                    if sample.secondary_pitch_rad is not None:
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
                        _publish_cam_state(pub, sample, frame_id=frame_id, src_ts_ms=src_ts_ms)
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
                        _LOG.info(
                            "gimbal heartbeat pan=%.3f tilt=%.3f pan_rate=%.3f tilt_rate=%.3f frame_id=%s",
                            float(last_sample.pan_rad),
                            float(last_sample.tilt_rad),
                            float(pan_rate),
                            float(tilt_rate),
                            getattr(last_cmd, "frame_id", "n/a"),
                        )
            finally:
                try:
                    gimbal.stop()
                except Exception:  # noqa: BLE001
                    _LOG.warning("failed to send stop commands", exc_info=True)
                try:
                    if hasattr(gimbal.pitch_axis, "enable"):
                        gimbal.pitch_axis.enable(False)  # type: ignore[union-attr]
                    gimbal.yaw_axis.enable(False)
                except Exception:  # noqa: BLE001
                    _LOG.debug("axis disable failed", exc_info=True)
    finally:
        try:
            poller.unregister(sub)
        except Exception:  # noqa: BLE001
            pass
        try:
            sub.close(linger=0)
        except Exception:  # noqa: BLE001
            pass
        if manual_sub is not None:
            try:
                poller.unregister(manual_sub)
            except Exception:  # noqa: BLE001
                pass
            try:
                manual_sub.close(linger=0)
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
