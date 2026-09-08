#!/usr/bin/env python3
"""Record control-loop ZMQ telemetry to a JSONL trace file.

The recorder passively subscribes to the existing Jetson PUB sockets for
DetectionMsg, ControlCmd, and optional CamState telemetry. Each received
message is validated against the shared schema and written as one JSON object
per line so offline tools can replay or analyze the exact payloads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import zmq

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from common.config_sync import expand_config_paths, load_merged_config
from common.schemas import (
    CamState,
    control_cmd_from_json,
    detection_msg_from_json,
)
from common.shutdown import install_signal_handlers


@dataclass(frozen=True)
class _StreamSpec:
    name: str
    endpoint: str
    decoder: Callable[[bytes], Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml", help="Base YAML config")
    parser.add_argument(
        "--config-extra",
        default="configs/control.yaml,configs/system.yaml",
        help="Comma-separated YAML configs merged over --config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL output path (default: logs/control_trace_<unix_ts>.jsonl)",
    )
    parser.add_argument("--duration-s", type=float, default=None, help="Stop after N seconds")
    parser.add_argument(
        "--status-interval-s",
        type=float,
        default=2.0,
        help="Print receive counts at this interval; set 0 to disable.",
    )
    parser.add_argument(
        "--hwm",
        type=int,
        default=10000,
        help="SUB receive high-water mark per stream.",
    )
    parser.add_argument("--results-endpoint", default=None, help="Override net.zmq_results")
    parser.add_argument("--control-endpoint", default=None, help="Override net.zmq_control")
    parser.add_argument(
        "--camstate-endpoint",
        default=None,
        help="Override CamState endpoint (default: subscribe to net.zmq_camstate_trace and net.zmq_gimbal_state when configured).",
    )
    parser.add_argument("--no-detections", action="store_true", help="Do not record DetectionMsg")
    parser.add_argument("--no-control", action="store_true", help="Do not record ControlCmd")
    parser.add_argument("--no-camstate", action="store_true", help="Do not record CamState")
    parser.add_argument(
        "--pc-iface",
        default=None,
        help="Override net.pc_iface for Linux ZMQ BINDTODEVICE.",
    )
    return parser.parse_args()


def _bind_zmq_to_device_if_configured(socket: zmq.Socket, iface: Optional[str]) -> None:
    if not iface:
        return
    option = getattr(zmq, "BINDTODEVICE", None)
    if option is None:
        print("[record][WARN] net.pc_iface ignored: pyzmq/libzmq lacks BINDTODEVICE")
        return
    try:
        socket.setsockopt_string(option, iface)
    except Exception as exc:  # noqa: BLE001 - best-effort platform feature
        print(f"[record][WARN] net.pc_iface={iface!r} bind failed: {exc}")


def _endpoint(
    net_cfg: Mapping[str, Any],
    *,
    key: str,
    override: Optional[str],
) -> Optional[str]:
    raw = override if override is not None else net_cfg.get(key)
    if raw is None:
        return None
    endpoint = str(raw).strip()
    return endpoint or None


def _decode_cam_state(payload: bytes) -> CamState:
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"CamState payload must be mapping-like, got {type(raw)!r}")
    return CamState(**raw)


def _model_payload(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"decoded payload must be mapping-like, got {type(value)!r}")


def _make_sub(
    ctx: zmq.Context,
    spec: _StreamSpec,
    *,
    hwm: int,
    iface: Optional[str],
) -> zmq.Socket:
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, max(1, int(hwm)))
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    _bind_zmq_to_device_if_configured(sub, iface)
    sub.connect(spec.endpoint)
    print(f"[record] subscribed {spec.name} <- {spec.endpoint}")
    return sub


def _default_output_path() -> Path:
    return Path("logs") / f"control_trace_{int(time.time())}.jsonl"


def _write_jsonl(handle, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def main() -> int:
    args = _parse_args()
    if args.duration_s is not None and args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be positive when provided")
    if args.status_interval_s < 0.0:
        raise SystemExit("--status-interval-s must be >= 0")

    cfg = load_merged_config(expand_config_paths(args.config, args.config_extra))
    net_raw = cfg.get("net", {})
    net_cfg: Mapping[str, Any] = net_raw if isinstance(net_raw, Mapping) else {}
    iface = args.pc_iface
    if iface is None:
        iface_raw = net_cfg.get("pc_iface")
        iface = str(iface_raw).strip() if iface_raw else None

    streams: list[_StreamSpec] = []
    if not args.no_detections:
        endpoint = _endpoint(net_cfg, key="zmq_results", override=args.results_endpoint)
        if endpoint:
            streams.append(_StreamSpec("detection", endpoint, detection_msg_from_json))
    if not args.no_control:
        endpoint = _endpoint(net_cfg, key="zmq_control", override=args.control_endpoint)
        if endpoint:
            streams.append(_StreamSpec("control", endpoint, control_cmd_from_json))
    if not args.no_camstate:
        camstate_endpoints: list[str] = []
        if args.camstate_endpoint:
            camstate_endpoints.append(str(args.camstate_endpoint).strip())
        else:
            for key in ("zmq_camstate_trace", "zmq_gimbal_state"):
                endpoint = _endpoint(net_cfg, key=key, override=None)
                if endpoint and endpoint not in camstate_endpoints:
                    camstate_endpoints.append(endpoint)
        for endpoint in camstate_endpoints:
            if endpoint:
                streams.append(_StreamSpec("camstate", endpoint, _decode_cam_state))

    if not streams:
        raise SystemExit("no streams selected or configured")

    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stop_event = install_signal_handlers()
    ctx = zmq.Context()
    poller = zmq.Poller()
    sockets: dict[zmq.Socket, _StreamSpec] = {}
    counts = {spec.name: 0 for spec in streams}
    decode_errors = 0
    start_mono_ns = time.monotonic_ns()
    start_wall_ts_ms = int(time.time() * 1000)
    deadline_ns = (
        start_mono_ns + int(args.duration_s * 1e9)
        if args.duration_s is not None
        else None
    )

    try:
        for spec in streams:
            sock = _make_sub(ctx, spec, hwm=args.hwm, iface=iface)
            sockets[sock] = spec
            poller.register(sock, zmq.POLLIN)

        print(f"[record] writing {output_path}")
        next_status = time.monotonic() + args.status_interval_s
        with output_path.open("w", encoding="utf-8", buffering=1) as handle:
            _write_jsonl(
                handle,
                {
                    "type": "meta",
                    "tool": "record_control_trace",
                    "version": 1,
                    "start_monotonic_ns": start_mono_ns,
                    "start_wall_ts_ms": start_wall_ts_ms,
                    "streams": [
                        {"name": spec.name, "endpoint": spec.endpoint}
                        for spec in streams
                    ],
                },
            )

            while not stop_event.is_set():
                now_ns = time.monotonic_ns()
                if deadline_ns is not None and now_ns >= deadline_ns:
                    break

                events = dict(poller.poll(250))
                for sock, event in events.items():
                    if not (event & zmq.POLLIN):
                        continue
                    spec = sockets[sock]
                    while True:
                        try:
                            payload = sock.recv(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break

                        rx_mono_ns = time.monotonic_ns()
                        rx_wall_ts_ms = int(time.time() * 1000)
                        try:
                            decoded = spec.decoder(payload)
                            normalized = _model_payload(decoded)
                        except Exception as exc:  # noqa: BLE001 - keep the trace alive
                            decode_errors += 1
                            raw_text = payload.decode("utf-8", errors="replace")
                            _write_jsonl(
                                handle,
                                {
                                    "type": "decode_error",
                                    "stream": spec.name,
                                    "endpoint": spec.endpoint,
                                    "rx_monotonic_ns": rx_mono_ns,
                                    "rx_wall_ts_ms": rx_wall_ts_ms,
                                    "error": str(exc),
                                    "raw": raw_text[:2000],
                                },
                            )
                            continue

                        counts[spec.name] += 1
                        _write_jsonl(
                            handle,
                            {
                                "type": "event",
                                "stream": spec.name,
                                "endpoint": spec.endpoint,
                                "rx_monotonic_ns": rx_mono_ns,
                                "rx_wall_ts_ms": rx_wall_ts_ms,
                                "payload": normalized,
                            },
                        )

                if args.status_interval_s > 0.0 and time.monotonic() >= next_status:
                    summary = " ".join(f"{name}={count}" for name, count in counts.items())
                    print(f"[record] {summary} decode_errors={decode_errors}")
                    next_status += args.status_interval_s

    finally:
        for sock in list(sockets):
            try:
                poller.unregister(sock)
            except Exception:
                pass
            try:
                sock.close(0)
            except Exception:
                pass
        ctx.term()

    summary = " ".join(f"{name}={count}" for name, count in counts.items())
    print(f"[record] done {summary} decode_errors={decode_errors} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
