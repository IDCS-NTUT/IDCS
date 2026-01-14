"""Run the dedicated RS485 drain service on the Jetson."""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import zmq

from common.config_sync import parse_config_text, read_snapshot
from common.rs485 import (
    RS485CommandRequest,
    RS485CommandResponse,
    RS485DataUpdateRequest,
    RS485DataUpdateResponse,
    RS485HealthResponse,
    RS485HistoryResponse,
    RS485KeyLatestResponse,
    RS485LatestResponse,
    RS485Response,
    RS485Service,
    RS485ServiceConfig,
    rs485_request_from_json,
)
from common.shutdown import install_signal_handlers


def _load_config(path: Path) -> Mapping[str, Any]:
    snapshot = read_snapshot(path)
    return parse_config_text(snapshot.text, str(path))


def _build_service(cfg: Mapping[str, Any]) -> RS485Service:
    gimbal_cfg = cfg.get("gimbal")
    if not isinstance(gimbal_cfg, Mapping):
        raise SystemExit("config missing 'gimbal' section")

    backend = gimbal_cfg.get("backend")
    if backend != "mks_rs485":
        raise SystemExit(f"gimbal backend {backend!r} is not supported by the RS485 service")

    try:
        port = str(gimbal_cfg["serial_port"])
    except KeyError as exc:
        raise SystemExit("gimbal.serial_port is required") from exc

    config = RS485ServiceConfig(
        port=port,
        baudrate=int(gimbal_cfg.get("baudrate", 115200)),
        timeout=float(gimbal_cfg.get("timeout", 0.1)),
        max_retries=int(gimbal_cfg.get("retries", 1)),
        history_size=int(gimbal_cfg.get("rs485_history_size", 256)),
        schedule_interval_s=float(gimbal_cfg.get("rs485_schedule_interval_s", 0.1)),
    )
    return RS485Service(config)


def _resolve_endpoint(cfg: Mapping[str, Any]) -> str:
    gimbal_cfg = cfg.get("gimbal") or {}
    endpoint = gimbal_cfg.get("rs485_endpoint", "tcp://127.0.0.1:5559")
    if not isinstance(endpoint, str) or not endpoint:
        raise SystemExit("gimbal.rs485_endpoint must be a non-empty string")
    return endpoint


def _handle_request(service: RS485Service, request, *, req_id: str) -> RS485Response:
    logger = logging.getLogger(__name__)
    if isinstance(request, RS485CommandRequest):
        try:
            serial_start = time.monotonic()
            data = request.data or []
            resp = service.send_command(
                request.addr,
                request.func,
                data,
                response_expected=request.response_expected,
                expected_response_len=request.expected_response_len,
            )
            serial_ms = (time.monotonic() - serial_start) * 1000.0
            logger.info(
                "RS485 command id=%s addr=0x%02X func=0x%02X ok=true serial_ms=%.1f",
                req_id,
                request.addr,
                request.func,
                serial_ms,
            )
            return RS485CommandResponse(payload=list(resp), cached=False)
        except Exception as exc:  # noqa: BLE001
            serial_ms = (time.monotonic() - serial_start) * 1000.0
            logger.warning(
                "RS485 command id=%s addr=0x%02X func=0x%02X ok=false serial_ms=%.1f err=%s",
                req_id,
                request.addr,
                request.func,
                serial_ms,
                exc,
            )
            return RS485CommandResponse(ok=False, error=str(exc), payload=None, cached=False)

    if isinstance(request, RS485DataUpdateRequest):
        service.update_external_data(request.key, request.value)
        return RS485DataUpdateResponse(ok=True, cached=True)

    if request.type == "latest_key":
        latest = service.latest_by_key(request.key)
        payload = None
        if latest is not None:
            payload = {
                "raw": list(latest.raw),
                "received_ts": latest.received_ts,
                "addr": latest.addr,
                "func": latest.func,
            }
        return RS485KeyLatestResponse(payload=payload, cached=True, ok=payload is not None)

    if request.type == "latest":
        latest = service.latest_snapshot().get((request.addr, request.func))
        payload = None
        if latest is not None:
            payload = {
                "raw": list(latest.raw),
                "received_ts": latest.received_ts,
                "addr": latest.addr,
                "func": latest.func,
            }
        return RS485LatestResponse(payload=payload, cached=True, ok=payload is not None)

    if request.type == "history":
        history = service.history_snapshot().get((request.addr, request.func), [])
        payload = [
            {
                "raw": list(frame.raw),
                "received_ts": frame.received_ts,
                "addr": frame.addr,
                "func": frame.func,
            }
            for frame in history
        ]
        return RS485HistoryResponse(payload=payload, cached=True, ok=True)

    if request.type == "health":
        return RS485HealthResponse(payload=service.stats_snapshot(), cached=True, ok=True)

    return RS485Response(ok=False, error=f"Unsupported request type {request.type!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dev.yaml", help="Path to YAML config")
    ap.add_argument(
        "--endpoint",
        default=None,
        help="Optional override for gimbal.rs485_endpoint (e.g., tcp://0.0.0.0:5559)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    cfg = _load_config(Path(args.config))
    service = _build_service(cfg)
    schedule_cfg = cfg.get("rs485_commands")
    if isinstance(schedule_cfg, Mapping):
        service.load_schedule(schedule_cfg)
    endpoint = args.endpoint or _resolve_endpoint(cfg)
    stop_event = install_signal_handlers()
    service.start()
    logger = logging.getLogger(__name__)
    logger.info("RS485 service started")

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(endpoint)
    logger.info("RS485 IPC REP bound on %s", endpoint)

    poller = zmq.Poller()
    poller.register(rep, zmq.POLLIN)

    while not stop_event.is_set():
        events = dict(poller.poll(timeout=200))
        if rep in events and events[rep] == zmq.POLLIN:
            payload = rep.recv()
            req_id = uuid.uuid4().hex[:8]
            recv_ts = time.monotonic()
            request_type = "unknown"
            try:
                request = rs485_request_from_json(payload)
                request_type = getattr(request, "type", "unknown")
                response = _handle_request(service, request, req_id=req_id)
            except Exception as exc:  # noqa: BLE001
                response = RS485Response(ok=False, error=str(exc))
            send_ts = time.monotonic()
            rep.send_string(response.model_dump_json(exclude_none=True))
            logger.info(
                "RS485 IPC request id=%s type=%s ok=%s ipc_ms=%.1f",
                req_id,
                request_type,
                response.ok,
                (send_ts - recv_ts) * 1000.0,
            )
        else:
            time.sleep(0.01)

    rep.close(0)
    service.stop()
    logger.info("RS485 service stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
