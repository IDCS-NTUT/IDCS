"""Run the dedicated RS485 drain service on the Jetson."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import zmq

from common.config_sync import parse_config_text, read_snapshot
from common.rs485 import (
    RS485CommandRequest,
    RS485CommandResponse,
    RS485HealthResponse,
    RS485HistoryResponse,
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
    )
    return RS485Service(config)


def _resolve_endpoint(cfg: Mapping[str, Any]) -> str:
    gimbal_cfg = cfg.get("gimbal") or {}
    endpoint = gimbal_cfg.get("rs485_endpoint", "tcp://127.0.0.1:5559")
    if not isinstance(endpoint, str) or not endpoint:
        raise SystemExit("gimbal.rs485_endpoint must be a non-empty string")
    return endpoint


def _handle_request(service: RS485Service, request) -> RS485Response:
    if isinstance(request, RS485CommandRequest):
        try:
            data = request.data or []
            resp = service._bus.send_command(
                request.addr,
                request.func,
                data,
                response_expected=request.response_expected,
                expected_response_len=request.expected_response_len,
            )
            return RS485CommandResponse(payload=list(resp), cached=False)
        except Exception as exc:  # noqa: BLE001
            return RS485CommandResponse(ok=False, error=str(exc), payload=None, cached=False)

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
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    cfg = _load_config(Path(args.config))
    service = _build_service(cfg)
    endpoint = _resolve_endpoint(cfg)
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
            try:
                request = rs485_request_from_json(payload)
                response = _handle_request(service, request)
            except Exception as exc:  # noqa: BLE001
                response = RS485Response(ok=False, error=str(exc))
            rep.send_string(response.model_dump_json(exclude_none=True))
        else:
            time.sleep(0.01)

    rep.close(0)
    service.stop()
    logger.info("RS485 service stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
