"""Run the dedicated RS485 drain service on the Jetson."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping

from common.config_sync import parse_config_text, read_snapshot
from common.rs485 import RS485Service, RS485ServiceConfig
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dev.yaml", help="Path to YAML config")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    cfg = _load_config(Path(args.config))
    service = _build_service(cfg)
    stop_event = install_signal_handlers()
    service.start()
    logging.getLogger(__name__).info("RS485 service started")

    stop_event.wait()
    service.stop()
    logging.getLogger(__name__).info("RS485 service stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
