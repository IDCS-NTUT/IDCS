#!/usr/bin/env python3
"""Long-running config-sync companion for RPi2.

Keeps re-handshaking with Jetson config_sync while streaming is active so
Jetson restarts can still satisfy required peer sync checks (peer_id=rpi2).
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Mapping

import zmq

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Jetson config_sync endpoint")
    parser.add_argument(
        "--config-id",
        action="append",
        dest="config_ids",
        default=None,
        help="Config IDs to handshake (repeatable). Defaults to dev.yaml + dev_extra.yaml",
    )
    parser.add_argument("--peer-id", default="rpi2", help="Peer identity")
    parser.add_argument("--retry-interval", type=float, default=1.0, help="Retry delay seconds")
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=5.0,
        help="Delay between successful sync rounds",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=2.0,
        help="Per-request response timeout seconds",
    )
    return parser.parse_args()


def _recv_json(req: zmq.Socket, timeout_s: float) -> Mapping[str, Any]:
    poller = zmq.Poller()
    poller.register(req, zmq.POLLIN)
    timeout_ms = max(1, int(round(timeout_s * 1000)))
    events = dict(poller.poll(timeout_ms))
    if events.get(req) != zmq.POLLIN:
        raise TimeoutError("timed out waiting for config_sync response")
    payload = req.recv_json()
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"invalid config_sync payload type: {type(payload)!r}")
    return payload


def _handshake_once(
    endpoint: str,
    *,
    config_id: str,
    peer_id: str,
    request_timeout: float,
) -> None:
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.REQ) as req:
        req.setsockopt(zmq.LINGER, 0)
        req.connect(endpoint)

        req.send_json(
            {
                "type": "metadata",
                "config_id": config_id,
                "peer_id": peer_id,
                "metadata": {
                    "mtime_ns": 0,
                    "size": 0,
                    "sha256": EMPTY_SHA256,
                },
            }
        )

        reply = _recv_json(req, request_timeout)
        status = str(reply.get("status", ""))

        if status == "retry_later":
            return

        if status == "need_payload":
            # Extremely unlikely for empty metadata, but complete the protocol.
            req.send_json(
                {
                    "type": "content",
                    "config_id": config_id,
                    "peer_id": peer_id,
                    "metadata": {
                        "mtime_ns": 0,
                        "size": 0,
                        "sha256": EMPTY_SHA256,
                    },
                    "content": "",
                }
            )
            _recv_json(req, request_timeout)
            return

        if status != "ok":
            raise RuntimeError(f"unexpected sync status for {config_id}: {status!r}")


def main() -> int:
    args = parse_args()
    config_ids = args.config_ids or ["dev.yaml", "dev_extra.yaml"]

    if args.retry_interval <= 0:
        raise SystemExit("--retry-interval must be > 0")
    if args.heartbeat_interval <= 0:
        raise SystemExit("--heartbeat-interval must be > 0")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be > 0")

    print(
        "[rpi2-sync] companion running: endpoint=%s peer_id=%s configs=%s"
        % (args.endpoint, args.peer_id, ",".join(config_ids))
    )

    while True:
        try:
            for config_id in config_ids:
                _handshake_once(
                    args.endpoint,
                    config_id=config_id,
                    peer_id=args.peer_id,
                    request_timeout=args.request_timeout,
                )
            time.sleep(args.heartbeat_interval)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[rpi2-sync] sync round failed: {exc}")
            time.sleep(args.retry_interval)


if __name__ == "__main__":
    raise SystemExit(main())
