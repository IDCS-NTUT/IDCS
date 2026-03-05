#!/usr/bin/env python3
"""Wait for Jetson config-sync readiness and emit stream settings.

First-pass RPi gate:
- Blocks until Jetson's config-sync REP endpoint responds for required config IDs.
- Fetches server-side config content for those IDs.
- Resolves effective video/net settings and prints shell assignments.

This allows the RPi streaming service to start only after Jetson is up and to
match the active Jetson configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml
import zmq

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class SyncGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteConfig:
    config_id: str
    text: str
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Jetson config_sync endpoint (e.g. tcp://192.168.55.1:5560)",
    )
    parser.add_argument(
        "--config-id",
        action="append",
        dest="config_ids",
        default=None,
        help="Config IDs to sync (repeatable). Defaults to dev.yaml + dev_extra.yaml",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Maximum seconds to wait for readiness",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=1.0,
        help="Seconds between retries",
    )
    parser.add_argument(
        "--shell-output",
        action="store_true",
        help="Print KEY='value' lines suitable for eval in shell",
    )
    parser.add_argument(
        "--peer-id",
        default="rpi2",
        help="Peer identity sent to Jetson config_sync server",
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
        raise SyncGateError(f"invalid response payload type: {type(payload)!r}")
    return payload


def _request_config(
    endpoint: str,
    config_id: str,
    per_try_timeout_s: float,
    peer_id: str,
) -> RemoteConfig:
    ctx = zmq.Context.instance()
    deadline = time.monotonic() + per_try_timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for config_sync response")

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

            reply = _recv_json(req, remaining)
            status = str(reply.get("status") or "")

            if status == "retry_later":
                time.sleep(min(0.2, max(0.05, remaining / 4.0)))
                continue

            if status == "need_payload":
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
                ack = _recv_json(req, remaining)
                status = str(ack.get("status") or "")
                if status == "retry_later":
                    time.sleep(min(0.2, max(0.05, remaining / 4.0)))
                    continue
                if status != "ok":
                    raise SyncGateError(
                        f"unexpected server status for {config_id!r}: {status!r}"
                    )
                reply = ack

            if status != "ok":
                raise SyncGateError(f"unexpected server status for {config_id!r}: {status!r}")

            if reply.get("config_id") != config_id:
                raise SyncGateError(
                    f"mismatched config_id in response: {reply.get('config_id')!r} != {config_id!r}"
                )

            winner = str(reply.get("winner", ""))
            metadata = dict(reply.get("metadata") or {})

            if winner == "server":
                text = str(reply.get("content") or "")
                return RemoteConfig(config_id=config_id, text=text, metadata=metadata)

            if winner in {"equal", "client"}:
                text = str(reply.get("content") or "")
                return RemoteConfig(config_id=config_id, text=text, metadata=metadata)

            raise SyncGateError(f"unexpected winner for {config_id!r}: {winner!r}")


def _merge_top_level(config_texts: Iterable[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for text in config_texts:
        data = yaml.safe_load(text) if text else {}
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise SyncGateError("config payload must be a top-level mapping")
        merged.update(data)
    return merged


def _resolve_video_cfg(cfg: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    video = cfg.get("video")
    if not isinstance(video, Mapping):
        raise SyncGateError("missing video section")

    profiles = video.get("profiles")
    if isinstance(profiles, Mapping) and profiles:
        active = video.get("active_profile")
        if not isinstance(active, str) or not active:
            raise SyncGateError("video.active_profile must be set when profiles are used")
        profile_cfg = profiles.get(active)
        if not isinstance(profile_cfg, Mapping):
            raise SyncGateError(f"video.profiles[{active!r}] must be a mapping")
        base = {k: v for k, v in video.items() if k not in {"profiles", "active_profile"}}
        merged = dict(base)
        merged.update(profile_cfg)
        return merged, active

    return dict(video), None


def _extract_settings(cfg: Mapping[str, Any]) -> Dict[str, str]:
    net = cfg.get("net")
    if not isinstance(net, Mapping):
        raise SyncGateError("missing net section")

    jetson_ip = str(net.get("jetson_ip") or "").strip()
    if not jetson_ip:
        raise SyncGateError("missing net.jetson_ip")

    rtp_port_raw = net.get("rtp_port")
    if rtp_port_raw is None:
        raise SyncGateError("missing net.rtp_port")
    rtp_port = int(rtp_port_raw)

    header_push = str(net.get("header_push") or "").strip()
    if not header_push.startswith("tcp://"):
        raise SyncGateError("missing/invalid net.header_push")
    try:
        header_push_port = int(header_push.rsplit(":", 1)[1])
    except Exception as exc:  # noqa: BLE001
        raise SyncGateError("invalid net.header_push port") from exc

    video_cfg, _ = _resolve_video_cfg(cfg)
    width_raw = video_cfg.get("width")
    height_raw = video_cfg.get("height")
    fps_raw = video_cfg.get("fps")
    if width_raw is None or height_raw is None or fps_raw is None:
        raise SyncGateError("missing video.width/video.height/video.fps")
    width = int(width_raw)
    height = int(height_raw)
    fps = float(fps_raw)

    bitrate_raw = video_cfg.get("bitrate_kbps", 4000)
    bitrate_kbps = int(bitrate_raw)

    settings = {
        "STREAM_JETSON_IP": jetson_ip,
        "STREAM_JETSON_PORT": str(rtp_port),
        "STREAM_WIDTH": str(width),
        "STREAM_HEIGHT": str(height),
        "STREAM_FPS": str(fps),
        "STREAM_BITRATE_KBPS": str(bitrate_kbps),
        "STREAM_HEADER_PUSH_PORT": str(header_push_port),
    }

    camera = cfg.get("camera")
    if isinstance(camera, Mapping):
        libcamera = camera.get("libcamera")
        if isinstance(libcamera, Mapping):
            tuning_file = libcamera.get("tuning_file")
            if tuning_file is not None and str(tuning_file).strip():
                settings["STREAM_CAM_TUNING_FILE"] = str(tuning_file).strip()

            shutter_us = libcamera.get("shutter_us")
            if shutter_us is not None and str(shutter_us).strip():
                settings["STREAM_CAM_SHUTTER_US"] = str(shutter_us).strip()

            gain = libcamera.get("gain")
            if gain is not None and str(gain).strip():
                settings["STREAM_CAM_GAIN"] = str(gain).strip()

    return settings


def _wait_for_sync(
    endpoint: str,
    config_ids: Iterable[str],
    timeout_s: float,
    retry_interval_s: float,
    peer_id: str,
) -> Dict[str, str]:
    deadline = time.monotonic() + timeout_s
    per_try_timeout = min(2.0, max(0.2, retry_interval_s))
    last_error: Optional[Exception] = None

    while time.monotonic() < deadline:
        try:
            configs = [
                _request_config(
                    endpoint,
                    config_id=config_id,
                    per_try_timeout_s=per_try_timeout,
                    peer_id=peer_id,
                )
                for config_id in config_ids
            ]
            merged = _merge_top_level(cfg.text for cfg in configs)
            return _extract_settings(merged)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(retry_interval_s)

    raise SyncGateError(f"timeout waiting for config sync readiness: {last_error}")


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    args = parse_args()
    config_ids = args.config_ids or ["dev.yaml", "dev_extra.yaml"]
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if args.retry_interval <= 0:
        raise SystemExit("--retry-interval must be > 0")

    settings = _wait_for_sync(
        endpoint=args.endpoint,
        config_ids=config_ids,
        timeout_s=float(args.timeout),
        retry_interval_s=float(args.retry_interval),
        peer_id=str(args.peer_id),
    )

    if args.shell_output:
        for key in sorted(settings):
            print(f"{key}={_shell_quote(settings[key])}")
    else:
        print(json.dumps(settings, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
