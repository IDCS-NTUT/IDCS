"""Publish joystick-derived pan/tilt rates for the gimbal bridge to consume."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import zmq
import yaml

from common.schemas import ControlCmd
from rpi.joystick_common import JoystickReader, log_sample, load_config, map_adc_to_rate


_LOG = logging.getLogger(__name__)


@dataclass
class PublishConfig:
    endpoint: str = "tcp://0.0.0.0:5565"
    bind: bool = True


def _load_publish_config(path: Path) -> PublishConfig:
    data = yaml.safe_load(path.read_text()) or {}
    publish_cfg = data.get("manual_publish") if isinstance(data, dict) else {}
    if publish_cfg is None:
        publish_cfg = {}

    endpoint = publish_cfg.get("endpoint", PublishConfig.endpoint)
    bind = bool(publish_cfg.get("bind", PublishConfig.bind))
    return PublishConfig(endpoint=str(endpoint), bind=bind)


def _make_pub(endpoint: str, bind: bool) -> zmq.Socket:
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.setsockopt(zmq.LINGER, 0)
    if bind:
        pub.bind(endpoint)
        _LOG.info("publishing joystick ControlCmd on %s (bind)", endpoint)
    else:
        pub.connect(endpoint)
        _LOG.info("publishing joystick ControlCmd to %s (connect)", endpoint)
    return pub


def _build_control_cmd(frame_id: int, yaw_rate: float, pitch_rate: float) -> ControlCmd:
    now_ms = int(time.monotonic_ns() / 1e6)
    return ControlCmd(
        frame_id=frame_id,
        src_ts_ms=now_ms,
        cmd_ts_ms=now_ms,
        target_ok=True,
        target_uv=(0.0, 0.0),
        err_uv=(0.0, 0.0),
        err_rad=(0.0, 0.0),
        pan_rate_cmd=yaw_rate,
        tilt_rate_cmd=pitch_rate,
        controller_mode="manual",
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="rpi/manual_config.yaml", type=Path, help="Path to YAML config")
    ap.add_argument("--endpoint", type=str, help="Override manual_publish.endpoint")
    ap.add_argument(
        "--connect",
        action="store_true",
        help="Connect to the endpoint instead of binding (overrides config)",
    )
    ap.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    _, joystick_cfg = load_config(args.config)
    publish_cfg = _load_publish_config(args.config)
    endpoint = args.endpoint or publish_cfg.endpoint
    bind = not args.connect and publish_cfg.bind

    joystick = JoystickReader(joystick_cfg)
    pub = _make_pub(endpoint, bind)
    poll_interval = max(joystick_cfg.poll_interval_s, 0.01)

    _LOG.info(
        "starting joystick publisher | poll=%.2f s deadzone=%d max_rate yaw=%.2f pitch=%.2f",
        poll_interval,
        joystick_cfg.deadzone,
        joystick_cfg.max_rate_yaw,
        joystick_cfg.max_rate_pitch,
    )

    frame_id = 0
    try:
        while True:
            raw_yaw = joystick.read_channel(joystick_cfg.yaw_channel)
            raw_pitch = joystick.read_channel(joystick_cfg.pitch_channel)

            yaw_rate = map_adc_to_rate(raw_yaw, joystick_cfg.max_rate_yaw, joystick_cfg.deadzone)
            pitch_rate = map_adc_to_rate(raw_pitch, joystick_cfg.max_rate_pitch, joystick_cfg.deadzone)

            cmd = _build_control_cmd(frame_id, yaw_rate, pitch_rate)
            pub.send_string(cmd.model_dump_json(exclude_none=True))
            log_sample(raw_yaw, raw_pitch, yaw_rate, pitch_rate)

            frame_id += 1
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        _LOG.info("joystick publisher interrupted")
    finally:
        try:
            pub.close(linger=0)
        except Exception:  # noqa: BLE001
            _LOG.debug("failed to close publisher", exc_info=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
