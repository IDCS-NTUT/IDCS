"""RPi return-video consumer for Jetson annotated RTP stream.

Consumes the same Jetson return stream used by ``pc.ui`` and renders to a
selected HDMI output via configurable GStreamer sink selection.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any, Mapping

import gi

from common.config_sync import (
    merge_config_maps,
    parse_config_text,
    read_snapshot,
    resolve_active_return_video_profile,
)

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore[attr-defined]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dev.yaml", help="Base YAML config path")
    parser.add_argument(
        "--config-extra",
        default="configs/dev_extra.yaml",
        help="Optional second YAML config merged over --config",
    )
    parser.add_argument("--sink", default=None, help="GStreamer sink type (autovideosink/kmssink)")
    parser.add_argument(
        "--hdmi-port",
        type=int,
        default=None,
        help="Logical HDMI port index for kmssink connector map lookup",
    )
    parser.add_argument(
        "--kmssink-connector-id",
        type=int,
        default=None,
        help="Explicit kmssink connector-id override",
    )
    parser.add_argument(
        "--wayland-display",
        default=None,
        help="Override WAYLAND_DISPLAY for desktop sinks",
    )
    parser.add_argument(
        "--xdg-runtime-dir",
        default=None,
        help="Override XDG_RUNTIME_DIR for desktop sinks",
    )
    parser.add_argument(
        "--display",
        default=None,
        help="Override DISPLAY for X11 sinks",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def install_stop_event() -> Event:
    stop_event = Event()

    def _handler(signum, _frame):
        stop_event.set()
        return None

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return stop_event


def _load_cfg(config_path: Path, extra_path: Path | None) -> Mapping[str, Any]:
    config_paths = [config_path] + ([extra_path] if extra_path else [])
    snapshots = {path: read_snapshot(path) for path in config_paths}
    return merge_config_maps(*(parse_config_text(snapshot.text, str(path)) for path, snapshot in snapshots.items()))


def _parse_connector_map(raw: Any) -> dict[int, int]:
    if not isinstance(raw, Mapping):
        return {}
    result: dict[int, int] = {}
    for key, value in raw.items():
        try:
            k = int(key)
            v = int(value)
        except Exception:
            continue
        result[k] = v
    return result


def _resolve_sink_clause(
    *,
    sink_name: str,
    hdmi_port: int | None,
    kmssink_connector_id: int | None,
    connector_map: dict[int, int],
) -> str:
    if sink_name == "waylandsink":
        return "waylandsink fullscreen=true"

    if sink_name != "kmssink":
        return sink_name

    connector_id = kmssink_connector_id
    if connector_id is None and hdmi_port is not None:
        connector_id = connector_map.get(int(hdmi_port))

    if connector_id is None:
        return "kmssink sync=false"

    return f"kmssink connector-id={int(connector_id)} sync=false"


def _build_pipeline(*, port: int, sink_clause: str, decoder_element: str) -> str:
    return (
        f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=97 ! "
        "rtpjitterbuffer latency=0 mode=0 drop-on-latency=true do-lost=true ! "
        "rtph264depay ! h264parse ! "
        "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        f"{decoder_element} ! videoconvert ! "
        "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        f"{sink_clause}"
    )


def _select_decoder_element() -> str:
    # Prefer Pi hardware decode when available; fall back for compatibility.
    if Gst.ElementFactory.find("v4l2h264dec") is not None:
        return "v4l2h264dec"
    return "avdec_h264"


def _session_env_from_config(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    parsed: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        parsed[str(key)] = str(value)
    return parsed


def _apply_sink_session_env(
    *,
    sink_name: str,
    return_cfg: Mapping[str, Any],
    wayland_display_arg: str | None,
    xdg_runtime_dir_arg: str | None,
    display_arg: str | None,
    log: logging.Logger,
) -> None:
    env_updates = _session_env_from_config(return_cfg.get("session_env"))
    if wayland_display_arg is not None:
        env_updates["WAYLAND_DISPLAY"] = wayland_display_arg
    if xdg_runtime_dir_arg is not None:
        env_updates["XDG_RUNTIME_DIR"] = xdg_runtime_dir_arg
    if display_arg is not None:
        env_updates["DISPLAY"] = display_arg

    for key, value in env_updates.items():
        if value.strip():
            os.environ[key] = value

    if sink_name in {"waylandsink", "autovideosink"}:
        missing_wayland = [name for name in ("WAYLAND_DISPLAY", "XDG_RUNTIME_DIR") if not os.environ.get(name)]
        if missing_wayland:
            log.warning(
                "desktop sink may fail; missing environment variable(s): %s "
                "(set rpi.return_video.session_env in config or pass CLI overrides)",
                ", ".join(missing_wayland),
            )
    if sink_name in {"ximagesink", "xvimagesink"} and not os.environ.get("DISPLAY"):
        log.warning(
            "x11 sink may fail; missing DISPLAY "
            "(set rpi.return_video.session_env.DISPLAY in config or pass --display)",
        )


def main() -> int:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("rpi.return_video")

    cfg = _load_cfg(Path(args.config), Path(args.config_extra) if args.config_extra else None)

    net_cfg = cfg.get("net") if isinstance(cfg, Mapping) else None
    if not isinstance(net_cfg, Mapping):
        raise SystemExit("config missing net section")

    try:
        return_port = int(net_cfg["rtp_return_port"])
    except KeyError as exc:
        raise SystemExit("config missing net.rtp_return_port") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit("net.rtp_return_port must be an integer") from exc

    _video_cfg, active_profile = resolve_active_return_video_profile(cfg)

    rpi_cfg = cfg.get("rpi") if isinstance(cfg, Mapping) else None
    if not isinstance(rpi_cfg, Mapping):
        rpi_cfg = {}
    return_cfg_raw = rpi_cfg.get("return_video")
    return_cfg: Mapping[str, Any] = return_cfg_raw if isinstance(return_cfg_raw, Mapping) else {}

    sink_name = str(args.sink or return_cfg.get("sink") or "autovideosink").strip()
    hdmi_port = args.hdmi_port
    hdmi_port_cfg = return_cfg.get("hdmi_port")
    if hdmi_port is None and hdmi_port_cfg is not None:
        hdmi_port = int(hdmi_port_cfg)

    connector_id = args.kmssink_connector_id
    connector_id_cfg = return_cfg.get("kmssink_connector_id")
    if connector_id is None and connector_id_cfg is not None:
        connector_id = int(connector_id_cfg)

    connector_map = _parse_connector_map(return_cfg.get("kmssink_connector_map"))

    _apply_sink_session_env(
        sink_name=sink_name,
        return_cfg=return_cfg,
        wayland_display_arg=args.wayland_display,
        xdg_runtime_dir_arg=args.xdg_runtime_dir,
        display_arg=args.display,
        log=log,
    )

    sink_clause = _resolve_sink_clause(
        sink_name=sink_name,
        hdmi_port=hdmi_port,
        kmssink_connector_id=connector_id,
        connector_map=connector_map,
    )

    Gst.init(None)
    stop_event = install_stop_event()
    decoder_element = _select_decoder_element()
    if decoder_element == "v4l2h264dec":
        log.info("using decoder: v4l2h264dec (hardware)")
    else:
        log.warning("v4l2h264dec unavailable; falling back to avdec_h264")

    pipeline = _build_pipeline(
        port=return_port,
        sink_clause=sink_clause,
        decoder_element=decoder_element,
    )

    log.info(
        "opening return feed on port %d (profile=%s sink=%s hdmi_port=%s)",
        return_port,
        active_profile or "legacy",
        sink_name,
        hdmi_port,
    )
    log.debug("pipeline: %s", pipeline)

    player = Gst.parse_launch(pipeline)
    bus = player.get_bus()

    player.set_state(Gst.State.PLAYING)
    try:
        while not stop_event.is_set():
            msg = bus.timed_pop_filtered(
                int(100 * Gst.MSECOND),
                Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.STATE_CHANGED,
            )
            if msg is None:
                continue
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                raise RuntimeError(f"GStreamer error: {err} ({dbg})")
            if msg.type == Gst.MessageType.EOS:
                log.info("return stream ended")
                break
    except KeyboardInterrupt:
        pass
    finally:
        player.set_state(Gst.State.NULL)
        time.sleep(0.05)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
