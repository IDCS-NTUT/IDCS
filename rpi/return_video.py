"""RPi return-video consumer for Jetson annotated RTP stream.

Consumes the same Jetson return stream used by ``pc.ui`` and renders to a
selected HDMI output via configurable GStreamer sink selection.
In addition to Jetson-drawn annotations, this script now draws:
    - Bottom-right status text (frame id, e2e latency, FPS estimate).
    - Optional MPC debug overlay (when control.debug_overlay.enabled).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

import cv2
import gi
import numpy as np
import zmq

from common.config_sync import (
    merge_config_maps,
    parse_config_text,
    read_snapshot,
    resolve_active_return_video_profile,
)
from common.control import ControlConfig, ControlConfigError, ControlDebugOverlayConfig
from common.schemas import ControlCmd, control_cmd_from_json, detection_msg_from_json

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore[attr-defined]

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class _OverlaySample:
    timestamp: float
    terms: Dict[str, float]
    term_directions: Dict[str, float]
    status: str
    u0: Optional[float]


class MpcDebugOverlay:
    """Render MPC term history on the return video."""

    TERM_COLOURS: Dict[str, Tuple[int, int, int]] = {
        "theta": (64, 192, 255),
        "theta_linear": (64, 128, 255),
        "omega": (0, 160, 255),
        "dtheta": (255, 140, 0),
        "dtheta_linear": (255, 110, 0),
        "effort": (144, 214, 72),
        "slew": (198, 118, 255),
        "slew_linear": (170, 90, 220),
        "slack": (96, 96, 96),
    }

    SIGNED_TERMS = {"theta_linear", "dtheta_linear", "slew_linear"}

    def __init__(self, cfg: ControlDebugOverlayConfig) -> None:
        self._cfg = cfg
        self._history: Dict[str, Deque[_OverlaySample]] = {
            "yaw": deque(),
            "pitch": deque(),
        }

    def ingest(self, cmd: ControlCmd, now: float) -> None:
        if not cmd.mpc:
            return
        for axis in ("yaw", "pitch"):
            diag = cmd.mpc.get(axis)
            if diag is None or not diag.terms:
                continue
            sample = _OverlaySample(
                timestamp=now,
                terms=dict(diag.terms),
                term_directions=dict(getattr(diag, "term_directions", {}) or {}),
                status=diag.status,
                u0=diag.u0,
            )
            self._history[axis].append(sample)
            self._prune_history(axis, now)

    def render(self, frame: np.ndarray, now: float) -> None:
        for axis in ("yaw", "pitch"):
            self._prune_history(axis, now)

        overlay_needed = any(self._history[axis] for axis in ("yaw", "pitch"))
        if not overlay_needed:
            return

        overlay = frame.copy()
        height, width = frame.shape[:2]
        section_height = self._cfg.bar_height_px + 18 * (len(self._cfg.show_terms) + 2)
        margin = 12
        spacing = 10

        total_height = 2 * section_height + spacing
        y_base = max(margin, height - margin - total_height)

        for idx, axis in enumerate(("yaw", "pitch")):
            sample = self._latest_sample(axis)
            if sample is None:
                continue
            y_origin = y_base + idx * (section_height + spacing)
            self._draw_axis_section(overlay, width, y_origin, axis, sample)

        cv2.addWeighted(overlay, self._cfg.opacity, frame, 1.0 - self._cfg.opacity, 0, frame)

    def _prune_history(self, axis: str, now: float) -> None:
        window = self._cfg.history_window_s
        dq = self._history[axis]
        while dq and (now - dq[0].timestamp) > window:
            dq.popleft()

    def _latest_sample(self, axis: str) -> Optional[_OverlaySample]:
        dq = self._history[axis]
        return dq[-1] if dq else None

    def _max_abs_term(self, axis: str) -> float:
        max_magnitude = 0.0
        for sample in self._history[axis]:
            for term in self._cfg.show_terms:
                max_magnitude = max(max_magnitude, abs(float(sample.terms.get(term, 0.0))))
        return max_magnitude

    def _draw_axis_section(
        self,
        overlay: np.ndarray,
        frame_width: int,
        y_origin: int,
        axis: str,
        sample: _OverlaySample,
    ) -> None:
        bar_width = min(int(frame_width * 0.32), 420)
        x_origin = 12
        bar_height = self._cfg.bar_height_px
        bar_rect = (x_origin, y_origin, x_origin + bar_width, y_origin + bar_height)
        center_y = int(round((bar_rect[1] + bar_rect[3]) / 2))

        cv2.rectangle(
            overlay,
            (bar_rect[0], bar_rect[1]),
            (bar_rect[2], bar_rect[3]),
            (32, 32, 32),
            thickness=cv2.FILLED,
        )
        cv2.rectangle(
            overlay,
            (bar_rect[0], bar_rect[1]),
            (bar_rect[2], bar_rect[3]),
            (64, 64, 64),
            thickness=1,
        )

        weights = [float(sample.terms.get(term, 0.0)) for term in self._cfg.show_terms]
        max_term = max(self._max_abs_term(axis), 1e-6)
        term_count = max(1, len(self._cfg.show_terms))
        slot_width = bar_width / term_count
        padding = min(6, int(slot_width * 0.15))

        for idx, (term, value) in enumerate(zip(self._cfg.show_terms, weights)):
            colour = self.TERM_COLOURS.get(term, (200, 200, 200))
            bar_center_x = int(round(x_origin + slot_width * idx + slot_width / 2))
            half_width = max(2, int((slot_width / 2) - padding))
            direction_hint = float(sample.term_directions.get(term, 0.0))
            if abs(direction_hint) > 0.0:
                directional = True
                direction_sign = 1.0 if direction_hint > 0.0 else -1.0
            elif term in self.SIGNED_TERMS and abs(value) > 0.0:
                directional = True
                direction_sign = 1.0 if value > 0.0 else -1.0
            else:
                directional = False
                direction_sign = 0.0
            scale = (bar_height / 2) / max_term
            magnitude = int(round(abs(value) * scale))
            if magnitude == 0:
                continue

            if directional and direction_sign < 0.0:
                y0, y1 = center_y, center_y + magnitude
            else:
                if directional:
                    y0, y1 = center_y - magnitude, center_y
                else:
                    y0, y1 = center_y - magnitude, center_y + magnitude
            x0, x1 = bar_center_x - half_width, bar_center_x + half_width
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))

            cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, thickness=cv2.FILLED)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), thickness=1)

        cv2.line(
            overlay,
            (bar_rect[0], center_y),
            (bar_rect[2], center_y),
            (90, 90, 90),
            thickness=1,
        )

        label = f"{axis.upper()}  {sample.status or 'n/a'}"
        if sample.u0 is not None:
            label += f"  u0={sample.u0:+0.2f}"
        self._draw_text(overlay, label, (x_origin, max(12, y_origin - 6)), 0.5, (255, 255, 255))

        text_y = y_origin + bar_height + 16
        for term, value in zip(self._cfg.show_terms, weights):
            colour = self.TERM_COLOURS.get(term, (200, 200, 200))
            text = f"{term}: {value:+0.2f}"
            self._draw_text(overlay, text, (x_origin, text_y), 0.45, colour)
            text_y += 16

    def _draw_text(
        self,
        overlay: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        scale: float,
        colour: Tuple[int, int, int],
    ) -> None:
        cv2.putText(overlay, text, origin, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, origin, FONT, scale, colour, 1, cv2.LINE_AA)


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
    parser.add_argument(
        "--wayland-fullscreen",
        action="store_true",
        help="Force waylandsink fullscreen mode",
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
    wayland_fullscreen: bool,
    hdmi_port: int | None,
    kmssink_connector_id: int | None,
    connector_map: dict[int, int],
) -> str:
    if sink_name == "waylandsink":
        if wayland_fullscreen:
            return "waylandsink fullscreen=true sync=false"
        return "waylandsink sync=false"

    if sink_name != "kmssink":
        return sink_name

    connector_id = kmssink_connector_id
    if connector_id is None and hdmi_port is not None:
        connector_id = connector_map.get(int(hdmi_port))

    if connector_id is None:
        return "kmssink sync=false"

    return f"kmssink connector-id={int(connector_id)} sync=false"


def _build_decode_pipeline(*, port: int, decoder_element: str) -> str:
    return (
        f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=97 ! "
        "rtpjitterbuffer latency=30 mode=0 drop-on-latency=true do-lost=true ! "
        "rtph264depay ! h264parse ! "
        "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        f"{decoder_element} ! videoconvert ! "
        "video/x-raw,format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
        "appsink name=sink drop=true sync=false max-buffers=1"
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


def resolve_return_timeout_ns(video_cfg: Mapping[str, Any]) -> int:
    timeout_ms = video_cfg.get("return_timeout_ms")
    if timeout_ms is not None:
        try:
            timeout_ms = float(timeout_ms)
        except (TypeError, ValueError) as exc:
            raise SystemExit("video.return_timeout_ms must be numeric") from exc
        if timeout_ms <= 0:
            raise SystemExit("video.return_timeout_ms must be positive")
        return int(round(timeout_ms * 1_000_000))

    fps = video_cfg.get("fps")
    if fps is None:
        return 50 * 1_000_000
    try:
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.fps must be numeric") from exc
    if fps <= 0:
        raise SystemExit("video.fps must be positive")
    frame_period_ms = 1000.0 / fps
    timeout_ms = frame_period_ms * 1.5
    return int(round(timeout_ms * 1_000_000))


def compute_e2e_ms(src_ts_ms: int) -> int:
    if not src_ts_ms:
        return 0

    # Support both historical monotonic timestamps and wall-clock epoch ms.
    if src_ts_ms >= 1_000_000_000_000:
        now_ms = int(time.time_ns() / 1_000_000)
    else:
        now_ms = int(time.monotonic_ns() / 1_000_000)

    delta = now_ms - int(src_ts_ms)
    if delta < 0 or delta > 600_000:
        return 0
    return int(delta)


class GstFrameReader:
    def __init__(self, *, port: int, decoder_element: str, pull_timeout_ns: int) -> None:
        pipeline = _build_decode_pipeline(port=port, decoder_element=decoder_element)
        self._pipeline = Gst.parse_launch(pipeline)
        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            raise RuntimeError("decode pipeline missing appsink named 'sink'")
        self._appsink.set_property("sync", False)
        self._appsink.set_property("max-buffers", 1)
        self._appsink.set_property("drop", True)
        self._bus = self._pipeline.get_bus()
        self._pipeline.set_state(Gst.State.PLAYING)
        self._pull_timeout_ns = pull_timeout_ns
        self._eos = False

    @property
    def eos(self) -> bool:
        return self._eos

    def _poll_bus(self) -> None:
        if self._bus is None or self._eos:
            return
        msg = self._bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg is None:
            return
        if msg.type == Gst.MessageType.EOS:
            self._eos = True
            return
        err, dbg = msg.parse_error()
        self._eos = True
        raise RuntimeError(f"decode pipeline error: {err} ({dbg})")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        self._poll_bus()
        if self._eos:
            return False, None
        sample = self._appsink.emit("try-pull-sample", self._pull_timeout_ns)
        if sample is None:
            return False, None
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None else None
        width = structure.get_value("width") if structure is not None else None
        height = structure.get_value("height") if structure is not None else None
        fmt = structure.get_value("format") if structure is not None else "BGR"
        success, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not success:
            return False, None
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
            if width and height:
                frame = frame.reshape((int(height), int(width), -1))
            fmt_name = str(fmt).upper()
            if fmt_name == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif fmt_name in {"RGBA", "RGBX"} and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif fmt_name == "BGRA" and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif fmt_name in {"BGRX", "BGRx"} and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            elif frame.ndim == 3 and frame.shape[2] == 4:
                # Unknown 4-channel layout; best-effort treat as BGRx-like.
                frame = frame[:, :, :3]
            frame = frame.copy()
        finally:
            buffer.unmap(mapinfo)
        return True, frame

    def release(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._pipeline = None
        self._appsink = None
        self._bus = None
        self._eos = True


class GstFrameSink:
    def __init__(self, *, sink_clause: str, pixel_format: str = "RGB") -> None:
        pipeline = (
            "appsrc name=src is-live=true block=false format=time do-timestamp=true ! "
            "videoconvert ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
            f"{sink_clause}"
        )
        pixel_format_norm = str(pixel_format).upper().strip()
        if pixel_format_norm not in {"RGB", "BGR"}:
            raise ValueError(f"unsupported sink pixel format: {pixel_format!r}")
        self._pixel_format = pixel_format_norm
        self._pipeline = Gst.parse_launch(pipeline)
        self._appsrc = self._pipeline.get_by_name("src")
        if self._appsrc is None:
            raise RuntimeError("render pipeline missing appsrc named 'src'")
        self._appsrc.set_property("is-live", True)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("do-timestamp", True)
        self._appsrc.set_property("block", False)
        self._bus = self._pipeline.get_bus()
        self._pipeline.set_state(Gst.State.PLAYING)
        self._configured_caps = False
        self._eos = False

    @property
    def eos(self) -> bool:
        return self._eos

    def _poll_bus(self) -> None:
        if self._bus is None or self._eos:
            return
        msg = self._bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg is None:
            return
        if msg.type == Gst.MessageType.EOS:
            self._eos = True
            return
        err, dbg = msg.parse_error()
        self._eos = True
        raise RuntimeError(f"render pipeline error: {err} ({dbg})")

    def push(self, frame: np.ndarray) -> None:
        self._poll_bus()
        if self._eos:
            return
        out_frame = frame
        if self._pixel_format == "RGB":
            out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not self._configured_caps:
            height, width = out_frame.shape[:2]
            caps = Gst.Caps.from_string(
                f"video/x-raw,format={self._pixel_format},width={int(width)},height={int(height)},framerate=0/1"
            )
            self._appsrc.set_property("caps", caps)
            self._configured_caps = True
        if not out_frame.flags["C_CONTIGUOUS"]:
            out_frame = np.ascontiguousarray(out_frame)
        payload = out_frame.tobytes()
        buffer = Gst.Buffer.new_allocate(None, len(payload), None)
        buffer.fill(0, payload)
        result = self._appsrc.emit("push-buffer", buffer)
        if result != Gst.FlowReturn.OK:
            raise RuntimeError(f"render pipeline push-buffer failed: {result!r}")

    def release(self) -> None:
        if self._appsrc is not None:
            try:
                self._appsrc.emit("end-of-stream")
            except Exception:
                pass
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._pipeline = None
        self._appsrc = None
        self._bus = None
        self._configured_caps = False
        self._eos = True


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

    video_cfg, active_profile = resolve_active_return_video_profile(cfg)

    rpi_cfg = cfg.get("rpi") if isinstance(cfg, Mapping) else None
    if not isinstance(rpi_cfg, Mapping):
        rpi_cfg = {}
    return_cfg_raw = rpi_cfg.get("return_video")
    return_cfg: Mapping[str, Any] = return_cfg_raw if isinstance(return_cfg_raw, Mapping) else {}

    sink_name = str(args.sink or return_cfg.get("sink") or "autovideosink").strip()
    wayland_fullscreen_cfg = bool(return_cfg.get("wayland_fullscreen", False))
    wayland_fullscreen = bool(args.wayland_fullscreen or wayland_fullscreen_cfg)
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
        wayland_fullscreen=wayland_fullscreen,
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

    log.info(
        "opening return feed on port %d (profile=%s sink=%s hdmi_port=%s)",
        return_port,
        active_profile or "legacy",
        sink_name,
        hdmi_port,
    )
    decode_timeout_ns = resolve_return_timeout_ns(video_cfg)
    log.debug(
        "decode pipeline: %s",
        _build_decode_pipeline(port=return_port, decoder_element=decoder_element),
    )
    log.debug(
        "render pipeline: %s",
        f"appsrc name=src ... ! videoconvert ! queue leaky=downstream ... ! {sink_clause}",
    )
    sink_pixel_format_raw = return_cfg.get("overlay_output_format", "RGB")
    sink_pixel_format = str(sink_pixel_format_raw).upper().strip()
    if sink_pixel_format not in {"RGB", "BGR"}:
        log.warning(
            "invalid rpi.return_video.overlay_output_format=%r; using RGB",
            sink_pixel_format_raw,
        )
        sink_pixel_format = "RGB"
    log.info("overlay sink pixel format: %s", sink_pixel_format)

    try:
        video_w = int(video_cfg["width"])
        video_h = int(video_cfg["height"])
    except Exception:
        video_w, video_h = 1280, 720
        log.warning(
            "video.width/video.height missing or invalid; defaulting overlay geometry to %dx%d",
            video_w,
            video_h,
        )

    overlay_renderer: Optional[MpcDebugOverlay] = None
    try:
        control_cfg = ControlConfig.from_raw_config(cfg, (video_w, video_h))
    except ControlConfigError as exc:
        log.warning("MPC debug overlay disabled: invalid control config (%s)", exc)
    else:
        if control_cfg.debug_overlay.enabled:
            overlay_renderer = MpcDebugOverlay(control_cfg.debug_overlay)
            log.info(
                "MPC overlay enabled (terms=%s, window=%.1fs)",
                ",".join(control_cfg.debug_overlay.show_terms),
                control_cfg.debug_overlay.history_window_s,
            )

    ctx = zmq.Context()
    poller = zmq.Poller()
    result_sub: Optional[zmq.Socket] = None
    control_sub: Optional[zmq.Socket] = None

    results_endpoint = net_cfg.get("zmq_results")
    if isinstance(results_endpoint, str) and results_endpoint.strip():
        result_sub = ctx.socket(zmq.SUB)
        result_sub.setsockopt(zmq.CONFLATE, 1)
        result_sub.setsockopt(zmq.RCVHWM, 1)
        result_sub.setsockopt(zmq.LINGER, 0)
        result_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        result_sub.connect(results_endpoint)
        poller.register(result_sub, zmq.POLLIN)
    else:
        log.warning("status overlay limited: net.zmq_results is not configured")

    if overlay_renderer is not None:
        control_endpoint = net_cfg.get("zmq_control")
        if isinstance(control_endpoint, str) and control_endpoint.strip():
            control_sub = ctx.socket(zmq.SUB)
            control_sub.setsockopt(zmq.CONFLATE, 1)
            control_sub.setsockopt(zmq.RCVHWM, 1)
            control_sub.setsockopt(zmq.LINGER, 0)
            control_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            control_sub.connect(control_endpoint)
            poller.register(control_sub, zmq.POLLIN)
        else:
            overlay_renderer = None
            log.warning("MPC debug overlay disabled: net.zmq_control is not configured")

    reader = GstFrameReader(
        port=return_port,
        decoder_element=decoder_element,
        pull_timeout_ns=decode_timeout_ns,
    )
    writer = GstFrameSink(sink_clause=sink_clause, pixel_format=sink_pixel_format)

    last_frame_id = -1
    last_e2e_ms = 0
    last_draw = time.time()
    fps_est = 0.0

    try:
        while not stop_event.is_set():
            ok, frame = reader.read()
            if reader.eos or writer.eos:
                log.info("return stream ended")
                break
            if not ok or frame is None:
                events = dict(poller.poll(timeout=10))
                if result_sub is not None and events.get(result_sub) == zmq.POLLIN:
                    payload = result_sub.recv()
                    msg = detection_msg_from_json(payload)
                    last_frame_id = msg.frame_id
                    last_e2e_ms = compute_e2e_ms(msg.src_ts_ms)
                if control_sub is not None and events.get(control_sub) == zmq.POLLIN and overlay_renderer is not None:
                    payload = control_sub.recv()
                    try:
                        cmd = control_cmd_from_json(payload)
                    except Exception as exc:
                        log.debug("failed to decode ControlCmd: %s", exc)
                    else:
                        overlay_renderer.ingest(cmd, time.time())
                continue

            events = dict(poller.poll(timeout=0))
            if result_sub is not None and events.get(result_sub) == zmq.POLLIN:
                payload = result_sub.recv()
                msg = detection_msg_from_json(payload)
                last_frame_id = msg.frame_id
                last_e2e_ms = compute_e2e_ms(msg.src_ts_ms)
            if control_sub is not None and events.get(control_sub) == zmq.POLLIN and overlay_renderer is not None:
                payload = control_sub.recv()
                try:
                    cmd = control_cmd_from_json(payload)
                except Exception as exc:
                    log.debug("failed to decode ControlCmd: %s", exc)
                else:
                    overlay_renderer.ingest(cmd, time.time())

            now = time.time()
            inst = 1.0 / max(1e-6, (now - last_draw))
            last_draw = now
            fps_est = inst if fps_est == 0.0 else (0.9 * fps_est + 0.1 * inst)
            status = (
                f"frame #{last_frame_id if last_frame_id >= 0 else '-'}  "
                f"e2e {int(last_e2e_ms)} ms  ~{fps_est:4.1f} fps"
            )

            if overlay_renderer is not None:
                overlay_renderer.render(frame, now)

            scale = 0.5
            thickness = 1
            margin = 8
            text_colour = (255, 255, 255)
            text_size, baseline = cv2.getTextSize(status, FONT, scale, thickness)
            text_w, text_h = text_size
            h, w = frame.shape[:2]
            origin_x = max(margin + 4, w - margin - text_w)
            origin_y = max(margin + text_h, h - margin - baseline)
            rect_tl = (int(origin_x - 4), int(max(0, origin_y - text_h - 4)))
            rect_br = (
                int(min(w - 1, origin_x + text_w + 4)),
                int(min(h - 1, origin_y + baseline + 4)),
            )
            cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), thickness=cv2.FILLED)
            cv2.putText(
                frame,
                status,
                (int(origin_x), int(origin_y)),
                FONT,
                scale,
                text_colour,
                thickness,
                cv2.LINE_AA,
            )

            writer.push(frame)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            reader.release()
        except Exception:
            pass
        try:
            writer.release()
        except Exception:
            pass
        if result_sub is not None:
            try:
                result_sub.close(0)
            except Exception:
                pass
        if control_sub is not None:
            try:
                control_sub.close(0)
            except Exception:
                pass
        try:
            ctx.term()
        except Exception:
            pass
        time.sleep(0.05)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
