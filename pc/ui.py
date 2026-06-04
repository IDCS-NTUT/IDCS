"""PC UI entrypoint.

Responsibilities:
    - Subscribe to Jetson detection metadata over ZMQ and display status overlays.
    - Receive the Jetson return video over RTP/UDP and present the annotated feed.
    - Optionally subscribe to control messages for MPC debug visualization.

Required ZMQ endpoints (from the config file):
    - net.zmq_results: SUB socket for DetectionMsg payloads.
    - net.zmq_control: SUB socket for ControlCmd payloads (required only when the
      MPC debug overlay is enabled).

Expected message types:
    - DetectionMsg via common.schemas.detection_msg_from_json().
    - ControlCmd via common.schemas.control_cmd_from_json().

Overlay configuration:
    - ControlConfig.from_raw_config() derives the debug overlay settings from
      the config (control.debug_overlay.*). These settings determine which MPC
      terms are shown, how far back to retain samples, and the rendering style.
"""
import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import zmq

try:
    import gi
except ModuleNotFoundError:
    gi = None
    Gst = None
else:
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

from common.config_sync import (
    ConfigSyncError,
    acquire_config_sync_lock,
    expand_config_paths,
    load_sync_marker,
    merge_config_maps,
    parse_config_text,
    read_snapshot,
    resolve_active_video_profile,
    resolve_config_sync_endpoint,
    sync_as_client,
    write_sync_marker,
    request_startup_state,
)
from common.control import (
    ControlConfig,
    ControlConfigError,
    ControlDebugOverlayConfig,
    LaserConfigError,
    LaserMountConfig,
)
from common.schemas import ControlCmd, detection_msg_from_json, control_cmd_from_json
from common.shutdown import install_signal_handlers

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bind_zmq_to_device_if_configured(socket: zmq.Socket, iface: Optional[str]) -> None:
    """Best-effort Linux interface bind for libzmq builds that expose it."""
    if not iface:
        return
    option = getattr(zmq, "BINDTODEVICE", None)
    if option is None:
        print("[ui][WARN] net.pc_iface ignored: pyzmq/libzmq lacks BINDTODEVICE")
        return
    try:
        socket.setsockopt_string(option, iface)
    except Exception as exc:
        print(f"[ui][WARN] net.pc_iface={iface!r} bind failed: {exc}")


@dataclass
class _OverlaySample:
    timestamp: float
    terms: Dict[str, float]
    term_directions: Dict[str, float]
    status: str
    u0: Optional[float]


@dataclass(frozen=True)
class _OverlayLayout:
    frame_width: int
    frame_height: int
    x0: int
    y0: int
    x1: int
    y1: int
    x_origin: int
    y_base: int
    bar_width: int
    section_height: int
    spacing: int


class MpcDebugOverlay:
    """Render MPC term history on the return video.

    The overlay draws one section per axis (yaw/pitch) using the latest ControlCmd
    sample in a rolling history window. Each bar corresponds to a term listed in
    ControlDebugOverlayConfig.show_terms (for example: theta, theta_linear,
    omega, dtheta, dtheta_linear, effort, slew, slew_linear, slack).
    Samples older than ControlDebugOverlayConfig.history_window_s are pruned, and
    the bar scale is normalized to the maximum absolute term magnitude seen in
    the remaining window for each axis.
    """

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

    def __init__(self, cfg: ControlDebugOverlayConfig) -> None:
        self._cfg = cfg
        self._render_interval_frames = max(1, int(cfg.render_interval_frames))
        self._history: Dict[str, Deque[_OverlaySample]] = {
            "yaw": deque(),
            "pitch": deque(),
        }
        self._dirty = True
        self._frames_since_render = self._render_interval_frames
        self._static_layout: Optional[_OverlayLayout] = None
        self._static_layer: Optional[np.ndarray] = None
        self._static_mask: Optional[np.ndarray] = None
        self._cached_layout: Optional[_OverlayLayout] = None
        self._cached_layer: Optional[np.ndarray] = None
        self._cached_mask: Optional[np.ndarray] = None

    def ingest(self, cmd: ControlCmd, now: float) -> None:
        if not cmd.mpc:
            return
        updated = False
        for axis in ("yaw", "pitch"):
            diag = cmd.mpc.get(axis)
            if diag is None or not diag.terms:
                continue
            terms = dict(diag.terms)

            sample = _OverlaySample(
                timestamp=now,
                terms=terms,
                term_directions=dict(getattr(diag, "term_directions", {}) or {}),
                status=diag.status,
                u0=diag.u0,
            )
            self._history[axis].append(sample)
            self._prune_history(axis, now)
            updated = True
        if updated:
            self._dirty = True

    def render(self, frame, now: float) -> None:
        pruned = False
        for axis in ("yaw", "pitch"):
            pruned = self._prune_history(axis, now) or pruned
        if pruned:
            self._dirty = True

        overlay_needed = any(self._history[axis] for axis in ("yaw", "pitch"))
        if not overlay_needed:
            self._cached_layout = None
            self._cached_layer = None
            self._cached_mask = None
            return

        height, width = frame.shape[:2]
        layout = self._make_layout(width, height)
        self._frames_since_render += 1

        cache_missing = (
            self._cached_layout != layout
            or self._cached_layer is None
            or self._cached_mask is None
        )
        render_due = self._frames_since_render >= self._render_interval_frames
        if cache_missing or (self._dirty and render_due):
            self._refresh_cached_overlay(layout)
            self._dirty = False
            self._frames_since_render = 0

        self._apply_cached_overlay(frame)

    def _make_layout(self, frame_width: int, frame_height: int) -> _OverlayLayout:
        section_height = self._cfg.bar_height_px + 18 * (len(self._cfg.show_terms) + 2)
        margin = 12
        spacing = 10
        total_height = 2 * section_height + spacing
        y_base = max(margin, frame_height - margin - total_height)
        bar_width = min(int(frame_width * 0.32), 420)
        x_origin = 12

        # Keep compositing bounded to the debug panel instead of blending the
        # whole video frame. The fixed width covers labels and numeric values.
        x0 = 0
        y0 = max(0, y_base - 28)
        x1 = min(frame_width, max(x_origin + bar_width + 24, 540))
        y1 = min(frame_height, y_base + total_height + 8)
        return _OverlayLayout(
            frame_width=frame_width,
            frame_height=frame_height,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            x_origin=x_origin,
            y_base=y_base,
            bar_width=bar_width,
            section_height=section_height,
            spacing=spacing,
        )

    def _refresh_cached_overlay(self, layout: _OverlayLayout) -> None:
        static_layer, static_mask = self._get_static_overlay(layout)
        layer = static_layer.copy()
        mask = static_mask.copy()

        for idx, axis in enumerate(("yaw", "pitch")):
            sample = self._latest_sample(axis)
            if sample is None:
                continue
            y_origin = layout.y_base + idx * (layout.section_height + layout.spacing)
            self._draw_axis_dynamic(layer, mask, layout, y_origin, axis, sample)

        self._cached_layout = layout
        self._cached_layer = layer
        self._cached_mask = mask

    def _apply_cached_overlay(self, frame) -> None:
        if (
            self._cached_layout is None
            or self._cached_layer is None
            or self._cached_mask is None
        ):
            return

        layout = self._cached_layout
        roi = frame[layout.y0 : layout.y1, layout.x0 : layout.x1]
        if roi.size == 0:
            return

        blended = cv2.addWeighted(
            self._cached_layer,
            self._cfg.opacity,
            roi,
            1.0 - self._cfg.opacity,
            0,
        )
        cv2.copyTo(blended, self._cached_mask, roi)

    def _get_static_overlay(
        self,
        layout: _OverlayLayout,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (
            self._cfg.cache_static_layout
            and self._static_layout == layout
            and self._static_layer is not None
            and self._static_mask is not None
        ):
            return self._static_layer, self._static_mask

        roi_height = max(1, layout.y1 - layout.y0)
        roi_width = max(1, layout.x1 - layout.x0)
        layer = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
        mask = np.zeros((roi_height, roi_width), dtype=np.uint8)

        for idx in range(2):
            y_origin = layout.y_base + idx * (layout.section_height + layout.spacing)
            self._draw_axis_static(layer, mask, layout, y_origin)

        if self._cfg.cache_static_layout:
            self._static_layout = layout
            self._static_layer = layer
            self._static_mask = mask
        return layer, mask

    def _prune_history(self, axis: str, now: float) -> bool:
        window = self._cfg.history_window_s
        dq = self._history[axis]
        removed = False
        while dq and (now - dq[0].timestamp) > window:
            dq.popleft()
            removed = True
        return removed

    def _latest_sample(self, axis: str) -> Optional[_OverlaySample]:
        dq = self._history[axis]
        return dq[-1] if dq else None

    def _max_abs_term(self, axis: str) -> float:
        max_magnitude = 0.0
        for sample in self._history[axis]:
            for term in self._cfg.show_terms:
                max_magnitude = max(
                    max_magnitude, abs(float(sample.terms.get(term, 0.0)))
                )
        return max_magnitude

    def _max_total(self, axis: str) -> float:
        max_total = 0.0
        for sample in self._history[axis]:
            total = sum(
                abs(float(sample.terms.get(term, 0.0)))
                for term in self._cfg.show_terms
            )
            max_total = max(max_total, total)
        return max_total

    def _display_term_value(self, sample: _OverlaySample, term: str) -> float:
        value = float(sample.terms.get(term, 0.0))
        direction_hint = float(sample.term_directions.get(term, 0.0))
        if abs(direction_hint) > 0.0:
            return abs(value) if direction_hint > 0.0 else -abs(value)
        return value

    def _draw_axis_static(
        self,
        layer,
        mask,
        layout: _OverlayLayout,
        y_origin: int,
    ) -> None:
        x_origin = layout.x_origin - layout.x0
        y_origin = y_origin - layout.y0
        bar_height = self._cfg.bar_height_px
        bar_rect = (
            x_origin,
            y_origin,
            x_origin + layout.bar_width,
            y_origin + bar_height,
        )

        cv2.rectangle(
            layer,
            (bar_rect[0], bar_rect[1]),
            (bar_rect[2], bar_rect[3]),
            (32, 32, 32),
            thickness=cv2.FILLED,
        )
        cv2.rectangle(
            mask,
            (bar_rect[0], bar_rect[1]),
            (bar_rect[2], bar_rect[3]),
            255,
            thickness=cv2.FILLED,
        )
        cv2.rectangle(
            layer,
            (bar_rect[0], bar_rect[1]),
            (bar_rect[2], bar_rect[3]),
            (64, 64, 64),
            thickness=1,
        )
        cv2.rectangle(
            mask,
            (bar_rect[0], bar_rect[1]),
            (bar_rect[2], bar_rect[3]),
            255,
            thickness=1,
        )

    def _draw_axis_dynamic(
        self,
        layer,
        mask,
        layout: _OverlayLayout,
        y_origin: int,
        axis: str,
        sample: _OverlaySample,
    ) -> None:
        x_origin = layout.x_origin - layout.x0
        y_origin = y_origin - layout.y0
        bar_width = layout.bar_width
        bar_height = self._cfg.bar_height_px
        bar_rect = (
            x_origin,
            y_origin,
            x_origin + bar_width,
            y_origin + bar_height,
        )

        center_y = int(round((bar_rect[1] + bar_rect[3]) / 2))

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
            elif abs(value) > 0.0:
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

            cv2.rectangle(
                layer,
                (x0, y0),
                (x1, y1),
                colour,
                thickness=cv2.FILLED,
            )
            cv2.rectangle(
                mask,
                (x0, y0),
                (x1, y1),
                255,
                thickness=cv2.FILLED,
            )
            cv2.rectangle(
                layer,
                (x0, y0),
                (x1, y1),
                (30, 30, 30),
                thickness=1,
            )
            cv2.rectangle(
                mask,
                (x0, y0),
                (x1, y1),
                255,
                thickness=1,
            )

        cv2.line(
            layer,
            (bar_rect[0], center_y),
            (bar_rect[2], center_y),
            (90, 90, 90),
            thickness=1,
        )
        cv2.line(
            mask,
            (bar_rect[0], center_y),
            (bar_rect[2], center_y),
            255,
            thickness=1,
        )

        label = f"{axis.upper()}  {sample.status or 'n/a'}"
        if sample.u0 is not None:
            label += f"  u0={sample.u0:+0.2f}"
        self._draw_text(
            layer,
            label,
            (x_origin, max(12, y_origin - 6)),
            0.5,
            (255, 255, 255),
            mask,
        )

        text_y = y_origin + bar_height + 16
        for term in self._cfg.show_terms:
            colour = self.TERM_COLOURS.get(term, (200, 200, 200))
            value = self._display_term_value(sample, term)
            text = f"{term}: {value:+0.2f}"
            self._draw_text(layer, text, (x_origin, text_y), 0.45, colour, mask)
            text_y += 16

    def _draw_text(self, overlay, text, origin, scale, colour, mask=None) -> None:
        cv2.putText(
            overlay,
            text,
            origin,
            FONT,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            origin,
            FONT,
            scale,
            colour,
            1,
            cv2.LINE_AA,
        )
        if mask is not None:
            cv2.putText(
                mask,
                text,
                origin,
                FONT,
                scale,
                255,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                mask,
                text,
                origin,
                FONT,
                scale,
                255,
                1,
                cv2.LINE_AA,
            )


class GstReturnVideo:
    def __init__(self, port: int, pull_timeout_ns: int, bind_ip: Optional[str] = None) -> None:
        if Gst is None:
            raise RuntimeError("PyGObject/GStreamer bindings are required for return video")
        udp_bind = f"address={bind_ip} " if bind_ip else ""
        pipeline = (
            f"udpsrc {udp_bind}port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=97,clock-rate=90000 ! "
            "rtpjitterbuffer latency=120 ! rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR ! queue leaky=downstream max-size-buffers=5 ! "
            "appsink name=sink drop=true sync=false max-buffers=1"
        )
        self._pipeline = Gst.parse_launch(pipeline)
        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            raise RuntimeError("return video pipeline missing appsink named 'sink'")
        self._appsink.set_property("sync", False)
        self._appsink.set_property("max-buffers", 1)
        self._appsink.set_property("drop", True)
        self._bus = self._pipeline.get_bus()
        self._pipeline.set_state(Gst.State.PLAYING)
        self._eos = False
        self._pull_timeout_ns = pull_timeout_ns

    @property
    def eos(self) -> bool:
        return self._eos

    def isOpened(self) -> bool:
        return not self._eos and self._pipeline is not None

    def read(self):
        if self._eos:
            return False, None
        if self._bus is not None:
            msg = self._bus.timed_pop_filtered(
                0, Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
            if msg is not None:
                if msg.type == Gst.MessageType.EOS:
                    print("[ui] return stream EOS received")
                else:
                    err, dbg = msg.parse_error()
                    print(f"[ui] return stream error: {err} ({dbg})")
                self._eos = True
                return False, None
        if self._appsink is None:
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
            if fmt == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif fmt == "RGBA" and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif fmt == "BGRx" and frame.shape[2] == 4:
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


def open_return_video(port: int, pull_timeout_ns: int, bind_ip: Optional[str] = None) -> GstReturnVideo:
    return GstReturnVideo(port, pull_timeout_ns, bind_ip)


def resolve_return_timeout_ns(video_cfg: Dict[str, object]) -> int:
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

def main():
    if Gst is None:
        raise SystemExit("PyGObject/GStreamer bindings are required to run the UI")
    Gst.init(None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/network.yaml")
    ap.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
        help="Comma-separated YAML configs merged over --config.",
    )
    ap.add_argument(
        "--config-sync-timeout",
        type=float,
        default=None,
        help=(
            "Maximum seconds to wait for Jetson config sync before continuing. "
            "Default waits indefinitely. "
            "Use 0 to skip the handshake."
        ),
    )
    ap.add_argument(
        "--config-sync-mode",
        choices=("auto", "force", "skip"),
        default="auto",
        help=(
            "auto: bounded startup wait and fallback when streamer sync is unavailable; "
            "force: require startup handshake and sync; "
            "skip: never perform the handshake."
        ),
    )
    args = ap.parse_args()

    if args.config_sync_timeout is not None and args.config_sync_timeout < 0:
        raise SystemExit("--config-sync-timeout must be >= 0")

    config_paths = expand_config_paths(args.config, args.config_extra)

    initial_snapshots = {path: read_snapshot(path) for path in config_paths}
    preview_cfg = merge_config_maps(
        *(
            parse_config_text(snapshot.text, str(path))
            for path, snapshot in initial_snapshots.items()
        )
    )
    sync_endpoint = resolve_config_sync_endpoint(preview_cfg)
    preview_source = str(preview_cfg.get("source", "") or "").strip().lower()
    effective_source = preview_source
    if args.config_sync_timeout != 0 and args.config_sync_mode != "skip":
        startup_probe_wait: Optional[float]
        if args.config_sync_mode == "auto":
            startup_probe_wait = args.config_sync_timeout if args.config_sync_timeout is not None else 1.0
        else:
            startup_probe_wait = args.config_sync_timeout
        try:
            startup_state = request_startup_state(
                sync_endpoint,
                peer_id="pc",
                max_wait=startup_probe_wait,
                retry_interval=0.2,
            )
            startup_source = str(startup_state.get("effective_source", "") or "").strip().lower()
            if startup_source:
                effective_source = startup_source
            if startup_source and startup_source != preview_source:
                print(
                    "[ui] Startup source override received from Jetson: "
                    f"{startup_source} (local={preview_source or '<unset>'})"
                )
        except ConfigSyncError as exc:
            if args.config_sync_mode == "auto":
                print(
                    "[ui][WARN] Config sync: startup probe unavailable; "
                    f"continuing with local source ({exc})"
                )
            else:
                raise SystemExit(f"startup handshake failed: {exc}") from exc

    source_is_sim = effective_source.startswith("sim")

    final_texts = {path: snapshot.text for path, snapshot in initial_snapshots.items()}
    final_metas = {
        path: snapshot.metadata for path, snapshot in initial_snapshots.items()
    }

    marker_metas = {
        path: (load_sync_marker(path) or (None, None))[0] for path in config_paths
    }

    skip_reason: Optional[str] = None
    if args.config_sync_timeout == 0:
        skip_reason = "--config-sync-timeout=0"
    elif args.config_sync_mode == "skip":
        skip_reason = "--config-sync-mode=skip"
    elif args.config_sync_mode == "auto":
        if not source_is_sim:
            skip_reason = "source!=sim"
        elif all(
            marker is not None and marker.sha256 == initial_snapshots[path].metadata.sha256
            for path, marker in marker_metas.items()
        ):
            skip_reason = "streamer markers match local configuration"

    if skip_reason is not None:
        print(f"[ui] Config sync: skipping handshake ({skip_reason})")
    else:
        try:
            with acquire_config_sync_lock(config_paths[0], args.config_sync_timeout):
                for path in config_paths:
                    snapshot = initial_snapshots[path]
                    final_text, final_meta = sync_as_client(
                        path,
                        sync_endpoint,
                        config_id=path.name,
                        peer_id="pc",
                        max_wait=args.config_sync_timeout,
                    )

                    if final_meta.sha256 != snapshot.metadata.sha256:
                        print(
                            "[ui] Config sync: updated local configuration "
                            f"(sha256={final_meta.sha256})"
                        )
                    write_sync_marker(path, final_meta)
                    final_texts[path] = final_text
                    final_metas[path] = final_meta
        except ConfigSyncError as exc:
            if args.config_sync_mode == "auto":
                print(
                    "[ui][WARN] Config sync: skipping handshake "
                    f"(lock unavailable: {exc})"
                )
            else:
                raise SystemExit(f"config synchronization failed: {exc}") from exc

    cfg = merge_config_maps(
        *(
            parse_config_text(final_texts[path], str(path))
            for path in config_paths
        )
    )

    video_cfg, active_profile = resolve_active_video_profile(cfg)
    try:
        w = int(video_cfg["width"])
        h = int(video_cfg["height"])
    except KeyError as exc:
        raise SystemExit("config missing video.width/video.height") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.width/video.height must be integers") from exc
    pull_timeout_ns = resolve_return_timeout_ns(video_cfg)

    try:
        control_cfg = ControlConfig.from_raw_config(cfg, (w, h))
    except ControlConfigError as exc:
        raise SystemExit(f"invalid control configuration: {exc}") from exc

    try:
        laser_cfg = LaserMountConfig.from_raw_config(cfg)
    except LaserConfigError as exc:
        raise SystemExit(f"invalid laser configuration: {exc}") from exc

    try:
        return_port = int(cfg["net"]["rtp_return_port"])
    except KeyError as exc:
        raise SystemExit("config missing net.rtp_return_port") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit("net.rtp_return_port must be an integer") from exc
    net_cfg = cfg.get("net", {}) if isinstance(cfg, dict) else {}
    pc_bind_ip_raw = net_cfg.get("pc_bind_ip")
    pc_bind_ip = str(pc_bind_ip_raw).strip() if pc_bind_ip_raw else None
    pc_iface_raw = net_cfg.get("pc_iface")
    pc_iface = str(pc_iface_raw).strip() if pc_iface_raw else None

    stop_event = install_signal_handlers()

    if active_profile:
        print(
            "[ui] Using video profile %s (%dx%d)" % (active_profile, w, h)
        )

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detections", w, h)

    cap: Optional[GstReturnVideo] = None
    last_cap_open = 0.0

    # ZMQ
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.LINGER, 0)
    _bind_zmq_to_device_if_configured(sub, pc_iface)
    sub.connect(cfg['net']['zmq_results'])
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    ctrl_sub: Optional[zmq.Socket] = None
    overlay_renderer: Optional[MpcDebugOverlay] = None
    if control_cfg.debug_overlay.enabled:
        ctrl_endpoint = cfg["net"].get("zmq_control")
        if not ctrl_endpoint:
            print("[ui] MPC overlay disabled: net.zmq_control is not configured")
        else:
            ctrl_sub = ctx.socket(zmq.SUB)
            ctrl_sub.setsockopt(zmq.CONFLATE, 1)
            ctrl_sub.setsockopt(zmq.RCVHWM, 1)
            ctrl_sub.setsockopt(zmq.LINGER, 0)
            ctrl_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            _bind_zmq_to_device_if_configured(ctrl_sub, pc_iface)
            ctrl_sub.connect(ctrl_endpoint)
            overlay_renderer = MpcDebugOverlay(control_cfg.debug_overlay)
            print(
                "[ui] MPC overlay enabled "
                "(terms=%s, window=%.1fs, interval=%d, static_cache=%s)"
                % (
                    ",".join(control_cfg.debug_overlay.show_terms),
                    control_cfg.debug_overlay.history_window_s,
                    control_cfg.debug_overlay.render_interval_frames,
                    control_cfg.debug_overlay.cache_static_layout,
                )
            )

    last_frame_id = -1
    last_e2e_ms = 0
    last_draw = time.time()
    fps_est = 0.0

    try:
        while not stop_event.is_set():
            now = time.time()
            if cap is not None and cap.eos:
                stop_event.set()
                break
            if (cap is None or not cap.isOpened()) and (now - last_cap_open) > 0.5:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                print(f"[ui] opening return video (port {return_port})")
                cap = open_return_video(return_port, pull_timeout_ns, pc_bind_ip)
                last_cap_open = now

            okv, video = (cap.read() if cap and cap.isOpened() else (False, None))
            if cap is not None and cap.eos:
                stop_event.set()
                break
            if stop_event.is_set():
                break
            if okv and video is not None:
                frame = video
            # Hold the last decoded frame across temporary read misses to
            # avoid turning stream jitter into visible black flashes.

            while True:
                try:
                    payload = sub.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                msg = detection_msg_from_json(payload)
                last_frame_id = msg.frame_id
                last_e2e_ms = compute_e2e_ms(msg.src_ts_ms)
                # (Optional) you disabled local drawing; keep it off
            if ctrl_sub is not None:
                while True:
                    try:
                        payload = ctrl_sub.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        cmd = control_cmd_from_json(payload)
                    except Exception as exc:
                        print(f"[ui] failed to decode ControlCmd: {exc}")
                    else:
                        if overlay_renderer is not None:
                            overlay_renderer.ingest(cmd, time.time())

            now = time.time()
            inst = 1.0 / max(1e-6, (now - last_draw))
            last_draw = now
            fps_est = inst if fps_est == 0.0 else (0.9*fps_est + 0.1*inst)
            status = (
                f"frame #{last_frame_id if last_frame_id>=0 else '-'}  "
                f"e2e {int(last_e2e_ms)} ms  ~{fps_est:4.1f} fps"
            )

            font = FONT
            scale = 0.5
            thickness = 1
            margin = 8
            text_colour = (255, 255, 255)

            text_size, baseline = cv2.getTextSize(status, font, scale, thickness)
            text_w, text_h = text_size
            h, w = frame.shape[:2]

            origin_x = max(margin + 4, w - margin - text_w)
            origin_y = max(margin + text_h, h - margin - baseline)

            rect_tl = (int(origin_x - 4), int(max(0, origin_y - text_h - 4)))
            rect_br = (
                int(min(w - 1, origin_x + text_w + 4)),
                int(min(h - 1, origin_y + baseline + 4)),
            )

            if overlay_renderer is not None:
                overlay_renderer.render(frame, time.time())

            cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), thickness=cv2.FILLED)
            cv2.putText(
                frame,
                status,
                (int(origin_x), int(origin_y)),
                font,
                scale,
                text_colour,
                thickness,
                cv2.LINE_AA,
            )
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("[ui] shutting down...")
        try:
            if cap:
                cap.release()
        except Exception:
            pass
        try:
            sub.close(0)
        except Exception:
            pass
        if ctrl_sub is not None:
            try:
                ctrl_sub.close(0)
            except Exception:
                pass
        try:
            ctx.term()
        except Exception:
            pass
        # make sure window goes away on all platforms
        for _ in range(3):
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        time.sleep(0.05)

if __name__ == "__main__":
    main()
