# pc/ui.py
import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import zmq

from common.config_sync import (
    ConfigSyncError,
    DEFAULT_CONFIG_SYNC_TIMEOUT,
    load_sync_marker,
    parse_config_text,
    read_snapshot,
    resolve_active_video_profile,
    resolve_config_sync_endpoint,
    sync_as_client,
    write_sync_marker,
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


@dataclass
class _OverlaySample:
    timestamp: float
    terms: Dict[str, float]
    status: str
    u0: Optional[float]


class MpcDebugOverlay:
    """Utility that renders MPC cost-term history on the return feed."""

    TERM_COLOURS: Dict[str, Tuple[int, int, int]] = {
        "theta": (64, 192, 255),
        "omega": (0, 160, 255),
        "approach": (255, 176, 59),
        "effort": (144, 214, 72),
        "slew": (198, 118, 255),
        "slack": (96, 96, 96),
    }

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
                status=diag.status,
                u0=diag.u0,
            )
            self._history[axis].append(sample)
            self._prune_history(axis, now)

    def render(self, frame, now: float) -> None:
        for axis in ("yaw", "pitch"):
            self._prune_history(axis, now)

        overlay_needed = any(self._history[axis] for axis in ("yaw", "pitch"))
        if not overlay_needed:
            return

        overlay = frame.copy()
        height, width = frame.shape[:2]
        section_height = self._cfg.bar_height_px * 2 + 18 * (len(self._cfg.show_terms) + 2)
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
        max_abs = 0.0
        for sample in self._history[axis]:
            for term in self._cfg.show_terms:
                value = float(sample.terms.get(term, 0.0))
                max_abs = max(max_abs, abs(value))
        return max_abs

    def _draw_axis_section(
        self,
        overlay,
        frame_width: int,
        y_origin: int,
        axis: str,
        sample: _OverlaySample,
    ) -> None:
        bar_width = min(int(frame_width * 0.32), 420)
        x_origin = 12
        half_height = self._cfg.bar_height_px
        bar_rect = (
            x_origin,
            y_origin,
            x_origin + bar_width,
            y_origin + 2 * half_height,
        )

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

        zero_y = y_origin + half_height
        cv2.line(
            overlay,
            (x_origin, zero_y),
            (bar_rect[2], zero_y),
            (96, 96, 96),
            thickness=1,
        )

        values = [float(sample.terms.get(term, 0.0)) for term in self._cfg.show_terms]
        max_abs = max(
            self._max_abs_term(axis), max((abs(v) for v in values), default=0.0), 1e-6
        )
        scale = half_height / max_abs
        n_terms = len(self._cfg.show_terms)
        spacing = 4 if n_terms > 0 else 0
        available_width = max(bar_width - spacing * max(0, n_terms - 1), 1)
        term_width = max(4, available_width // max(1, n_terms))
        cursor = x_origin
        for term, value in zip(self._cfg.show_terms, values):
            height_px = int(round(abs(value) * scale))
            if height_px <= 0:
                cursor += term_width + spacing
                continue
            top = zero_y - height_px if value >= 0 else zero_y
            bottom = zero_y if value >= 0 else min(bar_rect[3], zero_y + height_px)
            colour = self.TERM_COLOURS.get(term, (200, 200, 200))
            cv2.rectangle(
                overlay,
                (cursor, top),
                (min(bar_rect[2], cursor + term_width), bottom),
                colour,
                thickness=cv2.FILLED,
            )
            cursor += term_width + spacing

        direction_hint = {
            "yaw": "+ right / - left",
            "pitch": "+ up / - down",
        }.get(axis, "")
        direction_text = f" ({direction_hint})" if direction_hint else ""
        label = f"{axis.upper()}{direction_text}  {sample.status or 'n/a'}"
        if sample.u0 is not None:
            label += f"  u0={sample.u0:+0.2f}"
        self._draw_text(overlay, label, (x_origin, max(12, y_origin - 6)), 0.5, (255, 255, 255))

        text_y = y_origin + 2 * half_height + 16
        for term, value in zip(self._cfg.show_terms, values):
            colour = self.TERM_COLOURS.get(term, (200, 200, 200))
            text = f"{term}: {value:+0.2f}"
            self._draw_text(overlay, text, (x_origin, text_y), 0.45, colour)
            text_y += 16

    def _draw_text(self, overlay, text, origin, scale, colour) -> None:
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

def open_return_video(port, w, h):
    pipeline = (
    f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=97,clock-rate=90000 ! "
    "rtpjitterbuffer latency=120 ! rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! queue leaky=downstream max-size-buffers=5 ! appsink drop=true sync=false max-buffers=1"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    ap.add_argument(
        "--config-sync-timeout",
        type=float,
        default=DEFAULT_CONFIG_SYNC_TIMEOUT,
        help=(
            "Maximum seconds to wait for Jetson config sync before continuing "
            f"(default: {DEFAULT_CONFIG_SYNC_TIMEOUT:g}). "
            "Use 0 to skip the handshake."
        ),
    )
    ap.add_argument(
        "--config-sync-mode",
        choices=("auto", "force", "skip"),
        default="auto",
        help=(
            "auto: reuse the streamer sync marker when available; "
            "force: always perform the handshake; "
            "skip: never perform the handshake."
        ),
    )
    args = ap.parse_args()

    if args.config_sync_timeout is not None and args.config_sync_timeout < 0:
        raise SystemExit("--config-sync-timeout must be >= 0")

    config_path = Path(args.config)
    initial_snapshot = read_snapshot(config_path)
    preview_cfg = parse_config_text(initial_snapshot.text, str(config_path))
    sync_endpoint = resolve_config_sync_endpoint(preview_cfg)

    final_text = initial_snapshot.text
    final_meta = initial_snapshot.metadata

    marker_info = load_sync_marker(config_path)
    marker_meta = marker_info[0] if marker_info else None

    skip_reason: Optional[str] = None
    if args.config_sync_timeout == 0:
        skip_reason = "--config-sync-timeout=0"
    elif args.config_sync_mode == "skip":
        skip_reason = "--config-sync-mode=skip"
    elif args.config_sync_mode == "auto" and marker_meta is not None:
        if marker_meta.sha256 == initial_snapshot.metadata.sha256:
            skip_reason = "streamer marker matches local configuration"

    if skip_reason is not None:
        print(f"[ui] Config sync: skipping handshake ({skip_reason})")
    else:
        try:
            final_text, final_meta = sync_as_client(
                config_path,
                sync_endpoint,
                max_wait=args.config_sync_timeout,
            )
        except ConfigSyncError as exc:
            raise SystemExit(f"config synchronization failed: {exc}") from exc

        if final_meta.sha256 != initial_snapshot.metadata.sha256:
            print(
                "[ui] Config sync: updated local configuration "
                f"(sha256={final_meta.sha256})"
            )
        write_sync_marker(config_path, final_meta)

    cfg = parse_config_text(final_text, str(config_path))

    video_cfg, active_profile = resolve_active_video_profile(cfg)
    try:
        w = int(video_cfg["width"])
        h = int(video_cfg["height"])
    except KeyError as exc:
        raise SystemExit("config missing video.width/video.height") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.width/video.height must be integers") from exc

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

    stop_event = install_signal_handlers()

    if active_profile:
        print(
            "[ui] Using video profile %s (%dx%d)" % (active_profile, w, h)
        )

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detections", w, h)

    cap = None
    last_cap_open = 0.0

    # ZMQ
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.setsockopt(zmq.LINGER, 0)
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
            ctrl_sub.connect(ctrl_endpoint)
            overlay_renderer = MpcDebugOverlay(control_cfg.debug_overlay)
            print(
                "[ui] MPC overlay enabled (terms=%s, window=%.1fs)"
                % (",".join(control_cfg.debug_overlay.show_terms), control_cfg.debug_overlay.history_window_s)
            )

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    if ctrl_sub is not None:
        poller.register(ctrl_sub, zmq.POLLIN)

    last_frame_id = -1
    last_e2e_ms = 0
    last_draw = time.time()
    fps_est = 0.0

    try:
        while not stop_event.is_set():
            now = time.time()
            if (cap is None or not cap.isOpened()) and (now - last_cap_open) > 0.5:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                print(f"[ui] opening return video (port {return_port})")
                cap = open_return_video(return_port, w, h)
                last_cap_open = now

            okv, video = (cap.read() if cap and cap.isOpened() else (False, None))
            if okv and video is not None:
                frame = video
            else:
                frame[:] = 0

            events = dict(poller.poll(timeout=50))
            if sub in events and events[sub] == zmq.POLLIN:
                payload = sub.recv()
                msg = detection_msg_from_json(payload)
                now_ms = int(time.monotonic_ns() / 1e6)
                last_frame_id = msg.frame_id
                last_e2e_ms = (now_ms - msg.src_ts_ms) if msg.src_ts_ms else 0
                # (Optional) you disabled local drawing; keep it off
            if ctrl_sub is not None and events.get(ctrl_sub) == zmq.POLLIN:
                payload = ctrl_sub.recv()
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
