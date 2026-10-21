"""PC UI entrypoint.

Responsibilities:
    - Subscribe to Jetson detection metadata over ZMQ and display status overlays.
    - Receive the Jetson return video over RTP/UDP and present the annotated feed.
    - Compose the operational V2 HUD from perception, CamState, ControlCmd, and
      read-only controller diagnostics.

Required ZMQ endpoint (from the config file):
    - net.zmq_perception_v2: SUB socket for PerceptionSnapshot V2 payloads.

Expected message types:
    - PerceptionSnapshotV2 via common.perception.perception_snapshot_from_json().
    - ControlCmd via common.schemas.control_cmd_from_json().
    - ControlDiagnostics via its strict versioned schema.

"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import zmq
from urllib.parse import urlsplit

try:
    import gi
except ModuleNotFoundError:
    gi = None
    Gst = None
else:
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

from common.live_config import runtime_bundle
from common.config import (
    ConfigError,
    resolve_active_return_video_profile,
    resolve_active_video_profile,
    resolve_config_paths,
)
from common.control import (
    ControlConfig,
    ControlConfigError,
    LaserConfigError,
    LaserMountConfig,
)
from common.perception import PerceptionSnapshotV2, perception_snapshot_from_json
from common.hud_geometry import perception_progresses
from common.sim_mode import require_simulation_loopback_endpoint
from common.schemas import CamState, ControlCmd, ControlDiagnostics, control_cmd_from_json
from common.shutdown import install_signal_handlers
from pc.v2_hud import V2HudRenderer, resolve_hud_fov
from pc.ui_text import text as ui_text
from common import ui_style as ui_theme
from common.video_health import ReturnVideoHealth

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bind_zmq_to_device_if_configured(
    socket: zmq.Socket, iface: Optional[str], endpoint: Optional[str] = None,
) -> None:
    """Best-effort Linux interface bind for libzmq builds that expose it.

    Loopback endpoints (simulation) are left unbound: binding to the LAN
    device would make them unreachable.
    """
    if not iface:
        return
    if endpoint is not None and urlsplit(str(endpoint)).hostname in {"127.0.0.1", "localhost", "::1"}:
        return
    option = getattr(zmq, "BINDTODEVICE", None)
    if option is None:
        print("[ui][WARN] net.pc_iface ignored: pyzmq/libzmq lacks BINDTODEVICE")
        return
    try:
        socket.setsockopt_string(option, iface)
    except Exception as exc:
        print(f"[ui][WARN] net.pc_iface={iface!r} bind failed: {exc}")


class GstReturnVideo:
    def __init__(
        self,
        port: int,
        pull_timeout_ns: int,
        bind_ip: Optional[str] = None,
        jitter_ms: int = 20,
        queue_buffers: int = 1,
        feedback_host: Optional[str] = None,
        feedback_port: int = 5003,
        sender_report_port: int = 5004,
    ) -> None:
        if Gst is None:
            raise RuntimeError("PyGObject/GStreamer bindings are required for return video")
        from common.return_rtp import receiver_head, ReceiverRecovery
        decoder = "nvh264dec" if Gst.ElementFactory.find("nvh264dec") is not None else "avdec_h264"
        self.decoder_name = decoder
        pipeline = (
            # 4 MB receive buffer: a 20 Mbps keyframe arrives as a burst larger than
            # the 212 KB kernel default and would lose packets.
            receiver_head(port,jitter_ms,bind_ip=bind_ip,feedback_host=feedback_host,
                          feedback_port=feedback_port,sender_report_port=sender_report_port) +
            f"h264parse ! {decoder} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            f"queue leaky=downstream max-size-buffers={queue_buffers} ! "
            "appsink name=sink drop=true sync=false max-buffers=1"
        )
        print(f"[ui] return video decoder: {decoder}")
        self._pipeline = Gst.parse_launch(pipeline)
        self.recovery = ReceiverRecovery(self._pipeline)
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
        for _ in range(64 if self._bus is not None else 0):
            msg = self._bus.timed_pop_filtered(
                0, Gst.MessageType.EOS | Gst.MessageType.ERROR | Gst.MessageType.ELEMENT
            )
            if msg is None:
                break
            if msg is not None:
                if msg.type == Gst.MessageType.ELEMENT:
                    self.recovery.on_message(self._bus,msg)
                elif msg.type == Gst.MessageType.EOS:
                    print("[ui] return stream EOS received")
                else:
                    err, dbg = msg.parse_error()
                    print(f"[ui] return stream error: {err} ({dbg})")
                if msg.type != Gst.MessageType.ELEMENT:
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
        pipeline = self._pipeline
        self._pipeline = None
        self._appsink = None
        self._bus = None
        self._eos = True
        if pipeline is None:
            return
        try:
            pipeline.set_state(Gst.State.NULL)
            # State changes involving hardware decoders may complete
            # asynchronously. Retain the final pipeline reference until its
            # pad tasks have stopped instead of allowing GObject finalization
            # to race the transition.
            pipeline.get_state(2 * Gst.SECOND)
        except Exception:
            pass


def open_return_video(
    port: int,
    pull_timeout_ns: int,
    bind_ip: Optional[str] = None,
    jitter_ms: int = 20,
    queue_buffers: int = 1,
    feedback_host: Optional[str] = None,
    feedback_port: int = 5003,
    sender_report_port: int = 5004,
) -> GstReturnVideo:
    return GstReturnVideo(port, pull_timeout_ns, bind_ip, jitter_ms, queue_buffers,
                          feedback_host,feedback_port,sender_report_port)


def resolve_return_buffering(video_cfg: Dict[str, object]) -> tuple[int, int]:
    try:
        jitter_ms = int(video_cfg.get("return_jitter_ms", 20))
        queue_buffers = int(video_cfg.get("return_queue_buffers", 1))
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "video.return_jitter_ms and return_queue_buffers must be integers"
        ) from exc
    if jitter_ms < 0:
        raise SystemExit("video.return_jitter_ms must be non-negative")
    if queue_buffers <= 0:
        raise SystemExit("video.return_queue_buffers must be positive")
    return jitter_ms, queue_buffers


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


def write_ui_report(path: Path, report: Dict[str, object]) -> bool:
    """Persist a diagnostic report without interrupting runtime cleanup."""

    try:
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[ui][WARN] failed to write report {path}: {exc}", flush=True)
        return False
    return True

def main():
    if Gst is None:
        raise SystemExit("PyGObject/GStreamer bindings are required to run the UI")
    Gst.init(None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base")
    ap.add_argument(
        "--config-extra",
        default="",
        help="Comma-separated YAML configs merged over --config.",
    )
    ap.add_argument(
        "--duration-s",
        type=float,
        help="Stop after this many seconds (for bounded validation runs).",
    )
    ap.add_argument(
        "--control-sub",
        help="optional non-production ControlCmd endpoint for operational HUD state",
    )
    ap.add_argument("--camstate-sub", help="optional CamState endpoint for heading/elevation HUD")
    ap.add_argument("--perception-sub", help="optional loopback simulator control-source perception (read-only)")
    ap.add_argument(
        "--diagnostics-sub",
        help="controller diagnostics endpoint (default: net.zmq_control_diagnostics)",
    )
    ap.add_argument("--no-hud", action="store_true", help="disable the operational V2 HUD")
    ap.add_argument("--authority-label", default="passive", help="short HUD authority/source label")
    ap.add_argument("--check", action="store_true", help="validate config without opening sockets or video")
    ap.add_argument("--report", type=Path, help="write UI video/metadata counters on shutdown")
    args = ap.parse_args()

    if args.duration_s is not None and args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    config_paths = resolve_config_paths(args.config, args.config_extra)
    try:
        bundle = runtime_bundle(config_paths, required_sections=("net", "video"))
        cfg = bundle.mutable_copy()
    except ConfigError as exc:
        raise SystemExit(f"invalid configuration: {exc}") from exc

    video_cfg, active_profile = resolve_active_video_profile(cfg)
    return_video_cfg, active_return_profile = resolve_active_return_video_profile(cfg)
    try:
        w = int(video_cfg["width"])
        h = int(video_cfg["height"])
    except KeyError as exc:
        raise SystemExit("config missing video.width/video.height") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.width/video.height must be integers") from exc
    pull_timeout_ns = resolve_return_timeout_ns(return_video_cfg)
    return_jitter_ms, return_queue_buffers = resolve_return_buffering(return_video_cfg)

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
    diagnostics_endpoint_raw = args.diagnostics_sub or net_cfg.get("zmq_control_diagnostics")
    diagnostics_endpoint = (
        str(diagnostics_endpoint_raw).strip() if diagnostics_endpoint_raw else None
    )

    if args.check:
        print(json.dumps({
            "mode": "v2_local_config",
            "perception_endpoint": (require_simulation_loopback_endpoint(args.perception_sub, "--perception-sub")
                                    if args.perception_sub else net_cfg.get("zmq_perception_v2")),
            "return_port": return_port,
            "return_profile": active_return_profile,
            "return_fps": return_video_cfg.get("fps"),
            "return_pull_timeout_ms": round(pull_timeout_ns / 1_000_000, 3),
            "control_endpoint": args.control_sub,
            "camstate_endpoint": args.camstate_sub,
            "diagnostics_endpoint": diagnostics_endpoint,
            "operational_hud": not args.no_hud,

            **bundle.provenance(),
        }, sort_keys=True))
        return

    stop_event = install_signal_handlers()
    print(json.dumps({**bundle.provenance(), "pending_config_keys": []}, sort_keys=True))

    if active_profile:
        print(
            "[ui] Using video profile %s (%dx%d)" % (active_profile, w, h)
        )
    if active_return_profile:
        print(
            "[ui] Return profile %s (%sx%s @ %s FPS, pull timeout %.1f ms)"
            % (
                active_return_profile,
                return_video_cfg.get("width"),
                return_video_cfg.get("height"),
                return_video_cfg.get("fps"),
                pull_timeout_ns / 1_000_000,
            )
        )

    last_video_frame = np.zeros((h, w, 3), dtype=np.uint8)
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
    _bind_zmq_to_device_if_configured(sub, pc_iface, cfg['net']['zmq_perception_v2'])
    perception_endpoint = (require_simulation_loopback_endpoint(args.perception_sub, "--perception-sub")
                           if args.perception_sub else cfg['net']['zmq_perception_v2'])
    sub.connect(perception_endpoint)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    ctrl_sub: Optional[zmq.Socket] = None
    if args.control_sub:
        ctrl_endpoint = str(args.control_sub)
        if ctrl_endpoint == str(cfg["net"].get("zmq_control", "")):
            raise SystemExit("--control-sub must not use production net.zmq_control in passive V2 UI")
        ctrl_sub = ctx.socket(zmq.SUB)
        ctrl_sub.setsockopt(zmq.CONFLATE, 1)
        ctrl_sub.setsockopt(zmq.RCVHWM, 1)
        ctrl_sub.setsockopt(zmq.LINGER, 0)
        ctrl_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        _bind_zmq_to_device_if_configured(ctrl_sub, pc_iface, ctrl_endpoint)
        ctrl_sub.connect(ctrl_endpoint)

    camstate_sub: Optional[zmq.Socket] = None
    if args.camstate_sub:
        camstate_sub = ctx.socket(zmq.SUB)
        camstate_sub.setsockopt(zmq.CONFLATE, 1)
        camstate_sub.setsockopt(zmq.RCVHWM, 1)
        camstate_sub.setsockopt(zmq.LINGER, 0)
        camstate_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        _bind_zmq_to_device_if_configured(camstate_sub, pc_iface, str(args.camstate_sub))
        camstate_sub.connect(str(args.camstate_sub))

    diagnostics_sub: Optional[zmq.Socket] = None
    if diagnostics_endpoint and not args.no_hud:
        diagnostics_sub = ctx.socket(zmq.SUB)
        diagnostics_sub.setsockopt(zmq.CONFLATE, 1)
        diagnostics_sub.setsockopt(zmq.RCVHWM, 1)
        diagnostics_sub.setsockopt(zmq.LINGER, 0)
        diagnostics_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        _bind_zmq_to_device_if_configured(diagnostics_sub, pc_iface, diagnostics_endpoint)
        diagnostics_sub.connect(diagnostics_endpoint)

    hud_renderer: Optional[V2HudRenderer] = None
    if not args.no_hud:
        hfov_deg, vfov_deg = resolve_hud_fov(cfg, (w, h))
        hud_renderer = V2HudRenderer(
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            authority_label=args.authority_label,
        )
        print(
            "[ui] operational V2 HUD enabled "
            f"(fov={hfov_deg:.1f}x{vfov_deg:.1f})"
        )

    last_frame_id = -1
    last_e2e_ms = 0
    last_detection_count = 0
    last_track_count = 0
    last_selected_track_id: Optional[int] = None
    last_snapshot: Optional[PerceptionSnapshotV2] = None
    last_snapshot_mono: Optional[float] = None
    last_control_cmd: Optional[ControlCmd] = None
    last_control_mono: Optional[float] = None
    last_cam_state: Optional[CamState] = None
    last_cam_state_mono: Optional[float] = None
    last_control_diagnostics: Optional[ControlDiagnostics] = None
    last_diagnostics_mono: Optional[float] = None
    last_hud_elements: tuple[str, ...] = ()
    last_feedforward_state = "unavailable"
    last_feedforward_yaw: Optional[float] = None
    last_feedforward_pitch: Optional[float] = None
    fps_est = 0.0
    video_health = ReturnVideoHealth()
    video_frames = 0
    metadata_messages = 0
    return_decoder: Optional[str] = None
    deadline = None if args.duration_s is None else time.monotonic() + args.duration_s

    try:
        while not stop_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
            now = time.time()
            # A return-stream error/EOS is a source gap (mode switch, runtime
            # restart), not the end of the UI: drop the capture and reopen it
            # below. Exiting here left Camera mode unready (agent "mixed").
            if (cap is None or not cap.isOpened()) and (now - last_cap_open) > 0.5:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                print(f"[ui] opening return video (port {return_port})")
                cap = open_return_video(
                    return_port,
                    pull_timeout_ns,
                    pc_bind_ip,
                    return_jitter_ms,
                    return_queue_buffers,
                    str(cfg['net'].get('jetson_ip') or '') or None,
                    int(cfg['net'].get('return_rtcp_feedback_port',5003)),
                    int(cfg['net'].get('return_rtcp_sender_report_port',5004)),
                )
                return_decoder = cap.decoder_name
                last_cap_open = now

            okv, video = (cap.read() if cap and cap.isOpened() else (False, None))
            if stop_event.is_set():
                break
            if okv and video is not None:
                video_health.mark(time.monotonic())
                last_video_frame = video
                video_frames += 1
            # A 30 FPS return naturally has gaps longer than a 60 FPS pull
            # cadence. Retain the newest decoded image across a timeout and
            # draw UI state on a fresh copy so neither black flashes nor local
            # overlay residue can accumulate.
            frame = last_video_frame.copy()

            while True:
                try:
                    payload = sub.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    snapshot = perception_snapshot_from_json(payload)
                except (TypeError, ValueError) as exc:
                    print(f"[ui] failed to decode PerceptionSnapshot V2: {exc}")
                    continue
                if not perception_progresses(last_snapshot, snapshot):
                    continue
                last_snapshot_mono = time.monotonic()
                last_frame_id = snapshot.frame.frame_id
                last_snapshot = snapshot
                metadata_messages += 1
                if snapshot.frame.source_clock_domain in {"pc.monotonic", "pc_monotonic"}:
                    last_e2e_ms = compute_e2e_ms(snapshot.frame.source_time_ns // 1_000_000)
                else:
                    last_e2e_ms = 0
                last_detection_count = len(snapshot.detections)
                last_track_count = len(snapshot.tracks)
                last_selected_track_id = (
                    snapshot.selection.track_id if snapshot.selection is not None else None
                )
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
                        last_control_cmd = cmd
                        last_control_mono = time.monotonic()
            if camstate_sub is not None:
                while True:
                    try:
                        payload = camstate_sub.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        state = CamState.model_validate_json(payload)
                    except Exception as exc:
                        print(f"[ui] failed to decode CamState: {exc}")
                    else:
                        last_cam_state = state
                        last_cam_state_mono = time.monotonic()
            if diagnostics_sub is not None:
                while True:
                    try:
                        payload = diagnostics_sub.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        diagnostics = ControlDiagnostics.model_validate_json(payload)
                    except Exception as exc:
                        print(f"[ui] failed to decode ControlDiagnostics: {exc}")
                    else:
                        last_control_diagnostics = diagnostics
                        last_diagnostics_mono = time.monotonic()

            now = time.time()
            fps_est = video_health.fps(time.monotonic())
            # Counts/identity belong to Jetson labels; timing diagnostics remain
            # in reports, not a second opaque banner across the returned image.
            status = video_health.warning(time.monotonic())

            h, w = frame.shape[:2]


            if hud_renderer is not None:
                now_mono = time.monotonic()
                cam_age = None if last_cam_state_mono is None else max(0.0, now_mono - last_cam_state_mono)
                control_age = None if last_control_mono is None else max(0.0, now_mono - last_control_mono)
                diagnostics_age = (
                    None
                    if last_diagnostics_mono is None
                    else max(0.0, now_mono - last_diagnostics_mono)
                )
                report = hud_renderer.render(
                    frame,
                    snapshot=last_snapshot,
                    cam_state=last_cam_state if cam_age is None or cam_age <= 1.0 else None,
                    control_cmd=last_control_cmd if control_age is None or control_age <= 1.0 else None,
                    control_diagnostics=last_control_diagnostics,
                    cam_state_age_s=cam_age,
                    control_age_s=control_age,
                    diagnostics_age_s=diagnostics_age,
                    perception_age_s=None if last_snapshot_mono is None else now_mono - last_snapshot_mono,
                )
                last_hud_elements = report.elements
                last_feedforward_state = report.feedforward_state
                last_feedforward_yaw = report.feedforward_yaw_rad_s
                last_feedforward_pitch = report.feedforward_pitch_rad_s

            s = ui_theme.scale_for(h)
            receiver = ui_theme.HudLayout(w,h).receiver_status
            if status:
                ui_text(frame,status,(receiver.x,receiver.y+15*s),ui_theme.bgr(ui_theme.AMBER),
                        size=ui_theme.pixel_size(ui_theme.SMALL_TEXT_PX,h),background=True,
                        max_width=receiver.width-8*s)
            cv2.imshow("Detections", frame)
            if okv and video_frames % 120 == 0:
                print(
                    "[ui] decoded=%d metadata=%d frame=%d objects=%d tracks=%d selected=%s fps=%.1f hud=%s"
                    % (
                        video_frames,
                        metadata_messages,
                        last_frame_id,
                        max(last_detection_count, last_track_count),
                        last_track_count,
                        last_selected_track_id if last_selected_track_id is not None else "-",
                        fps_est,
                        ",".join(last_hud_elements) if last_hud_elements else "-",

                    ),
                    flush=True,
                )
            if cv2.waitKey(1) == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("[ui] shutting down...")
        report = {
            "schema": "PerceptionSnapshotV2",
            "decoded_video_frames": video_frames,
            "metadata_messages": metadata_messages,
            "last_frame_id": last_frame_id,
            "last_object_count": max(last_detection_count, last_track_count),
            "last_track_count": last_track_count,
            "last_selected_track_id": last_selected_track_id,
            "return_decoder": return_decoder,
            "hud_elements": list(last_hud_elements),

            "diagnostics_endpoint": diagnostics_endpoint,
            "feedforward": {
                "state": last_feedforward_state,
                "yaw_rad_s": last_feedforward_yaw,
                "pitch_rad_s": last_feedforward_pitch,
            },
        }
        print("[ui] report " + json.dumps(report, sort_keys=True), flush=True)
        if args.report is not None:
            write_ui_report(args.report, report)
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
        if camstate_sub is not None:
            try:
                camstate_sub.close(0)
            except Exception:
                pass
        if diagnostics_sub is not None:
            try:
                diagnostics_sub.close(0)
            except Exception:
                pass
        try:
            ctx.destroy(linger=0)
        except Exception:
            pass
        # make sure window goes away on all platforms
        for _ in range(3):
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        time.sleep(0.05)

if __name__ == "__main__":
    main()
