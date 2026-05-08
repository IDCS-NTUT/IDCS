"""PC-side video streamer for webcam, file, or simulated sources.

The streamer opens the configured video source, encodes frames with a
GStreamer H.264 pipeline, and pushes frame headers (plus optional simulated
camera state) to the Jetson over ZMQ.
"""

import argparse
import queue
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import cv2
import gi
import zmq
from pydantic import ValidationError

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from common.control import (
    ControlConfig,
    ControlConfigError,
    LaserConfigError,
    LaserMountConfig,
)
from common.config_sync import (
    ConfigSyncError,
    acquire_config_sync_lock,
    clear_sync_marker,
    expand_config_paths,
    merge_config_maps,
    parse_config_text,
    read_snapshot,
    request_startup_state,
    resolve_active_video_profile,
    resolve_config_sync_endpoint,
    sync_as_client,
    write_sync_marker,
)
from common.schemas import CamState, ControlCmd
from common.shutdown import install_signal_handlers
from pc.sim_camera import SimCamera


PIPELINE_TEMPLATE = (
    "appsrc name=src is-live=true block=false do-timestamp=true format=time "  # <-- non-blocking, self timestamps
    "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
    "videoconvert ! "
    "{pre_encode_caps} ! "
    "{encoder_chain} ! "
    "h264parse ! "
    "queue leaky=downstream max-size-buffers=120 max-size-bytes=0 max-size-time=0 ! "  # <-- drop if downstream slow
    "rtph264pay pt=96 config-interval=1 ! "
    "udpsink {udp_bind}host={host} port={port} sync=false async=false"
)


ENCODER_CANDIDATES = (
    {
        "name": "nvh264enc",
        "pre_encode_caps": "video/x-raw,format=NV12,colorimetry=bt709,interlace-mode=progressive,chromasite=mpeg2",
        "encoder_chain": "nvh264enc preset=low-latency-hq zerolatency=true rc-mode=cbr bframes=0 gop-size=30 bitrate={br}",
    },
    {
        "name": "vah264enc",
        "pre_encode_caps": "video/x-raw,format=NV12,colorimetry=bt709,interlace-mode=progressive,chromasite=mpeg2",
        "encoder_chain": "vah264enc rate-control=cbr bitrate={br} keyframe-period=30",
    },
    {
        "name": "x264enc",
        "pre_encode_caps": "video/x-raw,format=I420,colorimetry=bt709,interlace-mode=progressive,chromasite=mpeg2",
        "encoder_chain": "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bitrate={br} byte-stream=true",
    },
)


def build_uplink_pipeline(
    *,
    w: int,
    h: int,
    fps: int,
    br: int,
    host: str,
    port: int,
    bind_ip: Optional[str] = None,
    pre_encode_caps: str,
    encoder_chain: str,
) -> str:
    pre_encode_caps_resolved = pre_encode_caps.format(br=br)
    encoder_chain_resolved = encoder_chain.format(br=br)
    udp_bind = f"bind-address={bind_ip} " if bind_ip else ""
    return PIPELINE_TEMPLATE.format(
        w=w,
        h=h,
        fps=fps,
        br=br,
        host=host,
        port=port,
        udp_bind=udp_bind,
        pre_encode_caps=pre_encode_caps_resolved,
        encoder_chain=encoder_chain_resolved,
    )


def create_video_writer_with_auto_encoder(
    *,
    w: int,
    h: int,
    fps: int,
    br: int,
    host: str,
    port: int,
    bind_ip: Optional[str] = None,
) -> Tuple["GstVideoWriter", str]:
    last_error: Optional[Exception] = None
    for candidate in ENCODER_CANDIDATES:
        enc_name = str(candidate["name"])
        if Gst.ElementFactory.find(enc_name) is None:
            continue
        pipeline = build_uplink_pipeline(
            w=w,
            h=h,
            fps=fps,
            br=br,
            host=host,
            port=port,
            bind_ip=bind_ip,
            pre_encode_caps=str(candidate["pre_encode_caps"]),
            encoder_chain=str(candidate["encoder_chain"]),
        )
        try:
            writer = GstVideoWriter(pipeline, fps=fps)
        except Exception as exc:
            last_error = exc
            print(f"[streamer] Encoder {enc_name} unavailable at runtime ({exc}); trying next.")
            continue
        print(f"[streamer] Using H.264 encoder: {enc_name}")
        return writer, enc_name

    known = ", ".join(str(c["name"]) for c in ENCODER_CANDIDATES)
    if last_error is not None:
        raise SystemExit(
            f"No usable H.264 encoder found ({known}). Last error: {last_error}"
        ) from last_error
    raise SystemExit(
        f"No H.264 encoder plugin found. Install one of: {known}"
    )


def _bind_zmq_to_device_if_configured(socket: zmq.Socket, iface: Optional[str]) -> None:
    """Best-effort Linux interface bind for libzmq builds that expose it."""
    if not iface:
        return
    option = getattr(zmq, "BINDTODEVICE", None)
    if option is None:
        print("[streamer][WARN] net.pc_iface ignored: pyzmq/libzmq lacks BINDTODEVICE")
        return
    try:
        socket.setsockopt_string(option, iface)
    except Exception as exc:
        print(f"[streamer][WARN] net.pc_iface={iface!r} bind failed: {exc}")


class GstVideoWriter:
    def __init__(self, pipeline: str, *, fps: int) -> None:
        self._pipeline = Gst.parse_launch(pipeline)
        self._appsrc = self._pipeline.get_by_name("src")
        if self._appsrc is None:
            raise RuntimeError("GStreamer pipeline missing appsrc named 'src'")
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._frame_count = 0
        self._frame_duration_ns = int(1e9 / fps) if fps > 0 else None
        self._pipeline.set_state(Gst.State.PLAYING)
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame) -> bool:
        if not self._opened:
            return False
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        if self._frame_duration_ns is not None:
            buf.duration = self._frame_duration_ns
            buf.pts = self._frame_count * self._frame_duration_ns
            buf.dts = buf.pts
        self._frame_count += 1
        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            self._opened = False
            return False
        return True

    def end_of_stream(self) -> None:
        if self._appsrc is not None:
            self._appsrc.end_of_stream()

    def release(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self._opened = False


'''
PIPELINE_X264 = (
    "appsrc is-live=true block=true format=time "
    "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
    "videoconvert ! "
    "video/x-raw,format=I420,colorimetry=bt709,interlace-mode=progressive,chromasite=mpeg2 ! "
    "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bitrate={br} byte-stream=true ! "
    "h264parse ! "
    "rtph264pay pt=96 config-interval=1 ! "
    "udpsink host={host} port={port} sync=false async=false"
)
'''

def open_source(
    spec: str,
    w: int,
    h: int,
    fps: int,
    cfg=None,
    *,
    control_cfg: Optional[ControlConfig] = None,
    laser_mount: Optional[LaserMountConfig] = None,
):
    """Open a capture source based on the configured spec.

    Supported specs:
    - ``webcam:<index>`` to open a local webcam device.
    - ``file:<path>`` to read frames from a video file.
    - ``sim`` to use the :class:`pc.sim_camera.SimCamera` generator.

    For ``sim``, renderer settings are sourced from ``cfg["sim"]``:
    ``renderer`` chooses the renderer implementation, ``renderer_opts`` is
    forwarded verbatim to the renderer constructor, and ``debug`` toggles the
    orbit/debug rendering mode.

    ``control_cfg`` influences the simulator by setting the maximum pan/tilt
    rate limits used to clamp incoming control commands. ``laser_mount`` is
    retained on the simulator wrapper for renderers that want access to the
    physical laser mounting metadata, but it does not otherwise affect frame
    generation here.
    """
    spec_clean = str(spec or "").strip()
    spec_lower = spec_clean.lower()

    if spec_lower.startswith("webcam:"):
        idx = int(spec_clean.split(":",1)[1])
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap
    elif spec_lower.startswith("file:"):
        return cv2.VideoCapture(spec_clean.split(":",1)[1])
    elif spec_lower.startswith("sim"):
        sim_cfg = {}
        if cfg is not None:
            try:
                sim_cfg = cfg.get("sim", {})
            except AttributeError:
                sim_cfg = {}
        gimbal_cfg = {}
        if cfg is not None:
            try:
                raw_gimbal_cfg = cfg.get("gimbal", {})
                if isinstance(raw_gimbal_cfg, Mapping):
                    gimbal_cfg = raw_gimbal_cfg
            except AttributeError:
                gimbal_cfg = {}

        def _opt_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        yaw_min_rad = _opt_float(gimbal_cfg.get("yaw_min_rad"))
        yaw_max_rad = _opt_float(gimbal_cfg.get("yaw_max_rad"))
        pitch_min_rad = _opt_float(gimbal_cfg.get("pitch_min_rad"))
        pitch_max_rad = _opt_float(gimbal_cfg.get("pitch_max_rad"))

        if yaw_min_rad is not None and yaw_max_rad is not None and yaw_min_rad >= yaw_max_rad:
            yaw_min_rad = None
            yaw_max_rad = None
        if pitch_min_rad is not None and pitch_max_rad is not None and pitch_min_rad >= pitch_max_rad:
            pitch_min_rad = None
            pitch_max_rad = None

        renderer_name = sim_cfg.get("renderer")
        renderer_opts = sim_cfg.get("renderer_opts")
        debug_mode = sim_cfg.get("debug")
        scene_cfg = sim_cfg.get("scene")
        # Wrap SimCamera into a VideoCapture-like object
        class _SimCap:
            def __init__(
                self,
                W,
                H,
                fps,
                renderer_name=None,
                renderer_opts=None,
                debug_mode=None,
                control_cfg: Optional[ControlConfig] = None,
                laser_mount: Optional[LaserMountConfig] = None,
                encoder_pose_enabled: bool = False,
                encoder_pose_stale_timeout_s: float = 0.5,
                yaw_min_rad: Optional[float] = None,
                yaw_max_rad: Optional[float] = None,
                pitch_min_rad: Optional[float] = None,
                pitch_max_rad: Optional[float] = None,
            ):
                sim_kwargs = {"width": W, "height": H}
                sim_kwargs["fps_hz"] = float(fps)
                if renderer_name is not None:
                    sim_kwargs["renderer_name"] = renderer_name
                if renderer_opts is not None:
                    sim_kwargs["renderer_opts"] = renderer_opts
                if debug_mode is not None:
                    sim_kwargs["debug"] = bool(debug_mode)
                if scene_cfg is not None:
                    sim_kwargs["scene"] = scene_cfg
                self.gen = SimCamera(**sim_kwargs)
                self.period = 1.0 / max(1, fps)
                self._t = time.monotonic()
                self._cmd_timeout = 0.5
                self._last_cmd: Optional[ControlCmd] = None
                self._last_cmd_time: Optional[float] = None
                self._max_pan_rate = (
                    float(control_cfg.rate_limits.yaw)
                    if control_cfg is not None and control_cfg.rate_limits is not None
                    else 1.5
                )
                self._max_tilt_rate = (
                    float(control_cfg.rate_limits.pitch)
                    if control_cfg is not None and control_cfg.rate_limits is not None
                    else 1.0
                )
                self._pan_rate = 0.0
                self._tilt_rate = 0.0
                self._last_pose = self.gen.get_pose()
                self._laser_mount = laser_mount
                self._encoder_pose_enabled = bool(encoder_pose_enabled)
                self._encoder_pose_stale_timeout_s = max(float(encoder_pose_stale_timeout_s), 0.05)
                self._last_cam_state: Optional[CamState] = None
                self._last_cam_state_mono: Optional[float] = None
                self._last_cam_state_log_mono: float = 0.0
                self._cam_state_rx_count: int = 0
                self._yaw_min_rad = yaw_min_rad
                self._yaw_max_rad = yaw_max_rad
                self._pitch_min_rad = pitch_min_rad
                self._pitch_max_rad = pitch_max_rad

            def isOpened(self):
                return True

            def read(self):
                # pace to approx fps
                now = time.monotonic()
                sleep = self.period - (now - self._t)
                if sleep > 0:
                    time.sleep(sleep)
                now = time.monotonic()
                dt = max(0.0, now - self._t)
                self._t = now
                if self._apply_encoder_pose_if_fresh(now):
                    pass
                else:
                    pan_rate, tilt_rate = self._resolve_command(now)
                    self.gen.apply_control_rates(pan_rate, tilt_rate, dt)
                    self._pan_rate = pan_rate
                    self._tilt_rate = tilt_rate
                self._last_pose = self.gen.get_pose()
                return self.gen.next_frame()

            def release(self):
                pass

            def handle_control_cmd(self, payload: dict) -> None:
                try:
                    cmd = ControlCmd(**payload)
                except (ValidationError, TypeError, ValueError):
                    return
                self._last_cmd = cmd
                self._last_cmd_time = time.monotonic()

            def handle_cam_state(self, payload: Mapping[str, Any]) -> None:
                try:
                    cam_state = CamState(**payload)
                except (ValidationError, TypeError, ValueError):
                    return
                self._last_cam_state = cam_state
                self._last_cam_state_mono = time.monotonic()
                self._cam_state_rx_count += 1

            def _resolve_command(self, now: float) -> Tuple[float, float]:
                cmd = self._last_cmd
                if cmd is None:
                    return (0.0, 0.0)
                if self._last_cmd_time is None or (now - self._last_cmd_time) > self._cmd_timeout:
                    return (0.0, 0.0)
                pan = max(-self._max_pan_rate, min(self._max_pan_rate, float(cmd.pan_rate_cmd)))
                tilt = max(-self._max_tilt_rate, min(self._max_tilt_rate, float(cmd.tilt_rate_cmd)))

                pose = self.gen.get_pose() if hasattr(self.gen, "get_pose") else {}
                cur_pan = float(pose.get("pan", 0.0))
                cur_tilt = float(pose.get("tilt", 0.0))

                if self._yaw_max_rad is not None and cur_pan >= self._yaw_max_rad and pan > 0.0:
                    pan = 0.0
                if self._yaw_min_rad is not None and cur_pan <= self._yaw_min_rad and pan < 0.0:
                    pan = 0.0
                if self._pitch_max_rad is not None and cur_tilt >= self._pitch_max_rad and tilt > 0.0:
                    tilt = 0.0
                if self._pitch_min_rad is not None and cur_tilt <= self._pitch_min_rad and tilt < 0.0:
                    tilt = 0.0

                if not cmd.target_ok and abs(pan) < 1e-6 and abs(tilt) < 1e-6:
                    return (0.0, 0.0)
                return (pan, tilt)

            def _apply_encoder_pose_if_fresh(self, now: float) -> bool:
                if not self._encoder_pose_enabled:
                    return False
                if self._last_cam_state is None or self._last_cam_state_mono is None:
                    return False
                age_s = now - self._last_cam_state_mono
                if age_s > self._encoder_pose_stale_timeout_s:
                    if (now - self._last_cam_state_log_mono) >= 1.0:
                        self._last_cam_state_log_mono = now
                        print(
                            "[streamer] CamState stale for %.3fs (> %.3fs); falling back to ControlCmd integration"
                            % (age_s, self._encoder_pose_stale_timeout_s)
                        )
                    return False
                self.gen.apply_cam_state(
                    pan=float(self._last_cam_state.pan),
                    tilt=float(self._last_cam_state.tilt),
                    pan_rate=(
                        float(self._last_cam_state.pan_rate)
                        if self._last_cam_state.pan_rate is not None
                        else None
                    ),
                    tilt_rate=(
                        float(self._last_cam_state.tilt_rate)
                        if self._last_cam_state.tilt_rate is not None
                        else None
                    ),
                )
                self._pan_rate = float(self._last_cam_state.pan_rate or 0.0)
                self._tilt_rate = float(self._last_cam_state.tilt_rate or 0.0)
                return True

            def cam_state_stats(self, now: float) -> Optional[dict[str, float]]:
                if self._last_cam_state_mono is None:
                    return None
                return {
                    "age_s": max(0.0, now - self._last_cam_state_mono),
                    "rx_count": float(self._cam_state_rx_count),
                }

            def build_cam_state(self, frame_id: int, src_ts_ms: int) -> Optional[dict]:
                pose = self._last_pose or {}
                home = {}
                if hasattr(self.gen, "get_home_pose"):
                    try:
                        home = dict(self.gen.get_home_pose() or {})
                    except Exception:
                        home = {}
                return {
                    "type": "CamState",
                    "origin": "pc",
                    "frame_id": frame_id,
                    "src_ts_ms": src_ts_ms,
                    "pan": float(pose.get("pan", 0.0)),
                    "tilt": float(pose.get("tilt", 0.0)),
                    "pan_rate": float(self._pan_rate),
                    "tilt_rate": float(self._tilt_rate),
                    "home_pan": float(home.get("pan", pose.get("pan", 0.0))),
                    "home_tilt": float(home.get("tilt", pose.get("tilt", 0.0))),
                }

        return _SimCap(
            w,
            h,
            fps,
            renderer_name,
            renderer_opts,
            debug_mode,
            control_cfg,
            laser_mount,
            encoder_pose_enabled=bool(sim_cfg.get("use_jetson_cam_state", False)),
            encoder_pose_stale_timeout_s=float(sim_cfg.get("jetson_cam_state_stale_timeout_s", 0.5)),
            yaw_min_rad=yaw_min_rad,
            yaw_max_rad=yaw_max_rad,
            pitch_min_rad=pitch_min_rad,
            pitch_max_rad=pitch_max_rad,
        )
    else:
        raise ValueError(
            "Unknown source, use webcam:<idx> | file:<path> | sim "
            "(or run source:rpi on Jetson receiver)"
        )


def main():
    """Entry point for the PC streamer CLI."""
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
            "Use 0 to skip the handshake and keep the local file."
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
    if args.config_sync_timeout != 0:
        startup_probe_wait: Optional[float] = args.config_sync_timeout
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
                    "[streamer] Startup source override received from Jetson: "
                    f"{startup_source} (local={preview_source or '<unset>'})"
                )
        except ConfigSyncError as exc:
            raise SystemExit(
                "startup handshake failed: "
                f"{exc}. Use --config-sync-timeout=0 only when intentionally skipping sync."
            ) from exc

    source_is_sim = effective_source.startswith("sim")

    skip_sync = args.config_sync_timeout == 0 if args.config_sync_timeout is not None else False
    if not source_is_sim:
        skip_sync = True
    if skip_sync:
        if not source_is_sim:
            print("[streamer] Config sync: skipping handshake (source!=sim)")
        else:
            print("[streamer] Config sync: skipping handshake (--config-sync-timeout=0)")
        final_texts = {path: snapshot.text for path, snapshot in initial_snapshots.items()}
        final_metas = {
            path: snapshot.metadata for path, snapshot in initial_snapshots.items()
        }
        for path in config_paths:
            clear_sync_marker(path)
    else:
        final_texts = {}
        final_metas = {}
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
                            "[streamer] Config sync: updated local configuration "
                            f"(sha256={final_meta.sha256})"
                        )
                    write_sync_marker(path, final_meta)
                    final_texts[path] = final_text
                    final_metas[path] = final_meta
        except ConfigSyncError as exc:
            raise SystemExit(f"config synchronization failed: {exc}") from exc

    cfg = merge_config_maps(
        *(
            parse_config_text(final_texts[path], str(path))
            for path in config_paths
        )
    )
    if effective_source:
        cfg = dict(cfg)
        cfg["source"] = effective_source

    video_cfg, active_profile = resolve_active_video_profile(cfg)
    try:
        w = int(video_cfg["width"])
        h = int(video_cfg["height"])
    except KeyError as exc:
        raise SystemExit("config missing video.width/video.height") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.width/video.height must be integers") from exc
    try:
        fps_value = video_cfg["fps"]
    except KeyError as exc:
        raise SystemExit("config missing video.fps") from exc
    try:
        fps = int(round(float(fps_value)))
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.fps must be numeric") from exc
    if fps <= 0:
        raise SystemExit("video.fps must be positive")
    try:
        br_value = video_cfg["bitrate_kbps"]
    except KeyError as exc:
        raise SystemExit("config missing video.bitrate_kbps") from exc
    try:
        br = int(br_value)
    except (TypeError, ValueError) as exc:
        raise SystemExit("video.bitrate_kbps must be an integer") from exc
    if br <= 0:
        raise SystemExit("video.bitrate_kbps must be positive")

    if active_profile:
        print(
            "[streamer] Using video profile %s (%dx%d @ %d FPS, %d kbps)"
            % (active_profile, w, h, fps, br)
        )

    try:
        control_cfg = ControlConfig.from_raw_config(cfg, (w, h))
    except ControlConfigError as exc:
        raise SystemExit(f"invalid control configuration: {exc}") from exc

    try:
        laser_cfg = LaserMountConfig.from_raw_config(cfg)
    except LaserConfigError as exc:
        raise SystemExit(f"invalid laser configuration: {exc}") from exc
    net_cfg = cfg.get("net", {}) if isinstance(cfg, Mapping) else {}
    host,port = net_cfg['jetson_ip'], net_cfg['rtp_port']
    pc_bind_ip_raw = net_cfg.get("pc_bind_ip")
    pc_bind_ip = str(pc_bind_ip_raw).strip() if pc_bind_ip_raw else None
    pc_iface_raw = net_cfg.get("pc_iface")
    pc_iface = str(pc_iface_raw).strip() if pc_iface_raw else None

    source_spec = str(cfg.get('source', 'webcam:0'))
    source_lower = source_spec.strip().lower()
    if source_lower.startswith("webcam") or source_lower.startswith("rpi"):
        print("[streamer] source configured for Jetson-side camera ingest; streamer disabled on PC. Exiting.")
        return

    # --- signals
    stop_event = install_signal_handlers()

    # --- ZMQ (local context so we can term())
    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.SNDHWM, 1)
    push.setsockopt(zmq.LINGER, 0)
    _bind_zmq_to_device_if_configured(push, pc_iface)
    push.connect(net_cfg['header_push'])
    is_file_source = source_lower.startswith('file:')
    is_sim_source = source_lower.startswith('sim')

    ctrl_ep = net_cfg.get('zmq_control')
    ctrl_sub: Optional[zmq.Socket] = None
    gimbal_state_sub: Optional[zmq.Socket] = None
    if ctrl_ep and not is_file_source:
        ctrl_sub = ctx.socket(zmq.SUB)
        ctrl_sub.setsockopt(zmq.RCVHWM, 1)
        ctrl_sub.setsockopt(zmq.CONFLATE, 1)
        ctrl_sub.setsockopt(zmq.LINGER, 0)
        ctrl_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        _bind_zmq_to_device_if_configured(ctrl_sub, pc_iface)
        ctrl_sub.connect(ctrl_ep)
        ctrl_sub.RCVTIMEO = 0

    sim_cfg = cfg.get("sim", {}) if isinstance(cfg, Mapping) else {}
    use_jetson_cam_state = bool(sim_cfg.get("use_jetson_cam_state", False))
    gimbal_state_ep = net_cfg.get("zmq_gimbal_state") if isinstance(net_cfg, Mapping) else None
    if is_sim_source and use_jetson_cam_state and gimbal_state_ep:
        gimbal_state_sub = ctx.socket(zmq.SUB)
        gimbal_state_sub.setsockopt(zmq.RCVHWM, 1)
        gimbal_state_sub.setsockopt(zmq.CONFLATE, 1)
        gimbal_state_sub.setsockopt(zmq.LINGER, 0)
        gimbal_state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        _bind_zmq_to_device_if_configured(gimbal_state_sub, pc_iface)
        gimbal_state_sub.connect(str(gimbal_state_ep))
        gimbal_state_sub.RCVTIMEO = 0
        print(f"[streamer] Sim camera pose source: Jetson CamState from {gimbal_state_ep}")

    cap = open_source(
        source_spec,
        w,
        h,
        fps,
        cfg,
        control_cfg=control_cfg,
        laser_mount=laser_cfg,
    )
    if not cap.isOpened():
        raise SystemExit("Failed to open source")

    out, _ = create_video_writer_with_auto_encoder(
        w=w,
        h=h,
        fps=fps,
        br=br,
        host=host,
        port=port,
        bind_ip=pc_bind_ip,
    )
    if not out.isOpened():
        raise SystemExit("Failed to open GStreamer pipeline")

    frame_id = 0
    t0 = time.monotonic_ns()

    frame_queue: queue.Queue = queue.Queue(maxsize=1)
    capture_thread: Optional[threading.Thread] = None

    if not is_sim_source:
        def _capture_worker() -> None:
            can_poll = callable(getattr(cap, "grab", None)) and callable(getattr(cap, "retrieve", None))
            while not stop_event.is_set():
                if can_poll:
                    grabbed = cap.grab()
                    if grabbed:
                        ok, frame = cap.retrieve()
                    else:
                        ok, frame = False, None
                else:
                    ok, frame = cap.read()
                if not ok:
                    continue
                try:
                    frame_queue.put_nowait((ok, frame))
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put_nowait((ok, frame))
                    except queue.Full:
                        pass

        capture_thread = threading.Thread(target=_capture_worker, name="capture-reader", daemon=True)
        capture_thread.start()

    try:
        while not stop_event.is_set():
            if ctrl_sub is not None and hasattr(cap, "handle_control_cmd"):
                try:
                    while True:
                        payload = ctrl_sub.recv_json(flags=zmq.NOBLOCK)
                        cap.handle_control_cmd(payload)
                except zmq.Again:
                    pass

            if gimbal_state_sub is not None and hasattr(cap, "handle_cam_state"):
                try:
                    while True:
                        payload = gimbal_state_sub.recv_json(flags=zmq.NOBLOCK)
                        cap.handle_cam_state(payload)
                except zmq.Again:
                    pass

            if is_sim_source:
                # OpenGL/ModernGL contexts are thread-affine; sim capture must stay on one thread.
                ok, frame = cap.read()
            else:
                try:
                    ok, frame = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            if stop_event.is_set():
                break
            if not ok:
                continue
            frame_id += 1
            src_ts_ms = int(time.monotonic_ns() / 1e6)
            if hasattr(cap, "build_cam_state"):
                cam_state = cap.build_cam_state(frame_id, src_ts_ms)
                if cam_state:
                    try:
                        push.send_json(cam_state, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
            # non-blocking header send
            try:
                push.send_json(
                    {
                        "origin": "pc",
                        "frame_id": frame_id,
                        "src_ts_ms": src_ts_ms,
                    },
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                pass

            h_src, w_src = frame.shape[:2]
            frame_to_write = frame
            if (w_src, h_src) != (w, h):
                frame_to_write = cv2.resize(frame, (w, h))
            if not frame_to_write.flags.c_contiguous:
                frame_to_write = frame_to_write.copy()
            if frame_to_write.shape[0] != h or frame_to_write.shape[1] != w:
                raise RuntimeError(
                    f"encoder frame shape mismatch: got {frame_to_write.shape[1]}x{frame_to_write.shape[0]},"
                    f" expected {w}x{h}"
                )
            if not out.write(frame_to_write):
                stop_event.set()
                break

            if frame_id % max(1,fps*2) == 0:
                dt = (time.monotonic_ns() - t0)/1e9
                print(f"[streamer] Sent {frame_id} frames, ~{frame_id/dt:.1f} FPS")
                if is_sim_source and hasattr(cap, "cam_state_stats"):
                    stats = cap.cam_state_stats(time.monotonic())
                    if stats is not None:
                        print(
                            "[streamer] CamState rx=%d latest_age=%.3fs"
                            % (int(stats["rx_count"]), float(stats["age_s"]))
                        )
    except KeyboardInterrupt:
        pass
    finally:
        print("[streamer] shutting down...")
        stop_event.set()
        if capture_thread is not None:
            capture_thread.join(timeout=2.0)
        try: cap.release()
        except Exception: pass
        try:
            out.end_of_stream()
        except Exception:
            pass
        try:
            out.release()
        except: pass
        try: push.close(0)
        except: pass
        if ctrl_sub is not None:
            try: ctrl_sub.close(0)
            except: pass
        if gimbal_state_sub is not None:
            try: gimbal_state_sub.close(0)
            except: pass
        try: ctx.term()
        except: pass
        # give GStreamer a tick to flush
        time.sleep(0.05)

if __name__ == "__main__":
    main()
