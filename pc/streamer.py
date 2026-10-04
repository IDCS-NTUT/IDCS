"""PC-side video streamer for webcam, file, or simulated sources.

The streamer opens the configured video source, encodes frames with a
GStreamer H.264 pipeline, and pushes frame headers (plus optional simulated
camera state) to the Jetson over ZMQ.
"""

import argparse
from collections import OrderedDict, deque
import json
import math
import queue
import threading
import time
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import urlsplit

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
from common.config import (
    ConfigError,
    load_config_bundle,
    resolve_active_video_profile,
    resolve_config_paths,
)
from common.perception import perception_snapshot_from_json, perception_snapshot_to_json
from common.gimbal.mks_servo42_rs485 import MksServo42Axis
from common.schemas import CamState, ControlCmd, ControlIntent
from common.shutdown import install_signal_handlers
from common.sim_mode import require_simulation_loopback_endpoint, resolve_simulation_motion_mode
from common.rtp_identity import parse_rtp_identity
from pc.sim_camera import SimCamera, build_plant_model
from pc.clock_sync_service import ClockSyncResponder


PIPELINE_TEMPLATE = (
    "appsrc name=src is-live=true block=false do-timestamp=true format=time "  # <-- non-blocking, self timestamps
    "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
    "videoconvert ! "
    "{pre_encode_caps} ! "
    "{encoder_chain} ! "
    "h264parse ! "
    "queue leaky=downstream max-size-buffers=120 max-size-bytes=0 max-size-time=0 ! "  # <-- drop if downstream slow
    "rtph264pay name=rtp_pay pt=96 config-interval=1 ! "
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


POSE_DELAY_MAX_NS = 150_000_000
POSE_DELAY_MARGIN_NS = 5_000_000
# Pose samples per axis before the render delay is fixed (at the p99 gap
# plus margin). With 40 the p99 of 39 gaps was simply the largest gap, so one
# serial retry during warm-up (~100 ms) set a delay above the 100 ms truth
# budget and the HIL streamer refused to start (2026-09-28: 103.5 ms, then
# 121 ms). Over a 5 min HIL run the gaps were p50 43 ms, p99 64-71 ms, max
# 104 ms; 250 samples (~11 s at 23 Hz) estimate that p99.
POSE_WARMUP_SAMPLES = 250
POSE_MAX_UNCOVERED_FRACTION = 0.05


class MeasuredPoseTimeline:
    """Measured gimbal pose for hardware-in-loop rendering, without prediction.

    Each axis keeps the bridge's measured samples relative to its startup
    home, stamped in this host's monotonic clock as
    ``received - (published - measured)``: the subtracted part is an exact
    Jetson-clock difference, so only network delay (~1 ms) remains. After a
    warm-up the render delay ``D`` is fixed just above the observed sample
    gap, so a frame captured at ``now - D`` is always bracketed by two real
    samples and its pose is interpolated, never predicted. ``D`` follows the
    feedback rate automatically (smaller gaps give a smaller ``D``).
    """

    def __init__(self, *, warmup_samples: int = POSE_WARMUP_SAMPLES,
                 margin_ns: int = POSE_DELAY_MARGIN_NS, max_samples: int = 512) -> None:
        if warmup_samples < 3 or margin_ns < 0:
            raise ValueError("invalid pose timeline settings")
        self.warmup_samples = warmup_samples
        self.margin_ns = margin_ns
        # Upper bound for the render delay: the truth latency budget when
        # truth is published (a later render could not meet it).
        self.max_delay_ns = POSE_DELAY_MAX_NS
        # Share of warm-up time a frame would find no arrived sample to
        # bracket it at the chosen delay (those frames stream without truth).
        self.uncovered_fraction: Optional[float] = None
        self._axes: dict[str, deque[tuple[int, int, float]]] = {
            "pan": deque(maxlen=max_samples), "tilt": deque(maxlen=max_samples),
        }
        self.delay_ns: Optional[int] = None
        self.sample_hz: dict[str, float] = {}

    def add(self, state: CamState, received_ns: int) -> bool:
        """Record new per-axis measurements; True if any axis gained one."""

        if (state.home_pan is None or state.home_tilt is None
                or state.state_monotonic_ns is None):
            return False
        pan_delta = float(state.pan) - float(state.home_pan)
        values = {
            "pan": (state.pan_sample_monotonic_ns, math.atan2(math.sin(pan_delta), math.cos(pan_delta))),
            "tilt": (state.tilt_sample_monotonic_ns, float(state.tilt) - float(state.home_tilt)),
        }
        added = False
        for axis, (measured_ns, value) in values.items():
            if measured_ns is None or measured_ns > state.state_monotonic_ns:
                continue
            series = self._axes[axis]
            if series and measured_ns <= series[-1][0]:
                continue  # republished sample
            local_ns = int(received_ns) - (int(state.state_monotonic_ns) - int(measured_ns))
            series.append((int(measured_ns), local_ns, value))
            added = True
        if added and self.delay_ns is None:
            self._maybe_freeze_delay()
        return added

    def _maybe_freeze_delay(self) -> None:
        if min(len(series) for series in self._axes.values()) < self.warmup_samples:
            return
        worst = 0
        all_gaps = {}
        for axis, series in self._axes.items():
            gaps = sorted(b[1] - a[1] for a, b in zip(series, list(series)[1:]))
            all_gaps[axis] = gaps
            worst = max(worst, gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))])
            span_s = (series[-1][1] - series[0][1]) / 1e9
            self.sample_hz[axis] = (len(series) - 1) / span_s if span_s > 0 else 0.0
        self.delay_ns = min(worst + self.margin_ns, self.max_delay_ns)
        # Gaps are receipt-time gaps, so they include delivery jitter; a
        # capture instant is bracketed if the next sample arrives within D.
        self.uncovered_fraction = max(
            sum(max(0, gap - self.delay_ns) for gap in gaps) / max(1, sum(gaps))
            for gaps in all_gaps.values()
        )

    def pose_at(self, t_ns: int) -> Optional[Tuple[float, float, float, float]]:
        """(pan, tilt, pan_rate, tilt_rate) at local time ``t_ns`` if both axes bracket it."""

        out = []
        for axis in ("pan", "tilt"):
            series = list(self._axes[axis])
            pair = next(((a, b) for a, b in zip(series, series[1:]) if a[1] <= t_ns <= b[1]), None)
            if pair is None:
                return None
            (_, t0, v0), (_, t1, v1) = pair
            rate = (v1 - v0) / ((t1 - t0) / 1e9) if t1 > t0 else 0.0
            out.append((v0 + (t_ns - t0) / 1e9 * rate, rate))
        (pan, pan_rate), (tilt, tilt_rate) = out
        return pan, tilt, pan_rate, tilt_rate


def require_simulation_perception_endpoint(
    endpoint: str, name: str, pc_bind_ip: str | None
) -> str:
    """Allow read-only sim truth on loopback or the configured PC LAN address."""

    value = str(endpoint or "").strip()
    parsed = urlsplit(value)
    allowed = {"127.0.0.1", "localhost", "::1"}
    if pc_bind_ip:
        allowed.add(str(pc_bind_ip).strip())
    if parsed.scheme != "tcp" or parsed.hostname not in allowed:
        raise ValueError(
            f"{name} must use loopback or configured net.pc_bind_ip"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError(f"{name} must include a valid port")
    return value


class SourceFrameIds:
    """Generate transport frame IDs that remain ordered across restarts.

    SimCamera owns its own small, sequential render-frame timeline. The outer
    streamer ID is transport provenance, so it uses the process start time in
    Unix microseconds as an epoch and increments locally. A restarted streamer
    therefore cannot fall behind the still-running Jetson header correlator
    merely because its local frame counter returned to one.
    """

    def __init__(self, *, start_time_ns: Optional[int] = None) -> None:
        epoch_ns = time.time_ns() if start_time_ns is None else int(start_time_ns)
        if epoch_ns < 0:
            raise ValueError("start_time_ns must be non-negative")
        self._epoch = epoch_ns // 1_000
        self.frames_sent = 0

    def next(self) -> int:
        self.frames_sent += 1
        return self._epoch + self.frames_sent


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
    verified_rtp_headers: bool = False,
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
            writer = GstVideoWriter(pipeline, fps=fps, verified_rtp_headers=verified_rtp_headers)
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
    def __init__(self, pipeline: str, *, fps: int, verified_rtp_headers: bool = False) -> None:
        self._pipeline = Gst.parse_launch(pipeline)
        self._appsrc = self._pipeline.get_by_name("src")
        if self._appsrc is None:
            raise RuntimeError("GStreamer pipeline missing appsrc named 'src'")
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._frame_count = 0
        self._frame_duration_ns = int(1e9 / fps) if fps > 0 else None
        self._verified_rtp_headers = verified_rtp_headers
        self._identity_headers: queue.SimpleQueue[dict[str, int | str]] = queue.SimpleQueue()
        self._rtp_packets = 0
        self._rtp_markers = 0
        self._rtp_missing_meta = 0
        self._identity_queued = 0
        self._frame_id_caps = Gst.Caps.from_string("timestamp/x-idcs-frame-counter")
        self._source_time_caps = Gst.Caps.from_string("timestamp/x-system-monotonic")
        if verified_rtp_headers:
            payloader = self._pipeline.get_by_name("rtp_pay")
            if payloader is None:
                raise RuntimeError("verified RTP headers require named payloader")
            payloader.get_static_pad("src").add_probe(
                Gst.PadProbeType.BUFFER | Gst.PadProbeType.BUFFER_LIST, self._rtp_probe
            )
        self._pipeline.set_state(Gst.State.PLAYING)
        self._opened = True

    def _rtp_probe(self, _pad, info):
        if info.type & Gst.PadProbeType.BUFFER_LIST:
            buffer_list = info.get_buffer_list()
            for index in range(buffer_list.length()):
                self._inspect_rtp_buffer(buffer_list.get(index))
        elif info.type & Gst.PadProbeType.BUFFER:
            buffer = info.get_buffer()
            if buffer is not None:
                self._inspect_rtp_buffer(buffer)
        return Gst.PadProbeReturn.OK

    def _inspect_rtp_buffer(self, buffer) -> None:
        self._rtp_packets += 1
        ok, mapping = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return
        try:
            packet = parse_rtp_identity(bytes(mapping.data[:12]))
        except ValueError:
            return
        finally:
            buffer.unmap(mapping)
        if not packet.marker:
            return
        self._rtp_markers += 1
        frame_meta = buffer.get_reference_timestamp_meta(self._frame_id_caps)
        time_meta = buffer.get_reference_timestamp_meta(self._source_time_caps)
        if frame_meta is None or time_meta is None:
            self._rtp_missing_meta += 1
            return
        source_time_ns = int(time_meta.timestamp)
        self._identity_headers.put({
            "origin": "pc",
            "frame_id": int(frame_meta.timestamp),
            "src_ts_ms": source_time_ns // 1_000_000,
            "source_time_ns": source_time_ns,
            "source_clock_domain": "pc_monotonic",
            "rtp_ssrc": packet.key.ssrc,
            "rtp_timestamp": packet.key.timestamp,
        })
        self._identity_queued += 1

    def identity_report(self) -> dict[str, int]:
        return {
            "rtp_packets": self._rtp_packets,
            "rtp_markers": self._rtp_markers,
            "rtp_missing_meta": self._rtp_missing_meta,
            "identity_queued": self._identity_queued,
        }

    def wait_identity_header(self, timeout_s: float) -> dict[str, int | str] | None:
        try:
            return self._identity_headers.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame, *, frame_id: int | None = None, source_time_ns: int | None = None) -> bool:
        if not self._opened:
            return False
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        if self._frame_duration_ns is not None:
            buf.duration = self._frame_duration_ns
            buf.pts = self._frame_count * self._frame_duration_ns
            buf.dts = buf.pts
        if self._verified_rtp_headers:
            if frame_id is None or source_time_ns is None:
                raise ValueError("verified RTP headers require frame ID and source time")
            if buf.add_reference_timestamp_meta(
                self._frame_id_caps, frame_id, Gst.CLOCK_TIME_NONE
            ) is None or buf.add_reference_timestamp_meta(
                self._source_time_caps, source_time_ns, Gst.CLOCK_TIME_NONE
            ) is None:
                raise RuntimeError("cannot attach source-frame metadata to video buffer")
        self._frame_count += 1
        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            self._opened = False
            return False
        return True

    def end_of_stream(self) -> None:
        if self._appsrc is not None:
            self._appsrc.emit("end-of-stream")

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

class CamStateReceiver:
    """Receive gimbal CamState on its own thread, stamping each on arrival.

    The pose timeline maps the bridge's sample times onto this host's clock
    from the receipt time, so receipt must not wait for the render loop (the
    OpenGL renderer can stall for 100+ ms). Every message is kept: with a
    latest-only socket read once per frame, a slow frame would drop samples.
    """

    def __init__(self, ctx: zmq.Context, endpoint: str, pc_iface: Optional[str] = None) -> None:
        self._socket = ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1000)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        _bind_zmq_to_device_if_configured(self._socket, pc_iface)
        self._socket.connect(str(endpoint))
        self._queue: deque[tuple[dict, int]] = deque(maxlen=1000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camstate-rx", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                return
            received_ns = time.monotonic_ns()
            try:
                self._queue.append((json.loads(raw), received_ns))
            except (ValueError, UnicodeDecodeError):
                continue

    def drain(self) -> list[tuple[dict, int]]:
        items = []
        while self._queue:
            items.append(self._queue.popleft())
        return items

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close(0)


def open_source(
    spec: str,
    w: int,
    h: int,
    fps: int,
    cfg=None,
    *,
    control_cfg: Optional[ControlConfig] = None,
    laser_mount: Optional[LaserMountConfig] = None,
    sim_control_enabled: bool = False,
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
    rate limits used to clamp incoming control commands.  Plant dynamics are
    advanced only when ``sim_control_enabled`` confirms that an explicit
    simulator-only command source is attached; otherwise a passive video run
    holds camera pose. ``laser_mount`` is retained on the simulator wrapper for
    renderers that want access to the physical laser mounting metadata, but it
    does not otherwise affect frame generation here.
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
        # ControlIntent rates run through the same measured F6 speed model and
        # caps as the gimbal bridge, so the simulated mount moves in the real
        # actuator's speed steps.
        intent_rate_caps = (
            _opt_float(gimbal_cfg.get("yaw_rate_limit_rad_s")),
            _opt_float(gimbal_cfg.get("pitch_rate_limit_rad_s")),
        )

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
        camera_cfg = sim_cfg.get("camera")
        plant_model_cfg = sim_cfg.get("plant_model")
        freeze_frame = bool(sim_cfg.get("freeze_frame", False))
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
                freeze_frame: bool = False,
                camera_cfg: Any = None,
                plant_model_cfg: Any = None,
                sim_control_enabled: bool = False,
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
                if camera_cfg is not None:
                    sim_kwargs["camera"] = camera_cfg
                if plant_model_cfg is not None:
                    sim_kwargs["plant_model"] = plant_model_cfg
                if control_cfg is not None:
                    sim_kwargs["threat_eval"] = control_cfg.threat_eval
                if laser_mount is not None:
                    sim_kwargs["laser_mount"] = laser_mount
                self.gen = SimCamera(**sim_kwargs)
                self._render_frame_by_transport: "OrderedDict[int, int]" = OrderedDict()
                print(
                    json.dumps(
                        {
                            "sim_camera_model": self.gen.get_camera_model_info(),
                            "sim_plant_model": self.gen.get_plant_model_info(),
                        },
                        sort_keys=True,
                    )
                )
                self.period = 1.0 / max(1, fps)
                self._t = time.monotonic()
                self._cmd_timeout = 0.5
                self._last_cmd: Optional[ControlCmd] = None
                self._last_cmd_time: Optional[float] = None
                self._last_intent: Optional[ControlIntent] = None
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
                self._last_cam_state_mono: Optional[float] = None
                self._cam_state_rx_count: int = 0
                # Hardware-in-loop: render from measured motor pose only.
                self.pose_timeline = MeasuredPoseTimeline() if self._encoder_pose_enabled else None
                self.frame_has_measured_pose = not self._encoder_pose_enabled
                self.frames_without_measured_pose = 0
                self._yaw_min_rad = yaw_min_rad
                self._yaw_max_rad = yaw_max_rad
                self._pitch_min_rad = pitch_min_rad
                self._pitch_max_rad = pitch_max_rad
                self._freeze_frame = bool(freeze_frame)
                self._sim_control_enabled = bool(sim_control_enabled)
                self._frozen_frame = None

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
                self.last_frame_source_ns = time.monotonic_ns()
                if self.pose_timeline is not None:
                    self._apply_measured_pose(self.last_frame_source_ns)
                elif self._sim_control_enabled:
                    pan_rate, tilt_rate = self._resolve_command(now)
                    self.gen.apply_control_rates(pan_rate, tilt_rate, dt)
                self._last_pose = self.gen.get_pose()
                self._pan_rate = float(self._last_pose.get("pan_rate", 0.0))
                self._tilt_rate = float(self._last_pose.get("tilt_rate", 0.0))
                if self._freeze_frame:
                    if self._frozen_frame is None:
                        ok, rendered = self.gen.next_frame()
                        if not ok:
                            return False, None
                        self._frozen_frame = rendered.copy()
                    return True, self._frozen_frame
                return self.gen.next_frame()

            def release(self):
                pass

            def handle_control_cmd(self, payload: dict) -> None:
                if isinstance(payload, Mapping) and payload.get("type") == "ControlIntent":
                    try:
                        self._last_intent = ControlIntent(**payload)
                    except (ValidationError, TypeError, ValueError):
                        return
                    self._last_cmd = None
                    return
                try:
                    cmd = ControlCmd(**payload)
                except (ValidationError, TypeError, ValueError):
                    return
                self._last_cmd = cmd
                self._last_cmd_time = time.monotonic()
                self._last_intent = None

            def _resolve_intent(self, intent: ControlIntent) -> Tuple[float, float]:
                """Live, unexpired intent rates as the F6 actuator would run them.

                The controller shares this host's monotonic clock, so the
                lease is checked exactly; an expired or shadow intent is zero.
                """
                if intent.mode != "live" or time.monotonic_ns() > intent.valid_until_monotonic_ns:
                    return (0.0, 0.0)
                return (
                    MksServo42Axis.quantized_speed_rad_s(intent.yaw_rate_rad_s, 1.0, intent_rate_caps[0]),
                    MksServo42Axis.quantized_speed_rad_s(intent.pitch_rate_rad_s, 1.0, intent_rate_caps[1]),
                )

            def handle_cam_state(self, payload: Mapping[str, Any], received_ns: Optional[int] = None) -> None:
                try:
                    cam_state = CamState(**payload)
                except (ValidationError, TypeError, ValueError):
                    return
                received_ns = time.monotonic_ns() if received_ns is None else int(received_ns)
                self._last_cam_state_mono = received_ns / 1e9
                self._cam_state_rx_count += 1
                if self.pose_timeline is not None:
                    self.pose_timeline.add(cam_state, received_ns)

            def planner_eval_enabled(self) -> bool:
                enabled = getattr(self.gen, "planner_eval_enabled", None)
                return bool(enabled()) if callable(enabled) else False

            def note_transport_frame(self, transport_frame_id: int) -> None:
                """Remember which render frame went out under this transport id."""
                render_frame_id = getattr(self.gen, "_frame_id", None)
                if render_frame_id is None:
                    return
                self._render_frame_by_transport[int(transport_frame_id)] = int(render_frame_id)
                while len(self._render_frame_by_transport) > 1024:
                    self._render_frame_by_transport.popitem(last=False)

            def handle_perception_feedback(self, payload: Any) -> None:
                apply_feedback = getattr(self.gen, "apply_perception_feedback", None)
                if not callable(apply_feedback):
                    return
                try:
                    snapshot = perception_snapshot_from_json(payload)
                except (ValidationError, TypeError, ValueError):
                    return
                # Snapshots carry the transport frame id; the simulator's
                # timeline is its own render frame. Unknown frames are dropped.
                render_frame_id = self._render_frame_by_transport.get(int(snapshot.frame.frame_id))
                if render_frame_id is None:
                    return
                apply_feedback(snapshot, frame_id=render_frame_id)

            def _resolve_command(self, now: float) -> Tuple[float, float]:
                cmd = self._last_cmd
                if self._last_intent is not None:
                    pan, tilt = self._resolve_intent(self._last_intent)
                elif cmd is None:
                    return (0.0, 0.0)
                elif self._last_cmd_time is None or (now - self._last_cmd_time) > self._cmd_timeout:
                    return (0.0, 0.0)
                else:
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

                if cmd is not None and not cmd.target_ok and abs(pan) < 1e-6 and abs(tilt) < 1e-6:
                    return (0.0, 0.0)
                return (pan, tilt)

            def _apply_measured_pose(self, now_ns: int) -> None:
                """Render the world at ``now - D`` using the interpolated measured pose.

                The frame is stamped with that capture time. Until ``D`` is fixed,
                or when no measured sample brackets the capture time (a late
                sample), the frame is still streamed but carries no perception.
                """
                timeline = self.pose_timeline
                assert timeline is not None
                delay = timeline.delay_ns
                capture_ns = now_ns - (delay if delay is not None else POSE_DELAY_MAX_NS)
                pose = timeline.pose_at(capture_ns) if delay is not None else None
                self.last_frame_source_ns = capture_ns
                self.frame_has_measured_pose = pose is not None
                if pose is None:
                    if delay is not None:
                        self.frames_without_measured_pose += 1
                    return
                pan, tilt, pan_rate, tilt_rate = pose
                self.gen.apply_cam_state(pan=pan, tilt=tilt, pan_rate=pan_rate, tilt_rate=tilt_rate)
                self._pan_rate = pan_rate
                self._tilt_rate = tilt_rate

            def cam_state_stats(self, now: float) -> Optional[dict[str, float]]:
                if self._last_cam_state_mono is None:
                    return None
                stats = {
                    "age_s": max(0.0, now - self._last_cam_state_mono),
                    "rx_count": float(self._cam_state_rx_count),
                }
                if self.pose_timeline is not None:
                    stats["render_delay_ms"] = (
                        -1.0 if self.pose_timeline.delay_ns is None
                        else self.pose_timeline.delay_ns / 1e6
                    )
                    stats["frames_without_measured_pose"] = float(self.frames_without_measured_pose)
                return stats

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
                    **({} if self._encoder_pose_enabled else {
                        # Simulated mount: the pose is exact at the frame's
                        # render instant, on this host's monotonic clock.
                        "pan_sample_monotonic_ns": int(self.last_frame_source_ns),
                        "tilt_sample_monotonic_ns": int(self.last_frame_source_ns),
                        "state_monotonic_ns": time.monotonic_ns(),
                    }),
                }

            def build_ground_truth_snapshot(self, frame_id: int, source_time_ns: int):
                """Exact target truth for this frame; none if the frame had no measured pose."""
                build_snapshot = getattr(self.gen, "build_ground_truth_snapshot", None)
                if not callable(build_snapshot) or not self.frame_has_measured_pose:
                    return None
                return build_snapshot(frame_id, int(source_time_ns))

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
            freeze_frame=freeze_frame,
            camera_cfg=camera_cfg,
            plant_model_cfg=plant_model_cfg,
            sim_control_enabled=sim_control_enabled,
        )
    else:
        raise ValueError(
            "Unknown source, use webcam:<idx> | file:<path> | sim "
            "(or run source:rpi on Jetson receiver)"
        )


class FrameDump:
    """Write every sent frame (MJPEG AVI) plus a per-frame sidecar for offline studies.

    The sidecar line for frame ``index`` in the AVI carries its transport frame id,
    source time, camera pose and the simulator's truth boxes. Writing runs on its
    own thread behind a bounded queue: a full queue drops the frame (and its
    sidecar line) instead of stalling the stream; drops are counted.
    """

    def __init__(self, directory: str, fps: float, size: Tuple[int, int]) -> None:
        from pathlib import Path

        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(self.dir / "frames.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                       float(fps), size)
        if not self._writer.isOpened():
            raise RuntimeError(f"cannot open frame dump writer in {self.dir}")
        self._sidecar = open(self.dir / "frames.jsonl", "w", encoding="utf-8")
        self._queue: "queue.Queue" = queue.Queue(maxsize=240)
        self.written = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="frame-dump", daemon=True)
        self._thread.start()

    def put(self, frame, record: dict) -> None:
        try:
            self._queue.put_nowait((frame.copy(), record))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, record = item
            self._writer.write(frame)
            record["index"] = self.written
            self._sidecar.write(json.dumps(record) + "\n")
            self.written += 1

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30.0)
        self._writer.release()
        self._sidecar.close()
        print(json.dumps({"frame_dump": {"dir": str(self.dir), "written": self.written,
                                         "dropped": self.dropped}}), flush=True)


def main():
    """Entry point for the PC streamer CLI."""
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
    ap.add_argument("--check", action="store_true", help="validate config without opening sockets or video")
    ap.add_argument(
        "--verified-rtp-headers", action="store_true",
        help="opt-in frame metadata keyed by the encoded RTP SSRC/timestamp",
    )
    ap.add_argument("--source", help="explicit source override (for example file:/tmp/sweep.avi)")
    ap.add_argument("--dump-frames", metavar="DIR",
                    help="also write every sent frame (MJPEG AVI) and its id, pose and truth to DIR")
    ap.add_argument(
        "--pace-file",
        action="store_true",
        help="read file sources at the configured video FPS instead of latest-only capture",
    )
    ap.add_argument(
        "--sim-control-sub",
        help="explicit non-production ControlCmd endpoint for simulator experiments",
    )
    ap.add_argument(
        "--sim-camstate-pub",
        help="explicit loopback CamState PUB endpoint for simulator experiments",
    )
    ap.add_argument(
        "--sim-perception-pub",
        help="explicit loopback ground-truth PerceptionSnapshot V2 endpoint",
    )
    ap.add_argument(
        "--sim-events-pub",
        help="loopback PUB for planner-eval events (spawn/eliminated/breach) for the flight recorder",
    )
    ap.add_argument(
        "--sim-total-latency-ms",
        type=float,
        default=0.0,
        help="capture-to-publication latency of simulator truth (includes the measured-pose "
             "render delay in hardware-in-loop); study-only",
    )
    args = ap.parse_args()

    if args.duration_s is not None and args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    if not 0.0 <= args.sim_total_latency_ms <= 400.0:
        raise SystemExit("--sim-total-latency-ms must be in [0, 400]")
    if args.sim_total_latency_ms and not args.sim_perception_pub:
        raise SystemExit("--sim-total-latency-ms requires --sim-perception-pub")
    config_paths = resolve_config_paths(args.config, args.config_extra)
    try:
        bundle = load_config_bundle(config_paths, required_sections=("net", "video"))
        cfg = bundle.mutable_copy()
    except ConfigError as exc:
        raise SystemExit(f"invalid configuration: {exc}") from exc
    print(json.dumps({"mode": "v2_local_config", **bundle.provenance()}, sort_keys=True))

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
    clock_sync_endpoint = net_cfg.get("zmq_source_clock_sync")
    if clock_sync_endpoint:
        try:
            clock_sync_endpoint = require_simulation_perception_endpoint(
                str(clock_sync_endpoint), "net.zmq_source_clock_sync", pc_bind_ip
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    pc_iface_raw = net_cfg.get("pc_iface")
    pc_iface = str(pc_iface_raw).strip() if pc_iface_raw else None

    source_spec = str(args.source or cfg.get('source', 'webcam:0'))
    source_lower = source_spec.strip().lower()
    if source_lower.startswith("webcam") or source_lower.startswith("rpi"):
        print("[streamer] source configured for Jetson-side camera ingest; streamer disabled on PC. Exiting.")
        return

    is_sim_source = source_lower.startswith('sim')
    try:
        sim_control_endpoint = (
            require_simulation_loopback_endpoint(
                args.sim_control_sub, "--sim-control-sub"
            )
            if args.sim_control_sub
            else None
        )
        sim_camstate_endpoint = (
            require_simulation_loopback_endpoint(
                args.sim_camstate_pub, "--sim-camstate-pub"
            )
            if args.sim_camstate_pub
            else None
        )
        sim_events_endpoint = (
            require_simulation_loopback_endpoint(args.sim_events_pub, "--sim-events-pub")
            if args.sim_events_pub
            else None
        )
        sim_perception_endpoint = (
            require_simulation_perception_endpoint(
                args.sim_perception_pub, "--sim-perception-pub", pc_bind_ip
            )
            if args.sim_perception_pub
            else None
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (sim_control_endpoint or sim_camstate_endpoint or sim_perception_endpoint) and not is_sim_source:
        raise SystemExit("simulator control/state endpoints require source: sim")
    sim_endpoints = [
        value for value in
        (sim_control_endpoint, sim_camstate_endpoint, sim_perception_endpoint)
        if value is not None
    ]
    if len(set(sim_endpoints)) != len(sim_endpoints):
        raise SystemExit("simulator control, CamState, and perception endpoints must be distinct")
    sim_cfg = cfg.get("sim", {}) if isinstance(cfg, Mapping) else {}
    try:
        sim_motion_mode = resolve_simulation_motion_mode(
            sim_cfg if isinstance(sim_cfg, Mapping) else {}
        )
    except ValueError as exc:
        raise SystemExit(f"invalid simulator motion mode: {exc}") from exc
    if is_sim_source and sim_motion_mode.moves_physical_mount and sim_control_endpoint:
        raise SystemExit(
            "hardware-in-loop sim.use_jetson_cam_state=true cannot use a "
            "simulated ControlCmd endpoint"
        )

    if args.check:
        plant_model_info = None
        if source_lower.startswith("sim"):
            plant_model = build_plant_model(
                sim_cfg.get("plant_model") if isinstance(sim_cfg, Mapping) else None
            )
            plant_model_info = (
                plant_model.describe() if plant_model is not None else {"mode": "ideal"}
            )
            camera_cfg = sim_cfg.get("camera", {}) if isinstance(sim_cfg, Mapping) else {}
            camera_fov_y_deg = float(camera_cfg.get("fov_y_deg", 60.0)) if isinstance(camera_cfg, Mapping) else 60.0
            camera_fov_x_deg = (
                float(camera_cfg["fov_x_deg"]) if isinstance(camera_cfg, Mapping) and camera_cfg.get("fov_x_deg") is not None
                else math.degrees(2 * math.atan(w / h * math.tan(math.radians(camera_fov_y_deg) / 2)))
            )
        print(json.dumps({
            "source": source_spec,
            "video": {"width": w, "height": h, "fps": fps, "bitrate_kbps": br},
            "header_endpoint": net_cfg.get("header_push"),
            "perception_endpoint": net_cfg.get("zmq_perception_v2"),
            "sim_control_endpoint": sim_control_endpoint,
            "sim_camstate_endpoint": sim_camstate_endpoint,
            "sim_perception_endpoint": sim_perception_endpoint,
            "sim_total_latency_ms": args.sim_total_latency_ms,
            "source_clock_sync_bind": clock_sync_endpoint,
            "sim_plant_model": plant_model_info,
            "sim_camera_fov_y_deg": camera_fov_y_deg if source_lower.startswith("sim") else None,
            "sim_camera_fov_x_deg": camera_fov_x_deg if source_lower.startswith("sim") else None,
            "sim_motion_mode": sim_motion_mode.name if is_sim_source else None,
            "moves_physical_mount": sim_motion_mode.moves_physical_mount if is_sim_source else False,
        }, sort_keys=True))
        return

    # --- signals
    stop_event = install_signal_handlers()
    clock_responder = ClockSyncResponder(clock_sync_endpoint) if clock_sync_endpoint else None
    if clock_responder is not None:
        clock_responder.start()
        print(f"[streamer] Source clock sync REP: {clock_sync_endpoint}")

    # --- ZMQ (local context so we can term())
    ctx = zmq.Context()
    push: Optional[zmq.Socket] = None
    if not args.verified_rtp_headers:
        push = ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.SNDHWM, 1)
        push.setsockopt(zmq.LINGER, 0)
        _bind_zmq_to_device_if_configured(push, pc_iface)
        push.connect(net_cfg['header_push'])
    is_file_source = source_lower.startswith('file:')
    paced_file_source = is_file_source and args.pace_file

    ctrl_sub: Optional[zmq.Socket] = None
    sim_state_pub: Optional[zmq.Socket] = None
    sim_perception_pub: Optional[zmq.Socket] = None
    gimbal_state_sub: Optional[CamStateReceiver] = None
    perception_sub: Optional[zmq.Socket] = None
    ctrl_ep = sim_control_endpoint
    if ctrl_ep and is_sim_source:
        if str(ctrl_ep) == str(net_cfg.get("zmq_control", "")):
            raise SystemExit("--sim-control-sub must not use production net.zmq_control")
        ctrl_sub = ctx.socket(zmq.SUB)
        ctrl_sub.setsockopt(zmq.RCVHWM, 1)
        ctrl_sub.setsockopt(zmq.CONFLATE, 1)
        ctrl_sub.setsockopt(zmq.LINGER, 0)
        ctrl_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        _bind_zmq_to_device_if_configured(ctrl_sub, pc_iface)
        ctrl_sub.connect(ctrl_ep)
        ctrl_sub.RCVTIMEO = 0
    if sim_camstate_endpoint and is_sim_source:
        sim_state_pub = ctx.socket(zmq.PUB)
        sim_state_pub.setsockopt(zmq.SNDHWM, 1)
        sim_state_pub.setsockopt(zmq.LINGER, 0)
        sim_state_pub.bind(sim_camstate_endpoint)
        print(f"[streamer] Sim CamState PUB: {sim_camstate_endpoint}")
    sim_events_pub = None
    if sim_events_endpoint and is_sim_source:
        sim_events_pub = ctx.socket(zmq.PUB)
        sim_events_pub.setsockopt(zmq.SNDHWM, 1000)
        sim_events_pub.setsockopt(zmq.LINGER, 0)
        sim_events_pub.bind(sim_events_endpoint)
        print(f"[streamer] Sim planner-eval events PUB: {sim_events_endpoint}")
    if sim_perception_endpoint and is_sim_source:
        sim_perception_pub = ctx.socket(zmq.PUB)
        sim_perception_pub.setsockopt(zmq.SNDHWM, 1)
        sim_perception_pub.setsockopt(zmq.LINGER, 0)
        sim_perception_pub.bind(sim_perception_endpoint)
        print(f"[streamer] Sim ground-truth PerceptionSnapshot PUB: {sim_perception_endpoint}")

    use_jetson_cam_state = sim_motion_mode.use_jetson_cam_state
    gimbal_state_ep = net_cfg.get("zmq_gimbal_state") if isinstance(net_cfg, Mapping) else None
    if is_sim_source and use_jetson_cam_state and gimbal_state_ep:
        gimbal_state_sub = CamStateReceiver(ctx, str(gimbal_state_ep), pc_iface)
        print(f"[streamer] Sim camera pose source: Jetson CamState from {gimbal_state_ep}")

    cap = open_source(
        source_spec,
        w,
        h,
        fps,
        cfg,
        control_cfg=control_cfg,
        laser_mount=laser_cfg,
        sim_control_enabled=bool(ctrl_sub is not None),
    )
    if not cap.isOpened():
        raise SystemExit("Failed to open source")

    planner_eval_enabled = (
        is_sim_source
        and hasattr(cap, "planner_eval_enabled")
        and bool(cap.planner_eval_enabled())
    )
    perception_ep = net_cfg.get("zmq_perception_v2") if isinstance(net_cfg, Mapping) else None
    if planner_eval_enabled and perception_ep:
        perception_sub = ctx.socket(zmq.SUB)
        perception_sub.setsockopt(zmq.RCVHWM, 1)
        perception_sub.setsockopt(zmq.CONFLATE, 1)
        perception_sub.setsockopt(zmq.LINGER, 0)
        perception_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        _bind_zmq_to_device_if_configured(perception_sub, pc_iface)
        perception_sub.connect(str(perception_ep))
        perception_sub.RCVTIMEO = 0
        print(f"[streamer] Planner-eval feedback source: PerceptionSnapshot V2 from {perception_ep}")

    out, _ = create_video_writer_with_auto_encoder(
        w=w,
        h=h,
        fps=fps,
        br=br,
        host=host,
        port=port,
        bind_ip=pc_bind_ip,
        verified_rtp_headers=args.verified_rtp_headers,
    )
    encoded_headers_sent = 0
    encoded_headers_dropped = 0

    header_sender_stop = threading.Event()
    header_sender_thread: threading.Thread | None = None

    def send_encoded_headers() -> None:
        nonlocal encoded_headers_sent, encoded_headers_dropped
        sender = ctx.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDHWM, 256)
        sender.setsockopt(zmq.LINGER, 0)
        _bind_zmq_to_device_if_configured(sender, pc_iface)
        sender.connect(str(net_cfg['header_push']))
        while not header_sender_stop.is_set():
            encoded_header = out.wait_identity_header(0.01)
            if encoded_header is None:
                continue
            try:
                sender.send_json(encoded_header, flags=zmq.NOBLOCK)
                encoded_headers_sent += 1
            except zmq.Again:
                encoded_headers_dropped += 1
        sender.close(0)
    if not out.isOpened():
        raise SystemExit("Failed to open GStreamer pipeline")
    if args.verified_rtp_headers:
        header_sender_thread = threading.Thread(target=send_encoded_headers, name="rtp-header-sender", daemon=True)
        header_sender_thread.start()

    source_frame_ids = SourceFrameIds()
    frame_dump = FrameDump(args.dump_frames, fps, (w, h)) if args.dump_frames else None
    total_latency_ns = int(args.sim_total_latency_ms * 1_000_000)
    if sim_perception_pub is not None and getattr(cap, "pose_timeline", None) is not None:
        cap.pose_timeline.max_delay_ns = min(POSE_DELAY_MAX_NS, total_latency_ns)
    render_delay_reported = False
    if args.sim_total_latency_ms:
        print(f"[streamer] Simulator truth capture-to-publication latency: {args.sim_total_latency_ms:.1f} ms")
    delayed_perception: deque[tuple[float, str]] = deque()
    t0 = time.monotonic_ns()
    deadline = None if args.duration_s is None else time.monotonic() + args.duration_s
    next_file_frame_at = time.monotonic()

    frame_queue: queue.Queue = queue.Queue(maxsize=1)
    capture_thread: Optional[threading.Thread] = None

    if not is_sim_source and not paced_file_source:
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
                capture_ts_ns = time.monotonic_ns()
                try:
                    frame_queue.put_nowait((ok, frame, capture_ts_ns))
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put_nowait((ok, frame, capture_ts_ns))
                    except queue.Full:
                        pass

        capture_thread = threading.Thread(target=_capture_worker, name="capture-reader", daemon=True)
        capture_thread.start()

    try:
        while not stop_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
            if ctrl_sub is not None and hasattr(cap, "handle_control_cmd"):
                try:
                    while True:
                        payload = ctrl_sub.recv_json(flags=zmq.NOBLOCK)
                        cap.handle_control_cmd(payload)
                except zmq.Again:
                    pass

            if gimbal_state_sub is not None and hasattr(cap, "handle_cam_state"):
                for payload, received_ns in gimbal_state_sub.drain():
                    cap.handle_cam_state(payload, received_ns)

            if perception_sub is not None and hasattr(cap, "handle_perception_feedback"):
                try:
                    while True:
                        payload = perception_sub.recv(flags=zmq.NOBLOCK)
                        cap.handle_perception_feedback(payload)
                except zmq.Again:
                    pass

            if is_sim_source:
                # OpenGL/ModernGL contexts are thread-affine; sim capture must stay on one thread.
                ok, frame = cap.read()
                source_ts_ns = int(getattr(cap, "last_frame_source_ns", time.monotonic_ns()))
            elif paced_file_source:
                next_file_frame_at += 1.0 / fps
                delay = next_file_frame_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                ok, frame = cap.read()
                source_ts_ns = time.monotonic_ns()
                if not ok:
                    break
            else:
                try:
                    ok, frame, source_ts_ns = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            if stop_event.is_set():
                break
            if not ok:
                continue
            frame_id = source_frame_ids.next()
            if hasattr(cap, "note_transport_frame"):
                cap.note_transport_frame(frame_id)
            if sim_events_pub is not None:
                for event in cap.gen.drain_planner_events():
                    try:
                        sim_events_pub.send_string(json.dumps({
                            "type": "planner_eval_event", "transport_frame_id": frame_id,
                            "monotonic_ns": time.monotonic_ns(), **event,
                        }), flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
            src_ts_ms = int(source_ts_ns // 1_000_000)
            header = None
            if hasattr(cap, "build_cam_state"):
                cam_state = cap.build_cam_state(frame_id, src_ts_ms)
                if cam_state:
                    header = cam_state
            if header is None:
                header = {
                    "origin": "pc",
                    "frame_id": frame_id,
                    "src_ts_ms": src_ts_ms,
                }
            if sim_state_pub is not None:
                try:
                    sim_state_pub.send_json(header, flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
            timeline = getattr(cap, "pose_timeline", None)
            if timeline is not None and timeline.delay_ns is not None and not render_delay_reported:
                render_delay_reported = True
                print(json.dumps({
                    "sim_render_delay_ms": timeline.delay_ns / 1e6,
                    "frames_without_pose_expected": timeline.uncovered_fraction,
                    "measured_pose_hz": timeline.sample_hz,
                    "sim_total_latency_ms": args.sim_total_latency_ms,
                }), flush=True)
                # The delay is capped at the truth budget; frames whose pose
                # sample arrives later stream without truth. Refuse only when
                # that would be a substantial share of frames.
                if (sim_perception_pub is not None and timeline.uncovered_fraction is not None
                        and timeline.uncovered_fraction > POSE_MAX_UNCOVERED_FRACTION):
                    print(f"[streamer] {100 * timeline.uncovered_fraction:.1f}% of frames would have no "
                          f"measured pose within --sim-total-latency-ms {args.sim_total_latency_ms:.1f} "
                          f"(pose rate {timeline.sample_hz}); stopping")
                    stop_event.set()
                    break
            truth_snapshot = None
            if sim_perception_pub is not None and hasattr(cap, "build_ground_truth_snapshot"):
                snapshot = truth_snapshot = cap.build_ground_truth_snapshot(frame_id, source_ts_ns)
                if snapshot is not None:
                    # Publish exactly ``total`` after capture, whatever the render delay.
                    delayed_perception.append((
                        (source_ts_ns + total_latency_ns) / 1e9,
                        perception_snapshot_to_json(snapshot),
                    ))
                while delayed_perception and delayed_perception[0][0] <= time.monotonic():
                    _, payload = delayed_perception.popleft()
                    try:
                        sim_perception_pub.send_string(payload, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
            if not args.verified_rtp_headers:
                # Legacy order-based correlation remains an explicit rollback.
                assert push is not None
                try:
                    push.send_json(header, flags=zmq.NOBLOCK)
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
            if not out.write(
                frame_to_write,
                frame_id=frame_id if args.verified_rtp_headers else None,
                source_time_ns=source_ts_ns if args.verified_rtp_headers else None,
            ):
                stop_event.set()
                break
            if frame_dump is not None:
                if truth_snapshot is None and hasattr(cap, "build_ground_truth_snapshot"):
                    truth_snapshot = cap.build_ground_truth_snapshot(frame_id, source_ts_ns)
                frame_dump.put(frame_to_write, {
                    "frame_id": frame_id, "source_time_ns": source_ts_ns,
                    "pan": header.get("pan"), "tilt": header.get("tilt"),
                    "truth": [] if truth_snapshot is None else [
                        {"id": t.track_id, "x": t.box.x, "y": t.box.y, "w": t.box.w, "h": t.box.h}
                        for t in truth_snapshot.tracks],
                })

            if source_frame_ids.frames_sent % max(1, fps * 2) == 0:
                dt = (time.monotonic_ns() - t0)/1e9
                frames_sent = source_frame_ids.frames_sent
                print(f"[streamer] Sent {frames_sent} frames, ~{frames_sent/dt:.1f} FPS")
                if planner_eval_enabled:
                    planner_stats = cap.gen.get_planner_eval_stats()
                    if planner_stats is not None:
                        print(json.dumps({"planner_eval": planner_stats}, sort_keys=True))
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
        if frame_dump is not None:
            frame_dump.close()
        if clock_responder is not None:
            clock_responder.close()
        if capture_thread is not None:
            capture_thread.join(timeout=2.0)
        try: cap.release()
        except Exception: pass
        try:
            out.end_of_stream()
        except Exception:
            pass
        if args.verified_rtp_headers:
            header_sender_stop.set()
            if header_sender_thread is not None:
                header_sender_thread.join(timeout=1.0)
            print(json.dumps({
                "verified_rtp_header_sender": out.identity_report(),
                "headers_sent": encoded_headers_sent,
                "headers_dropped": encoded_headers_dropped,
            }, sort_keys=True))
        try:
            out.release()
        except: pass
        if push is not None:
            try: push.close(0)
            except: pass
        if ctrl_sub is not None:
            try: ctrl_sub.close(0)
            except: pass
        if sim_state_pub is not None:
            try: sim_state_pub.close(0)
            except: pass
        if sim_perception_pub is not None:
            try: sim_perception_pub.close(0)
            except: pass
        if gimbal_state_sub is not None:
            try: gimbal_state_sub.close()
            except: pass
        if perception_sub is not None:
            try: perception_sub.close(0)
            except: pass
        try: ctx.destroy(linger=0)
        except: pass
        # give GStreamer a tick to flush
        time.sleep(0.05)

if __name__ == "__main__":
    main()
