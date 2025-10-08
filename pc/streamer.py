import argparse
import time
from typing import Optional, Tuple

import cv2
import yaml
import zmq
from pydantic import ValidationError

from common.control import (
    ControlConfig,
    ControlConfigError,
    LaserConfigError,
    LaserMountConfig,
)
from common.debug import DebugConfig, DebugConfigError
from common.schemas import ControlCmd
from common.shutdown import install_signal_handlers
from pc.sim_camera import SimCamera


PIPELINE = (
    "appsrc is-live=true block=false do-timestamp=true format=time "  # <-- non-blocking, self timestamps
    "caps=video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1 ! "
    "videoconvert ! "
    "video/x-raw,format=NV12,colorimetry=bt709,interlace-mode=progressive,chromasite=mpeg2 ! "
    "nvh264enc preset=low-latency-hq zerolatency=true rc-mode=cbr bframes=0 gop-size=30 bitrate={br} ! "
    "h264parse ! "
    "queue leaky=downstream max-size-buffers=120 max-size-bytes=0 max-size-time=0 ! "  # <-- drop if downstream slow
    "rtph264pay pt=96 config-interval=1 ! "
    "udpsink host={host} port={port} sync=false async=false"
)


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
    debug_cfg: Optional[DebugConfig] = None,
):
    if spec.startswith("webcam:"):
        idx = int(spec.split(":",1)[1])
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap
    elif spec.startswith("file:"):
        return cv2.VideoCapture(spec.split(":",1)[1])
    elif spec.startswith("sim"):
        sim_cfg = {}
        if cfg is not None:
            try:
                sim_cfg = cfg.get("sim", {})
            except AttributeError:
                sim_cfg = {}
        renderer_name = sim_cfg.get("renderer")
        renderer_opts = sim_cfg.get("renderer_opts")
        debug_mode = sim_cfg.get("debug")
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
                debug_cfg: Optional[DebugConfig] = None,
            ):
                sim_kwargs = {"width": W, "height": H}
                if renderer_name is not None:
                    sim_kwargs["renderer_name"] = renderer_name
                if renderer_opts is not None:
                    sim_kwargs["renderer_opts"] = renderer_opts
                if debug_mode is not None:
                    sim_kwargs["debug"] = bool(debug_mode)
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
                self._debug_cfg = debug_cfg

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

            def _resolve_command(self, now: float) -> Tuple[float, float]:
                cmd = self._last_cmd
                if cmd is None:
                    return (0.0, 0.0)
                if self._last_cmd_time is None or (now - self._last_cmd_time) > self._cmd_timeout:
                    return (0.0, 0.0)
                pan = max(-self._max_pan_rate, min(self._max_pan_rate, float(cmd.pan_rate_cmd)))
                tilt = max(-self._max_tilt_rate, min(self._max_tilt_rate, float(cmd.tilt_rate_cmd)))
                if not cmd.target_ok and abs(pan) < 1e-6 and abs(tilt) < 1e-6:
                    return (0.0, 0.0)
                return (pan, tilt)

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
            debug_cfg,
        )
    else:
        raise ValueError("Unknown source, use webcam:<idx> | file:<path> | sim")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dev.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        debug_cfg = DebugConfig.from_raw_config(cfg)
    except DebugConfigError as exc:
        raise SystemExit(f"invalid debug configuration: {exc}") from exc

    w,h,fps = cfg['video']['width'], cfg['video']['height'], cfg['video']['fps']
    try:
        control_cfg = ControlConfig.from_raw_config(cfg, (w, h))
    except ControlConfigError as exc:
        raise SystemExit(f"invalid control configuration: {exc}") from exc

    try:
        laser_cfg = LaserMountConfig.from_raw_config(cfg)
    except LaserConfigError as exc:
        raise SystemExit(f"invalid laser configuration: {exc}") from exc
    br = cfg['video']['bitrate_kbps']
    host,port = cfg['net']['jetson_ip'], cfg['net']['rtp_port']

    # --- signals
    stop_event = install_signal_handlers()

    # --- ZMQ (local context so we can term())
    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.SNDHWM, 1)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(cfg['net']['header_push'])

    source_spec = str(cfg.get('source', 'webcam:0'))
    is_file_source = source_spec.startswith('file:')

    ctrl_ep = cfg['net'].get('zmq_control')
    ctrl_sub: Optional[zmq.Socket] = None
    if ctrl_ep and not is_file_source:
        ctrl_sub = ctx.socket(zmq.SUB)
        ctrl_sub.setsockopt(zmq.RCVHWM, 1)
        ctrl_sub.setsockopt(zmq.CONFLATE, 1)
        ctrl_sub.setsockopt(zmq.LINGER, 0)
        ctrl_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        ctrl_sub.connect(ctrl_ep)
        ctrl_sub.RCVTIMEO = 0

    cap = open_source(
        source_spec,
        w,
        h,
        fps,
        cfg,
        control_cfg=control_cfg,
        laser_mount=laser_cfg,
        debug_cfg=debug_cfg,
    )
    if not cap.isOpened():
        raise SystemExit("Failed to open source")

    gst = PIPELINE.format(w=w,h=h,fps=fps, br=br, host=host, port=port)
    out = cv2.VideoWriter(gst, cv2.CAP_GSTREAMER, 0, float(fps), (w,h))
    if not out.isOpened():
        raise SystemExit("Failed to open GStreamer pipeline")

    frame_id = 0
    t0 = time.monotonic_ns()
    try:
        while not stop_event.is_set():
            if ctrl_sub is not None and hasattr(cap, "handle_control_cmd"):
                try:
                    while True:
                        payload = ctrl_sub.recv_json(flags=zmq.NOBLOCK)
                        cap.handle_control_cmd(payload)
                except zmq.Again:
                    pass

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
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
                push.send_json({"frame_id": frame_id, "src_ts_ms": src_ts_ms}, flags=zmq.NOBLOCK)
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
            out.write(frame_to_write)

            if frame_id % max(1,fps*2) == 0:
                dt = (time.monotonic_ns() - t0)/1e9
                print(f"[streamer] Sent {frame_id} frames, ~{frame_id/dt:.1f} FPS")
    except KeyboardInterrupt:
        pass
    finally:
        print("[streamer] shutting down...")
        try: cap.release()
        except: pass
        try: out.release()
        except: pass
        try: push.close(0)
        except: pass
        if ctrl_sub is not None:
            try: ctrl_sub.close(0)
            except: pass
        try: ctx.term()
        except: pass
        # give GStreamer a tick to flush
        time.sleep(0.05)

if __name__ == "__main__":
    main()