"""Build and run the shared metadata-aware DeepStream pipeline on the Jetson.

Examples:
    python -m jetson.deepstream.verify_pipeline /path/to/input.mp4 --paced
    python -m jetson.deepstream.verify_pipeline /path/to/input.mp4 --paced --nvsort
    python -m jetson.deepstream.verify_pipeline --live-argus --duration-s 15
    python -m jetson.deepstream.verify_pipeline --live-argus --duration-s 15 \\
        --gpu-osd --return-h264

The generic YOLO26 COCO engine is only valid for DeepStream parser and
throughput verification. This module is shared by the passive production
launcher and the verification CLI. It has no controller or gimbal dependency
and never emits a control command.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.perception import TrackAssessmentV2
from common.rtp_identity import parse_rtp_identity
from common.shutdown import install_signal_handlers
from jetson.deepstream.async_target_selection import AsyncDeepStreamTargetSelector
from jetson.deepstream.metadata_adapter import (
    FrameTiming, MissedFrameCounter, perception_snapshot_from_metadata, pts_ns_to_ms,
)
from jetson.deepstream.snapshot_transport import SnapshotTransport


REPO_ROOT = Path(__file__).resolve().parents[2]
NVINFER_CONFIG = REPO_ROOT / "configs/deepstream/nvinfer_yolo26n_960.txt"
DS_ROOT = Path("/opt/nvidia/deepstream/deepstream")
# nvtracker profiles. NvSORT associates detections by motion and overlap only;
# NvDCF adds a per-target visual correlation filter that keeps tracking when
# the detector misses (the repo copy is tuned for this system).
TRACKER_CONFIGS = {
    "nvsort": DS_ROOT / "samples/configs/deepstream-app/config_tracker_NvSORT.yml",
    "nvdcf": REPO_ROOT / "configs/deepstream/tracker_nvdcf.yml",
}
UNTRACKED_OBJECT_ID = (1 << 64) - 1
_INVALID_PTS_NS = (1 << 63) - 1


def _valid_pts_ns(value: int) -> int | None:
    """Return a usable PTS or ``None`` for GStreamer CLOCK_TIME_NONE."""

    value = int(value)
    return value if 0 <= value < _INVALID_PTS_NS else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


@dataclass
class StageClock:
    """Correlate ordered single-stream replay stages using same-host clocks.

    ``nvstreammux`` may rewrite buffer PTS, so PTS cannot safely connect a
    decoder buffer to the metadata buffer.  This verifier has exactly one
    source and batch size one; its element order is preserved, allowing FIFO
    correlation for timing only.  PTS remains the source timestamp carried in
    the V2 perception frame provenance (and the explicit legacy projection).
    """

    decoded_at_s: deque[float] = field(default_factory=deque)
    infer_input_at_s: deque[tuple[float | None, float]] = field(default_factory=deque)
    decoded_buffers: int = 0
    infer_input_buffers: int = 0
    metadata_buffers: int = 0
    matched_infer_inputs: int = 0

    def record_decode(self, now_s: float) -> None:
        self.decoded_buffers += 1
        self.decoded_at_s.append(now_s)

    def record_infer_input(self, now_s: float) -> None:
        self.infer_input_buffers += 1
        decoded_at_s = self.decoded_at_s.popleft() if self.decoded_at_s else None
        self.infer_input_at_s.append((decoded_at_s, now_s))

    def consume(self, infer_output_at_s: float) -> tuple[float | None, float | None, float]:
        """Return decode→infer-input, infer, and infer-input timestamps in ms."""

        self.metadata_buffers += 1
        if not self.infer_input_at_s:
            return None, None, infer_output_at_s * 1000.0
        decoded_at_s, infer_input_at_s = self.infer_input_at_s.popleft()
        decode_to_infer_input_ms = (
            None if decoded_at_s is None or infer_input_at_s is None else (infer_input_at_s - decoded_at_s) * 1000.0
        )
        infer_ms = None if infer_input_at_s is None else (infer_output_at_s - infer_input_at_s) * 1000.0
        if infer_input_at_s is not None:
            self.matched_infer_inputs += 1
        rx_ts_ms = (infer_input_at_s or infer_output_at_s) * 1000.0
        return decode_to_infer_input_ms, infer_ms, rx_ts_ms

    def report(self) -> dict[str, int]:
        return {
            "decoded_buffers": self.decoded_buffers,
            "infer_input_buffers": self.infer_input_buffers,
            "metadata_buffers": self.metadata_buffers,
            "matched_infer_inputs": self.matched_infer_inputs,
            "unmatched_infer_inputs": max(self.metadata_buffers - self.matched_infer_inputs, 0),
            "unmatched_decoded_buffers": len(self.decoded_at_s),
            "correlation": "fifo_single_source_batch_1",
        }


@dataclass
class VerificationStats:
    started_at_s: float = field(default_factory=time.monotonic)
    frames: int = 0
    frames_with_objects: int = 0
    objects: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)
    tracker_ids: set[int] = field(default_factory=set)
    # Objects the tracker carried without a detector match (NvDCF visual tracking).
    tracker_only_objects: int = 0
    missed_frames: MissedFrameCounter = field(default_factory=MissedFrameCounter)
    first_pts_ns: int | None = None
    last_pts_ns: int | None = None
    first_frame_at_s: float | None = None
    last_frame_at_s: float | None = None
    decode_to_infer_input_ms: list[float] = field(default_factory=list)
    infer_stage_ms: list[float] = field(default_factory=list)
    encoded_buffers: int = 0
    return_frames: int = 0
    ready_file: Path | None = None
    _ready_written: bool = False
    health_file: Path | None = None
    _last_health_write_s: float = 0.0

    def current_pipeline_fps(self) -> float:
        return self.frames / max(time.monotonic() - self.started_at_s, 1e-9)

    def record_frame(
        self,
        frame_meta: Any,
        *,
        decode_to_infer_input_ms: float | None,
        infer_stage_ms: float | None,
    ) -> None:
        now_s = time.monotonic()
        self.frames += 1
        self.first_frame_at_s = now_s if self.first_frame_at_s is None else self.first_frame_at_s
        if self.frames == 1 and self.ready_file is not None:
            try:
                self.ready_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.ready_file.with_suffix(self.ready_file.suffix + ".tmp")
                temporary.write_text(json.dumps({"frames": 1, "ready_at_monotonic_s": now_s}) + "\n", encoding="utf-8")
                temporary.replace(self.ready_file)
                self._ready_written = True
            except OSError as exc:
                print(f"[deepstream.verify] unable to write readiness file: {exc}", flush=True)
        if self.health_file is not None and (self.frames == 1 or now_s - self._last_health_write_s >= 1.0):
            try:
                self.health_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.health_file.with_suffix(self.health_file.suffix + ".tmp")
                elapsed_s = max(now_s - self.started_at_s, 1e-9)
                temporary.write_text(json.dumps({
                    "frames": self.frames,
                    "last_frame_monotonic_s": now_s,
                    "pipeline_fps": round(self.current_pipeline_fps(), 3),
                    "return_frames": self.return_frames,
                    "return_fps": round(self.return_frames / elapsed_s, 3),
                }) + "\n", encoding="utf-8")
                temporary.replace(self.health_file)
                self._last_health_write_s = now_s
            except OSError as exc:
                print(f"[deepstream.verify] unable to write health file: {exc}", flush=True)
        self.last_frame_at_s = now_s
        pts = _valid_pts_ns(frame_meta.buf_pts)
        if pts is not None:
            self.first_pts_ns = pts if self.first_pts_ns is None else self.first_pts_ns
            self.last_pts_ns = pts
        if decode_to_infer_input_ms is not None:
            self.decode_to_infer_input_ms.append(decode_to_infer_input_ms)
        if infer_stage_ms is not None:
            self.infer_stage_ms.append(infer_stage_ms)
        if self.frames == 1 or self.frames % 300 == 0:
            elapsed_s = max(time.monotonic() - self.started_at_s, 1e-9)
            print(
                f"[deepstream.verify] frames={self.frames} "
                f"wall_fps={self.frames / elapsed_s:.2f} "
                f"return_fps={self.return_frames / elapsed_s:.2f}",
                flush=True,
            )

    def report(
        self,
        *,
        tracker_enabled: bool,
        stage_clock: StageClock,
        gpu_osd_enabled: bool,
        h264_return_enabled: bool,
    ) -> dict[str, Any]:
        elapsed_s = max(time.monotonic() - self.started_at_s, 1e-9)
        steady_wall_elapsed_s = None
        if self.first_frame_at_s is not None and self.last_frame_at_s is not None:
            steady_wall_elapsed_s = max(self.last_frame_at_s - self.first_frame_at_s, 0.0)
        source_elapsed_s = None
        if self.first_pts_ns is not None and self.last_pts_ns is not None:
            source_elapsed_s = max((self.last_pts_ns - self.first_pts_ns) / 1e9, 0.0)
        return {
            "frames": self.frames,
            "wall_elapsed_s": round(elapsed_s, 3),
            "pipeline_fps": round(self.frames / elapsed_s, 3),
            "steady_wall_elapsed_s": (
                None if steady_wall_elapsed_s is None else round(steady_wall_elapsed_s, 3)
            ),
            "steady_pipeline_fps": (
                None
                if not steady_wall_elapsed_s
                else round((self.frames - 1) / steady_wall_elapsed_s, 3)
            ),
            "source_elapsed_s": None if source_elapsed_s is None else round(source_elapsed_s, 3),
            "source_fps": (
                None
                if not source_elapsed_s
                else round((self.frames - 1) / source_elapsed_s, 3)
            ),
            "frames_with_objects": self.frames_with_objects,
            "objects": self.objects,
            "class_counts": dict(sorted(self.class_counts.items())),
            "tracker_enabled": tracker_enabled,
            "gpu_osd_enabled": gpu_osd_enabled,
            "h264_return_enabled": h264_return_enabled,
            # This is intentionally counted after h264parse, where frame metadata
            # is no longer available.  It proves that the GPU OSD/encoder tail
            # actually produced encoded access units rather than merely accepting
            # upstream DeepStream metadata.
            "encoded_h264_buffers": self.encoded_buffers,
            "return_frames": self.return_frames,
            "return_fps": round(self.return_frames / elapsed_s, 3),
            "unique_tracker_ids": len(self.tracker_ids),
            "tracker_only_objects": self.tracker_only_objects,
            "stage_timing_ms": {
                "decode_to_infer_input_samples": len(self.decode_to_infer_input_ms),
                "decode_to_infer_input_p50": _rounded_percentile(self.decode_to_infer_input_ms, 0.50),
                "decode_to_infer_input_p95": _rounded_percentile(self.decode_to_infer_input_ms, 0.95),
                "infer_and_metadata_samples": len(self.infer_stage_ms),
                "infer_and_metadata_p50": _rounded_percentile(self.infer_stage_ms, 0.50),
                "infer_and_metadata_p95": _rounded_percentile(self.infer_stage_ms, 0.95),
                "pts_match": stage_clock.report(),
            },
        }


def _rounded_percentile(values: list[float], percentile: float) -> float | None:
    result = _percentile(values, percentile)
    return None if result is None else round(result, 3)


def _require_bindings() -> tuple[Any, Any, Any]:
    try:
        import gi
        import pyds

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst
    except Exception as exc:  # pragma: no cover - Jetson-only import boundary
        raise RuntimeError(
            "DeepStream Python bindings are unavailable; activate the Jetson project venv"
        ) from exc
    return Gst, GLib, pyds


def _pipeline_description(
    *,
    input_file: Path | None,
    live_argus: bool,
    rtp_input_port: int | None,
    argus_sensor_id: int,
    argus_sensor_mode: int,
    argus_width: int,
    argus_height: int,
    argus_fps: int,
    nvinfer_config: Path,
    paced: bool,
    tracker: str,
    gpu_osd: bool,
    return_h264: bool,
    return_udp_host: str | None,
    return_udp_port: int | None,
    return_h264_file: Path | None,
    return_width: int = 1280,
    return_height: int = 720,
    return_fps: int = 60,
    return_bitrate_kbps: int = 8000,
) -> str:
    if input_file is not None and "'" in str(input_file):
        raise ValueError("input path cannot contain a single quote")
    if live_argus:
        source = (
            f"nvarguscamerasrc name=camera sensor_id={argus_sensor_id} sensor-mode={argus_sensor_mode} ! "
            f"video/x-raw(memory:NVMM),width={argus_width},height={argus_height},"
            f"framerate={argus_fps}/1,format=NV12 ! "
            "queue max-size-buffers=2 leaky=downstream ! mux.sink_0 "
        )
        mux = (
            f"nvstreammux name=mux batch-size=1 width={argus_width} height={argus_height} "
            "live-source=true batched-push-timeout=16666 ! "
        )
    elif rtp_input_port is not None:
        source = (
            f"udpsrc name=rtp_input port={rtp_input_port} "
            "caps=application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! "
            "rtpjitterbuffer name=rtp_jitter latency=50 drop-on-latency=true ! rtph264depay ! h264parse ! "
            "nvv4l2decoder name=decoder enable-max-performance=1 ! "
            "queue max-size-buffers=2 leaky=downstream ! mux.sink_0 "
        )
        mux = "nvstreammux name=mux batch-size=1 width=1280 height=720 live-source=true batched-push-timeout=16666 ! "
    else:
        assert input_file is not None
        pace = "identity sync=true !" if paced else ""
        source = (
            f"filesrc location={input_file} ! qtdemux name=demux "
            # Never drop compressed access units: losing an H.264 reference frame can
            # prevent the hardware decoder from producing any output for the replay.
            "demux.video_0 ! queue max-size-buffers=2 ! h264parse ! "
            f"nvv4l2decoder name=decoder enable-max-performance=1 ! {pace}"
            "queue max-size-buffers=2 ! mux.sink_0 "
        )
        mux = "nvstreammux name=mux batch-size=1 width=1280 height=720 live-source=false batched-push-timeout=16666 ! "
    tracker_element = ""
    if tracker != "none":
        tracker_element = (
            f"! nvtracker name=tracker "
            f"ll-lib-file={DS_ROOT}/lib/libnvds_nvmultiobjecttracker.so "
            f"ll-config-file={TRACKER_CONFIGS[tracker]} "
            "tracker-width=640 tracker-height=384 "
        )
    # GPU-mode nvdsosd renders correctly on RGBA NVMM surfaces.  Feeding the
    # tracker's NV12 surface directly can leave partial glyph/rectangle writes
    # behind as the overlay moves, which appears as UI smearing after encode.
    osd = (
        "! nvvideoconvert name=osd_rgba_convert ! "
        "video/x-raw(memory:NVMM),format=RGBA ! "
        "nvdsosd name=osd process-mode=1 "
        if gpu_osd
        else ""
    )
    if return_h264:
        encoded_sink = "fakesink name=sink sync=false"
        if return_h264_file is not None:
            encoded_sink = f"filesink name=encoded_file location={return_h264_file.resolve()} sync=false"
        if return_udp_host is not None:
            assert return_udp_port is not None
            encoded_sink = (
                "rtph264pay name=rtp_pay pt=97 config-interval=1 ! "
                f"udpsink name=return_udp host={return_udp_host} port={return_udp_port} "
                "sync=false async=false"
            )
        tail = (
            f"{osd}! nvvideoconvert ! "
            f"video/x-raw(memory:NVMM),format=NV12,width={return_width},height={return_height} ! "
            f"videorate name=return_rate drop-only=true max-rate={return_fps} ! "
            f"video/x-raw(memory:NVMM),format=NV12,width={return_width},height={return_height},"
            f"framerate={return_fps}/1 ! queue leaky=downstream max-size-buffers=1 ! "
            f"nvv4l2h264enc name=encoder maxperf-enable=1 control-rate=1 bitrate={return_bitrate_kbps * 1000} "
            "iframeinterval=1 idrinterval=1 num-B-Frames=0 num-Ref-Frames=1 "
            "insert-sps-pps=true insert-aud=true insert-vui=true copy-timestamp=true preset-level=1 ! "
            f"h264parse name=h264parse config-interval=-1 ! {encoded_sink}"
        )
    else:
        tail = f"{osd}! fakesink name=sink sync=false"
    return (
        source
        + mux
        + f"nvinfer name=primary config-file-path={nvinfer_config.resolve()} {tracker_element}"
        + tail
    )


def _load_nvinfer_labels(nvinfer_config: Path) -> dict[int, str]:
    """Read the optional ``labelfile-path`` from a DeepStream nvinfer profile."""
    try:
        profile_lines = nvinfer_config.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    label_path = None
    for line in profile_lines:
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "labelfile-path":
            label_path = Path(value.strip())
            if not label_path.is_absolute():
                label_path = nvinfer_config.parent / label_path
            break
    if label_path is None:
        return {}
    try:
        return {index: label.strip() for index, label in enumerate(label_path.read_text(encoding="utf-8").splitlines()) if label.strip()}
    except OSError:
        return {}


def _iterate_meta(pyds: Any, list_meta: Any, cast: Any):
    while list_meta is not None:
        try:
            value = cast(list_meta.data)
        except StopIteration:
            return
        yield value
        try:
            list_meta = list_meta.next
        except StopIteration:
            return


def _stage_probe(pad: Any, info: Any, user_data: tuple[Any, StageClock, str]):
    gst, stage_clock, stage = user_data
    buffer = info.get_buffer()
    if buffer is None:
        return gst.PadProbeReturn.OK
    now_s = time.monotonic()
    if stage == "decode":
        stage_clock.record_decode(now_s)
    elif stage == "infer_input":
        stage_clock.record_infer_input(now_s)
    else:  # Defensive: probes are internal and their stage must stay explicit.
        raise RuntimeError(f"unsupported stage probe: {stage}")
    return gst.PadProbeReturn.OK


def _rtp_identity_probe(pad: Any, info: Any, user_data: tuple[Any, SnapshotTransport]):
    gst, transport = user_data
    buffer = info.get_buffer()
    if buffer is None:
        return gst.PadProbeReturn.OK
    ok, mapping = buffer.map(gst.MapFlags.READ)
    if not ok:
        return gst.PadProbeReturn.OK
    try:
        packet = parse_rtp_identity(bytes(mapping.data[:12]))
    except ValueError:
        return gst.PadProbeReturn.OK
    finally:
        buffer.unmap(mapping)
    pts_ns = _valid_pts_ns(buffer.pts)
    if packet.marker and pts_ns is not None:
        transport.push_rtp_marker(decoded_pts_ns=pts_ns, key=packet.key)
    return gst.PadProbeReturn.OK


def _metadata_probe(
    pad: Any,
    info: Any,
    user_data: tuple[
        Any,
        VerificationStats,
        StageClock,
        SnapshotTransport | None,
        bool,
        AsyncDeepStreamTargetSelector | None,
        Mapping[int, str],
    ],
):
    _gst, stats, stage_clock, snapshot_transport, gpu_osd_enabled, target_selector, class_labels = user_data
    pyds = sys.modules["pyds"]
    buffer = info.get_buffer()
    if buffer is None:
        return _gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    if batch_meta is None:
        return _gst.PadProbeReturn.OK
    for frame_meta in _iterate_meta(pyds, batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
        infer_output_at_s = time.monotonic()
        pts_ns = _valid_pts_ns(frame_meta.buf_pts)
        decode_to_infer_input_ms, infer_stage_ms, rx_ts_ms = stage_clock.consume(infer_output_at_s)
        stats.record_frame(
            frame_meta,
            decode_to_infer_input_ms=decode_to_infer_input_ms,
            infer_stage_ms=infer_stage_ms,
        )
        object_metas = list(_iterate_meta(pyds, frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast))
        for object_meta in object_metas:
            stats.objects += 1
            stats.tracker_only_objects += float(object_meta.confidence) < 0.0
            stats.class_counts[int(object_meta.class_id)] += 1
            object_id = int(object_meta.object_id)
            if object_id != UNTRACKED_OBJECT_ID:
                stats.tracker_ids.add(object_id)
        if object_metas:
            stats.frames_with_objects += 1
        if snapshot_transport is not None:
            snapshot_transport.drain_headers()
        if snapshot_transport is not None or target_selector is not None:
            image_width = int(frame_meta.source_frame_width)
            image_height = int(frame_meta.source_frame_height)
            if image_width <= 0 or image_height <= 0:
                raise RuntimeError("DeepStream frame metadata has invalid source dimensions")
            header = (
                snapshot_transport.next_header(decoded_pts_ns=pts_ns)
                if snapshot_transport is not None and snapshot_transport.requires_headers else None
            )
            if snapshot_transport is not None and snapshot_transport.requires_headers and header is None:
                if gpu_osd_enabled:
                    _decorate_osd_metadata(pyds, batch_meta, frame_meta, object_metas, class_labels=class_labels, pipeline_fps=stats.current_pipeline_fps())
                continue
            timing = FrameTiming(
                frame_id=(header.frame_id if header is not None else int(frame_meta.frame_num) + 1),
                src_ts_ms=(header.src_ts_ms if header is not None else pts_ns_to_ms(0 if pts_ns is None else pts_ns)),
                rx_ts_ms=round(rx_ts_ms),
                infer_ts_ms=round(infer_output_at_s * 1000.0),
                img_w=image_width,
                img_h=image_height,
                source_clock_domain=(
                    "pc_monotonic" if header is not None else "gstreamer_pts_relative"
                ),
                observation_clock_domain="jetson_monotonic",
                src_ts_ns=(header.source_time_ns if header is not None else None),
                source_identity_verified=(header.source_identity_verified if header is not None else None),
            )
            snapshot = perception_snapshot_from_metadata(timing, object_metas, stats.missed_frames)
            if target_selector is not None:
                snapshot = target_selector.submit_and_apply_snapshot(snapshot)
            target_track_id = (
                snapshot.selection.track_id if snapshot.selection is not None else None
            )
            target_assessment = next(
                (
                    assessment
                    for assessment in snapshot.assessments
                    if assessment.track_id == target_track_id
                ),
                None,
            )
            if gpu_osd_enabled:
                _decorate_osd_metadata(
                    pyds,
                    batch_meta,
                    frame_meta,
                    object_metas,
                    target_track_id=target_track_id,
                    class_labels=class_labels,
                    target_assessment=target_assessment,
                    frame_id=snapshot.frame.frame_id,
                    infer_stage_ms=infer_stage_ms,
                    pipeline_fps=stats.current_pipeline_fps(),
                )
            if snapshot_transport is not None:
                snapshot_transport.publish(snapshot)
        elif gpu_osd_enabled:
            _decorate_osd_metadata(pyds, batch_meta, frame_meta, object_metas, class_labels=class_labels, pipeline_fps=stats.current_pipeline_fps())
    return _gst.PadProbeReturn.OK


def _set_rgba(color: Any, red: float, green: float, blue: float, alpha: float = 1.0) -> None:
    """Set a PyDS color parameter without allocating CPU video surfaces."""

    color.set(float(red), float(green), float(blue), float(alpha))


def _decorate_osd_metadata(
    pyds: Any,
    batch_meta: Any,
    frame_meta: Any,
    object_metas: list[Any],
    *,
    target_track_id: int | None = None,
    class_labels: Mapping[int, str] | None = None,
    target_assessment: TrackAssessmentV2 | None = None,
    frame_id: int | None = None,
    infer_stage_ms: float | None = None,
    pipeline_fps: float | None = None,
) -> None:
    """Attach only GPU-renderable OSD metadata to an NVMM DeepStream frame.

    This deliberately renders neutral detector/tracker state, rather than the
    legacy controller, laser, or predicted-aim overlays.  Those values do not
    exist in this isolated, control-disabled verifier.  ``nvdsosd`` consumes
    the attached metadata on the GPU; no OpenCV frame copy or BGR appsrc is
    introduced here.
    """

    for object_meta in object_metas:
        rect = object_meta.rect_params
        rect.border_width = 3
        tracker_id = int(object_meta.object_id)
        selected = target_track_id is not None and tracker_id == int(target_track_id)
        # Negative detector confidence: carried by the tracker alone this frame.
        tracker_only = float(object_meta.confidence) < 0.0
        if selected:
            _set_rgba(rect.border_color, 1.0, 0.2, 0.1)
        elif tracker_only:
            _set_rgba(rect.border_color, 1.0, 0.75, 0.0)
        else:
            _set_rgba(rect.border_color, 0.1, 1.0, 0.1)
        text = object_meta.text_params
        track_suffix = "" if tracker_id == UNTRACKED_OBJECT_ID else f" id={tracker_id}"
        selected_prefix = "TARGET " if selected else ""
        class_id = int(object_meta.class_id)
        label = (class_labels or {}).get(class_id, f"class={class_id}")
        target_suffix = _target_osd_suffix(target_assessment) if selected else ""
        score = "tracked" if tracker_only else f"{float(object_meta.confidence):.2f}"
        text.display_text = f"{selected_prefix}{label} {score}{track_suffix}{target_suffix}"
        text.x_offset = int(max(float(rect.left), 0.0))
        text.y_offset = int(max(float(rect.top) - 24.0, 0.0))
        text.font_params.font_name = "Sans"
        text.font_params.font_size = 14
        _set_rgba(text.font_params.font_color, 1.0, 1.0, 1.0)
        text.set_bg_clr = 1
        _set_rgba(text.text_bg_clr, 0.0, 0.0, 0.0, 0.75)

    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    if display_meta is None:
        return
    display_meta.num_labels = 1
    status = display_meta.text_params[0]
    status_bits = ["DeepStream GPU OSD", "control disabled"]
    if frame_id is not None:
        status_bits.append(f"frame={frame_id}")
    if infer_stage_ms is not None:
        status_bits.append(f"infer={infer_stage_ms:.1f}ms")
    if pipeline_fps is not None:
        status_bits.append(f"fps={pipeline_fps:.1f}")
    status.display_text = " | ".join(status_bits)
    status.x_offset = 12
    status.y_offset = 12
    status.font_params.font_name = "Sans"
    status.font_params.font_size = 18
    _set_rgba(status.font_params.font_color, 0.2, 1.0, 0.2)
    status.set_bg_clr = 1
    _set_rgba(status.text_bg_clr, 0.0, 0.0, 0.0, 0.75)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def _target_osd_suffix(assessment: TrackAssessmentV2 | None) -> str:
    """Controller-independent target facts suitable for a compact GPU label."""
    if assessment is None:
        return ""
    details: list[str] = []
    if assessment.distance_m is not None:
        details.append(f"r={assessment.distance_m:.1f}m")
    if assessment.threat_level:
        details.append(assessment.threat_level)
    if assessment.engagement_rank is not None:
        details.append(f"rank={assessment.engagement_rank}")
    return " " + " ".join(details) if details else ""


def _encoded_output_probe(pad: Any, info: Any, user_data: tuple[Any, VerificationStats]):
    gst, stats = user_data
    if info.get_buffer() is not None:
        stats.encoded_buffers += 1
    return gst.PadProbeReturn.OK


def _return_rate_probe(
    pad: Any,
    info: Any,
    user_data: tuple[Any, VerificationStats],
):
    gst, stats = user_data
    if info.get_buffer() is not None:
        stats.return_frames += 1
    return gst.PadProbeReturn.OK


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="H.264 MP4 replay source")
    parser.add_argument("--live-argus", action="store_true", help="use the local CSI/Argus camera instead of replay")
    parser.add_argument("--rtp-input-port", type=int, help="receive PC-compatible RTP/H.264 payload 96 on this UDP port")
    parser.add_argument("--duration-s", type=float, help="bounded live-camera duration in seconds")
    parser.add_argument("--argus-sensor-id", type=int, default=0)
    parser.add_argument("--argus-sensor-mode", type=int, default=4)
    parser.add_argument("--argus-width", type=int, default=1280)
    parser.add_argument("--argus-height", type=int, default=720)
    parser.add_argument("--argus-fps", type=int, default=60)
    parser.add_argument(
        "--nvinfer-config",
        type=Path,
        default=NVINFER_CONFIG,
        help="DeepStream nvinfer config; defaults to the generic YOLO26n smoke profile",
    )
    parser.add_argument("--tracker", choices=sorted(TRACKER_CONFIGS) + ["none"], default="none",
                        help="nvtracker profile: nvsort (motion only) or nvdcf (visual correlation filter)")
    parser.add_argument("--nvsort", action="store_true", help="deprecated alias for --tracker nvsort")
    parser.add_argument("--paced", action="store_true", help="pace replay using source PTS")
    parser.add_argument(
        "--gpu-osd",
        action="store_true",
        help="attach detector/tracker display metadata and render it with GPU-mode nvdsosd",
    )
    parser.add_argument(
        "--return-h264",
        action="store_true",
        help="hardware-encode the post-OSD NVMM frame; defaults to a local fakesink",
    )
    parser.add_argument(
        "--return-udp-host",
        help="optional RTP/H.264 destination host; requires --return-h264 and --return-udp-port",
    )
    parser.add_argument(
        "--return-udp-port",
        type=int,
        help="optional RTP/H.264 destination UDP port; requires --return-h264 and --return-udp-host",
    )
    parser.add_argument("--return-width", type=int, default=1280)
    parser.add_argument("--return-height", type=int, default=720)
    parser.add_argument("--return-fps", type=int, default=60)
    parser.add_argument("--return-bitrate-kbps", type=int, default=8000)
    parser.add_argument("--return-h264-file", type=Path, help="write the post-OSD Annex-B H.264 stream for local visual inspection")
    parser.add_argument("--report", type=Path, help="write JSON report")
    parser.add_argument("--ready-file", type=Path, help="create after the first DeepStream metadata frame")
    parser.add_argument("--health-file", type=Path, help="refresh at most once per second while frames arrive")
    parser.add_argument("--header-bind", help="optional ZMQ PULL bind endpoint for PC headers")
    parser.add_argument(
        "--verified-rtp-headers", action="store_true",
        help="require exact RTP SSRC/timestamp plus decoded PTS correlation",
    )
    parser.add_argument("--snapshot-result-bind", help="ZMQ PUB bind endpoint for PerceptionSnapshot V2 metadata")
    parser.add_argument(
        "--target-selection",
        action="store_true",
        help="run the trained, control-free swarm target selector on V2 metadata",
    )
    parser.add_argument(
        "--idcs-config",
        action="append",
        type=Path,
        default=[],
        help="IDCS YAML config for --target-selection; repeat in merge order",
    )
    args = parser.parse_args(argv)
    if args.nvsort:
        if args.tracker not in ("none", "nvsort"):
            parser.error("--nvsort conflicts with --tracker")
        args.tracker = "nvsort"
    if sum(value is not None for value in (args.input, args.rtp_input_port)) + int(args.live_argus) != 1:
        parser.error("select exactly one input: file, --live-argus, or --rtp-input-port")
    if args.input is not None and not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.duration_s is not None and (not args.live_argus or args.duration_s <= 0):
        if args.rtp_input_port is None or args.duration_s <= 0:
            parser.error("--duration-s must be positive and is only valid with a live input")
    if args.rtp_input_port is not None and not 1 <= args.rtp_input_port <= 65535:
        parser.error("--rtp-input-port must be between 1 and 65535")
    if args.return_h264 and not args.gpu_osd:
        parser.error("--return-h264 requires --gpu-osd so the encoded path exercises GPU OSD")
    if (args.return_udp_host is None) != (args.return_udp_port is None):
        parser.error("--return-udp-host and --return-udp-port must be supplied together")
    if args.return_udp_host is not None and not args.return_h264:
        parser.error("--return-udp-host/--return-udp-port require --return-h264")
    if args.return_udp_host is not None and not re.fullmatch(r"[A-Za-z0-9._:-]+", args.return_udp_host):
        parser.error("--return-udp-host must be a hostname, IPv4 address, or IPv6 address without brackets")
    if args.return_udp_port is not None and not 1 <= args.return_udp_port <= 65535:
        parser.error("--return-udp-port must be between 1 and 65535")
    if args.return_h264_file is not None and not args.return_h264:
        parser.error("--return-h264-file requires --return-h264")
    if args.return_h264_file is not None and args.return_udp_host is not None:
        parser.error("--return-h264-file cannot be combined with RTP return output")
    if min(
        args.return_width,
        args.return_height,
        args.return_fps,
        args.return_bitrate_kbps,
    ) <= 0:
        parser.error("return width, height, fps, and bitrate must be positive")
    has_metadata_output = args.snapshot_result_bind is not None
    if args.header_bind is not None and not has_metadata_output:
        parser.error("--header-bind requires --snapshot-result-bind")
    if args.verified_rtp_headers and (
        args.rtp_input_port is None or args.header_bind is None or not has_metadata_output
    ):
        parser.error("--verified-rtp-headers requires RTP input, header bind, and snapshot output")
    if has_metadata_output and args.header_bind is None and not args.live_argus:
        parser.error("headerless metadata publication is supported only with --live-argus")
    if args.target_selection and not has_metadata_output:
        parser.error("--target-selection requires --snapshot-result-bind")
    if args.target_selection and not args.idcs_config:
        parser.error("--target-selection requires at least one --idcs-config")
    for config_path in args.idcs_config:
        if not config_path.is_file():
            parser.error(f"IDCS config does not exist: {config_path}")
    if not args.nvinfer_config.is_file():
        parser.error(f"nvinfer config does not exist: {args.nvinfer_config}")

    Gst, GLib, pyds = _require_bindings()
    Gst.init(None)
    if args.ready_file is not None:
        try:
            args.ready_file.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"unable to clear readiness file {args.ready_file}: {exc}") from exc
    if args.health_file is not None:
        try:
            args.health_file.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"unable to clear health file {args.health_file}: {exc}") from exc
    stats = VerificationStats(ready_file=args.ready_file, health_file=args.health_file)
    stage_clock = StageClock()
    snapshot_transport: SnapshotTransport | None = None
    target_selector: AsyncDeepStreamTargetSelector | None = None
    try:
        pipeline = Gst.parse_launch(
            _pipeline_description(
                input_file=None if (args.live_argus or args.rtp_input_port is not None) else args.input.resolve(),
                live_argus=args.live_argus,
                rtp_input_port=args.rtp_input_port,
                argus_sensor_id=args.argus_sensor_id,
                argus_sensor_mode=args.argus_sensor_mode,
                argus_width=args.argus_width,
                argus_height=args.argus_height,
                argus_fps=args.argus_fps,
                nvinfer_config=args.nvinfer_config,
                paced=args.paced,
                tracker=args.tracker,
                gpu_osd=args.gpu_osd,
                return_h264=args.return_h264,
                return_udp_host=args.return_udp_host,
                return_udp_port=args.return_udp_port,
                return_h264_file=args.return_h264_file,
                return_width=args.return_width,
                return_height=args.return_height,
                return_fps=args.return_fps,
                return_bitrate_kbps=args.return_bitrate_kbps,
            )
        )
    except Exception as exc:
        raise RuntimeError("unable to create DeepStream verification pipeline") from exc

    metadata_source = pipeline.get_by_name("tracker" if args.tracker != "none" else "primary")
    if metadata_source is None:
        raise RuntimeError("metadata source element is unavailable")
    src_pad = metadata_source.get_static_pad("src")
    if src_pad is None:
        raise RuntimeError("metadata source pad is unavailable")
    if args.snapshot_result_bind:
        snapshot_transport = SnapshotTransport(
            header_bind=args.header_bind,
            snapshot_bind=args.snapshot_result_bind,
            verified_rtp_headers=args.verified_rtp_headers,
        )
    if args.verified_rtp_headers:
        jitter = pipeline.get_by_name("rtp_jitter")
        if jitter is None or snapshot_transport is None:
            raise RuntimeError("verified RTP identity requires jitterbuffer and snapshot transport")
        jitter.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER,
            _rtp_identity_probe,
            (Gst, snapshot_transport),
        )
    if args.target_selection:
        target_selector = AsyncDeepStreamTargetSelector(args.idcs_config)
        print("[deepstream.verify] latest-only target selection service enabled; control remains disabled", flush=True)
    class_labels = _load_nvinfer_labels(args.nvinfer_config)
    src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        _metadata_probe,
        (Gst, stats, stage_clock, snapshot_transport, args.gpu_osd, target_selector, class_labels),
    )
    if args.return_h264:
        return_rate = pipeline.get_by_name("return_rate")
        if return_rate is None:
            raise RuntimeError("return video rate element is unavailable")
        return_rate_src_pad = return_rate.get_static_pad("src")
        if return_rate_src_pad is None:
            raise RuntimeError("unable to attach return-rate enforcement probe")
        return_rate_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            _return_rate_probe,
            (Gst, stats),
        )
        h264parse = pipeline.get_by_name("h264parse")
        if h264parse is None:
            raise RuntimeError("H.264 return parser is unavailable")
        h264parse_src_pad = h264parse.get_static_pad("src")
        if h264parse_src_pad is None:
            raise RuntimeError("unable to attach encoded-output timing probe")
        h264parse_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            _encoded_output_probe,
            (Gst, stats),
        )
    source_element = pipeline.get_by_name("camera" if args.live_argus else "decoder")
    if source_element is None:
        raise RuntimeError("camera/decoder source element is unavailable")
    source_src_pad = source_element.get_static_pad("src")
    primary = pipeline.get_by_name("primary")
    if primary is None:
        raise RuntimeError("primary nvinfer element is unavailable")
    primary_sink_pad = primary.get_static_pad("sink")
    if source_src_pad is None or primary_sink_pad is None:
        raise RuntimeError("unable to attach DeepStream stage timing probes")
    source_src_pad.add_probe(Gst.PadProbeType.BUFFER, _stage_probe, (Gst, stage_clock, "decode"))
    primary_sink_pad.add_probe(Gst.PadProbeType.BUFFER, _stage_probe, (Gst, stage_clock, "infer_input"))

    loop = GLib.MainLoop()
    outcome = {"error": None}
    stop_event = install_signal_handlers()

    def on_bus_message(_bus: Any, message: Any, _unused: Any) -> bool:
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            outcome["error"] = f"{error}: {debug}"
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, None)
    if args.duration_s is not None:
        GLib.timeout_add(int(args.duration_s * 1000), lambda: (loop.quit(), False)[1])
    # systemd and Ctrl+C must take the same orderly path as EOS: release
    # DeepStream resources, emit the final report, and remove health files.
    GLib.timeout_add(100, lambda: (loop.quit(), False)[1] if stop_event.is_set() else True)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
        if snapshot_transport is not None:
            snapshot_transport.close()
        if target_selector is not None:
            target_selector.close()
        if args.ready_file is not None:
            try:
                args.ready_file.unlink(missing_ok=True)
            except OSError:
                pass
        if args.health_file is not None:
            try:
                args.health_file.unlink(missing_ok=True)
            except OSError:
                pass

    if outcome["error"]:
        raise RuntimeError(f"DeepStream pipeline failed: {outcome['error']}")
    report = stats.report(
        tracker_enabled=args.tracker != "none",
        stage_clock=stage_clock,
        gpu_osd_enabled=args.gpu_osd,
        h264_return_enabled=args.return_h264,
    )
    report["source_mode"] = (
        f"argus_sensor_{args.argus_sensor_id}_mode_{args.argus_sensor_mode}_"
        f"{args.argus_width}x{args.argus_height}_{args.argus_fps}fps"
        if args.live_argus
        else f"rtp_h264_payload_96_udp_{args.rtp_input_port}"
        if args.rtp_input_port is not None
        else "file_replay"
    )
    report["return_output"] = (
        None
        if not args.return_h264
        else {
            "mode": "rtp_udp" if args.return_udp_host is not None else "encoded_fakesink",
            "rtp_payload_type": 97 if args.return_udp_host is not None else None,
            "host": args.return_udp_host,
            "port": args.return_udp_port,
            "control_disabled": True,
        }
    )
    report["snapshot_transport"] = None if snapshot_transport is None else snapshot_transport.report()
    report["target_selection"] = None if target_selector is None else target_selector.report()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if stats.frames == 0:
        raise RuntimeError("pipeline completed without DeepStream frame metadata")
    if args.return_h264 and stats.encoded_buffers == 0:
        raise RuntimeError("H.264 return path completed without an encoded access unit")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
