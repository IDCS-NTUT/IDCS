"""Read-only Jetson probe: RTP marker PTS versus decoded DeepStream frame PTS."""

from __future__ import annotations

import argparse
import json
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds  # noqa: E402


def _rtp_key(buffer: Gst.Buffer) -> tuple[int, int, bool]:
    ok, mapping = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("cannot map RTP buffer")
    try:
        header = bytes(mapping.data[:12])
        if len(header) < 12 or header[0] >> 6 != 2:
            raise RuntimeError("invalid RTP header")
        return (
            int.from_bytes(header[8:12], "big"),
            int.from_bytes(header[4:8], "big"),
            bool(header[1] & 0x80),
        )
    finally:
        buffer.unmap(mapping)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--duration-s", type=float, default=7.0)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 0 < args.duration_s <= 30:
        parser.error("invalid port or duration")
    Gst.init(None)
    pipeline = Gst.parse_launch(
        f"udpsrc name=rtp_input port={args.port} "
        "caps=application/x-rtp,media=video,encoding-name=H264,"
        "payload=96,clock-rate=90000 ! "
        "rtpjitterbuffer name=jitter latency=50 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! "
        "nvv4l2decoder name=decoder enable-max-performance=1 ! "
        "queue max-size-buffers=2 leaky=downstream ! mux.sink_0 "
        "nvstreammux name=mux batch-size=1 width=320 height=240 "
        "live-source=true batched-push-timeout=16666 ! "
        "fakesink name=sink sync=false"
    )
    rtp_by_pts: dict[int, tuple[int, int]] = {}
    pts_collisions = 0
    deepstream_pts: list[int] = []
    batch_buffer_pts: list[int] = []

    def rtp_probe(_pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        nonlocal pts_collisions
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        ssrc, rtp_timestamp, marker = _rtp_key(buffer)
        if marker:
            pts = int(buffer.pts)
            old = rtp_by_pts.get(pts)
            if old is not None and old != (ssrc, rtp_timestamp):
                pts_collisions += 1
            rtp_by_pts[pts] = (ssrc, rtp_timestamp)
        return Gst.PadProbeReturn.OK

    def mux_probe(_pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        batch_buffer_pts.append(int(buffer.pts))
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch is None:
            return Gst.PadProbeReturn.OK
        item = batch.frame_meta_list
        while item is not None:
            frame = pyds.NvDsFrameMeta.cast(item.data)
            deepstream_pts.append(int(frame.buf_pts))
            try:
                item = item.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    pipeline.get_by_name("jitter").get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, rtp_probe
    )
    pipeline.get_by_name("mux").get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, mux_probe
    )
    try:
        pipeline.set_state(Gst.State.PLAYING)
        print("READY", flush=True)
        deadline = time.monotonic() + args.duration_s
        bus = pipeline.get_bus()
        while time.monotonic() < deadline:
            message = bus.timed_pop_filtered(
                100 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            if message is not None and message.type == Gst.MessageType.ERROR:
                error, detail = message.parse_error()
                raise RuntimeError(f"DeepStream pipeline error: {error}: {detail}")
    finally:
        pipeline.set_state(Gst.State.NULL)
    matched = sum(pts in rtp_by_pts for pts in deepstream_pts)
    report = {
        "rtp_marker_frames": len(rtp_by_pts),
        "deepstream_frames": len(deepstream_pts),
        "deepstream_pts_matched_to_rtp_marker": matched,
        "deepstream_pts_unmatched": len(deepstream_pts) - matched,
        "rtp_pts_collisions": pts_collisions,
        "batch_buffer_pts_first_five": batch_buffer_pts[:5],
        "deepstream_frame_pts_first_five": deepstream_pts[:5],
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if deepstream_pts and matched == len(deepstream_pts) and pts_collisions == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
