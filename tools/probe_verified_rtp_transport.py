"""Bounded video-only host canary with independent RTP/header frame loss.

Uses the real GstVideoWriter verified-header path. It has no controller,
serial, laser, or hardware state inputs. Destination ports must be isolated
high ports; the matching Jetson candidate must be started separately.
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import cv2
import numpy as np
import zmq

from pc.streamer import Gst, GstVideoWriter


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary", action="store_true", help="acknowledge isolated video-only test")
    parser.add_argument("--destination-host", required=True)
    parser.add_argument("--rtp-port", type=int, required=True)
    parser.add_argument("--header-endpoint", required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--drop-video-every", type=int, default=0)
    parser.add_argument("--drop-header-every", type=int, default=0)
    args = parser.parse_args()
    if not args.canary or not 50000 <= args.rtp_port <= 65535:
        parser.error("--canary and an isolated RTP port >= 50000 are required")
    if not args.header_endpoint.startswith("tcp://") or not 50000 <= int(args.header_endpoint.rsplit(":", 1)[1]) <= 65535:
        parser.error("an isolated TCP header port >= 50000 is required")
    if not 1 <= args.frames <= 300 or not 0 <= args.drop_video_every <= 300 or not 0 <= args.drop_header_every <= 300:
        parser.error("frames and loss intervals must be in 1..300 and 0..300")
    if not args.drop_video_every and not args.drop_header_every:
        parser.error("select at least one independent loss mode")

    Gst.init(None)
    pipeline = (
        "appsrc name=src is-live=true block=false do-timestamp=true format=time "
        "caps=video/x-raw,format=BGR,width=320,height=240,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=NV12,colorimetry=bt709,"
        "interlace-mode=progressive,chromasite=mpeg2 ! "
        "nvh264enc preset=low-latency-hq zerolatency=true rc-mode=cbr "
        "bframes=0 gop-size=30 bitrate=1000 ! h264parse ! "
        "queue leaky=downstream max-size-buffers=120 max-size-bytes=0 max-size-time=0 ! "
        "rtph264pay name=rtp_pay pt=96 config-interval=1 ! "
        f"udpsink name=loss_gate host={args.destination_host} port={args.rtp_port} "
        "sync=false async=false"
    )
    writer = GstVideoWriter(pipeline, fps=30, verified_rtp_headers=True)
    frame_caps = Gst.Caps.from_string("timestamp/x-idcs-frame-counter")
    dropped_video_ids: set[int] = set()
    mixed_buffer_lists = 0

    def drop_video(_pad, info):
        nonlocal mixed_buffer_lists
        if info.type & Gst.PadProbeType.BUFFER_LIST:
            buffers = info.get_buffer_list()
            frames = {
                int(meta.timestamp)
                for index in range(buffers.length())
                if (meta := buffers.get(index).get_reference_timestamp_meta(frame_caps)) is not None
            }
        elif info.type & Gst.PadProbeType.BUFFER:
            buffer = info.get_buffer()
            meta = None if buffer is None else buffer.get_reference_timestamp_meta(frame_caps)
            frames = set() if meta is None else {int(meta.timestamp)}
        else:
            return Gst.PadProbeReturn.OK
        if len(frames) > 1:
            mixed_buffer_lists += 1
            return Gst.PadProbeReturn.OK
        if frames and args.drop_video_every and next(iter(frames)) % args.drop_video_every == 0:
            dropped_video_ids.update(frames)
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    writer._pipeline.get_by_name("loss_gate").get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER | Gst.PadProbeType.BUFFER_LIST, drop_video
    )
    sender_stop = threading.Event()
    sender_stats = {"headers_sent": 0, "headers_skipped": 0, "headers_backpressure": 0}

    def send_headers() -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.setsockopt(zmq.SNDHWM, 256)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(args.header_endpoint)
        try:
            while not sender_stop.is_set():
                header = writer.wait_identity_header(0.01)
                if header is None:
                    continue
                if args.drop_header_every and int(header["frame_id"]) % args.drop_header_every == 0:
                    sender_stats["headers_skipped"] += 1
                    continue
                try:
                    socket.send_json(header, flags=zmq.NOBLOCK)
                    sender_stats["headers_sent"] += 1
                except zmq.Again:
                    sender_stats["headers_backpressure"] += 1
        finally:
            socket.close(0)
            context.term()

    sender_thread = threading.Thread(target=send_headers, daemon=True)
    sender_thread.start()
    next_at = time.monotonic()
    try:
        for frame_id in range(1, args.frames + 1):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            left = 20 + frame_id * 3 % 240
            cv2.rectangle(frame, (left, 90), (left + 35, 125), (255, 255, 255), -1)
            if not writer.write(frame, frame_id=frame_id, source_time_ns=time.monotonic_ns()):
                raise RuntimeError(f"encoder rejected frame {frame_id}")
            next_at += 1.0 / 30.0
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        writer.end_of_stream()
        time.sleep(0.5)
    finally:
        sender_stop.set()
        sender_thread.join(timeout=1.0)
        writer.release()
    report = {
        "requested_frames": args.frames,
        "drop_video_every": args.drop_video_every,
        "drop_header_every": args.drop_header_every,
        "dropped_video_frame_ids": sorted(dropped_video_ids),
        "mixed_buffer_lists": mixed_buffer_lists,
        **sender_stats,
        **writer.identity_report(),
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if dropped_video_ids or sender_stats["headers_skipped"] else 1


if __name__ == "__main__":
    raise SystemExit(run())
