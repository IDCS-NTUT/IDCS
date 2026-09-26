"""Isolated RTP/H.264 probe of packet timestamp to decoded-frame PTS identity.

This loopback test does not prove DeepStream preserves the same key. It only
checks the proposed join across the host's software H.264/RTP elements,
including deterministic whole-frame RTP loss.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rtp_identity(buffer: Gst.Buffer) -> tuple[int, int, int, bool]:
    ok, mapping = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("cannot map RTP buffer")
    try:
        header = bytes(mapping.data[:12])
        if len(header) < 12 or header[0] >> 6 != 2:
            raise RuntimeError("invalid RTP header")
        return (
            int.from_bytes(header[4:8], "big"),
            int.from_bytes(header[2:4], "big"),
            int.from_bytes(header[8:12], "big"),
            bool(header[1] & 0x80),
        )
    finally:
        buffer.unmap(mapping)


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--drop-every", type=int, default=0)
    parser.add_argument("--encoder", choices=("openh264", "nvh264"), default="openh264")
    parser.add_argument("--destination-host", default="127.0.0.1")
    parser.add_argument("--destination-port", type=int)
    parser.add_argument("--sender-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames <= 300 or not 0 <= args.drop_every <= 100:
        parser.error("frames must be 1..300 and drop-every must be 0..100")
    if args.sender_only and args.destination_port is None:
        parser.error("--sender-only requires --destination-port")
    Gst.init(None)
    port = args.destination_port or _free_udp_port()
    receiver = None if args.sender_only else Gst.parse_launch(
        f"udpsrc port={port} caps=application/x-rtp,media=video,"
        "encoding-name=H264,payload=96,clock-rate=90000 ! "
        "rtpjitterbuffer name=jitter latency=100 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! avdec_h264 ! "
        "fakesink name=decoded sync=false"
    )
    encoder = (
        "videoconvert ! video/x-raw,format=I420 ! "
        "openh264enc bitrate=1000000 gop-size=30 enable-frame-skip=false"
        if args.encoder == "openh264" else
        "videoconvert ! video/x-raw,format=NV12,colorimetry=bt709,"
        "interlace-mode=progressive,chromasite=mpeg2 ! "
        "nvh264enc preset=low-latency-hq zerolatency=true rc-mode=cbr "
        "bframes=0 gop-size=30 bitrate=1000"
    )
    sender = Gst.parse_launch(
        "appsrc name=source is-live=true block=false format=time "
        "caps=video/x-raw,format=I420,width=320,height=240,framerate=30/1 ! "
        f"{encoder} ! h264parse ! "
        "rtph264pay name=pay pt=96 config-interval=1 ! "
        f"udpsink host={args.destination_host} port={port} sync=false async=false"
    )
    sender_pts_by_rtp: dict[int, int] = {}
    raw_pts: list[int] = []
    metadata_caps = Gst.Caps.from_string("timestamp/x-idcs-frame-counter")
    source_time_caps = Gst.Caps.from_string("timestamp/x-system-monotonic")
    source_meta_count = 0
    payloader_meta_by_rtp: dict[int, int] = {}
    source_time_by_frame: dict[int, int] = {}
    payloader_source_time_by_rtp: dict[int, int] = {}
    receiver_pts_by_rtp: dict[int, int] = {}
    receiver_rtp_by_pts: dict[int, int] = {}
    decoded_pts: list[int] = []
    frame_order: dict[int, int] = {}
    dropped: set[int] = set()
    collisions = 0

    def sent(_pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        rtp_ts, _seq, _ssrc, _marker = _rtp_identity(buffer)
        if rtp_ts not in frame_order:
            frame_order[rtp_ts] = len(frame_order) + 1
        if args.drop_every and frame_order[rtp_ts] % args.drop_every == 0:
            dropped.add(rtp_ts)
            return Gst.PadProbeReturn.DROP
        sender_pts_by_rtp[rtp_ts] = int(buffer.pts)
        metadata = buffer.get_reference_timestamp_meta(metadata_caps)
        if metadata is not None:
            payloader_meta_by_rtp[rtp_ts] = int(metadata.timestamp)
        source_time_meta = buffer.get_reference_timestamp_meta(source_time_caps)
        if source_time_meta is not None:
            payloader_source_time_by_rtp[rtp_ts] = int(source_time_meta.timestamp)
        return Gst.PadProbeReturn.OK

    def received(_pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        nonlocal collisions
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        rtp_ts, _seq, _ssrc, marker = _rtp_identity(buffer)
        if marker:
            pts = int(buffer.pts)
            old = receiver_rtp_by_pts.get(pts)
            if old is not None and old != rtp_ts:
                collisions += 1
            receiver_rtp_by_pts[pts] = rtp_ts
            receiver_pts_by_rtp[rtp_ts] = pts
        return Gst.PadProbeReturn.OK

    def decoded(_pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        buffer = info.get_buffer()
        if buffer is not None:
            decoded_pts.append(int(buffer.pts))
        return Gst.PadProbeReturn.OK

    sender.get_by_name("pay").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, sent)
    if receiver is not None:
        receiver.get_by_name("jitter").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, received)
        receiver.get_by_name("decoded").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, decoded)
    try:
        if receiver is not None:
            receiver.set_state(Gst.State.PLAYING)
            time.sleep(0.2)
        sender.set_state(Gst.State.PLAYING)
        source_element = sender.get_by_name("source")
        frame_bytes = bytes([16]) * (320 * 240) + bytes([128]) * (320 * 240 // 2)
        next_at = time.monotonic()
        for frame_id in range(1, args.frames + 1):
            buffer = Gst.Buffer.new_allocate(None, len(frame_bytes), None)
            buffer.fill(0, frame_bytes)
            buffer.pts = (frame_id - 1) * Gst.SECOND // 30
            buffer.dts = buffer.pts
            buffer.duration = Gst.SECOND // 30
            buffer.offset = frame_id
            raw_pts.append(int(buffer.pts))
            source_time_by_frame[frame_id] = time.monotonic_ns()
            metadata = buffer.add_reference_timestamp_meta(
                metadata_caps, frame_id, Gst.CLOCK_TIME_NONE
            )
            source_time_meta = buffer.add_reference_timestamp_meta(
                source_time_caps, source_time_by_frame[frame_id], Gst.CLOCK_TIME_NONE
            )
            if metadata is not None and source_time_meta is not None:
                source_meta_count += 1
            if source_element.emit("push-buffer", buffer) != Gst.FlowReturn.OK:
                raise RuntimeError("appsrc rejected frame")
            next_at += 1.0 / 30.0
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        source_element.emit("end-of-stream")
        bus = sender.get_bus()
        message = bus.timed_pop_filtered(
            15 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        if message is None:
            raise RuntimeError("sender timed out")
        if message.type == Gst.MessageType.ERROR:
            error, detail = message.parse_error()
            raise RuntimeError(f"sender error: {error}: {detail}")
        time.sleep(0.5)
    finally:
        sender.set_state(Gst.State.NULL)
        if receiver is not None:
            receiver.set_state(Gst.State.NULL)
    matched = [pts for pts in decoded_pts if pts in receiver_rtp_by_pts]
    wrong_sender_key = sum(
        receiver_rtp_by_pts[pts] not in sender_pts_by_rtp for pts in matched
    )
    report = {
        "requested_frames": args.frames,
        "encoder": args.encoder,
        "sender_only": args.sender_only,
        "sender_rtp_frames": len(frame_order),
        "dropped_rtp_frames": len(dropped),
        "receiver_marker_frames": len(receiver_pts_by_rtp),
        "decoded_frames": len(decoded_pts),
        "decoded_pts_matched_to_rtp_marker": len(matched),
        "decoded_pts_unmatched": len(decoded_pts) - len(matched),
        "receiver_pts_collisions": collisions,
        "matched_rtp_missing_sender_key": wrong_sender_key,
        "sender_payloader_pts_missing_raw_source": sum(
            pts not in raw_pts for pts in sender_pts_by_rtp.values()
        ),
        "source_reference_meta_added": source_meta_count,
        "payloader_frames_with_reference_meta": len(payloader_meta_by_rtp),
        "payloader_frames_with_source_time_meta": len(payloader_source_time_by_rtp),
        "payloader_reference_meta_mismatches": sum(
            payloader_meta_by_rtp[key] != frame_order[key]
            for key in payloader_meta_by_rtp
        ),
        "payloader_source_time_mismatches": sum(
            payloader_source_time_by_rtp.get(key) != source_time_by_frame.get(frame_order[key])
            for key in sender_pts_by_rtp
        ),
        "raw_pts_first_five_ns": raw_pts[:5],
        "payloader_pts_first_five_ns": list(sender_pts_by_rtp.values())[:5],
        "ordered_raw_to_payloader_shift_first_five_ns": [
            pay - raw for raw, pay in zip(raw_pts, sender_pts_by_rtp.values())
        ][:5],
        "sender_to_receiver_pts_shift_range_ns": [min(shifts), max(shifts)] if (
            shifts := [
            receiver_pts_by_rtp[key] - sender_pts_by_rtp[key]
            for key in receiver_pts_by_rtp if key in sender_pts_by_rtp
            ]
        ) else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    sender_valid = (
        source_meta_count == args.frames
        and report["payloader_frames_with_reference_meta"] == len(sender_pts_by_rtp)
        and report["payloader_frames_with_source_time_meta"] == len(sender_pts_by_rtp)
        and not report["payloader_reference_meta_mismatches"]
        and not report["payloader_source_time_mismatches"]
    )
    receiver_valid = (
        decoded_pts and len(matched) == len(decoded_pts)
        and not collisions and not wrong_sender_key
    )
    return 0 if sender_valid and (args.sender_only or receiver_valid) else 1


if __name__ == "__main__":
    raise SystemExit(run())
