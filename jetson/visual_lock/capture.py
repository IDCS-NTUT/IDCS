"""Bounded pre-OSD Annex-B access-unit evidence, with exact decoded-PTS join.

Only compressed bytes are copied. Stops at a whole-access-unit boundary; never
wraps or silently overwrites the seed. Capture limits do not stop live tracking.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class InputCapture:
    def __init__(self, output: Path, *, max_bytes=256_000_000, max_duration_ns=150_000_000_000):
        if type(max_bytes) is not int or max_bytes <= 0 or type(max_duration_ns) is not int or max_duration_ns <= 0:
            raise ValueError("positive integer capture limits required")
        self.video = (output / "input.h264").open("xb")
        self.index = (output / "input.jsonl").open("x", encoding="utf-8")
        self.max_bytes, self.max_duration_ns = max_bytes, max_duration_ns
        self.count = self.bytes = 0
        self.first_pts = self.last_pts = None
        self.stop_reason = None

    def append(self, packet: bytes, pts_ns: int):
        if self.stop_reason is not None:
            return
        if type(pts_ns) is not int or not 0 <= pts_ns < (1 << 63)-1:
            raise ValueError("input access unit lacks valid PTS")
        if not packet or not packet.startswith((b"\x00\x00\x01", b"\x00\x00\x00\x01")):
            raise ValueError("input capture requires Annex-B access units")
        if self.last_pts is not None and pts_ns <= self.last_pts:
            raise ValueError("input capture PTS repeated or reordered")
        if self.bytes+len(packet) > self.max_bytes:
            self.stop_reason = "byte limit"
            return
        if self.first_pts is not None and pts_ns-self.first_pts > self.max_duration_ns:
            self.stop_reason = "duration limit"
            return
        row = dict(encoded_index=self.count, decoded_pts_ns=pts_ns, offset=self.bytes,
                   size=len(packet), sha256=hashlib.sha256(packet).hexdigest())
        self.video.write(packet)
        self.index.write(json.dumps(row)+"\n")
        if self.first_pts is None: self.first_pts = pts_ns
        self.last_pts = pts_ns
        self.count += 1
        self.bytes += len(packet)
        if self.count % 60 == 0:
            self.video.flush(); self.index.flush()

    def report(self):
        return dict(access_units=self.count, bytes=self.bytes, first_pts_ns=self.first_pts,
                    last_pts_ns=self.last_pts, stop_reason=self.stop_reason,
                    max_bytes=self.max_bytes, max_duration_ns=self.max_duration_ns,
                    unannotated=True, format="Annex-B H264 alignment=au")

    def close(self):
        self.video.close()
        self.index.close()
