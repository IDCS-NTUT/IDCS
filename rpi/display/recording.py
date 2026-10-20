"""Bounded recording of the received return video on the panel Pi.

The encoded H.264 stream is tapped from a tee after parsing, before decoding,
so nothing is re-encoded. Each segment is its own branch that starts at a
keyframe and at time 0:00 (so every file plays on its own; splitmuxsink keeps
one timeline across segments, which players show as starting at 1:00, 2:00...).
Stopping sends EOS so the last file is finalised. Files are named
``<recording start>-<segment>-return.mkv``; an unfinalised file (power loss)
remains playable. Before each new segment the oldest recordings in the
directory are removed until the kept files plus one segment fit the budget.
Only ``*-return.mkv`` files are touched.
"""
from __future__ import annotations

import time
from pathlib import Path

SUFFIX = "-return.mkv"


def prune(directory: Path, budget_bytes: int, reserve_bytes: int) -> list[Path]:
    """Delete oldest recordings until kept + reserve <= budget; return deleted paths."""
    files = sorted((p for p in directory.glob("*" + SUFFIX) if p.is_file() and not p.is_symlink()),
                   key=lambda p: p.stat().st_mtime)
    sizes = {p: p.stat().st_size for p in files}
    deleted = []
    while files and sum(sizes.values()) + reserve_bytes > budget_bytes:
        oldest = files.pop(0)
        oldest.unlink(missing_ok=True)
        sizes.pop(oldest)
        deleted.append(oldest)
    return deleted


class KeyframeGate:
    """Pass nothing until the first keyframe, then everything."""

    def __init__(self) -> None:
        self.keyed = False
        self.passed = 0

    def accept(self, is_delta: bool) -> bool:
        if not self.keyed:
            if is_delta:
                return False
            self.keyed = True
        self.passed += 1
        return True


class ReturnRecorder:
    """One branch per segment on the stream's tee; every file starts at 0:00.

    A segment's branch (leaky queue -> h264parse -> matroskamux -> filesink)
    passes nothing before its first keyframe and then shifts that keyframe to
    time zero with a pad offset, so each file plays on its own. At a segment
    boundary the next branch attaches first; the old one gets EOS only after
    the new one has its keyframe, so there is no gap (a few frames may appear
    in both files).
    """

    FINALIZE_S = 2.0

    def __init__(self, pipeline, tee, directory: Path, *, segment_s: int, budget_bytes: int, gst) -> None:
        self.gst, self.pipeline, self.tee = gst, pipeline, tee
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segment_s, self.budget_bytes = int(segment_s), int(budget_bytes)
        # Reserve one segment of the 20 Mbit/s all-IDR worst case before it starts.
        self.reserve_bytes = int(self.segment_s * 20e6 / 8)
        self.active = None      # current segment branch
        self.next = None        # following segment, attached but waiting for its keyframe
        self._closing = []      # (deadline, elements)
        self.recording_start = None
        self.segment_index = 0
        self.deleted = self.recordings = 0
        self.last_files: list[str] = []

    @property
    def recording(self) -> bool:
        return self.recording_start is not None

    def set_recording(self, wanted: bool) -> None:
        now = time.monotonic()
        if wanted and self.recording_start is None:
            self.recording_start, self.segment_index = time.strftime("%Y%m%dT%H%M%S"), 0
            self.recordings += 1
            self.active = self._attach()
        elif not wanted and self.recording_start is not None:
            for branch in (self.next, self.active):
                if branch is not None:
                    self._detach(branch)
            self.active = self.next = None
            self.recording_start = None
        elif self.recording_start is not None:
            if self.next is None and self.active["gate"].keyed and now - self.active["opened"] >= self.segment_s:
                self.next = self._attach()
            if self.next is not None and self.next["gate"].keyed:
                self._detach(self.active)
                self.active, self.next = self.next, None
        self._reap(now)

    def _attach(self) -> dict:
        gst = self.gst
        self.deleted += len(prune(self.directory, self.budget_bytes, self.reserve_bytes))
        path = str(self.directory / f"{self.recording_start}-{self.segment_index:03d}{SUFFIX}")
        self.segment_index += 1
        queue = gst.ElementFactory.make("queue")
        queue.set_property("leaky", 2)  # downstream: a slow card drops old data, never stalls display
        queue.set_property("max-size-time", 2_000_000_000)
        queue.set_property("max-size-buffers", 0)
        queue.set_property("max-size-bytes", 0)
        parse = gst.ElementFactory.make("h264parse")
        parse.set_property("config-interval", -1)
        mux = gst.ElementFactory.make("matroskamux")
        sink = gst.ElementFactory.make("filesink")
        sink.set_property("location", path)
        sink.set_property("async", False)
        elements = (queue, parse, mux, sink)
        for element in elements:
            self.pipeline.add(element)
        queue.link(parse); parse.link(mux); mux.link(sink)
        for element in elements:
            element.sync_state_with_parent()
        pad = self.tee.request_pad_simple("src_%u")
        gate = KeyframeGate()
        out_pad = queue.get_static_pad("src")

        def probe(_pad, info):
            buffer = info.get_buffer()
            delta = buffer is not None and buffer.has_flags(gst.BufferFlags.DELTA_UNIT)
            first = not gate.keyed
            if not gate.accept(delta):
                return gst.PadProbeReturn.DROP
            if first and buffer.pts != gst.CLOCK_TIME_NONE:
                out_pad.set_offset(-int(buffer.pts))  # this segment starts at 0:00
            return gst.PadProbeReturn.OK
        pad.add_probe(gst.PadProbeType.BUFFER, probe)
        pad.link(queue.get_static_pad("sink"))
        return dict(elements=elements, pad=pad, gate=gate, path=path, opened=time.monotonic())

    def _detach(self, branch: dict) -> None:
        queue = branch["elements"][0]
        branch["pad"].unlink(queue.get_static_pad("sink"))
        self.tee.release_request_pad(branch["pad"])
        # EOS finalises the file; the branch is removed once that is done.
        queue.get_static_pad("sink").send_event(self.gst.Event.new_eos())
        self._closing.append((time.monotonic() + self.FINALIZE_S, branch["elements"]))
        if branch["gate"].keyed:
            self.last_files = (self.last_files + [branch["path"]])[-8:]
        else:
            Path(branch["path"]).unlink(missing_ok=True)  # never received a frame

    def _reap(self, now: float, force: bool = False) -> None:
        keep = []
        for deadline, elements in self._closing:
            if force or now >= deadline:
                for element in elements:
                    element.set_state(self.gst.State.NULL)
                    self.pipeline.remove(element)
            else:
                keep.append((deadline, elements))
        self._closing = keep

    def close(self) -> None:
        if self.recording_start is not None:
            self.set_recording(False)
        time.sleep(0.2)
        self._reap(time.monotonic(), force=True)

    def report(self) -> dict:
        active = self.active
        return dict(recording=self.recording, directory=str(self.directory),
                    current=active["path"] if active else None, started=self.recording_start,
                    buffers=active["gate"].passed if active else 0, recordings=self.recordings,
                    last_files=self.last_files, pruned=self.deleted, budget_bytes=self.budget_bytes)
