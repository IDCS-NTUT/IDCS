"""Bounded recording of the received return video on the panel Pi.

The encoded H.264 stream is tapped from a tee after parsing, before decoding,
so nothing is re-encoded. Each recording attaches its own branch (leaky queue
-> h264parse -> splitmuxsink with Matroska segments) starting at a keyframe;
stopping detaches it and sends EOS so its last file is finalised. Files are
named ``<recording start>-<segment>-return.mkv``; an unfinalised file (power
loss) remains playable. Before each new segment the oldest recordings in the
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
    """Attaches one recording branch per recording to the stream's tee."""

    FINALIZE_S = 2.0

    def __init__(self, pipeline, tee, directory: Path, *, segment_s: int, budget_bytes: int, gst) -> None:
        self.gst, self.pipeline, self.tee = gst, pipeline, tee
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segment_s, self.budget_bytes = int(segment_s), int(budget_bytes)
        # Reserve one segment of the 20 Mbit/s all-IDR worst case before it starts.
        self.reserve_bytes = int(self.segment_s * 20e6 / 8)
        self.active = None          # dict(elements, pad, gate, start, files)
        self._closing = []          # (deadline, elements)
        self.deleted = self.recordings = 0
        self.last_files: list[str] = []

    @property
    def recording(self) -> bool:
        return self.active is not None

    def set_recording(self, wanted: bool) -> None:
        if wanted and self.active is None:
            self._start()
        elif not wanted and self.active is not None:
            self._stop()
        self._reap()

    def _start(self) -> None:
        gst = self.gst
        start = time.strftime("%Y%m%dT%H%M%S")
        queue = gst.ElementFactory.make("queue")
        queue.set_property("leaky", 2)  # downstream: a slow card drops old data, never stalls display
        queue.set_property("max-size-time", 2_000_000_000)
        queue.set_property("max-size-buffers", 0)
        queue.set_property("max-size-bytes", 0)
        parse = gst.ElementFactory.make("h264parse")
        parse.set_property("config-interval", -1)
        sink = gst.ElementFactory.make("splitmuxsink")
        sink.set_property("muxer-factory", "matroskamux")
        sink.set_property("max-size-time", self.segment_s * 1_000_000_000)
        sink.set_property("async-finalize", False)
        files = []

        def location(_sink, fragment, _first):
            self.deleted += len(prune(self.directory, self.budget_bytes, self.reserve_bytes))
            path = str(self.directory / f"{start}-{fragment:03d}{SUFFIX}")
            files.append(path)
            return path
        sink.connect("format-location-full", location)
        elements = (queue, parse, sink)
        for element in elements:
            self.pipeline.add(element)
        queue.link(parse)
        parse.link(sink)
        for element in elements:
            element.sync_state_with_parent()
        pad = self.tee.request_pad_simple("src_%u")
        gate = KeyframeGate()

        def probe(_pad, info):
            buffer = info.get_buffer()
            delta = buffer is not None and buffer.has_flags(gst.BufferFlags.DELTA_UNIT)
            return gst.PadProbeReturn.OK if gate.accept(delta) else gst.PadProbeReturn.DROP
        pad.add_probe(gst.PadProbeType.BUFFER, probe)
        pad.link(queue.get_static_pad("sink"))
        self.active = dict(elements=elements, pad=pad, gate=gate, start=start, files=files)
        self.recordings += 1

    def _stop(self) -> None:
        active, self.active = self.active, None
        queue = active["elements"][0]
        pad = active["pad"]
        pad.unlink(queue.get_static_pad("sink"))
        self.tee.release_request_pad(pad)
        # EOS finalises the open segment; the branch is removed once that is done.
        queue.get_static_pad("sink").send_event(self.gst.Event.new_eos())
        self._closing.append((time.monotonic() + self.FINALIZE_S, active["elements"]))
        self.last_files = list(active["files"])

    def _reap(self, force: bool = False) -> None:
        now = time.monotonic()
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
        if self.active is not None:
            self._stop()
        time.sleep(0.2)
        self._reap(force=True)

    def report(self) -> dict:
        active = self.active
        return dict(recording=active is not None, directory=str(self.directory),
                    current=(active["files"][-1] if active and active["files"] else None),
                    started=active["start"] if active else None,
                    buffers=active["gate"].passed if active else 0, recordings=self.recordings,
                    last_files=self.last_files, pruned=self.deleted, budget_bytes=self.budget_bytes)
