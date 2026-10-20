"""Frame-owned ROI adapter for the normal DeepStream source.

No devices are opened here. Idle/selecting frames retain detector overlap.
The router drains earlier frames only for a pending/locked box. The single
RGBA streaming owner creates, uses and closes native tracker/cache resources.
"""
from __future__ import annotations

import ctypes
import json
import threading
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Full, Queue

from common.perception import NormalizedBoxV2, PerceptionSnapshotV2
from common.visual_roi_seed import VisualRoiRelease, VisualRoiSeed, VisualRoiSelect
from jetson.visual_lock.acquisition import acquire_cached
from jetson.visual_lock.frame_cache import ExactFrameCache, bounded_catchup
from jetson.visual_lock.managed import BoundedLog, evidence_directory
from jetson.visual_lock.session import VisualLockSession


def native_factory(build: Path):
    """Validate release libraries without creating a VPI instance or EGL owner."""
    library = build / "libidcs_visual_lock.so"
    surface = build / "libidcs_visual_lock_live_surface.so"
    if not library.is_file() or not surface.is_file():
        raise RuntimeError(f"ROI unavailable: release native libraries missing under {build}")
    lib = ctypes.CDLL(str(library.resolve()))
    lib.vl_abi_version.restype = ctypes.c_int
    if lib.vl_abi_version() != 4:
        raise RuntimeError("ROI unavailable: visual-lock ABI4 required")
    from jetson.visual_lock.live_surface import BorrowedBackend, NVMMBorrower
    borrower = NVMMBorrower(surface)
    # Resolve the clone ABI before advertising capability.
    for name in ("vll_clone", "vll_clone_image", "vll_clone_release"):
        getattr(borrower.lib, name)
    return borrower, lambda w, h: BorrowedBackend(library, w, h)


class RoiRuntime:
    def __init__(self, source, tracker_session, *, build, evidence_root,
                 factory=native_factory, clock=time.monotonic):
        self.source, self.tracker_session = source, tracker_session
        self.clock = clock
        self.commands, self.messages = Queue(maxsize=8), Queue(maxsize=128)
        self.cache = ExactFrameCache(capacity=16)
        self.pending = self.active = None
        self.acquiring = None
        self.current_frame_id = None
        self.terminal = {"request_id": None, "reason": None}
        self._reset = False
        self._overflow = False
        self.barrier_requested = threading.Event()
        self.native_idle = threading.Event()
        self.native_idle.set()
        self.abort_reason = None
        self.selecting = None
        self.selection_started_s = self.selection_seen_s = 0.0
        self.expired_selection = None
        self.session = None
        self.borrower = None
        self.evidence_root = Path(evidence_root)
        self.logs = {}
        self.output = None
        self.counts = Counter()
        self.timings = deque(maxlen=600)
        self.pipeline_timings = deque(maxlen=600)
        self.frame_pipeline_ms = 0.0
        self.frame_work_ms = self.vpi_ms = 0.0
        self.selector_pause_ms = 0.0
        self.yolo_skipped = False
        self.last_verified_s = None
        self.sequence = 0
        self.error = None
        self.owner_ident = None
        try:
            self.borrower, self.backend_factory = factory(Path(build))
        except (OSError, RuntimeError, AttributeError) as exc:
            self.error = str(exc)
        self.capable = self.borrower is not None

    @property
    def owns_selection(self):
        return self.active is not None or self.pending is not None or self.acquiring is not None

    @property
    def cache_enabled(self):
        if not self.capable or self.selecting is None or self.abort_reason is not None:
            return False
        now = self.clock()
        return (0 <= now-self.selection_seen_s <= .5
                and 0 <= now-self.selection_started_s <= 15.)

    def abort(self, reason):
        # Signal only: native teardown and state cancellation belong to the
        # streaming owner, including when metadata is delayed or in flight.
        self.abort_reason = reason
        self.barrier_requested.clear()

    def _abort_on_owner(self):
        if self.abort_reason is None:
            return False
        reason = self.abort_reason
        self._cancel(reason)
        self._close_lock()
        self._reset = False
        while True:
            try:
                request, _ = self.commands.get_nowait()
            except Empty:
                break
            if isinstance(request, VisualRoiSeed):
                self._notify(request, False, reason)
        self.abort_reason = None
        return True

    def enqueue(self, message):
        try:
            kind = message.get("type")
            cls = {"visual_roi_release_v1": VisualRoiRelease,
                   "visual_roi_select_v1": VisualRoiSelect}.get(kind, VisualRoiSeed)
            command = cls.from_mapping(message)
            self.commands.put_nowait((command, self.clock()))
            if isinstance(command, VisualRoiSeed):
                self.barrier_requested.set()
        except Full:
            # A lost release cancels all authority at the next frame boundary.
            self._overflow = True
            self.counts["command_overflow"] += 1
        except (ValueError, TypeError, AttributeError):
            self.counts["invalid_commands"] += 1

    def _notify(self, request, accepted, reason):
        if not accepted:
            self.terminal = {"request_id": request.request_id, "reason": reason}
            if not self.owns_selection:
                self.expired_selection = self.selecting or self.expired_selection
                self.selecting = None
        try:
            self.messages.put_nowait(dict(type="visual_roi_result_v1",
                request_id=request.request_id, frame_id=request.frame_id,
                accepted=accepted, reason=reason))
        except Full:
            self.counts["messages_dropped"] += 1

    def _cancel(self, reason):
        request = self.active or self.acquiring or (self.pending[0] if self.pending else None)
        if request is not None:
            self._notify(request, False, reason)
        self.active = self.pending = None
        self.acquiring = None
        if self.selecting is not None:
            self.expired_selection = self.selecting
        self.selecting = None
        self._reset = True

    def begin_frame(self, *, allow_seed=True):
        """Routing decision only; never call a native function on this thread."""
        if self.abort_reason is not None:
            return False
        if self._overflow:
            self._cancel("ROI command queue overflow")
            self._overflow = False
        while True:
            # Do not let a seed arriving after the router's drain decision
            # activate on an undrained frame. Leave it in order for the next
            # boundary, where barrier_requested engages the full drain.
            with self.commands.mutex:
                if not self.commands.queue:
                    break
                if not allow_seed and isinstance(self.commands.queue[0][0], VisualRoiSeed):
                    break
                request, received_s = self.commands.queue.popleft()
                self.commands.not_full.notify()
            current = self.active or (self.pending[0] if self.pending else None)
            if isinstance(request, VisualRoiSelect):
                if request.tracker_session != self.tracker_session or self.owns_selection:
                    continue
                if not request.enabled:
                    if self.selecting == request.selection_id:
                        self.expired_selection = self.selecting
                        self.selecting = None
                elif self.capable and request.selection_id != self.expired_selection:
                    if self.selecting != request.selection_id:
                        self.selecting = request.selection_id
                        self.selection_started_s = received_s
                    self.selection_seen_s = received_s
            elif isinstance(request, VisualRoiRelease):
                if (current is not None and request.request_id == current.request_id
                        and request.tracker_session == current.tracker_session):
                    self._cancel("operator released visual lock")
            elif not self.capable:
                self._notify(request, False, self.error or "ROI unavailable")
            elif self.owns_selection:
                self._notify(request, False, "ROI is already locked or awaiting a seed")
            elif request.tracker_session != self.tracker_session:
                self._notify(request, False, "source session mismatch")
            elif not self.cache_enabled:
                self._notify(request, False, "exact frame cache was not selecting; open Select box again")
            else:
                self.pending = (request, received_s)
        with self.commands.mutex:
            if any(isinstance(item[0], VisualRoiSeed) for item in self.commands.queue):
                self.barrier_requested.set()
            else:
                self.barrier_requested.clear()
        self.yolo_skipped = self.owns_selection
        self.counts["yolo_skipped" if self.yolo_skipped else "yolo_frames"] += 1
        return self.yolo_skipped

    def _close_lock(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        self.cache.clear()
        if self.output is not None:
            summary = {name: log.report() for name, log in self.logs.items()}
            for log in self.logs.values():
                log.close()
            try:
                (self.output / "report.json").write_text(json.dumps(dict(
                    logs=summary, terminal=self.terminal, counts=self.counts)) + "\n", encoding="utf-8")
            except OSError as exc:
                self.error = f"ROI evidence report unavailable: {exc}"
        self.logs, self.output = {}, None

    def expire_cache(self):
        """Serialized owner event also expires caches when video has stopped."""
        if self.owns_selection or self.cache_enabled:
            return
        if self.owner_ident is not None and self.owner_ident != threading.get_ident():
            raise RuntimeError("ROI cache cleanup moved off its frame owner")
        if self.selecting is not None:
            self.expired_selection = self.selecting
            self.selecting = None
            self.counts["selection_timeouts"] += 1
        self.cache.clear()

    def process(self, frame, buffer, *, decoded_pts_ns=None):
        """Return a VPI Output or None. YOLO decisions belong to the caller."""
        # Ordinary metadata still advertises exact identity/capability, but no
        # cache/native/timing work belongs to the idle path.
        if (not self.cache_enabled and not self.owns_selection and not self._reset
                and self.abort_reason is None and self.selecting is None
                and not self.cache._frames):
            expected_domain = "jetson_monotonic" if self.source == "camera" else "pc_monotonic"
            if (frame is not None and frame.source_identity_verified
                    and frame.tracker_session == self.tracker_session
                    and frame.source_clock_domain == expected_domain
                    and (frame.width, frame.height) == (1280, 720)):
                self.last_verified_s = self.clock()
                self.current_frame_id = frame.frame_id
            return None
        started = time.perf_counter_ns()
        if self.owner_ident is None:
            self.owner_ident = threading.get_ident()
        elif self.owner_ident != threading.get_ident():
            raise RuntimeError("ROI native work moved off its frame owner")
        self.vpi_ms = 0.0
        result = None
        self.native_idle.clear()
        try:
            if self._abort_on_owner():
                return None
            if self._reset:
                self._close_lock()
                self._reset = False
            if not self.capable:
                return None
            expected_domain = "jetson_monotonic" if self.source == "camera" else "pc_monotonic"
            if (frame is None or not frame.source_identity_verified
                    or frame.tracker_session != self.tracker_session
                    or frame.source_clock_domain != expected_domain
                    or (frame.width, frame.height) != (1280, 720)):
                self._cancel("verified source frame unavailable")
                self._close_lock()
                return None
            self.last_verified_s = self.clock()
            self.current_frame_id = frame.frame_id
            if not self.cache_enabled and not self.owns_selection:
                self.expire_cache()
                return None
            if self.active is None:
                self.cache.add(frame, self.borrower.clone(buffer), decoded_pts_ns=decoded_pts_ns)
                self.counts["cache_copies"] += 1
                if self.pending is not None:
                    request, received_s = self.pending
                    self.pending = None
                    self.acquiring = request
                    replay = self.cache.replay_from(request)
                    if self.clock() - received_s > .5 or replay is None:
                        self._notify(request, False, "exact source frame absent or stale")
                        self.acquiring = None
                        self._cancel("exact source frame absent or stale")
                        self._close_lock()
                        return None
                    self.output = evidence_directory(self.evidence_root)
                    self.output.mkdir()
                    for name, filename, limit in (("frames", "frames.jsonl", 32_000_000),
                            ("acquisitions", "roi-acquisitions.jsonl", 8_000_000)):
                        self.logs[name] = BoundedLog(self.output / filename, limit)
                    self.session = VisualLockSession(self.backend_factory, seed_output=self.output)
                    self.session.seed_diagnostics.max_writes = 64
                    box = NormalizedBoxV2(x=request.x/frame.width, y=request.y/frame.height,
                        w=request.w/frame.width, h=request.h/frame.height)
                    replay = bounded_catchup(replay, self.session.lock.gates.max_gap_ns)
                    acquisition = acquire_cached(self.session, replay, box)
                    self.counts["vpi_seed_steps"] += len(acquisition.steps)
                    result = acquisition.result
                    self.vpi_ms = sum(step.processing_ms for step in acquisition.steps)
                    self.logs["acquisitions"].write(json.dumps(dict(request=request.as_message(),
                        accepted=acquisition.accepted, reason=acquisition.reason,
                        gates=asdict(self.session.lock.gates),
                        steps=acquisition.audit(self.cache.decoded_pts))) + "\n")
                    self.logs["acquisitions"].flush()
                    self._notify(request, acquisition.accepted, acquisition.reason)
                    self.active = request if acquisition.accepted else None
                    self.acquiring = None
                    self.expired_selection = self.selecting
                    self.selecting = None
                    self.cache.clear()
            else:
                with self.borrower.borrow(buffer) as image:
                    result = self.session.update(frame, image)
                self.counts["vpi_updates"] += 1
                self.vpi_ms = result.processing_ms
                if result.decision.state == "lost":
                    self._cancel(result.decision.reason)
            if self._abort_on_owner():
                return None
            if result is not None:
                self.logs["frames"].write(json.dumps(dict(
                    frame=frame.model_dump(mode="json"), state=result.decision.state,
                    reason=result.decision.reason, box=result.decision.box,
                    selected=result.snapshot.selection is not None, vpi_ms=self.vpi_ms,
                    yolo_skipped=self.yolo_skipped,
                    frame_work_ms=(time.perf_counter_ns()-started)/1e6)) + "\n")
                if self.active is None:
                    self._close_lock()
            return result
        except (OSError, RuntimeError, ValueError, AttributeError) as exc:
            # Native and evidence failures withdraw capability, never the video owner.
            self.error = f"ROI unavailable: {exc}"
            self.capable = False
            self._cancel(self.error)
            self._close_lock()
            return None
        finally:
            self.frame_work_ms = (time.perf_counter_ns()-started)/1e6
            self.timings.append((self.frame_work_ms, self.vpi_ms, self.yolo_skipped))
            self.native_idle.set()

    def snapshot(self, frame, result):
        # Acquisition can fail on a historical catch-up image. Never publish it
        # as current geometry, nor inherit detector selection on a skipped frame.
        if self.abort_reason is None and result is not None and result.snapshot.frame == frame:
            return result.snapshot.model_copy(update={"sequence": frame.frame_id})
        return PerceptionSnapshotV2(sequence=frame.frame_id, frame=frame)

    def status(self):
        return dict(roi_capable=self.capable, roi_error=self.error,
            roi_active=self.owns_selection and self.abort_reason is None,
            roi_request_id=None if self.active is None or self.abort_reason is not None else self.active.request_id,
            frame_work_ms=self.frame_pipeline_ms, roi_frame_work_ms=self.frame_work_ms, vpi_ms=self.vpi_ms,
            selector_pause_ms=self.selector_pause_ms,
            roi_timing_scope="ROI workflow only; idle callbacks are not timed",
            yolo_skipped=self.yolo_skipped, roi_counts=dict(self.counts),
            roi_cache_enabled=self.cache_enabled, roi_cache_frames=len(self.cache._frames),
            roi_cache_capacity=self.cache.capacity, roi_cache_rgba_bytes=16*1280*720*4)

    def source_status(self):
        self.sequence += 1
        ready = (self.capable and self.last_verified_s is not None
                 and 0 <= self.clock()-self.last_verified_s <= .25)
        cancelling = self.active or self.acquiring or (self.pending[0] if self.pending else None)
        terminal = (dict(request_id=cancelling.request_id, reason=self.abort_reason)
                    if cancelling is not None and self.abort_reason is not None else self.terminal)
        return dict(type="visual_roi_status_v1", source_mode=self.source,
            tracker_session=self.tracker_session, sequence=self.sequence,
            source_ready=ready, control_disabled=False, managed_mode=self.source,
            selection_owner="visual_lock" if self.owns_selection and self.abort_reason is None else "detector",
            active_request_id=None if self.active is None or self.abort_reason is not None else self.active.request_id,
            roi_frame_id=self.current_frame_id,
            terminal_request_id=terminal["request_id"], reason=terminal["reason"],
            **self.status())

    def close(self):
        """Must run on the RGBA owner before its queue and EGL display stop."""
        if self.owner_ident is not None and self.owner_ident != threading.get_ident():
            raise RuntimeError("ROI teardown moved off its frame owner")
        self.abort("runtime stopped")
        self._abort_on_owner()
