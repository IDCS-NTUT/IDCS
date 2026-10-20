"""Single-native-owner ROI adapter for the normal DeepStream source.

No devices are opened here. Idle/selecting frames retain detector overlap.
The router drains earlier frames only for a pending/locked box. The single
native worker creates, uses and closes tracker/cache resources. RGBA callbacks
retain their buffer and wait for that worker while a workflow is attached.
"""
from __future__ import annotations

import ctypes
import json
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from queue import Full, Queue

from common.perception import NormalizedBoxV2, PerceptionSnapshotV2
from common.visual_roi_seed import VisualRoiRelease, VisualRoiSeed, VisualRoiSelect
from jetson.visual_lock.acquisition import acquire_cached
from jetson.visual_lock.frame_cache import ExactFrameCache, bounded_catchup
from jetson.visual_lock.managed import BoundedLog, evidence_directory
from jetson.visual_lock.session import VisualLockSession


def on_native_owner(method):
    @wraps(method)
    def dispatch(self, *args, **kwargs):
        if self.native_owner is not None and self.native_ident != threading.get_ident():
            return self.native_owner.submit(
                self._native_call, method, args, kwargs, self.generation).result()
        return method(self, *args, **kwargs)
    return dispatch


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
        self.terminal = {"request_id": None, "reason": None, "generation": 0}
        self._reset = False
        self._overflow = False
        self.barrier_requested = threading.Event()
        self.draining = threading.Event()
        self.native_idle = threading.Event()
        self.native_idle.set()
        self.abort_reason = None
        self.selecting = None
        # One generation covers admission through complete graph restoration.
        # Controls can cross this boundary in the queue; seeds/native jobs cannot.
        self.generation = 0
        self.managed_window = False
        self.retirement_requested = False
        self.probe_state = None
        self.selection_started_s = self.selection_seen_s = 0.0
        self.expired_selection = None
        self.session = None
        self.borrower = None
        self.evidence_root = Path(evidence_root)
        self.logs = {}
        self.output = None
        self.counts = Counter()
        self.probe_status = dict(roi_probes_active=False, roi_identity_probes_active=False,
            roi_probe_transition=None, roi_owner_present=False, roi_identity_history_frames=0)
        self.timings = deque(maxlen=600)
        self.pipeline_timings = deque(maxlen=600)
        self.frame_pipeline_ms = 0.0
        self.frame_work_ms = self.vpi_ms = 0.0
        self.selector_pause_ms = 0.0
        self.yolo_skipped = False
        self.last_verified_s = None
        self.sequence = 0
        self.sequence_lock = threading.Lock()
        self.error = None
        self.owner_ident = None
        self.native_owner = None
        self.native_ident = None
        try:
            self.borrower, self.backend_factory = factory(Path(build))
        except Exception as exc:
            self.error = str(exc)
        self.capable = self.borrower is not None

    @property
    def owns_selection(self):
        return self.active is not None or self.pending is not None or self.acquiring is not None

    @property
    def cache_enabled(self):
        if not self.capable or self.owns_selection or self.abort_reason is not None:
            return False
        return self.lease_live

    @property
    def lease_live(self):
        if self.selecting is None:
            return False
        now = self.clock()
        return (0 <= now-self.selection_seen_s <= .5
                and (self.owns_selection or 0 <= now-self.selection_started_s <= 15.))

    def start_native_owner(self):
        # No worker exists on the idle path. Buffer callbacks wait for their
        # native call, keeping the GstBuffer/surface alive until it completes.
        if self.native_owner is None:
            self.native_owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="roi-native")

    def _native_call(self, method, args, kwargs, generation):
        if not self.matches_generation(generation):
            return None
        self.native_ident = threading.get_ident()
        return method(self, *args, **kwargs)

    def submit_native(self, action):
        self.start_native_owner()
        return self.native_owner.submit(
            self._native_call, lambda _: action(), (), {}, self.generation)

    def matches_generation(self, generation):
        if generation is None or generation == self.generation:
            return True
        self.counts["stale_window_events"] += 1
        return False

    def finish_window(self):
        """Called only after worker shutdown and ordinary graph restoration."""
        if self.owns_selection or self.selecting is not None:
            raise RuntimeError("ROI window restored before selection retirement")
        self.generation += 1
        self.retirement_requested = False
        self.draining.clear()

    def stop_native_owner(self):
        if self.native_owner is not None:
            self.native_owner.shutdown(wait=True)
            self.native_owner = None
            self.native_ident = None

    def fault(self, reason):
        self.error = reason
        self.terminal = dict(self.terminal, reason=reason)
        self.capable = False
        self.counts["faults"] += 1
        self.abort(reason)
        print(f"[deepstream.roi] {reason}; YOLO remains available", flush=True)

    def abort(self, reason):
        # Signal only: native teardown and state cancellation belong to the
        # native owner, including when metadata is delayed or in flight.
        self.abort_reason = reason
        self.barrier_requested.clear()

    @on_native_owner
    def _abort_on_owner(self):
        if self.abort_reason is None:
            return False
        reason = self.abort_reason
        self._cancel(reason)
        if self.session is not None or self.cache._frames or self.logs or self.output is not None:
            self._close_lock()
        self._reset = False
        self._retire_seeds(reason)
        self.abort_reason = None
        return True

    @on_native_owner
    def service(self):
        """Timer work, serialized with native calls even when video stops."""
        if self._abort_on_owner():
            return
        if self.draining.is_set():
            return
        self.begin_frame(allow_seed=False, count_frame=False)
        if self.selecting is not None and not self.lease_live:
            self._cancel("ROI lease expired")
        if self._reset:
            self._close_lock()
            self._reset = False
        if not self.owns_selection and not self.cache_enabled:
            self.expire_cache()

    def enqueue(self, message, *, generation=None):
        try:
            kind = message.get("type")
            cls = {"visual_roi_release_v1": VisualRoiRelease,
                   "visual_roi_select_v1": VisualRoiSelect}.get(kind, VisualRoiSeed)
            command = cls.from_mapping(message)
            if isinstance(command, VisualRoiSelect):
                # Removal can outlast the bounded command queue. Coalesce
                # adjacent renewals without crossing a release/cancel/seed.
                with self.commands.mutex:
                    if self.commands.queue and self.commands.queue[-1][0] == command:
                        self.commands.queue[-1] = (command, self.clock(), None)
                        return
            self.commands.put_nowait((command, self.clock(),
                (self.generation if generation is None else generation)
                if isinstance(command, VisualRoiSeed) else None))
            if isinstance(command, VisualRoiSeed):
                self.barrier_requested.set()
        except Full:
            # A lost release cancels all authority at the next frame boundary.
            self._overflow = True
            self.counts["command_overflow"] += 1
            if isinstance(command, VisualRoiSelect):
                self._refuse_select(command, "ROI command queue overflow; retry selection")
        except (ValueError, TypeError, AttributeError):
            self.counts["invalid_commands"] += 1

    def _notify(self, request, accepted, reason, *, generation=None):
        # Refusal is request-scoped. A seed has no selection ID on the wire;
        # never infer authority to retire the current lease from its arrival.
        if not accepted and self.matches_generation(generation) and not self.owns_selection:
            self.terminal = dict(request_id=request.request_id,
                                 selection_id=None, reason=reason, generation=self.generation)
        try:
            self.messages.put_nowait(dict(type="visual_roi_result_v1",
                request_id=request.request_id, frame_id=request.frame_id,
                accepted=accepted, reason=reason))
        except Full:
            self.counts["messages_dropped"] += 1

    def _cancel(self, reason, *, allow_reselect=False):
        request = self.active or self.acquiring or (self.pending[0] if self.pending else None)
        if self.managed_window and (request is not None or self.selecting is not None):
            # Close admission at cancellation, before GLib notices an empty
            # window. Later controls remain ordered/coalesced for its successor.
            self.retirement_requested = True
        # Store the terminal before dropping ownership; readers on other
        # threads must never observe an unexplained ownership exit.
        if request is not None or self.selecting is not None:
            self.terminal = dict(request_id=None if request is None else request.request_id,
                                 selection_id=self.selecting, reason=reason, generation=self.generation)
        if request is not None:
            self._notify(request, False, reason)
        self.active = self.pending = None
        self.acquiring = None
        if self.selecting is not None:
            self.expired_selection = None if allow_reselect else self.selecting
        self.selecting = None
        self._reset = True
        if request is not None:
            self._publish_terminal()

    def _publish_terminal(self):
        try:
            self.messages.put_nowait(self.source_status())
        except Full:
            self.counts["messages_dropped"] += 1

    def _refuse_select(self, request, reason):
        self.terminal = dict(request_id=None, selection_id=request.selection_id,
                             reason=reason, generation=self.generation)
        self._publish_terminal()

    @on_native_owner
    def begin_frame(self, *, allow_seed=True, count_frame=True):
        """Decide routing; retire a cancelled payload before reopening YOLO."""
        # Checked on the worker, not just at submission on GLib/Gst. Calls
        # already queued at the start of removal cannot consume the next lease.
        if self.draining.is_set() or self.retirement_requested:
            return False
        if self.abort_reason is not None:
            if self.native_owner is not None:
                self._abort_on_owner()
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
                if self.draining.is_set() or self.retirement_requested:
                    break
                if not allow_seed and isinstance(self.commands.queue[0][0], VisualRoiSeed):
                    break
                request, received_s, generation = self.commands.queue[0]
                superseded = (self.managed_window and self.selecting is not None
                    and not self.owns_selection and isinstance(request, VisualRoiSelect)
                    and request.enabled and request.tracker_session == self.tracker_session
                    and request.selection_id not in (self.selecting, self.expired_selection))
                if not superseded:
                    self.commands.queue.popleft()
                    self.commands.not_full.notify()
            if superseded:
                self._cancel("ROI selection superseded")
                break
            current = self.active or self.acquiring or (self.pending[0] if self.pending else None)
            if isinstance(request, VisualRoiSelect):
                if request.tracker_session != self.tracker_session:
                    self._refuse_select(request, "source session mismatch")
                    continue
                if self.owns_selection and request.selection_id != self.selecting:
                    self._refuse_select(request, "ROI is already locked or awaiting a seed")
                    continue
                if not request.enabled:
                    if self.selecting == request.selection_id:
                        self._cancel("operator cancelled ROI lease")
                elif self.capable and request.selection_id != self.expired_selection:
                    if self.selecting != request.selection_id:
                        self.selecting = request.selection_id
                        self.selection_started_s = received_s
                    self.selection_seen_s = received_s
                elif request.enabled:
                    self._refuse_select(request,
                        self.error or "ROI lease retired; open Select box again")
            elif isinstance(request, VisualRoiRelease):
                if (current is not None and request.request_id == current.request_id
                        and request.tracker_session == current.tracker_session):
                    self._cancel("operator released visual lock", allow_reselect=True)
            elif not self.matches_generation(generation):
                self._notify(request, False, "ROI identity history retired; open Select box again",
                             generation=generation)
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
        if self.selecting is not None and not self.lease_live:
            self._cancel("ROI lease expired")
        if self._reset and self.native_owner is not None:
            self._close_lock()
            self._reset = False
        self.yolo_skipped = self.owns_selection
        if count_frame:
            self.counts["yolo_skipped" if self.yolo_skipped else "yolo_frames"] += 1
        return self.yolo_skipped

    @on_native_owner
    def _close_lock(self):
        # One failed destructor must not strand the rest of the resources or
        # escape into the ordinary metadata/pipeline owner. Never retry an
        # uncertain native destructor from a different thread.
        errors = []
        def cleanup(action):
            try:
                action()
            except Exception as exc:
                errors.append(str(exc))
        session, self.session = self.session, None
        if session is not None:
            cleanup(session.close)
        while self.cache._frames:
            cleanup(self.cache._evict)
        summary = {}
        for name, log in self.logs.items():
            cleanup(lambda: summary.update({name: log.report()}))
            cleanup(log.close)
        if self.output is not None:
            cleanup(lambda: (self.output / "report.json").write_text(json.dumps(dict(
                logs=summary, terminal=self.terminal, counts=self.counts)) + "\n", encoding="utf-8"))
        self.logs, self.output = {}, None
        if errors:
            self.error = "ROI cleanup fault: " + "; ".join(errors)
            self.capable = False
            self.terminal = dict(self.terminal, reason=self.error)
            self.counts["cleanup_faults"] += 1
            print(f"[deepstream.roi] {self.error}; YOLO remains available", flush=True)

    @on_native_owner
    def expire_cache(self, *, generation=None):
        """Native-owner task also expires caches when video has stopped."""
        if not self.matches_generation(generation):
            return
        if self.owns_selection or self.cache_enabled:
            return
        if self.owner_ident is not None and self.owner_ident != threading.get_ident():
            raise RuntimeError("ROI cache cleanup moved off its frame owner")
        if self.selecting is not None:
            self._cancel("ROI lease expired")
            self.counts["selection_timeouts"] += 1
        self._close_lock()
        self._reset = False

    @on_native_owner
    def retire(self, *, generation=None):
        """Release resources on their worker before removing the drained graph."""
        if not self.matches_generation(generation):
            return
        if self.owner_ident is not None and self.owner_ident != threading.get_ident():
            raise RuntimeError("ROI retirement moved off its frame owner")
        if self.owns_selection:
            raise RuntimeError("ROI retirement attempted during a lock")
        if self.selecting is not None:
            self._cancel("ROI window retired")
        self._close_lock()
        self._reset = False
        self.owner_ident = None
        self._retire_seeds("ROI identity history retired; open Select box again", generation=generation)

    def _retire_seeds(self, reason, *, generation=None):
        # Video can stop with an unconsumed heartbeat/seed in the router queue.
        # Keep control intent for the next lifecycle, but an old-window seed
        # must neither hold probes forever nor acquire from a new lease/cache.
        if not self.matches_generation(generation):
            return
        with self.commands.mutex:
            seeds = [item[0] for item in self.commands.queue
                     if isinstance(item[0], VisualRoiSeed) and item[2] == self.generation]
            controls = [item for item in self.commands.queue
                        if not isinstance(item[0], VisualRoiSeed) or item[2] != self.generation]
            self.commands.queue.clear()
            self.commands.queue.extend(controls)
            self.commands.not_full.notify_all()
        if any(isinstance(item[0], VisualRoiSeed) for item in controls):
            self.barrier_requested.set()
        else:
            self.barrier_requested.clear()
        for request in seeds:
            self._notify(request, False, reason, generation=generation)

    @on_native_owner
    def process(self, frame, buffer, *, decoded_pts_ns=None):
        """Return a VPI Output or None. YOLO decisions belong to the caller."""
        if self.draining.is_set():
            return None
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
            if self.retirement_requested:
                return None
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
                    self.acquiring = request
                    self.pending = None
                    replay = self.cache.replay_from(request)
                    if self.clock() - received_s > .5 or replay is None:
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
                    if acquisition.accepted:
                        self._notify(request, True, acquisition.reason)
                        self.active = request
                        self.acquiring = None
                    else:
                        self._cancel(acquisition.reason)
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
        except Exception as exc:
            # Native and evidence failures withdraw capability, never the video owner.
            self.error = f"ROI unavailable: {exc}"
            self.capable = False
            print(f"[deepstream.roi] {self.error}; YOLO remains available", flush=True)
            self._cancel(self.error)
            self._close_lock()
            return None
        finally:
            # Covers early returns (including withdrawn capability) as well
            # as an unexpected empty acquisition/update. Never leave a client
            # owning a request whose frame could not produce an ROI result.
            if result is None and self.owns_selection:
                self._cancel(self.error or "ROI frame produced no visual lock result")
                self._close_lock()
            self.frame_work_ms = (time.perf_counter_ns()-started)/1e6
            self.timings.append((self.frame_work_ms, self.vpi_ms, self.yolo_skipped))
            self.native_idle.set()

    def snapshot(self, frame, result):
        # Acquisition can fail on a historical catch-up image. Never publish it
        # as current geometry, nor inherit detector selection on a skipped frame.
        if (self.abort_reason is None and result is not None and result.snapshot.frame == frame
                and (result.snapshot.selection is None or self.owns_selection)):
            return result.snapshot.model_copy(update={"sequence": frame.frame_id})
        return PerceptionSnapshotV2(sequence=frame.frame_id, frame=frame)

    def status(self):
        probes = self.probe_status if self.probe_state is None else self.probe_state()
        return dict(**probes, roi_window_generation=self.generation,
            roi_window_retiring=self.retirement_requested,
            roi_selection_id=self.selecting,
            roi_capable=self.capable, roi_error=self.error,
            roi_active=self.owns_selection and self.abort_reason is None,
            roi_request_id=None if self.active is None or self.abort_reason is not None else self.active.request_id,
            frame_work_ms=self.frame_pipeline_ms, roi_frame_work_ms=self.frame_work_ms, vpi_ms=self.vpi_ms,
            selector_pause_ms=self.selector_pause_ms,
            roi_timing_scope="ROI workflow only; no idle ROI frame/packet callbacks",
            yolo_skipped=self.yolo_skipped, roi_counts=dict(self.counts),
            roi_cache_enabled=self.cache_enabled, roi_cache_frames=len(self.cache._frames),
            roi_cache_capacity=self.cache.capacity, roi_cache_rgba_bytes=16*1280*720*4)

    def next_status_sequence(self):
        with self.sequence_lock:
            self.sequence += 1
            return self.sequence

    def source_status(self):
        sequence = self.next_status_sequence()
        ready = (self.capable and self.last_verified_s is not None
                 and 0 <= self.clock()-self.last_verified_s <= .25)
        cancelling = self.active or self.acquiring or (self.pending[0] if self.pending else None)
        terminal = (dict(request_id=cancelling.request_id, selection_id=self.selecting,
                         reason=self.abort_reason)
                    if cancelling is not None and self.abort_reason is not None else self.terminal)
        return dict(type="visual_roi_status_v1", source_mode=self.source,
            tracker_session=self.tracker_session, sequence=sequence,
            source_ready=ready, control_disabled=False, managed_mode=self.source,
            selection_owner="visual_lock" if self.owns_selection and self.abort_reason is None else "detector",
            active_request_id=None if self.active is None or self.abort_reason is not None else self.active.request_id,
            roi_frame_id=self.current_frame_id,
            terminal_request_id=terminal["request_id"], reason=terminal["reason"],
            terminal_selection_id=terminal.get("selection_id"),
            terminal_window_generation=terminal.get("generation", self.generation),
            **self.status())

    @on_native_owner
    def close(self):
        """Must run on the native owner before the queue and EGL display stop."""
        if self.owner_ident is not None and self.owner_ident != threading.get_ident():
            raise RuntimeError("ROI teardown moved off its frame owner")
        self.abort("runtime stopped")
        self._abort_on_owner()
        self.owner_ident = None
