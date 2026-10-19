"""Single-owner image/session coordinator for replay and future live adapters.

No network, YOLO, simulator truth, controller or actuator dependencies. A caller
must supply the exact verified source frame and its pixels for an operator seed.
Backend failures latch loss, close GPU resources and suppress selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable

from common.perception import NormalizedBoxV2, PerceptionFrameV2, PerceptionSnapshotV2
from .fusion import Decision, Gates, Proposal, VisualLock
from .seed import SeedDiagnostics


@dataclass(frozen=True)
class Output:
    decision: Decision
    snapshot: PerceptionSnapshotV2
    proposal: Proposal | None
    processing_ms: float
    seed_report: dict | None = None


class VisualLockSession:
    def __init__(self, backend_factory: Callable, gates: Gates = Gates(), *, seed_output=None):
        self._factory = backend_factory
        self._backend = None
        self.lock = VisualLock(gates)
        self.sequence = 0
        self.seed_diagnostics = SeedDiagnostics(seed_output) if seed_output is not None else None

    def _close_backend(self):
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()

    def close(self):
        self.lock.fail("session closed")
        self._close_backend()

    def _output(self, frame, decision, proposal, start):
        self.sequence += 1
        # An empty failure output names the actual rejected frame, not an old
        # accepted image. A new source cannot inherit a selected track.
        snapshot = (self.lock.snapshot(self.sequence, decision) if self.lock.frame is not None
                    else PerceptionSnapshotV2(sequence=self.sequence,frame=frame))
        if not decision.accepted:
            snapshot = snapshot.model_copy(update={"frame":frame})
        report = asdict(proposal.seed_report) if proposal is not None and proposal.seed_report else None
        return Output(decision, snapshot, proposal, (time.perf_counter_ns()-start)/1e6, report)

    def seed(self, frame: PerceptionFrameV2, rgba, box: NormalizedBoxV2) -> Output:
        start = time.perf_counter_ns()
        self.lock.fail("new operator seed")
        self._close_backend()
        p = None
        # Validate identity/geometry before any native work. Initial texture
        # count is only a placeholder here; actual count decides acquisition.
        d = self.lock.seed(frame,box,self.lock.gates.min_points)
        if d.state == "lost":
            return self._output(frame,d,None,start)
        try:
            self._backend = self._factory(frame.width,frame.height)
            self._backend.seed_min_points = self.lock.gates.min_points
            p = self._backend.process(rgba,seed=self.lock.box)
            if p.seed_features < self.lock.gates.min_points:
                r = p.seed_report
                reason = ("seed lacks texture" if r is None or r.refined_box == r.operator_box
                          else r.reason if r.refined_box is None else "seed lacks texture after refinement")
                d = self.lock.fail(reason)
                self._close_backend()
            elif p.seed_report is not None:
                x,y,w,h = p.seed_report.refined_box
                d = self.lock.seed(frame,NormalizedBoxV2(x=x/frame.width,y=y/frame.height,
                    w=w/frame.width,h=h/frame.height),p.seed_features)
                if d.state == "lost": self._close_backend()
            if self.seed_diagnostics is not None and p.seed_report is not None:
                self.seed_diagnostics.write(p.seed_report,p.seed_gray,frame,d)
        except Exception as exc:
            d = self.lock.fail(f"backend fault: {exc}")
            self._close_backend()
        return self._output(frame,d,p,start)

    def update(self, frame: PerceptionFrameV2, rgba) -> Output:
        start = time.perf_counter_ns()
        p = None
        preflight = self.lock.check_frame(frame)
        if preflight is not None:
            if preflight.state == "lost": self._close_backend()
            return self._output(frame,preflight,None,start)
        try:
            p = self._backend.process(rgba)
            d = self.lock.evaluate(frame,p)
            if d.accepted:
                self._backend.commit(d.box,update_template=d.update_template,
                                     retain_mask=d.retain_mask,replenish=True)
            else:
                self._backend.discard()
            if d.state == "lost": self._close_backend()
        except Exception as exc:
            d = self.lock.fail(f"backend fault: {exc}")
            self._close_backend()
        return self._output(frame,d,p,start)
