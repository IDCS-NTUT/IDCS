"""One explicit exact-frame acquisition; the first failure is terminal.

The caller owns cached pixels. No retries, time estimates, reseeds or changes
to tracker gates are made here. Steps retain the real source frame and quality
result even when acquisition fails before reaching the current live image.
"""
from dataclasses import asdict, dataclass

from .session import Output


@dataclass(frozen=True)
class Acquisition:
    steps: tuple[Output, ...]

    @property
    def result(self):
        return self.steps[-1]

    @property
    def accepted(self):
        return self.result.decision.state != "lost"

    @property
    def reason(self):
        if self.accepted:
            return "exact-frame VPI seed accepted"
        stage = "seed" if len(self.steps) == 1 else "catch-up"
        return f"{stage}: {self.result.decision.reason}"

    def audit(self, decoded_pts):
        rows = []
        for index, output in enumerate(self.steps):
            frame, proposal = output.snapshot.frame, output.proposal
            rows.append(dict(stage="seed" if index == 0 else "catch-up",
                frame=frame.model_dump(mode="json"), decoded_pts_ns=decoded_pts(frame),
                decision=asdict(output.decision), processing_ms=output.processing_ms,
                seed_report=output.seed_report,
                proposal=None if proposal is None else dict(
                    box=proposal.box, psr=proposal.psr, peak=proposal.peak,
                    seed_features=proposal.seed_features,
                    input_features=proposal.input_features,
                    forward_features=proposal.forward_features,
                    backward_features=proposal.backward_features,
                    native_ms=proposal.native_ms)))
        return rows


def acquire_cached(session, replay, box):
    steps = []
    for index, (frame, image) in enumerate(replay):
        output = session.seed(frame, image, box) if index == 0 else session.update(frame, image)
        steps.append(output)
        if output.decision.state == "lost":
            break
    if not steps:
        raise ValueError("exact seed image required")
    return Acquisition(tuple(steps))
