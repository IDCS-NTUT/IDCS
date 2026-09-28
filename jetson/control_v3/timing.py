"""Causal timing evidence; never turn a cross-host estimate into an exact time."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeInterval:
    earliest_ns: int
    latest_ns: int

    def __post_init__(self) -> None:
        if self.earliest_ns > self.latest_ns:
            raise ValueError("invalid time interval")


@dataclass(frozen=True)
class ClockBounds:
    """Bounds for PC monotonic minus Jetson monotonic.

    A four-timestamp exchange bounds offset only if one-way network delays
    are nonnegative.  Drift between that exchange and a frame is separately
    bounded; callers must supply a *measured* or specified oscillator bound.
    """

    offset_min_ns: int
    offset_max_ns: int
    observed_jetson_ns: int
    max_drift_ppm: float | None = None

    @classmethod
    def from_exchange(
        cls,
        *,
        jetson_send_ns: int,
        pc_receive_ns: int,
        pc_send_ns: int,
        jetson_receive_ns: int,
        max_drift_ppm: float | None = None,
    ) -> ClockBounds:
        if jetson_send_ns > jetson_receive_ns or pc_receive_ns > pc_send_ns:
            raise ValueError("exchange timestamps out of order")
        minimum = pc_send_ns - jetson_receive_ns
        maximum = pc_receive_ns - jetson_send_ns
        if minimum > maximum:
            raise ValueError("impossible clock exchange")
        return cls(minimum, maximum, jetson_receive_ns, max_drift_ppm)

    def __post_init__(self) -> None:
        if self.offset_min_ns > self.offset_max_ns:
            raise ValueError("invalid clock offset bounds")
        if self.max_drift_ppm is not None and (
            not math.isfinite(self.max_drift_ppm)
            or not 0 <= self.max_drift_ppm < 1_000_000
        ):
            raise ValueError("drift bound must be finite and in [0, 1e6) ppm")

    def map_pc_event(self, pc_event_ns: int, *, jetson_now_ns: int) -> TimeInterval:
        elapsed_ns = jetson_now_ns - self.observed_jetson_ns
        if elapsed_ns < 0:
            raise ValueError("clock sample is from the future")
        if elapsed_ns and self.max_drift_ppm is None:
            raise ValueError("no clock drift bound")
        # The frame can precede the exchange. Bound drift over the larger
        # separation from calibration to either the frame or the query.
        span_ns = max(
            elapsed_ns,
            abs(pc_event_ns - self.offset_min_ns - self.observed_jetson_ns),
            abs(pc_event_ns - self.offset_max_ns - self.observed_jetson_ns),
        )
        drift_fraction = (self.max_drift_ppm or 0.0) / 1_000_000
        # Solve the bound conservatively because the true event position also
        # affects the time span over which oscillator drift accumulates.
        drift_ns = math.ceil(span_ns * drift_fraction / (1.0 - drift_fraction))
        return TimeInterval(
            pc_event_ns - self.offset_max_ns - drift_ns,
            pc_event_ns - self.offset_min_ns + drift_ns,
        )


@dataclass(frozen=True)
class FrameTimingEvidence:
    frame_id: int
    identity_verified: bool
    pc_source_ns: int
    jetson_received_ns: int
    jetson_observed_ns: int
    jetson_decision_ns: int


@dataclass(frozen=True)
class TimingVerdict:
    valid: bool
    reason: str
    capture_age_ns: TimeInterval | None = None
    receive_to_observe_ns: int | None = None
    observe_to_decision_ns: int | None = None


def verify_frame_timing(
    evidence: FrameTimingEvidence,
    clock: ClockBounds | None,
    *,
    max_clock_sample_age_ns: int,
    max_capture_age_ns: int,
) -> TimingVerdict:
    """Require verified frame identity and conservative capture-age bounds."""

    if not evidence.identity_verified:
        return TimingVerdict(False, "frame_identity_unverified")
    if clock is None:
        return TimingVerdict(False, "clock_unavailable")
    if not (
        evidence.jetson_received_ns
        <= evidence.jetson_observed_ns
        <= evidence.jetson_decision_ns
    ):
        return TimingVerdict(False, "local_timestamps_out_of_order")
    if not 0 <= evidence.jetson_decision_ns - clock.observed_jetson_ns <= max_clock_sample_age_ns:
        return TimingVerdict(False, "clock_sample_stale")
    try:
        source = clock.map_pc_event(
            evidence.pc_source_ns, jetson_now_ns=evidence.jetson_decision_ns
        )
    except ValueError as exc:
        return TimingVerdict(False, str(exc).replace(" ", "_"))
    if source.latest_ns > evidence.jetson_received_ns:
        return TimingVerdict(False, "capture_after_receive_possible")
    age = TimeInterval(
        evidence.jetson_decision_ns - source.latest_ns,
        evidence.jetson_decision_ns - source.earliest_ns,
    )
    if age.earliest_ns < 0 or age.latest_ns > max_capture_age_ns:
        return TimingVerdict(False, "capture_age_out_of_bounds", age)
    return TimingVerdict(
        True,
        "verified",
        age,
        evidence.jetson_observed_ns - evidence.jetson_received_ns,
        evidence.jetson_decision_ns - evidence.jetson_observed_ns,
    )
