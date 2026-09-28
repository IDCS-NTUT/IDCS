"""Bounded four-timestamp mapping from PC monotonic to Jetson monotonic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClockOffsetSample:
    # PC monotonic minus Jetson monotonic, in nanoseconds.
    offset_ns: int
    round_trip_ns: int
    uncertainty_ns: int
    observed_jetson_ns: int

    def map_pc_ns(self, source_ns: int) -> int:
        return source_ns - self.offset_ns


def calculate_clock_offset(
    jetson_send_ns: int,
    pc_receive_ns: int,
    pc_send_ns: int,
    jetson_receive_ns: int,
) -> ClockOffsetSample:
    if not (jetson_send_ns <= jetson_receive_ns and pc_receive_ns <= pc_send_ns):
        raise ValueError("clock exchange timestamps are out of order")
    round_trip_ns = (
        jetson_receive_ns - jetson_send_ns - (pc_send_ns - pc_receive_ns)
    )
    if round_trip_ns < 0:
        raise ValueError("clock exchange has negative network round trip")
    offset_ns = (
        (pc_receive_ns - jetson_send_ns)
        + (pc_send_ns - jetson_receive_ns)
    ) // 2
    return ClockOffsetSample(
        offset_ns=offset_ns,
        round_trip_ns=round_trip_ns,
        uncertainty_ns=round_trip_ns // 2,
        observed_jetson_ns=jetson_receive_ns,
    )
