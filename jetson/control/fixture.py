"""Explicit Jetson receive boundary for the deterministic simulator fixture.

This must never be applied to learned-detector output or used as proof that
the encoded RTP frame arrived. The fixture is controller-only ground truth.
"""

from __future__ import annotations

from common.perception import PerceptionFrameV2, PerceptionSnapshotV2


def stamp_sim_truth_receive(
    snapshot: PerceptionSnapshotV2, *, received_monotonic_ns: int
) -> PerceptionSnapshotV2:
    """Name the actual Jetson receive clock without inventing source timing."""

    frame = snapshot.frame
    if received_monotonic_ns <= 0:
        raise ValueError("Jetson receive time must be positive")
    if snapshot.selection is None or snapshot.selection.policy != "sim_ground_truth":
        raise ValueError("only simulator ground truth may use this boundary")
    if snapshot.selection.source_frame_id != frame.frame_id:
        raise ValueError("fixture selection and source frame differ")
    if frame.source_identity_verified is not True or frame.source_clock_domain != "pc_monotonic":
        raise ValueError("fixture source identity or clock is unverified")
    if frame.received_time_ns is not None or frame.receive_clock_domain is not None:
        raise ValueError("fixture already has a receive timestamp")
    stamped = PerceptionFrameV2.model_validate({
        **frame.model_dump(),
        "received_time_ns": received_monotonic_ns,
        "receive_clock_domain": "jetson_monotonic",
        "observed_time_ns": received_monotonic_ns,
        "observation_clock_domain": "jetson_monotonic",
    })
    return PerceptionSnapshotV2.model_validate({
        **snapshot.model_dump(),
        "frame": stamped.model_dump(),
    })
