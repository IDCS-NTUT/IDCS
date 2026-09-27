"""Causally stamp a verified PC-source snapshot at Jetson receipt/observation."""

from __future__ import annotations

from common.perception import PerceptionFrameV2, PerceptionSnapshotV2


def stamp_verified_snapshot(
    snapshot: PerceptionSnapshotV2, *, received_ns: int, observed_ns: int,
) -> PerceptionSnapshotV2:
    """Keep the original source time; add only locally measured Jetson times."""

    frame = snapshot.frame
    if frame.source_identity_verified is not True:
        raise ValueError("source frame identity is not verified")
    if frame.source_clock_domain != "pc_monotonic" or frame.source_time_ns <= 0:
        raise ValueError("source timestamp is not PC monotonic")
    if not 0 < received_ns <= observed_ns:
        raise ValueError("Jetson receipt/observation order is invalid")
    if frame.received_time_ns is not None:
        raise ValueError("snapshot already carries a receipt timestamp")
    stamped = PerceptionFrameV2.model_validate({
        **frame.model_dump(),
        "received_time_ns": received_ns,
        "receive_clock_domain": "jetson_monotonic",
        "observed_time_ns": observed_ns,
        "observation_clock_domain": "jetson_monotonic",
    })
    return PerceptionSnapshotV2.model_validate({
        **snapshot.model_dump(), "frame": stamped.model_dump(),
    })
