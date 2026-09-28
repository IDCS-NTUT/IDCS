"""Causally stamp a verified PC-source snapshot at Jetson receipt/observation."""

from __future__ import annotations

from common.perception import PerceptionFrameV2, PerceptionSnapshotV2


def stamp_verified_snapshot(
    snapshot: PerceptionSnapshotV2, *, received_ns: int, observed_ns: int,
    keep_upstream_receipt: bool = False, source_clock_domain: str = "pc_monotonic",
) -> PerceptionSnapshotV2:
    """Keep the original source time; add receipt/observation on the controller's clock.

    ``jetson_monotonic`` labels the controller-local clock. DeepStream already
    measures when each frame arrived on the Jetson; with
    ``keep_upstream_receipt`` (controller running on the Jetson) those earlier,
    truer times are kept. Otherwise any upstream receipt times are on another
    host's clock and are replaced with this process's own stamps.
    """

    frame = snapshot.frame
    if frame.source_identity_verified is not True:
        raise ValueError("source frame identity is not verified")
    if frame.source_clock_domain != source_clock_domain or frame.source_time_ns <= 0:
        raise ValueError(f"source timestamp is not {source_clock_domain}")
    if not 0 < received_ns <= observed_ns:
        raise ValueError("Jetson receipt/observation order is invalid")
    if frame.received_time_ns is not None:
        if keep_upstream_receipt and frame.receive_clock_domain == "jetson_monotonic" \
                and frame.observation_clock_domain == "jetson_monotonic":
            return snapshot
        # Receipt measured on another host's clock: re-stamp locally.
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
