"""V2 adapter from perception/state metadata to ``ControlObservation``.

The adapter intentionally refuses to infer missing input.  It keeps local
receipt timestamps for each source and marks each component invalid once it
ages out.  It neither publishes nor imports gimbal/serial drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from common.aiming import solve_snapshot_aiming
from common.control import ControlConfig, LaserMountConfig
from common.perception import PerceptionSnapshotV2
from common.schemas import (
    CamState,
    ControlGimbalObservation,
    ControlObservation,
    ControlSafetyObservation,
    ControlTargetObservation,
    ControlTransportObservation,
    ManualControlState,
)


@dataclass(frozen=True)
class ObservationAgeLimits:
    target_s: float = 0.100
    gimbal_s: float = 0.100
    safety_s: float = 0.750


class ControlObservationAssembler:
    """Assemble latest-only metadata into an atomic controller input.

    ``PerceptionSnapshotV2`` is the only perception input. Selection, range,
    and parallax geometry are resolved once and carried atomically.
    """

    def __init__(
        self,
        config: ControlConfig,
        *,
        laser_mount: Optional[LaserMountConfig] = None,
        age_limits: ObservationAgeLimits = ObservationAgeLimits(),
        sequence_base: int = 0,
    ) -> None:
        if sequence_base < 0:
            raise ValueError("sequence_base must be non-negative")
        self._config = config
        self._laser_mount = laser_mount
        self._limits = age_limits
        self._perception: Optional[Tuple[PerceptionSnapshotV2, float]] = None
        self._cam_state: Optional[Tuple[CamState, float]] = None
        self._manual: Optional[Tuple[ManualControlState, float]] = None
        self._sequence = int(sequence_base)

    def update_perception_snapshot(
        self, snapshot: PerceptionSnapshotV2, *, received_at: float
    ) -> None:
        """Accept an immutable V2 snapshot as the latest target input.

        The snapshot is retained unchanged; selection and geometry are read
        from its validated track/selection records at build time.
        """

        self._perception = (snapshot, float(received_at))

    def update_cam_state(self, state: CamState, *, received_at: float) -> None:
        self._cam_state = (state, float(received_at))

    def update_manual_state(self, state: ManualControlState, *, received_at: float) -> None:
        self._manual = (state, float(received_at))

    @staticmethod
    def _age_ms(now: float, received_at: float) -> float:
        return max(0.0, (now - received_at) * 1000.0)

    def _target(self, now: float) -> ControlTargetObservation:
        if self._perception is None:
            return ControlTargetObservation(valid=False)
        snapshot, received_at = self._perception
        age_ms = self._age_ms(now, received_at)
        solution = solve_snapshot_aiming(
            snapshot, self._config, self._laser_mount
        )
        if age_ms > self._limits.target_s * 1000.0 or solution is None:
            return ControlTargetObservation(valid=False, source_age_ms=age_ms)
        return ControlTargetObservation(
            valid=True,
            track_id=solution.track_id,
            class_id=solution.class_id,
            confidence=solution.confidence,
            target_center_px=solution.target_px,
            aim_reference_px=solution.aim_px,
            pixel_error=solution.pixel_error,
            bearing_error_rad=solution.bearing_error_rad,
            distance_m=solution.distance_m,
            distance_source=solution.distance_source,
            parallax_active=solution.parallax_active,
            on_target=solution.on_target,
            source_age_ms=age_ms,
        )

    def _gimbal(self, now: float) -> ControlGimbalObservation:
        if self._cam_state is None:
            return ControlGimbalObservation(valid=False)
        state, received_at = self._cam_state
        age_ms = self._age_ms(now, received_at)
        if age_ms > self._limits.gimbal_s * 1000.0:
            return ControlGimbalObservation(valid=False, sample_age_ms=age_ms)
        return ControlGimbalObservation(
            valid=True, yaw_rad=state.pan, pitch_rad=state.tilt,
            yaw_rate_rad_s=state.pan_rate, pitch_rate_rad_s=state.tilt_rate,
            sample_age_ms=age_ms,
        )

    def _safety(self, now: float) -> ControlSafetyObservation:
        if self._manual is None:
            return ControlSafetyObservation(valid=False, auto_allowed=False,
                                            manual_active=False, emergency_active=False)
        state, received_at = self._manual
        age_ms = self._age_ms(now, received_at)
        valid = age_ms <= self._limits.safety_s * 1000.0
        return ControlSafetyObservation(
            valid=valid,
            auto_allowed=bool(valid and not state.active and not state.emergency and state.control_cmd_enabled),
            manual_active=bool(state.active), emergency_active=bool(state.emergency), sample_age_ms=age_ms,
        )

    def _source_provenance(self) -> tuple[Optional[int], Optional[int], Optional[str]]:
        if self._perception is None:
            return None, None, None
        snapshot, _received_at = self._perception
        return (
            snapshot.frame.frame_id,
            snapshot.frame.source_time_ns,
            snapshot.frame.source_clock_domain,
        )

    def build(self, *, now: float, serial_acceptance_ms: Optional[float] = None,
              last_command_age_ms: Optional[float] = None) -> ControlObservation:
        """Return a fully validated snapshot; no absent measurement is fabricated."""

        self._sequence += 1
        source_frame_id, source_time_ns, source_clock_domain = self._source_provenance()
        return ControlObservation(
            sequence=self._sequence, created_monotonic_ns=int(now * 1_000_000_000),
            source_frame_id=source_frame_id, source_time_ns=source_time_ns,
            source_clock_domain=source_clock_domain,
            target=self._target(now), gimbal=self._gimbal(now),
            transport=ControlTransportObservation(
                serial_acceptance_ms=serial_acceptance_ms, last_command_age_ms=last_command_age_ms,
            ), safety=self._safety(now),
        )
