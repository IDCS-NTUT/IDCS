"""Observation-to-intent boundary for offline raw PID evaluation.

This module has no network, serial, or live-authority interface. Its output
is an immediately expired shadow intent so replay cannot be mistaken for a
motor command. Kalman state remains separate; optional feedforward is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.schemas import ControlIntent, ControlIntentLimits, ControlObservation
from jetson.control.pid import BasicPID, PIDDecision, PIDInput
from jetson.control.timing import (
    ClockBounds,
    FrameTimingEvidence,
    TimingVerdict,
    verify_frame_timing,
)


@dataclass(frozen=True)
class ShadowPIDResult:
    intent: ControlIntent
    pid: PIDDecision
    timing: TimingVerdict


class ShadowPIDController:
    """Fail-closed, stateful adapter; never publishes or actuates."""

    def __init__(
        self,
        pid: BasicPID,
        *,
        max_clock_sample_age_ns: int,
        max_capture_age_ns: int,
        max_gimbal_age_ns: int,
        max_safety_age_ns: int,
        source_clock_domain: str = "pc_monotonic",
    ) -> None:
        if min(
            max_clock_sample_age_ns, max_capture_age_ns,
            max_gimbal_age_ns, max_safety_age_ns,
        ) <= 0:
            raise ValueError("timing limits must be positive")
        self._pid = pid
        self._source_clock_domain = source_clock_domain
        self._max_clock_sample_age_ns = max_clock_sample_age_ns
        self._max_capture_age_ns = max_capture_age_ns
        self._max_gimbal_age_ns = max_gimbal_age_ns
        self._max_safety_age_ns = max_safety_age_ns
        self._last_sequence: int | None = None
        self._last_source_frame_id: int | None = None

    @property
    def track_id(self) -> int | None:
        """Track the PID is steering to, if any."""
        return self._pid.track_id

    def reset(self) -> None:
        self._pid.reset()
        self._last_sequence = None
        self._last_source_frame_id = None

    def _timing(self, obs: ControlObservation, clock: ClockBounds | None) -> TimingVerdict:
        if obs.source_identity_verified is not True:
            return TimingVerdict(False, "frame_identity_unverified")
        if obs.source_clock_domain != self._source_clock_domain:
            return TimingVerdict(False, "source_clock_domain_invalid")
        if obs.frame_receive_clock_domain != "jetson_monotonic":
            return TimingVerdict(False, "receive_clock_domain_invalid")
        if obs.frame_observation_clock_domain != "jetson_monotonic":
            return TimingVerdict(False, "observation_clock_domain_invalid")
        if any(value is None for value in (
            obs.source_frame_id, obs.source_time_ns,
            obs.frame_received_time_ns, obs.frame_observed_time_ns,
        )):
            return TimingVerdict(False, "frame_timing_missing")
        assert obs.source_frame_id is not None
        assert obs.source_time_ns is not None
        assert obs.frame_received_time_ns is not None
        assert obs.frame_observed_time_ns is not None
        if self._last_source_frame_id is not None and obs.source_frame_id < self._last_source_frame_id:
            return TimingVerdict(False, "source_frame_regressed")
        return verify_frame_timing(
            FrameTimingEvidence(
                frame_id=obs.source_frame_id,
                identity_verified=True,
                pc_source_ns=obs.source_time_ns,
                jetson_received_ns=obs.frame_received_time_ns,
                jetson_observed_ns=obs.frame_observed_time_ns,
                jetson_decision_ns=obs.created_monotonic_ns,
            ),
            clock,
            max_clock_sample_age_ns=self._max_clock_sample_age_ns,
            max_capture_age_ns=self._max_capture_age_ns,
        )

    def decide(
        self, obs: ControlObservation, clock: ClockBounds | None,
        *, feedforward_rad_s: tuple[float, float] = (0.0, 0.0),
        error_override_rad: tuple[float, float] | None = None,
        coast_error_rad: tuple[float, float] | None = None,
    ) -> ShadowPIDResult:
        """``error_override_rad`` replaces the frame bearing (e.g. a latency-
        compensated prediction); all timing, target, gimbal, and safety gates
        still apply unchanged."""
        timing = self._timing(obs, clock)
        if self._last_sequence is not None and obs.sequence <= self._last_sequence:
            timing = TimingVerdict(False, "observation_sequence_nonmonotonic")
        self._last_sequence = max(obs.sequence, self._last_sequence or 0)
        if timing.valid and obs.source_frame_id is not None:
            self._last_source_frame_id = obs.source_frame_id

        target = obs.target
        gimbal = obs.gimbal
        safety = obs.safety
        coasting = False
        sequence_ok = timing.reason != "observation_sequence_nonmonotonic"
        track_id = target.track_id
        error = error_override_rad if error_override_rad is not None else target.bearing_error_rad
        if not target.valid or target.track_id is None or target.bearing_error_rad is None:
            timing = TimingVerdict(False, "target_invalid")
            if coast_error_rad is not None and self._pid.track_id is not None and sequence_ok:
                # No detection this tick: keep steering the same track on the
                # caller's predicted error, with the PID state (and its rate
                # continuity) intact. Safety and gimbal gates still apply.
                coasting = True
                timing = TimingVerdict(True, "coasting")
                track_id = self._pid.track_id
                error = coast_error_rad
        gimbal_valid = bool(
            gimbal.valid
            and gimbal.yaw_rad is not None and gimbal.pitch_rad is not None
            and (not self._pid.requires_gimbal_rate or (
                gimbal.yaw_rate_rad_s is not None
                and gimbal.pitch_rate_rad_s is not None
            ))
            and gimbal.sample_age_ms is not None
            and gimbal.sample_age_ms * 1_000_000 <= self._max_gimbal_age_ns
        )
        safety_allowed = bool(
            safety.valid and safety.auto_allowed
            and not safety.manual_active and not safety.emergency_active
            and safety.sample_age_ms is not None
            and safety.sample_age_ms * 1_000_000 <= self._max_safety_age_ns
        )
        pid = self._pid.decide(PIDInput(
            decision_ns=obs.created_monotonic_ns,
            track_id=track_id if track_id is not None else -1,
            error_rad=error or (0.0, 0.0),
            gimbal_rate_rad_s=(
                gimbal.yaw_rate_rad_s or 0.0,
                gimbal.pitch_rate_rad_s or 0.0,
            ),
            timing=timing,
            safety_allowed=safety_allowed,
            gimbal_valid=gimbal_valid,
            feedforward_rad_s=feedforward_rad_s,
        ))
        intent = ControlIntent(
            sequence=obs.sequence,
            observation_sequence=obs.sequence,
            issued_monotonic_ns=obs.created_monotonic_ns,
            valid_until_monotonic_ns=obs.created_monotonic_ns,
            mode="shadow",
            yaw_rate_rad_s=pid.yaw.final_rad_s,
            pitch_rate_rad_s=pid.pitch.final_rad_s,
            limits=ControlIntentLimits(
                yaw_rate_limited=pid.yaw.rate_limited,
                pitch_rate_limited=pid.pitch.rate_limited,
                acceleration_limited=(
                    pid.yaw.acceleration_limited or pid.pitch.acceleration_limited
                ),
            ),
            reason="coasting" if coasting and pid.reason == "tracking" else pid.reason,
        )
        return ShadowPIDResult(intent, pid, timing)
