"""Deterministic, non-actuating rate policy for controller-overhaul replay.

This module deliberately has no ZMQ, serial, gimbal, or wall-clock dependency.
It consumes one validated :class:`ControlObservation` and emits a short-lived
``shadow`` :class:`ControlIntent`.  It is suitable for replay and parity work,
not for hardware command authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque
from typing import Literal, Optional, Tuple

from common.schemas import (
    ControlDiagnostics,
    ControlEstimatorAxisDiagnostics,
    ControlIntent,
    ControlIntentLimits,
    ControlObservation,
    CamState,
    ControlTimingDiagnostics,
)
from common.clock_sync import ClockOffsetSample
from jetson.los_kalman import AxisLOSKalman, LOSEstimate, LOSKalmanConfig


_LOCAL_MONOTONIC_CLOCK_DOMAINS = {"jetson_monotonic", "jetson.monotonic"}


@dataclass(frozen=True)
class ShadowRatePolicyConfig:
    """Explicit, local-monotonic bounds for :class:`ShadowRatePolicy`."""

    yaw_kp: float
    pitch_kp: float
    yaw_kd: float = 0.0
    pitch_kd: float = 0.0
    yaw_rate_limit_rad_s: float = 1.0
    pitch_rate_limit_rad_s: float = 1.0
    yaw_accel_limit_rad_s2: float = 2.0
    pitch_accel_limit_rad_s2: float = 2.0
    nominal_period_s: float = 0.02
    valid_for_ns: int = 50_000_000
    yaw_position_limits_rad: Optional[Tuple[float, float]] = None
    pitch_position_limits_rad: Optional[Tuple[float, float]] = None
    yaw_los_kalman: Optional[LOSKalmanConfig] = None
    pitch_los_kalman: Optional[LOSKalmanConfig] = None
    yaw_feedforward_gain: float = 0.0
    pitch_feedforward_gain: float = 0.0
    raw_gimbal_damping: bool = False
    intent_mode: Literal["shadow", "live"] = "shadow"
    sequence_base: int = 0
    source_clock_mapping_enabled: bool = False

    def __post_init__(self) -> None:
        finite_positive = (
            self.yaw_rate_limit_rad_s,
            self.pitch_rate_limit_rad_s,
            self.yaw_accel_limit_rad_s2,
            self.pitch_accel_limit_rad_s2,
            self.nominal_period_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("rate, acceleration, and period limits must be finite and > 0")
        if any(not math.isfinite(value) for value in (
            self.yaw_kp, self.pitch_kp, self.yaw_kd, self.pitch_kd,
            self.yaw_feedforward_gain, self.pitch_feedforward_gain,
        )):
            raise ValueError("gains must be finite")
        if (self.yaw_los_kalman is None) != (self.pitch_los_kalman is None):
            raise ValueError("yaw and pitch LOS Kalman configs must be enabled together")
        if self.valid_for_ns <= 0:
            raise ValueError("valid_for_ns must be > 0")
        if self.sequence_base < 0:
            raise ValueError("sequence_base must be non-negative")
        for bounds in (self.yaw_position_limits_rad, self.pitch_position_limits_rad):
            if bounds is not None and (not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[1]):
                raise ValueError("position limits must be finite (min, max) pairs")


class ShadowRatePolicy:
    """Bounded PD plus bearing-rate-feedforward policy with fail-safe holds.

    Positive yaw/pitch command is defined to increase the matching gimbal angle.
    ``bearing_rate_rad_s`` is the selected target's camera-relative bearing
    rate, so it is added directly as the derivative/feedforward term.  This
    convention is explicit for replay; sign/plant validation is still required
    before a live policy is designed.
    """

    def __init__(self, config: ShadowRatePolicyConfig) -> None:
        self._config = config
        self._intent_sequence = int(config.sequence_base)
        self._last_observation_sequence = -1
        self._last_issued_ns: Optional[int] = None
        self._last_rates = (0.0, 0.0)
        self._last_track_id: Optional[int] = None
        self._yaw_los = AxisLOSKalman(config.yaw_los_kalman) if config.yaw_los_kalman is not None else None
        self._pitch_los = AxisLOSKalman(config.pitch_los_kalman) if config.pitch_los_kalman is not None else None
        self._last_los_source: Optional[tuple[Optional[int], int]] = None
        self._last_estimates: tuple[Optional[LOSEstimate], Optional[LOSEstimate]] = (None, None)
        self._last_measurement_updated = False
        self._last_measurement_accepted: tuple[Optional[bool], Optional[bool]] = (None, None)
        self._last_diagnostics: Optional[ControlDiagnostics] = None
        self._source_clock_sample: Optional[ClockOffsetSample] = None
        self._last_estimator_time_source = "snapshot_receipt"
        self._last_source_frame_age_ms: Optional[float] = None
        self._last_frame_gimbal_pose_age_ms: Optional[float] = None
        self._camera_history: deque[tuple[int, float, float]] = deque(maxlen=100)

    def set_source_clock_sample(self, sample: Optional[ClockOffsetSample]) -> None:
        self._source_clock_sample = sample

    def record_cam_state(self, state: CamState, *, received_at_ns: int) -> None:
        sample_ns = state.state_monotonic_ns or received_at_ns
        if self._camera_history and sample_ns <= self._camera_history[-1][0]:
            return
        self._camera_history.append((
            sample_ns,
            float(state.pan),
            float(state.tilt),
        ))

    def _camera_pose_at(self, sample_ns: int) -> Optional[tuple[float, float, float]]:
        """Interpolate local camera poses, bounded to nearby recorded samples."""
        history = self._camera_history
        if not history:
            return None
        previous = None
        for current in history:
            if current[0] >= sample_ns:
                if previous is None:
                    age_ns = current[0] - sample_ns
                    return (current[1], current[2], age_ns / 1e6) if age_ns <= 30_000_000 else None
                span_ns = current[0] - previous[0]
                fraction = (sample_ns - previous[0]) / span_ns
                yaw_delta = math.atan2(
                    math.sin(current[1] - previous[1]),
                    math.cos(current[1] - previous[1]),
                )
                return (
                    previous[1] + yaw_delta * fraction,
                    previous[2] + (current[2] - previous[2]) * fraction,
                    min(sample_ns - previous[0], current[0] - sample_ns) / 1e6,
                )
            previous = current
        age_ns = sample_ns - history[-1][0]
        return (history[-1][1], history[-1][2], age_ns / 1e6) if age_ns <= 30_000_000 else None

    @property
    def last_diagnostics(self) -> Optional[ControlDiagnostics]:
        """Return non-authoritative evidence for the most recent decision."""

        return self._last_diagnostics

    def _intent(self, observation: ControlObservation, *, yaw: float, pitch: float,
                reason: str, limits: Optional[ControlIntentLimits] = None) -> ControlIntent:
        self._intent_sequence += 1
        issued = observation.created_monotonic_ns
        return ControlIntent(
            sequence=self._intent_sequence,
            observation_sequence=observation.sequence,
            issued_monotonic_ns=issued,
            valid_until_monotonic_ns=issued + self._config.valid_for_ns,
            mode=self._config.intent_mode,
            yaw_rate_rad_s=yaw,
            pitch_rate_rad_s=pitch,
            limits=limits or ControlIntentLimits(),
            reason=reason,
        )

    def _reset(self) -> None:
        self._last_issued_ns = None
        self._last_rates = (0.0, 0.0)
        self._last_track_id = None
        self._last_los_source = None
        self._last_estimates = (None, None)
        self._last_measurement_updated = False
        self._last_measurement_accepted = (None, None)
        self._last_estimator_time_source = "snapshot_receipt"
        self._last_source_frame_age_ms = None
        self._last_frame_gimbal_pose_age_ms = None
        if self._yaw_los is not None:
            self._yaw_los.reset()
        if self._pitch_los is not None:
            self._pitch_los.reset()

    @staticmethod
    def _nonnegative_delta_ms(later_ns: int, earlier_ns: Optional[int]) -> Optional[float]:
        if earlier_ns is None or later_ns < earlier_ns:
            return None
        return (later_ns - earlier_ns) / 1_000_000.0

    def _timing_diagnostics(self, observation: ControlObservation) -> ControlTimingDiagnostics:
        receive_local = observation.frame_receive_clock_domain in _LOCAL_MONOTONIC_CLOCK_DOMAINS
        observe_local = observation.frame_observation_clock_domain in _LOCAL_MONOTONIC_CLOCK_DOMAINS
        same_frame_clock = (
            observation.frame_receive_clock_domain is not None
            and observation.frame_receive_clock_domain == observation.frame_observation_clock_domain
        )
        return ControlTimingDiagnostics(
            snapshot_receipt_age_ms=observation.target.source_age_ms,
            frame_receive_to_tick_ms=(
                self._nonnegative_delta_ms(
                    observation.created_monotonic_ns, observation.frame_received_time_ns
                )
                if receive_local else None
            ),
            frame_observe_to_tick_ms=(
                self._nonnegative_delta_ms(
                    observation.created_monotonic_ns, observation.frame_observed_time_ns
                )
                if observe_local else None
            ),
            frame_receive_to_observe_ms=(
                self._nonnegative_delta_ms(
                    observation.frame_observed_time_ns or 0,
                    observation.frame_received_time_ns,
                )
                if same_frame_clock and observation.frame_observed_time_ns is not None else None
            ),
            gimbal_sample_age_ms=observation.gimbal.sample_age_ms,
            source_clock_domain=observation.source_clock_domain,
            frame_receive_clock_domain=observation.frame_receive_clock_domain,
            frame_observation_clock_domain=observation.frame_observation_clock_domain,
            source_to_local_mapping_available=(
                observation.source_clock_domain in _LOCAL_MONOTONIC_CLOCK_DOMAINS
                or self._last_estimator_time_source == "mapped_pc_source"
            ),
            estimator_time_source=self._last_estimator_time_source,
            source_frame_age_ms=self._last_source_frame_age_ms,
            source_clock_uncertainty_ms=(
                None if self._source_clock_sample is None else
                self._source_clock_sample.uncertainty_ns / 1_000_000.0
            ),
            frame_gimbal_pose_age_ms=self._last_frame_gimbal_pose_age_ms,
        )

    def _axis_diagnostics(
        self,
        *,
        axis: int,
        raw_error: Optional[float],
        estimated_error: Optional[float],
        feedback: Optional[float],
        damping: Optional[float],
        feedforward: Optional[float],
        pre_limit: Optional[float],
        post_limit: Optional[float],
        final_rate: float,
    ) -> ControlEstimatorAxisDiagnostics:
        estimate = self._last_estimates[axis]
        return ControlEstimatorAxisDiagnostics(
            estimator_enabled=self._yaw_los is not None,
            measurement_updated=self._last_measurement_updated,
            measurement_accepted=self._last_measurement_accepted[axis],
            measurement_reinitialized=(
                False if estimate is None else estimate.last_update_reinitialized
            ),
            raw_error_rad=raw_error,
            estimated_error_rad=estimated_error,
            estimated_target_angle_rad=None if estimate is None else estimate.angle_rad,
            estimated_target_rate_rad_s=None if estimate is None else estimate.rate_rad_s,
            estimate_sample_time_s=None if estimate is None else estimate.sample_time_s,
            estimate_query_time_s=None if estimate is None else estimate.query_time_s,
            prediction_horizon_ms=(
                None if estimate is None
                else max(0.0, (estimate.query_time_s - estimate.sample_time_s) * 1000.0)
            ),
            angle_variance_rad2=None if estimate is None else estimate.angle_variance_rad2,
            rate_variance_rad2_s2=None if estimate is None else estimate.rate_variance_rad2_s2,
            angle_rate_covariance_rad2_s=(
                None if estimate is None else estimate.angle_rate_covariance_rad2_s
            ),
            innovation_rad=None if estimate is None else estimate.innovation_rad,
            innovation_variance_rad2=(
                None if estimate is None else estimate.innovation_variance_rad2
            ),
            normalized_innovation_squared=(
                None if estimate is None else estimate.normalized_innovation_squared
            ),
            accepted_updates=0 if estimate is None else estimate.accepted_updates,
            rejected_updates=0 if estimate is None else estimate.rejected_updates,
            reinitialized_updates=0 if estimate is None else estimate.reinitialized_updates,
            consecutive_rejections=0 if estimate is None else estimate.consecutive_rejections,
            feedback_term_rad_s=feedback,
            damping_term_rad_s=damping,
            feedforward_term_rad_s=feedforward,
            desired_rate_pre_limit_rad_s=pre_limit,
            desired_rate_post_limit_rad_s=post_limit,
            final_rate_rad_s=final_rate,
        )

    def _record_diagnostics(
        self,
        observation: ControlObservation,
        intent: ControlIntent,
        *,
        raw_error: Optional[tuple[float, float]] = None,
        estimated_error: Optional[tuple[float, float]] = None,
        feedback: tuple[Optional[float], Optional[float]] = (None, None),
        damping: tuple[Optional[float], Optional[float]] = (None, None),
        feedforward: tuple[Optional[float], Optional[float]] = (None, None),
        pre_limit: tuple[Optional[float], Optional[float]] = (None, None),
        post_limit: tuple[Optional[float], Optional[float]] = (None, None),
    ) -> None:
        self._last_diagnostics = ControlDiagnostics(
            observation_sequence=observation.sequence,
            intent_sequence=intent.sequence,
            created_monotonic_ns=observation.created_monotonic_ns,
            reason=intent.reason,
            track_id=observation.target.track_id,
            timing=self._timing_diagnostics(observation),
            yaw=self._axis_diagnostics(
                axis=0,
                raw_error=None if raw_error is None else raw_error[0],
                estimated_error=None if estimated_error is None else estimated_error[0],
                feedback=feedback[0], damping=damping[0], feedforward=feedforward[0],
                pre_limit=pre_limit[0], post_limit=post_limit[0],
                final_rate=intent.yaw_rate_rad_s,
            ),
            pitch=self._axis_diagnostics(
                axis=1,
                raw_error=None if raw_error is None else raw_error[1],
                estimated_error=None if estimated_error is None else estimated_error[1],
                feedback=feedback[1], damping=damping[1], feedforward=feedforward[1],
                pre_limit=pre_limit[1], post_limit=post_limit[1],
                final_rate=intent.pitch_rate_rad_s,
            ),
        )

    def _estimated_target_terms(
        self,
        observation: ControlObservation,
        raw_error: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return current bearing error and absolute LOS rate for opt-in KF mode."""

        self._last_estimates = (None, None)
        self._last_measurement_updated = False
        self._last_measurement_accepted = (None, None)
        if self._yaw_los is None or self._pitch_los is None:
            return raw_error, observation.target.bearing_rate_rad_s or (0.0, 0.0)
        now_s = observation.created_monotonic_ns / 1_000_000_000.0
        target_age_s = (observation.target.source_age_ms or 0.0) / 1000.0
        target_sample_s = now_s - target_age_s
        self._last_estimator_time_source = "snapshot_receipt"
        self._last_source_frame_age_ms = None
        self._last_frame_gimbal_pose_age_ms = None
        if self._config.source_clock_mapping_enabled:
            sample = self._source_clock_sample
            if (
                sample is None
                or observation.source_clock_domain not in {"pc_monotonic", "pc.monotonic"}
                or observation.source_time_ns is None
                or not 0 <= observation.created_monotonic_ns - sample.observed_jetson_ns <= 5_000_000_000
                or sample.uncertainty_ns > 5_000_000
            ):
                # Continue bounded position feedback, but do not derive velocity
                # from a timestamp whose clock relationship is unknown.
                self._yaw_los.reset()
                self._pitch_los.reset()
                self._last_estimator_time_source = "unavailable"
                return raw_error, (0.0, 0.0)
            mapped_ns = sample.map_pc_ns(observation.source_time_ns)
            frame_age_ns = observation.created_monotonic_ns - mapped_ns
            if not 0 <= frame_age_ns <= 250_000_000:
                self._yaw_los.reset()
                self._pitch_los.reset()
                self._last_estimator_time_source = "invalid_mapped_age"
                return raw_error, (0.0, 0.0)
            target_sample_s = mapped_ns / 1_000_000_000.0
            self._last_source_frame_age_ms = frame_age_ns / 1_000_000.0
            self._last_estimator_time_source = "mapped_pc_source"
        source_key = (
            observation.source_frame_id,
            observation.source_time_ns
            if self._config.source_clock_mapping_enabled
            else int(round(target_sample_s * 1_000_000_000.0)),
        )
        yaw_position = observation.gimbal.yaw_rad or 0.0
        pitch_position = observation.gimbal.pitch_rad or 0.0
        yaw_rate = observation.gimbal.yaw_rate_rad_s or 0.0
        pitch_rate = observation.gimbal.pitch_rate_rad_s or 0.0
        gimbal_age_s = (observation.gimbal.sample_age_ms or 0.0) / 1000.0
        gimbal_sample_s = now_s - gimbal_age_s
        if source_key != self._last_los_source:
            self._last_measurement_updated = True
            if self._config.source_clock_mapping_enabled:
                camera_pose = self._camera_pose_at(int(target_sample_s * 1_000_000_000))
                if camera_pose is None:
                    self._yaw_los.reset()
                    self._pitch_los.reset()
                    self._last_estimator_time_source = "camera_pose_unavailable"
                    return raw_error, (0.0, 0.0)
                yaw_at_target, pitch_at_target, self._last_frame_gimbal_pose_age_ms = camera_pose
            else:
                yaw_at_target = yaw_position + yaw_rate * (target_sample_s - gimbal_sample_s)
                pitch_at_target = pitch_position + pitch_rate * (target_sample_s - gimbal_sample_s)
            try:
                yaw_accepted = self._yaw_los.update(
                    yaw_at_target + raw_error[0], sample_time_s=target_sample_s
                )
                pitch_accepted = self._pitch_los.update(
                    pitch_at_target + raw_error[1], sample_time_s=target_sample_s
                )
            except ValueError:
                # A regressed local sample timestamp cannot be fused.  Reset
                # both axes atomically and treat this sample as reacquisition.
                self._yaw_los.reset()
                self._pitch_los.reset()
                yaw_accepted = self._yaw_los.update(
                    yaw_at_target + raw_error[0], sample_time_s=target_sample_s
                )
                pitch_accepted = self._pitch_los.update(
                    pitch_at_target + raw_error[1], sample_time_s=target_sample_s
                )
            self._last_measurement_accepted = (yaw_accepted, pitch_accepted)
            self._last_los_source = source_key
        yaw_estimate = self._yaw_los.estimate(query_time_s=now_s)
        pitch_estimate = self._pitch_los.estimate(query_time_s=now_s)
        self._last_estimates = (yaw_estimate, pitch_estimate)
        if yaw_estimate is None or pitch_estimate is None:
            return raw_error, (0.0, 0.0)
        return (
            (yaw_estimate.angle_rad - yaw_position, pitch_estimate.angle_rad - pitch_position),
            (yaw_estimate.rate_rad_s, pitch_estimate.rate_rad_s),
        )

    def _hold(self, observation: ControlObservation, reason: str) -> ControlIntent:
        self._reset()
        intent = self._intent(observation, yaw=0.0, pitch=0.0, reason=reason)
        self._record_diagnostics(observation, intent)
        return intent

    @staticmethod
    def _clamp(value: float, limit: float) -> tuple[float, bool]:
        clipped = max(-limit, min(limit, value))
        return clipped, clipped != value

    @staticmethod
    def _outward(position: float, command: float, limits: Optional[Tuple[float, float]]) -> bool:
        return limits is not None and ((position <= limits[0] and command < 0.0) or (position >= limits[1] and command > 0.0))

    def _acceleration_limit(self, target: float, previous: float, limit: float, now_ns: int) -> tuple[float, bool]:
        if self._last_issued_ns is None:
            elapsed_s = self._config.nominal_period_s
        else:
            elapsed_s = max(self._config.nominal_period_s, (now_ns - self._last_issued_ns) / 1_000_000_000.0)
        maximum_delta = limit * elapsed_s
        bounded = max(previous - maximum_delta, min(previous + maximum_delta, target))
        return bounded, bounded != target

    def decide(self, observation: ControlObservation) -> ControlIntent:
        """Return one bounded shadow intent for ``observation``.

        Invalid, stale, manual, emergency, identity-switch, or out-of-order
        inputs yield an immediate zero-rate hold and reset limiter memory.
        """

        if observation.sequence <= self._last_observation_sequence:
            return self._hold(observation, "observation_out_of_order")
        self._last_observation_sequence = observation.sequence

        safety = observation.safety
        if safety.emergency_active:
            return self._hold(observation, "emergency_active")
        if safety.manual_active:
            return self._hold(observation, "manual_active")
        if not safety.valid:
            return self._hold(observation, "safety_invalid")
        if not safety.auto_allowed:
            return self._hold(observation, "auto_disallowed")
        if not observation.target.valid:
            return self._hold(observation, "target_invalid")
        if not observation.gimbal.valid:
            return self._hold(observation, "gimbal_invalid")
        if observation.target.bearing_error_rad is None:
            return self._hold(observation, "target_incomplete")

        track_id = observation.target.track_id
        if self._last_track_id is not None and track_id is not None and track_id != self._last_track_id:
            return self._hold(observation, "target_switch_hold")

        raw_error = observation.target.bearing_error_rad
        (yaw_error, pitch_error), (yaw_rate, pitch_rate) = self._estimated_target_terms(observation, raw_error)
        if self._yaw_los is None:
            # Compatibility path: historical ``kd`` multiplies the externally
            # supplied camera-relative rate.
            if self._config.raw_gimbal_damping:
                desired_yaw = (
                    self._config.yaw_kp * yaw_error
                    - self._config.yaw_kd * (observation.gimbal.yaw_rate_rad_s or 0.0)
                )
                desired_pitch = (
                    self._config.pitch_kp * pitch_error
                    - self._config.pitch_kd * (observation.gimbal.pitch_rate_rad_s or 0.0)
                )
                damping_terms = (
                    -self._config.yaw_kd * (observation.gimbal.yaw_rate_rad_s or 0.0),
                    -self._config.pitch_kd * (observation.gimbal.pitch_rate_rad_s or 0.0),
                )
            else:
                desired_yaw = self._config.yaw_kp * yaw_error + self._config.yaw_kd * yaw_rate
                desired_pitch = self._config.pitch_kp * pitch_error + self._config.pitch_kd * pitch_rate
                damping_terms = (
                    self._config.yaw_kd * yaw_rate,
                    self._config.pitch_kd * pitch_rate,
                )
            feedback_terms = (
                self._config.yaw_kp * yaw_error,
                self._config.pitch_kp * pitch_error,
            )
            feedforward_terms = (0.0, 0.0)
        else:
            desired_yaw = (
                self._config.yaw_kp * yaw_error
                - self._config.yaw_kd * (observation.gimbal.yaw_rate_rad_s or 0.0)
                + self._config.yaw_feedforward_gain * yaw_rate
            )
            desired_pitch = (
                self._config.pitch_kp * pitch_error
                - self._config.pitch_kd * (observation.gimbal.pitch_rate_rad_s or 0.0)
                + self._config.pitch_feedforward_gain * pitch_rate
            )
            feedback_terms = (
                self._config.yaw_kp * yaw_error,
                self._config.pitch_kp * pitch_error,
            )
            damping_terms = (
                -self._config.yaw_kd * (observation.gimbal.yaw_rate_rad_s or 0.0),
                -self._config.pitch_kd * (observation.gimbal.pitch_rate_rad_s or 0.0),
            )
            feedforward_terms = (
                self._config.yaw_feedforward_gain * yaw_rate,
                self._config.pitch_feedforward_gain * pitch_rate,
            )
        pre_limit_rates = (desired_yaw, desired_pitch)
        desired_yaw, yaw_limited = self._clamp(desired_yaw, self._config.yaw_rate_limit_rad_s)
        desired_pitch, pitch_limited = self._clamp(desired_pitch, self._config.pitch_rate_limit_rad_s)
        post_rate_limit_rates = (desired_yaw, desired_pitch)

        yaw_position_limited = self._outward(
            observation.gimbal.yaw_rad or 0.0, desired_yaw, self._config.yaw_position_limits_rad
        )
        if yaw_position_limited:
            desired_yaw = 0.0
            position_limited = True
        else:
            position_limited = False
        pitch_position_limited = self._outward(
            observation.gimbal.pitch_rad or 0.0, desired_pitch, self._config.pitch_position_limits_rad
        )
        if pitch_position_limited:
            desired_pitch = 0.0
            position_limited = True

        # A blocked axis must stop immediately, but the other axis still has
        # to honor its own acceleration bound.
        if yaw_position_limited:
            yaw, yaw_accel_limited = 0.0, False
        else:
            yaw, yaw_accel_limited = self._acceleration_limit(
                desired_yaw, self._last_rates[0], self._config.yaw_accel_limit_rad_s2, observation.created_monotonic_ns
            )
        if pitch_position_limited:
            pitch, pitch_accel_limited = 0.0, False
        else:
            pitch, pitch_accel_limited = self._acceleration_limit(
                desired_pitch, self._last_rates[1], self._config.pitch_accel_limit_rad_s2, observation.created_monotonic_ns
            )
        self._last_rates = (yaw, pitch)
        self._last_issued_ns = observation.created_monotonic_ns
        self._last_track_id = track_id
        intent = self._intent(
            observation, yaw=yaw, pitch=pitch, reason="position_limit_hold" if position_limited else "tracking",
            limits=ControlIntentLimits(yaw_rate_limited=yaw_limited, pitch_rate_limited=pitch_limited,
                                       acceleration_limited=yaw_accel_limited or pitch_accel_limited,
                                       position_limited=position_limited),
        )
        self._record_diagnostics(
            observation,
            intent,
            raw_error=raw_error,
            estimated_error=(yaw_error, pitch_error),
            feedback=feedback_terms,
            damping=damping_terms,
            feedforward=feedforward_terms,
            pre_limit=pre_limit_rates,
            post_limit=post_rate_limit_rates,
        )
        return intent
