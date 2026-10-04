"""Pure video observation -> PID plus separate target-rate feedforward.

This module never opens sockets or motors. A live intent is only a candidate
for a separately authorized runtime; the default output is expired shadow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from common.schemas import CamState, ControlIntent, ControlObservation
from jetson.control.pid import BasicPID
from jetson.control.shadow_pid import ShadowPIDController, ShadowPIDResult
from jetson.control.timing import ClockBounds
from jetson.control.video_feedforward import (
    VideoFeedforwardEstimate,
    VideoTargetRateEstimator,
)


@dataclass(frozen=True)
class VideoControllerPolicy:
    feedforward_scale: float = 0.0
    # Latency compensation: 0 uses the raw frame bearing (previous behaviour);
    # 1 uses predicted target minus camera angle at decision time.
    predict: float = 0.0
    feedforward_accel_sigma_rad_s2: float = 0.4
    live_authorized: bool = False
    live_intent_ttl_ns: int = 50_000_000
    max_capture_age_ns: int = 150_000_000
    max_travel_rad: float = 0.15
    source_clock_domain: str = "pc_monotonic"
    # With no target for this long, slew back to the origin (None: hold).
    idle_return_after_ns: int | None = None
    idle_return_rate_rad_s: float = 0.3
    idle_return_gain_per_s: float = 2.0
    idle_return_deadband_rad: float = 0.01
    # Keep steering on the predicted target for this long without a
    # detection (None: stop at once).
    coast_ns: int | None = None
    # Manual slew: the panel joystick's rate demand, clamped and slew-limited,
    # while the panel is in manual with the master arm on and no E-stop.
    manual_rate_limit_rad_s: float = 0.5
    manual_accel_limit_rad_s2: float = 2.0
    max_safety_age_ns: int = 750_000_000

    def __post_init__(self) -> None:
        if not math.isfinite(self.feedforward_scale) or not 0 <= self.feedforward_scale <= 1:
            raise ValueError("feedforward scale must be in [0, 1]")
        if not math.isfinite(self.predict) or not 0 <= self.predict <= 1:
            raise ValueError("predict must be in [0, 1]")
        if not math.isfinite(self.feedforward_accel_sigma_rad_s2) or self.feedforward_accel_sigma_rad_s2 <= 0:
            raise ValueError("feedforward acceleration sigma must be positive")
        if not 0 < self.live_intent_ttl_ns <= 50_000_000:
            raise ValueError("live intent TTL must be in (0, 50 ms]")
        if not 0 < self.max_capture_age_ns <= 250_000_000:
            raise ValueError("capture age gate must be in (0, 250 ms]")
        if not math.isfinite(self.max_travel_rad) or not 0 < self.max_travel_rad <= math.pi:
            raise ValueError("travel limit must be in (0, pi] rad")
        if self.coast_ns is not None and not 0 < self.coast_ns <= 2_000_000_000:
            raise ValueError("coast time must be in (0, 2 s]")
        if self.idle_return_after_ns is not None and self.idle_return_after_ns <= 0:
            raise ValueError("idle return delay must be positive")
        for name in ("idle_return_rate_rad_s", "idle_return_gain_per_s", "idle_return_deadband_rad",
                     "manual_rate_limit_rad_s", "manual_accel_limit_rad_s2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class VideoControllerDecision:
    intent: ControlIntent
    pid: ShadowPIDResult
    feedforward: VideoFeedforwardEstimate
    applied_feedforward_rad_s: tuple[float, float]
    pid_error_source: str = "frame_bearing"
    # (yaw, pitch) axes stopped at the travel envelope this tick.
    travel_held: tuple[bool, bool] = (False, False)


class VideoControllerCore:
    """One causal controller state; no transport and no clock-bound inference."""

    def __init__(self, pid: BasicPID, policy: VideoControllerPolicy) -> None:
        self.policy = policy
        self.feedforward = VideoTargetRateEstimator(
            max_sample_age_s=policy.max_capture_age_ns / 1e9,
            accel_sigma_rad_s2=policy.feedforward_accel_sigma_rad_s2,
            source_clock_domain=policy.source_clock_domain,
        )
        self.pid = ShadowPIDController(
            pid,
            max_clock_sample_age_ns=150_000_000,
            max_capture_age_ns=policy.max_capture_age_ns,
            max_gimbal_age_ns=100_000_000,
            max_safety_age_ns=policy.max_safety_age_ns,
            source_clock_domain=policy.source_clock_domain,
        )
        self._origin_rad: tuple[float, float] | None = None
        self._last_target_ns: int | None = None
        self._manual_rates: tuple[float, float] = (0.0, 0.0)
        self._manual_ns: int | None = None

    def reset(self) -> None:
        self.feedforward.reset()
        self.pid.reset()
        self._origin_rad = None
        self._last_target_ns = None
        self._manual_rates = (0.0, 0.0)
        self._manual_ns = None

    def manual_allowed(self, observation: ControlObservation) -> bool:
        safety = observation.safety
        return bool(
            safety.valid and safety.manual_active and safety.master_arm
            and not safety.emergency_active
            and safety.sample_age_ms is not None
            and safety.sample_age_ms * 1_000_000 <= self.policy.max_safety_age_ns
        )

    def _manual_slew(self, observation: ControlObservation) -> tuple[float, float]:
        """Joystick demand clamped to the manual rate limit, then slew-limited."""
        now_ns = observation.created_monotonic_ns
        dt = 0.0 if self._manual_ns is None else max(0.0, (now_ns - self._manual_ns) / 1e9)
        self._manual_ns = now_ns
        limit = self.policy.manual_rate_limit_rad_s
        step = self.policy.manual_accel_limit_rad_s2 * min(dt, 0.1)
        rates = []
        for demand, previous in zip(observation.safety.manual_rate_rad_s, self._manual_rates):
            demand = max(-limit, min(limit, float(demand)))
            rates.append(max(previous - step, min(previous + step, demand)))
        self._manual_rates = (rates[0], rates[1])
        return self._manual_rates

    def _travel_held(
        self, gimbal, rates: tuple[float, float],
    ) -> tuple[bool, bool]:
        """Axes whose projection leaves the envelope *and* moves farther out.

        Motion back toward the envelope stays allowed, so an overshoot cannot
        latch the controller. Per axis: yaw at its limit must not freeze pitch
        tracking (it did in HIL, on the fast target's climb at the far yaw end).
        """
        if self._origin_rad is None or gimbal.yaw_rad is None or gimbal.pitch_rad is None:
            return (False, False)
        ttl_s = self.policy.live_intent_ttl_ns / 1e9
        return tuple(
            abs(position + rate * ttl_s - origin) > self.policy.max_travel_rad
            and abs(position + rate * ttl_s - origin) > abs(position - origin)
            for position, rate, origin in zip(
                (gimbal.yaw_rad, gimbal.pitch_rad), rates, self._origin_rad,
            )
        )

    def _idle_return_rates(
        self, observation: ControlObservation, reason: str,
    ) -> tuple[float, float] | None:
        """Rates back to the origin once no target has been seen for the delay.

        Only on ``target_invalid``: the PID reports it after the safety and
        gimbal gates passed, so the mount may move exactly as for tracking.
        """
        now_ns = observation.created_monotonic_ns
        if reason != "target_invalid":
            self._last_target_ns = now_ns
            return None
        delay_ns = self.policy.idle_return_after_ns
        gimbal = observation.gimbal
        if (delay_ns is None or self._origin_rad is None
                or gimbal.yaw_rad is None or gimbal.pitch_rad is None):
            return None
        if self._last_target_ns is None:
            self._last_target_ns = now_ns
        if now_ns - self._last_target_ns < delay_ns:
            return None
        rates = []
        for position, origin in zip((gimbal.yaw_rad, gimbal.pitch_rad), self._origin_rad):
            error = origin - position
            rate = 0.0
            if abs(error) > self.policy.idle_return_deadband_rad:
                limit = self.policy.idle_return_rate_rad_s
                rate = max(-limit, min(limit, self.policy.idle_return_gain_per_s * error))
            rates.append(rate)
        return rates[0], rates[1]

    def observe_cam_state(self, state: CamState) -> bool:
        return self.feedforward.observe_cam_state(state)

    def decide(
        self, observation: ControlObservation, clock: ClockBounds | None,
        *, feedforward_scale: float | None = None,
    ) -> VideoControllerDecision:
        scale = self.policy.feedforward_scale if feedforward_scale is None else feedforward_scale
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError("feedforward scale must be in [0, 1]")
        estimate = self.feedforward.estimate(
            observation, clock, predict=self.policy.predict,
        )
        applied = (
            (scale * estimate.yaw_rate_rad_s,
             scale * estimate.pitch_rate_rad_s)
            if estimate.valid else (0.0, 0.0)
        )
        predicted = estimate.predicted_bearing_error_rad if estimate.valid else None
        coast_error = None
        if self.policy.coast_ns is not None and not observation.target.valid:
            # The prediction is from the last capture, which is up to one
            # capture-age older than the last decision on it.
            coast = self.feedforward.coast(
                observation, track_id=self.pid.track_id,
                max_age_s=(self.policy.coast_ns + self.policy.max_capture_age_ns) / 1e9,
            )
            if coast is not None:
                coast_error, target_rate = coast
                applied = (scale * target_rate[0], scale * target_rate[1])
        result = self.pid.decide(
            observation, clock, feedforward_rad_s=applied,
            error_override_rad=predicted, coast_error_rad=coast_error,
        )
        if result.intent.reason not in ("tracking", "coasting"):
            applied = (0.0, 0.0)
        gimbal = observation.gimbal
        if self._origin_rad is None and gimbal.valid and gimbal.yaw_rad is not None and gimbal.pitch_rad is not None:
            self._origin_rad = (gimbal.yaw_rad, gimbal.pitch_rad)
        held = (False, False)
        manual = self.manual_allowed(observation) and gimbal.valid
        if manual:
            # The PID is held by the same safety gate (manual active); the
            # joystick drives the mount inside the same travel envelope.
            rates = self._manual_slew(observation)
            held = self._travel_held(gimbal, rates)
        else:
            self._manual_rates, self._manual_ns = (0.0, 0.0), None
            if result.intent.reason in ("tracking", "coasting"):
                held = self._travel_held(
                    gimbal, (result.intent.yaw_rate_rad_s, result.intent.pitch_rate_rad_s))
        travel_hold = any(held)
        idle_rates = None if manual else self._idle_return_rates(observation, result.intent.reason)
        if manual:
            applied = (0.0, 0.0)
            guarded = ControlIntent.model_validate({
                **result.intent.model_dump(),
                "yaw_rate_rad_s": 0.0 if held[0] else rates[0],
                "pitch_rate_rad_s": 0.0 if held[1] else rates[1],
                "reason": "manual",
            })
        elif idle_rates is not None:
            guarded = ControlIntent.model_validate({
                **result.intent.model_dump(),
                "yaw_rate_rad_s": idle_rates[0], "pitch_rate_rad_s": idle_rates[1],
                "reason": "idle_return",
            })
        elif travel_hold:
            applied = tuple(0.0 if axis_held else rate for axis_held, rate in zip(held, applied))
            rates = (0.0 if held[0] else result.intent.yaw_rate_rad_s,
                     0.0 if held[1] else result.intent.pitch_rate_rad_s)
            guarded = ControlIntent.model_validate({
                **result.intent.model_dump(),
                "yaw_rate_rad_s": rates[0], "pitch_rate_rad_s": rates[1],
                # The free axis keeps tracking under the original reason.
                "reason": "travel_limit_hold" if all(held) else result.intent.reason,
            })
        else:
            guarded = result.intent
        if not self.policy.live_authorized:
            intent = guarded
        else:
            # A failed timing/safety/target gate is an explicit zero-rate
            # live stop, never a stale continuation of the previous demand.
            tracking = guarded.reason in ("tracking", "coasting", "idle_return", "manual")
            intent = ControlIntent.model_validate({
                **guarded.model_dump(),
                "mode": "live",
                "valid_until_monotonic_ns": (
                    observation.created_monotonic_ns + self.policy.live_intent_ttl_ns
                ),
                "yaw_rate_rad_s": guarded.yaw_rate_rad_s if tracking else 0.0,
                "pitch_rate_rad_s": guarded.pitch_rate_rad_s if tracking else 0.0,
            })
        return VideoControllerDecision(
            intent, result, estimate, applied,
            "predicted" if predicted is not None else "frame_bearing",
            tuple(held),
        )
