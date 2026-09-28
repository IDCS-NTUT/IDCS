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
        if not math.isfinite(self.max_travel_rad) or not 0 < self.max_travel_rad <= 1.0:
            raise ValueError("travel limit must be in (0, 1.0] rad")


@dataclass(frozen=True)
class VideoControllerDecision:
    intent: ControlIntent
    pid: ShadowPIDResult
    feedforward: VideoFeedforwardEstimate
    applied_feedforward_rad_s: tuple[float, float]
    pid_error_source: str = "frame_bearing"


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
            max_safety_age_ns=750_000_000,
            source_clock_domain=policy.source_clock_domain,
        )
        self._origin_rad: tuple[float, float] | None = None

    def reset(self) -> None:
        self.feedforward.reset()
        self.pid.reset()
        self._origin_rad = None

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
        result = self.pid.decide(
            observation, clock, feedforward_rad_s=applied,
            error_override_rad=predicted,
        )
        if result.intent.reason != "tracking":
            applied = (0.0, 0.0)
        gimbal = observation.gimbal
        if self._origin_rad is None and gimbal.valid and gimbal.yaw_rad is not None and gimbal.pitch_rad is not None:
            self._origin_rad = (gimbal.yaw_rad, gimbal.pitch_rad)
        travel_hold = False
        if self._origin_rad is not None and result.intent.reason == "tracking":
            assert gimbal.yaw_rad is not None and gimbal.pitch_rad is not None
            # Hold a command only if its projection leaves the envelope *and*
            # moves farther from the origin. Motion back toward the envelope
            # stays allowed, so an overshoot cannot latch the controller.
            ttl_s = self.policy.live_intent_ttl_ns / 1e9
            travel_hold = any(
                abs(position + rate * ttl_s - origin) > self.policy.max_travel_rad
                and abs(position + rate * ttl_s - origin) > abs(position - origin)
                for position, rate, origin in zip(
                    (gimbal.yaw_rad, gimbal.pitch_rad),
                    (result.intent.yaw_rate_rad_s, result.intent.pitch_rate_rad_s),
                    self._origin_rad,
                )
            )
        if travel_hold:
            applied = (0.0, 0.0)
            guarded = ControlIntent.model_validate({
                **result.intent.model_dump(),
                "yaw_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0,
                "reason": "travel_limit_hold",
            })
        else:
            guarded = result.intent
        if not self.policy.live_authorized:
            intent = guarded
        else:
            # A failed timing/safety/target gate is an explicit zero-rate
            # live stop, never a stale continuation of the previous demand.
            tracking = guarded.reason == "tracking"
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
        )
