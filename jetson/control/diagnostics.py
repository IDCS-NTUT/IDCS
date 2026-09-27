"""Read-only per-tick diagnostics for displays (the HUD), never for control.

Maps one controller decision onto the versioned ``ControlDiagnostics`` schema:
PID and feedforward terms per axis, the final commanded rate, the estimated
target rate, and the frame geometry (normalized) the aim cue is drawn from.
"""

from __future__ import annotations

from common.schemas import (
    ControlDiagnostics, ControlEstimatorAxisDiagnostics, ControlObservation, ControlTimingDiagnostics,
)
from jetson.control.video_controller import VideoControllerDecision


def build_diagnostics(
    observation: ControlObservation, decision: VideoControllerDecision, *,
    feedforward_scale: float, created_monotonic_ns: int,
    frame_size_px: tuple[int, int] | None = None,
) -> ControlDiagnostics:
    target = observation.target
    raw = target.bearing_error_rad if target is not None else None
    ff_rates = (decision.feedforward.yaw_rate_rad_s, decision.feedforward.pitch_rate_rad_s)
    intent = decision.intent

    def axis(index: int) -> ControlEstimatorAxisDiagnostics:
        terms = decision.pid.pid.yaw if index == 0 else decision.pid.pid.pitch
        return ControlEstimatorAxisDiagnostics(
            estimator_enabled=feedforward_scale > 0.0,
            raw_error_rad=None if raw is None else float(raw[index]),
            estimated_target_rate_rad_s=float(ff_rates[index]) if decision.feedforward.valid else None,
            feedback_term_rad_s=terms.proportional_rad_s + terms.integral_rad_s + terms.derivative_rad_s,
            feedforward_term_rad_s=float(decision.applied_feedforward_rad_s[index]),
            desired_rate_pre_limit_rad_s=terms.pre_limit_rad_s,
            desired_rate_post_limit_rad_s=terms.rate_limited_rad_s,
            final_rate_rad_s=float(intent.yaw_rate_rad_s if index == 0 else intent.pitch_rate_rad_s),
        )

    def norm(point: tuple[float, float] | None) -> tuple[float, float] | None:
        if point is None or frame_size_px is None or min(frame_size_px) <= 0:
            return None
        return (point[0] / frame_size_px[0], point[1] / frame_size_px[1])

    capture_age = decision.pid.timing.capture_age_ns
    return ControlDiagnostics(
        observation_sequence=observation.sequence,
        intent_sequence=intent.sequence,
        created_monotonic_ns=int(created_monotonic_ns),
        reason=intent.reason,
        track_id=None if target is None else target.track_id,
        timing=ControlTimingDiagnostics(
            gimbal_sample_age_ms=observation.gimbal.sample_age_ms,
            source_frame_age_ms=None if capture_age is None else capture_age.latest_ns / 1e6,
        ),
        yaw=axis(0),
        pitch=axis(1),
        target_center_norm=norm(target.target_center_px),
        aim_reference_norm=norm(target.aim_reference_px),
    )
