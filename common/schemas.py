"""Shared Pydantic schemas for ZMQ metadata payloads between PC and Jetson.

These models define the JSON payloads exchanged over the metadata sockets.
They aim to stay backward compatible by treating newly added fields as
optional and by omitting ``None`` values during serialization so older
consumers do not receive unexpected ``null`` keys.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

class MpcAxisDiagnostic(BaseModel):
    """Compact MPC diagnostics for a single axis (yaw or pitch).

    ``status`` mirrors the solver status string. ``cost`` is the objective
    value when available. ``u0`` is the first control command in the MPC
    sequence (typically a rate command in rad/s). ``slack``, ``solver``,
    ``terms``, and ``term_directions`` are optional diagnostic dictionaries
    containing solver and cost
    breakdowns. ``refs`` and ``pred`` optionally expose compact reference and
    prediction snapshots (for example ``theta_ref0`` or ``theta_pred0``).
    Optional fields are omitted when empty or non-finite to keep payloads
    compact and backward compatible.
    """

    status: str
    cost: Optional[float] = None
    u0: Optional[float] = None
    slack: Optional[Dict[str, float]] = None
    solver: Optional[Dict[str, float]] = None
    terms: Optional[Dict[str, float]] = None
    term_directions: Optional[Dict[str, float]] = None
    refs: Optional[Dict[str, float]] = None
    pred: Optional[Dict[str, float]] = None


class ControlCmd(BaseModel):
    """Jetson ??PC control command payload.

    ``pan_accel_cmd`` and ``tilt_accel_cmd`` are optional physical acceleration
    intents in rad/s^2. When absent, consumers should keep their configured
    acceleration behavior for backward compatibility.
    """

    type: Literal["ControlCmd"] = "ControlCmd"
    frame_id: int
    src_ts_ms: int
    cmd_ts_ms: int
    target_ok: bool
    target_uv: Tuple[float, float]
    err_uv: Tuple[float, float]
    err_rad: Tuple[float, float]
    pan_rate_cmd: float
    tilt_rate_cmd: float
    pan_accel_cmd: Optional[float] = None
    tilt_accel_cmd: Optional[float] = None
    pan_abs_cmd: Optional[float] = None
    tilt_abs_cmd: Optional[float] = None
    laser_origin_px: Optional[Tuple[float, float]] = None
    laser_dot_px: Optional[Tuple[float, float]] = None
    laser_on_target: Optional[bool] = None
    laser_range_m: Optional[float] = None
    laser_range_source: Optional[str] = None
    parallax_compensation_active: Optional[bool] = None
    controller_mode: Optional[Literal["pid", "mpc"]] = None
    mpc: Optional[Dict[str, MpcAxisDiagnostic]] = None


class CamState(BaseModel):
    """PC ??Jetson camera pose/state header."""

    type: Literal["CamState"] = "CamState"
    frame_id: int
    src_ts_ms: int
    state_monotonic_ns: Optional[int] = None
    pan: float
    tilt: float
    pan_rate: Optional[float] = None
    tilt_rate: Optional[float] = None
    home_pan: Optional[float] = None
    home_tilt: Optional[float] = None
    # Optional render-only pose predicted from the accepted, quantized motor
    # command and re-anchored to each encoder sample. ``pan``/``tilt`` remain
    # measured encoder truth for controller limits and fault handling.
    render_pan: Optional[float] = None
    render_tilt: Optional[float] = None
    render_pan_rate: Optional[float] = None
    render_tilt_rate: Optional[float] = None
    render_prediction_age_ms: Optional[float] = Field(default=None, ge=0.0)
    render_pan_correction_rad: Optional[float] = None
    render_tilt_correction_rad: Optional[float] = None
    encoder_pan_counts: Optional[int] = None
    encoder_tilt_counts: Optional[int] = None


class ManualControlState(BaseModel):
    """RPi ??Jetson manual-control state payload."""

    type: Literal["ManualControlState"] = "ManualControlState"
    src_ts_ms: int
    source: str
    active: bool
    emergency: bool
    active_changed: bool = False
    emergency_entered: bool = False
    emergency_exited: bool = False
    control_cmd_enabled: bool = False
    control_cmd_changed: bool = False
    joystick_raw: Tuple[int, int]
    joystick_rate_cmd: Tuple[float, float]
    serial_local_mode: bool = False
    note: Optional[str] = None


class _ControlProtocolModel(BaseModel):
    """Strict, immutable base for the controller-overhaul boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _reject_non_finite_numbers(self):
        def check(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("control protocol fields must be finite")
            if isinstance(value, Mapping):
                for item in value.values():
                    check(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check(item)

        check(self.__dict__)
        return self


class ControlTargetObservation(_ControlProtocolModel):
    """Complete selected-target geometry for one controller decision.

    Pixel geometry is carried explicitly so consumers never reconstruct the
    target from a signed bearing.  ``aim_reference_px`` is the optical centre
    for camera-centred control or the projected parallax aim point when that
    mode is active.
    """

    valid: bool
    track_id: Optional[int] = None
    class_id: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    target_center_px: Optional[Tuple[float, float]] = None
    aim_reference_px: Optional[Tuple[float, float]] = None
    pixel_error: Optional[Tuple[float, float]] = None
    bearing_error_rad: Optional[Tuple[float, float]] = None
    bearing_rate_rad_s: Optional[Tuple[float, float]] = None
    distance_m: Optional[float] = Field(default=None, gt=0.0)
    distance_source: Optional[str] = Field(default=None, min_length=1, max_length=80)
    parallax_active: bool = False
    on_target: Optional[bool] = None
    source_age_ms: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _valid_target_has_complete_geometry(self):
        geometry = (
            self.target_center_px,
            self.aim_reference_px,
            self.pixel_error,
            self.bearing_error_rad,
        )
        if self.valid and not all(value is not None for value in geometry):
            raise ValueError("valid control target requires complete image geometry")
        if self.parallax_active and (
            self.distance_m is None or self.aim_reference_px is None
        ):
            raise ValueError("active parallax requires range and aim reference")
        return self


class ControlGimbalObservation(_ControlProtocolModel):
    """Encoder-derived pose/rate sample in the controller sign convention."""

    valid: bool
    yaw_rad: Optional[float] = None
    pitch_rad: Optional[float] = None
    yaw_rate_rad_s: Optional[float] = None
    pitch_rate_rad_s: Optional[float] = None
    sample_age_ms: Optional[float] = Field(default=None, ge=0.0)


class ControlTransportObservation(_ControlProtocolModel):
    """Locally measured timing available to the controller at a tick."""

    last_command_age_ms: Optional[float] = Field(default=None, ge=0.0)
    serial_acceptance_ms: Optional[float] = Field(default=None, ge=0.0)


class ControlSafetyObservation(_ControlProtocolModel):
    """Authority state; invalid or stale safety input is fail-safe false."""

    valid: bool
    auto_allowed: bool
    manual_active: bool
    emergency_active: bool
    sample_age_ms: Optional[float] = Field(default=None, ge=0.0)


class ControlObservation(_ControlProtocolModel):
    """Atomic input snapshot for one fixed-rate controller decision."""

    type: Literal["ControlObservation"] = "ControlObservation"
    version: Literal[1] = 1
    sequence: int = Field(ge=0)
    created_monotonic_ns: int = Field(ge=0)
    source_frame_id: Optional[int] = Field(default=None, ge=0)
    source_time_ns: Optional[int] = Field(default=None, ge=0)
    source_clock_domain: Optional[str] = Field(default=None, min_length=1, max_length=80)
    source_identity_verified: Optional[bool] = None
    frame_received_time_ns: Optional[int] = Field(default=None, ge=0)
    frame_receive_clock_domain: Optional[str] = Field(default=None, min_length=1, max_length=80)
    frame_observed_time_ns: Optional[int] = Field(default=None, ge=0)
    frame_observation_clock_domain: Optional[str] = Field(default=None, min_length=1, max_length=80)
    target: ControlTargetObservation
    gimbal: ControlGimbalObservation
    transport: ControlTransportObservation
    safety: ControlSafetyObservation

    @model_validator(mode="after")
    def _source_provenance_is_atomic(self):
        present = (
            self.source_frame_id is not None,
            self.source_time_ns is not None,
            self.source_clock_domain is not None,
        )
        if any(present) and not all(present):
            raise ValueError("control source provenance fields must be set together")
        received = (
            self.frame_received_time_ns is not None,
            self.frame_receive_clock_domain is not None,
        )
        if any(received) and not all(received):
            raise ValueError("frame receive timing fields must be set together")
        observed = (
            self.frame_observed_time_ns is not None,
            self.frame_observation_clock_domain is not None,
        )
        if any(observed) and not all(observed):
            raise ValueError("frame observation timing fields must be set together")
        return self


class ControlIntentLimits(_ControlProtocolModel):
    """Explain which output constraints shaped a controller decision."""

    yaw_rate_limited: bool = False
    pitch_rate_limited: bool = False
    acceleration_limited: bool = False
    position_limited: bool = False


class ControlTimingDiagnostics(_ControlProtocolModel):
    """Controller-local timing evidence without cross-clock assumptions."""

    snapshot_receipt_age_ms: Optional[float] = Field(default=None, ge=0.0)
    frame_receive_to_tick_ms: Optional[float] = Field(default=None, ge=0.0)
    frame_observe_to_tick_ms: Optional[float] = Field(default=None, ge=0.0)
    frame_receive_to_observe_ms: Optional[float] = Field(default=None, ge=0.0)
    gimbal_sample_age_ms: Optional[float] = Field(default=None, ge=0.0)
    source_clock_domain: Optional[str] = Field(default=None, min_length=1, max_length=80)
    frame_receive_clock_domain: Optional[str] = Field(default=None, min_length=1, max_length=80)
    frame_observation_clock_domain: Optional[str] = Field(default=None, min_length=1, max_length=80)
    source_to_local_mapping_available: bool = False
    estimator_time_source: Optional[str] = None
    source_frame_age_ms: Optional[float] = Field(default=None, ge=0.0)
    source_clock_uncertainty_ms: Optional[float] = Field(default=None, ge=0.0)
    frame_gimbal_pose_age_ms: Optional[float] = Field(default=None, ge=0.0)


class ControlEstimatorAxisDiagnostics(_ControlProtocolModel):
    """One axis of estimator state and decomposed command evidence."""

    estimator_enabled: bool
    measurement_updated: bool = False
    measurement_accepted: Optional[bool] = None
    measurement_reinitialized: bool = False
    raw_error_rad: Optional[float] = None
    estimated_error_rad: Optional[float] = None
    estimated_target_angle_rad: Optional[float] = None
    estimated_target_rate_rad_s: Optional[float] = None
    estimate_sample_time_s: Optional[float] = None
    estimate_query_time_s: Optional[float] = None
    prediction_horizon_ms: Optional[float] = Field(default=None, ge=0.0)
    angle_variance_rad2: Optional[float] = Field(default=None, ge=0.0)
    rate_variance_rad2_s2: Optional[float] = Field(default=None, ge=0.0)
    angle_rate_covariance_rad2_s: Optional[float] = None
    innovation_rad: Optional[float] = None
    innovation_variance_rad2: Optional[float] = Field(default=None, ge=0.0)
    normalized_innovation_squared: Optional[float] = Field(default=None, ge=0.0)
    accepted_updates: int = Field(default=0, ge=0)
    rejected_updates: int = Field(default=0, ge=0)
    reinitialized_updates: int = Field(default=0, ge=0)
    consecutive_rejections: int = Field(default=0, ge=0)
    feedback_term_rad_s: Optional[float] = None
    damping_term_rad_s: Optional[float] = None
    feedforward_term_rad_s: Optional[float] = None
    desired_rate_pre_limit_rad_s: Optional[float] = None
    desired_rate_post_limit_rad_s: Optional[float] = None
    final_rate_rad_s: float = 0.0


class ControlDiagnostics(_ControlProtocolModel):
    """Versioned, non-authoritative diagnostics paired to one intent."""

    type: Literal["ControlDiagnostics"] = "ControlDiagnostics"
    version: Literal[1] = 1
    observation_sequence: int = Field(ge=0)
    intent_sequence: int = Field(ge=0)
    created_monotonic_ns: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=80)
    track_id: Optional[int] = None
    timing: ControlTimingDiagnostics
    yaw: ControlEstimatorAxisDiagnostics
    pitch: ControlEstimatorAxisDiagnostics


class ControlIntent(_ControlProtocolModel):
    """Bounded controller output, before any hardware/serial translation."""

    type: Literal["ControlIntent"] = "ControlIntent"
    version: Literal[1] = 1
    sequence: int = Field(ge=0)
    observation_sequence: int = Field(ge=0)
    issued_monotonic_ns: int = Field(ge=0)
    valid_until_monotonic_ns: int = Field(ge=0)
    mode: Literal["shadow", "live"] = "shadow"
    yaw_rate_rad_s: float
    pitch_rate_rad_s: float
    limits: ControlIntentLimits = Field(default_factory=ControlIntentLimits)
    reason: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def _intent_expiry_is_not_before_issue(self):
        if self.valid_until_monotonic_ns < self.issued_monotonic_ns:
            raise ValueError("ControlIntent expiry must not precede issue time")
        return self


def control_cmd_from_json(payload: Union[str, bytes, bytearray, Mapping[str, Any]]) -> ControlCmd:
    """Decode serialized control commands into :class:`ControlCmd` objects."""

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(f"ControlCmd payload must be mapping-like, got {type(payload)!r}")
    return ControlCmd(**payload)


def control_intent_from_json(
    payload: Union[str, bytes, bytearray, Mapping[str, Any]],
) -> ControlIntent:
    """Decode one versioned controller intent at the actuator boundary."""

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"ControlIntent payload must be mapping-like, got {type(payload)!r}"
        )
    return ControlIntent(**payload)


def manual_control_state_from_json(
    payload: Union[str, bytes, bytearray, Mapping[str, Any]]
) -> ManualControlState:
    """Decode serialized manual-control state into :class:`ManualControlState` objects."""

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"ManualControlState payload must be mapping-like, got {type(payload)!r}"
        )
    return ManualControlState(**payload)
