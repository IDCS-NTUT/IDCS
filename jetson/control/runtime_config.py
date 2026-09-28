"""Validated ``controller`` configuration for the video controller runtime.

One config section replaces the trial-era command-line locks. Every value is
bounded here, and the clock policy must name its basis so a report always
says why its drift bound is believed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from common.gimbal.mks_servo42_rs485 import min_f6_speed_rad_s

CLOCK_BASES = {
    # Both hosts' monotonic clocks are rate-limited: PC chrony maxslewrate 500
    # + maxdrift 500, Jetson timesyncd within the kernel's 500 ppm; the bound is
    # their sum (1500 ppm).
    "slew_limited_ntp",
    # No cross-host mapping is needed: the camera timestamps frames on the
    # controller host.
    "same_host",
    # An explicit assumption, not a derived bound.
    "assumed",
}


@dataclass(frozen=True)
class ControlRuntimeConfig:
    mode: Literal["shadow", "live"]
    yaw_kp: float
    pitch_kp: float
    rate_limit_rad_s: float
    accel_limit_rad_s2: float
    feedforward_scale: float
    predict: float
    feedforward_accel_sigma_rad_s2: float
    max_capture_age_ms: int
    max_travel_rad: float
    clock_drift_ppm: float
    clock_basis: str
    snapshot_endpoint: str
    gimbal_endpoint: str
    manual_bind: str
    clock_endpoint: str
    intent_bind: str
    diagnostics_bind: str
    # Host the controller runs on: DeepStream's Jetson receipt times are only
    # usable when it is the Jetson.
    local_clock: str = "jetson"
    # Clock the frames' source times are on: pc_monotonic (PC streamer,
    # mapped through the clock exchange) or jetson_monotonic (a camera on the
    # controller's own host: sensor start-of-frame times, no mapping).
    source_clock: str = "pc_monotonic"
    # Seconds without a target before slewing back to the origin; None holds.
    idle_return_s: float | None = None
    idle_return_rate_rad_s: float = 0.3
    # Seconds to keep steering on the predicted target through a detection
    # gap; None stops at once.
    coast_s: float | None = None
    camera_fov_y_deg: float | None = None
    # PUB for per-tick controller records and panel states (flight recorder);
    # None publishes nothing.
    record_bind: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "live"}:
            raise ValueError("controller.mode must be shadow or live")
        for name in ("yaw_kp", "pitch_kp"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 1.0 <= value <= 20.0:
                raise ValueError(f"controller.{name} must be in [1, 20]")
        if not math.isfinite(self.rate_limit_rad_s) or not min_f6_speed_rad_s() <= self.rate_limit_rad_s <= 1.0:
            raise ValueError(
                f"controller.rate_limit_rad_s must be in [{min_f6_speed_rad_s():.3f}, 1.0] "
                "(below the slowest F6 speed the axis cannot move)")
        if not math.isfinite(self.accel_limit_rad_s2) or not 0 < self.accel_limit_rad_s2 <= 20:
            raise ValueError("controller.accel_limit_rad_s2 must be in (0, 20]")
        for name in ("feedforward_scale", "predict"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"controller.{name} must be in [0, 1]")
        if not math.isfinite(self.feedforward_accel_sigma_rad_s2) or not 0 < self.feedforward_accel_sigma_rad_s2 <= 20:
            raise ValueError("controller.feedforward_accel_sigma_rad_s2 must be in (0, 20]")
        if not 0 < self.max_capture_age_ms <= 250:
            raise ValueError("controller.max_capture_age_ms must be in (0, 250]")
        # Up to half a turn: a swarm can come from any bearing. Hardware
        # overlays keep a small bench envelope.
        if not math.isfinite(self.max_travel_rad) or not 0 < self.max_travel_rad <= math.pi:
            raise ValueError("controller.max_travel_rad must be in (0, pi]")
        if self.clock_basis not in CLOCK_BASES:
            raise ValueError(f"controller.clock.basis must be one of {sorted(CLOCK_BASES)}")
        if self.local_clock not in ("jetson", "pc"):
            raise ValueError("controller.local_clock must be jetson or pc")
        if self.idle_return_s is not None and (
                not math.isfinite(self.idle_return_s) or not 0 < self.idle_return_s <= 60):
            raise ValueError("controller.idle_return_s must be in (0, 60] or null")
        if (not math.isfinite(self.idle_return_rate_rad_s)
                or not 0 < self.idle_return_rate_rad_s <= self.rate_limit_rad_s):
            raise ValueError("controller.idle_return_rate_rad_s must be in (0, rate_limit_rad_s]")
        if self.coast_s is not None and (not math.isfinite(self.coast_s) or not 0 < self.coast_s <= 2):
            raise ValueError("controller.coast_s must be in (0, 2] or null")
        if self.source_clock not in ("pc_monotonic", "jetson_monotonic"):
            raise ValueError("controller.source_clock must be pc_monotonic or jetson_monotonic")
        if self.source_clock == "jetson_monotonic" and (
                self.local_clock != "jetson" or self.clock_basis != "same_host"):
            raise ValueError("controller.source_clock jetson_monotonic needs local_clock jetson "
                             "and clock.basis same_host")
        if not math.isfinite(self.clock_drift_ppm) or not 0 <= self.clock_drift_ppm <= 2000:
            raise ValueError("controller.clock.drift_ppm must be in [0, 2000]")
        if self.camera_fov_y_deg is not None and (
                not math.isfinite(self.camera_fov_y_deg) or not 1 < self.camera_fov_y_deg < 179):
            raise ValueError("controller.camera_fov_y_deg must be in (1, 179)")
        endpoints = (self.snapshot_endpoint, self.gimbal_endpoint, self.manual_bind,
                     self.clock_endpoint, self.intent_bind, self.diagnostics_bind)
        if self.record_bind is not None:
            endpoints = (*endpoints, self.record_bind)
        if not all(isinstance(e, str) and e.startswith("tcp://") for e in endpoints):
            raise ValueError("controller endpoints must be tcp:// URLs")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ControlRuntimeConfig":
        raw = config.get("controller")
        if not isinstance(raw, Mapping):
            raise ValueError("missing controller configuration section")
        net = config.get("net") or {}
        clock = raw.get("clock") or {}
        endpoints = raw.get("endpoints") or {}

        def endpoint(key: str, net_key: str) -> str:
            value = endpoints.get(key) or net.get(net_key)
            if not value:
                raise ValueError(f"controller.endpoints.{key} (or net.{net_key}) is required")
            return str(value)

        return cls(
            mode=str(raw.get("mode", "shadow")),
            yaw_kp=float(raw["yaw_kp"]),
            pitch_kp=float(raw["pitch_kp"]),
            rate_limit_rad_s=float(raw["rate_limit_rad_s"]),
            accel_limit_rad_s2=float(raw.get("accel_limit_rad_s2", 3.5)),
            feedforward_scale=float(raw.get("feedforward_scale", 0.0)),
            predict=float(raw.get("predict", 0.0)),
            feedforward_accel_sigma_rad_s2=float(raw.get("feedforward_accel_sigma_rad_s2", 0.4)),
            max_capture_age_ms=int(raw.get("max_capture_age_ms", 150)),
            max_travel_rad=float(raw.get("max_travel_rad", 0.15)),
            clock_drift_ppm=float(clock["drift_ppm"]),
            clock_basis=str(clock["basis"]),
            snapshot_endpoint=endpoint("snapshot_sub", "zmq_perception_v2"),
            gimbal_endpoint=endpoint("gimbal_sub", "zmq_gimbal_state"),
            manual_bind=endpoint("manual_bind", "zmq_manual_state"),
            clock_endpoint=endpoint("clock", "zmq_source_clock_sync"),
            intent_bind=endpoint("intent_bind", "zmq_control"),
            diagnostics_bind=endpoint("diagnostics_bind", "zmq_control_diagnostics"),
            local_clock=str(raw.get("local_clock", "jetson")),
            source_clock=str(raw.get("source_clock", "pc_monotonic")),
            idle_return_s=(None if raw.get("idle_return_s") is None else float(raw["idle_return_s"])),
            idle_return_rate_rad_s=float(raw.get("idle_return_rate_rad_s", 0.3)),
            coast_s=(None if raw.get("coast_s") is None else float(raw["coast_s"])),
            record_bind=(str(endpoints["record_bind"]) if endpoints.get("record_bind") else None),
            camera_fov_y_deg=(None if raw.get("camera_fov_y_deg") is None
                              else float(raw["camera_fov_y_deg"])),
        )

    def describe(self) -> dict[str, Any]:
        return asdict(self)
