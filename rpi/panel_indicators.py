"""Read-only panel indications; no GPIO, actuator commands or remote-clock maths."""

from dataclasses import dataclass
import math

from common.perception import PerceptionSnapshotV2, perception_snapshot_from_json
from common.schemas import CamState


@dataclass(frozen=True)
class IndicatorState:
    fire_control_lamp: bool = False
    safety_lamp: bool = False
    yellow_light: bool = False


class _OffDelay:
    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self.on = False
        self.absent_since: float | None = None

    def update(self, present: bool, now_s: float) -> bool:
        if present:
            self.on, self.absent_since = True, None
        elif self.on:
            if self.absent_since is None:
                self.absent_since = now_s
            if now_s - self.absent_since >= self.delay_s:
                self.on = False
        return self.on

    def reset(self) -> None:
        self.on, self.absent_since = False, None


class PanelIndicators:
    """Lamp requests never depend on GPIO12/16 or on controller authority.

    Availability = a current detector registration or observed track.
    Tracking = the selected track is currently observed (not coasting).
    Local receive watchdogs are indications, not qualified control timing.
    """

    def __init__(self, *, fire_control_off_delay_s: float = 1.0,
                 safety_off_delay_s: float = 1.0, perception_stale_s: float = 0.5,
                 hardware_stale_s: float = 0.5):
        values = (fire_control_off_delay_s, safety_off_delay_s,
                  perception_stale_s, hardware_stale_s)
        if any(not math.isfinite(v) or v < 0 for v in values) or min(values[2:]) <= 0:
            raise ValueError("indicator delays must be finite/nonnegative; watchdogs must be positive")
        self.fire = _OffDelay(fire_control_off_delay_s)
        self.safety = _OffDelay(safety_off_delay_s)
        self.perception_stale_s, self.hardware_stale_s = values[2:]
        self._perception_rx: float | None = None
        self._hardware_rx: float | None = None
        self._snapshot: PerceptionSnapshotV2 | None = None
        self._source_times: dict[str, int] = {}
        self._hardware_stamp = -1
        self._hardware_available = False

    def on_perception(self, payload: bytes | str, now_s: float) -> bool:
        snapshot = perception_snapshot_from_json(payload)
        domain, stamp = snapshot.frame.source_clock_domain, snapshot.frame.source_time_ns
        if stamp <= self._source_times.get(domain, -1):
            return False  # replayed/out-of-order packets cannot refresh the watchdog
        if self._snapshot and domain != self._snapshot.frame.source_clock_domain:
            self.fire.reset()
            self.safety.reset()
        self._source_times[domain] = stamp
        self._snapshot, self._perception_rx = snapshot, now_s
        seen = {t.track_id for t in snapshot.tracks if t.missed_frames == 0 and t.confidence > 0}
        available = bool(seen or any(d.confidence > 0 for d in snapshot.detections))
        tracking = snapshot.selection is not None and snapshot.selection.track_id in seen
        # Observe every accepted absence, not just the latest state at a later tick.
        self.fire.update(available, now_s)
        self.safety.update(tracking, now_s)
        return True

    def on_hardware(self, payload: bytes | str, now_s: float) -> bool:
        state = CamState.model_validate_json(payload)
        stamp = state.state_monotonic_ns
        if stamp is None or stamp <= self._hardware_stamp:
            return False
        self._hardware_stamp, self._hardware_rx = stamp, now_s
        # Missing proof, synthetic pose, IMU-only state and explicit failure are off.
        self._hardware_available = state.hardware_available is True
        return True

    def state(self, now_s: float) -> IndicatorState:
        fresh = self._perception_rx is not None and 0 <= now_s - self._perception_rx < self.perception_stale_s
        if not fresh:
            self.fire.reset()
            self.safety.reset()
        elif self._snapshot is not None:
            seen = {t.track_id for t in self._snapshot.tracks if t.missed_frames == 0 and t.confidence > 0}
            self.fire.update(bool(seen or any(d.confidence > 0 for d in self._snapshot.detections)), now_s)
            self.safety.update(self._snapshot.selection is not None and self._snapshot.selection.track_id in seen, now_s)
        hardware = (self._hardware_rx is not None
                    and 0 <= now_s - self._hardware_rx < self.hardware_stale_s
                    and self._hardware_available)
        return IndicatorState(self.fire.on, self.safety.on, hardware)
