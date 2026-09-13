from __future__ import annotations

from dataclasses import dataclass

from common.schemas import (
    CamState,
    ControlGimbalObservation,
    ControlObservation,
    ControlSafetyObservation,
    ControlTargetObservation,
    ControlTransportObservation,
)
from jetson.fixed_rate_controller import FixedRateController


@dataclass
class _StubController:
    observations: list[tuple[int, float]]
    ticks: list[float]
    states: list[int]

    def update_control_observation(self, observation, *, received_at=None):
        self.observations.append((observation.sequence, received_at))

    def update_cam_state(self, state):
        self.states.append(state.frame_id)

    def tick(self, now=None):
        self.ticks.append(now)


def _observation(sequence: int = 1) -> ControlObservation:
    return ControlObservation(
        sequence=sequence,
        created_monotonic_ns=1,
        target=ControlTargetObservation(valid=False),
        gimbal=ControlGimbalObservation(valid=False),
        transport=ControlTransportObservation(),
        safety=ControlSafetyObservation(
            valid=False,
            auto_allowed=False,
            manual_active=False,
            emergency_active=False,
        ),
    )


def test_fixed_rate_uses_explicit_receipt_time_and_single_current_tick() -> None:
    stub = _StubController([], [], [])
    runner = FixedRateController(stub, loop_hz=50.0)
    runner.update_control_observation(_observation(7), received_at=10.0)
    assert runner.advance(10.0)
    assert not runner.advance(10.019)
    assert runner.advance(10.10)
    assert stub.observations == [(7, 10.0)]
    assert stub.ticks == [10.0, 10.10]
    assert runner.stats.missed_periods == 4
    assert runner.stats.observations == 1


def test_fixed_rate_forwards_cam_state() -> None:
    stub = _StubController([], [], [])
    runner = FixedRateController(stub, loop_hz=10.0)
    runner.update_cam_state(CamState(frame_id=2, src_ts_ms=1, pan=0.0, tilt=0.0))
    assert stub.states == [2]
    assert runner.stats.cam_states == 1
