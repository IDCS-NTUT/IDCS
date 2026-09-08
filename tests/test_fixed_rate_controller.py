from __future__ import annotations

from dataclasses import dataclass

from common.schemas import Box, CamState, DetectionMsg
from jetson.fixed_rate_controller import FixedRateController


@dataclass
class _StubController:
    detections: list[tuple[int, float]]
    ticks: list[float]
    states: list[int]

    def update_detection(self, msg, *, received_at=None):
        self.detections.append((msg.frame_id, received_at))

    def update_cam_state(self, state):
        self.states.append(state.frame_id)

    def tick(self, now=None):
        self.ticks.append(now)


def _detection(frame_id: int = 1) -> DetectionMsg:
    return DetectionMsg(frame_id=frame_id, src_ts_ms=1, rx_ts_ms=2, infer_ts_ms=3,
                        img_w=1280, img_h=720,
                        boxes=[Box(x=0.1, y=0.1, w=0.1, h=0.1, cls="person", conf=0.9)])


def test_fixed_rate_uses_explicit_receipt_time_and_single_current_tick() -> None:
    stub = _StubController([], [], [])
    runner = FixedRateController(stub, loop_hz=50.0)
    runner.update_detection(_detection(7), received_at=10.0)
    assert runner.advance(10.0)
    assert not runner.advance(10.019)
    assert runner.advance(10.10)
    assert stub.detections == [(7, 10.0)]
    assert stub.ticks == [10.0, 10.10]
    assert runner.stats.missed_periods == 4


def test_fixed_rate_forwards_cam_state() -> None:
    stub = _StubController([], [], [])
    runner = FixedRateController(stub, loop_hz=10.0)
    runner.update_cam_state(CamState(frame_id=2, src_ts_ms=1, pan=0.0, tilt=0.0))
    assert stub.states == [2]
    assert runner.stats.cam_states == 1
