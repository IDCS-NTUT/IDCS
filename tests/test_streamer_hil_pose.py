import math

import pytest

from common.schemas import CamState
from pc.streamer import MeasuredPoseTimeline, require_simulation_perception_endpoint


def _state(measured_ns: int, pan: float, tilt: float, *, published_ns: int | None = None,
           home=(1.0, -0.5)) -> CamState:
    published = measured_ns + 5_000_000 if published_ns is None else published_ns
    return CamState(frame_id=1, src_ts_ms=0, state_monotonic_ns=published,
                    pan_sample_monotonic_ns=measured_ns, tilt_sample_monotonic_ns=measured_ns,
                    pan=pan, tilt=tilt, home_pan=home[0], home_tilt=home[1])


def _feed(timeline, count, period_ns, *, start_ns=1_000_000_000, rate=0.0):
    for i in range(count):
        t = start_ns + i * period_ns
        timeline.add(_state(t, 1.0 + rate * i * period_ns / 1e9, -0.5), received_ns=t + 6_000_000)


def test_pose_is_relative_to_bridge_home_and_wraps_pan() -> None:
    timeline = MeasuredPoseTimeline(warmup_samples=3)
    for i, pan in enumerate((1.2, 1.2, 1.2)):
        timeline.add(_state(1_000_000_000 + i * 40_000_000, pan, -0.4), received_ns=0 + 1_000_000_000 + i * 40_000_000)
    pan, tilt, _, _ = timeline.pose_at(1_030_000_000)
    assert pan == pytest.approx(0.2) and tilt == pytest.approx(0.1)
    wrap = MeasuredPoseTimeline(warmup_samples=3)
    for i in range(3):
        wrap.add(_state(1_000_000_000 + i * 40_000_000, -math.pi + 0.1, 0.0, home=(math.pi - 0.1, 0.0)),
                 received_ns=1_000_000_000 + i * 40_000_000)
    assert wrap.pose_at(1_030_000_000)[0] == pytest.approx(0.2)


def test_local_time_is_receipt_minus_the_jetson_publication_lag() -> None:
    timeline = MeasuredPoseTimeline(warmup_samples=3)
    # Measured at 1.00/1.04/1.08 s, published 15 ms later, received 1 ms after that
    # in a local clock offset by +5 s. Local sample times are therefore +5.001 s.
    for i, pan in enumerate((1.0, 1.04, 1.08)):
        t = 1_000_000_000 + i * 40_000_000
        timeline.add(_state(t, pan, -0.5, published_ns=t + 15_000_000), received_ns=t + 5_016_000_000)
    pan, _, pan_rate, _ = timeline.pose_at(6_021_000_000)  # local time of measured 1.020 s
    assert pan == pytest.approx(0.02) and pan_rate == pytest.approx(1.0)


def test_republished_samples_are_ignored() -> None:
    timeline = MeasuredPoseTimeline(warmup_samples=3)
    assert timeline.add(_state(1_000_000_000, 1.0, -0.5), received_ns=1_006_000_000)
    assert not timeline.add(_state(1_000_000_000, 1.0, -0.5, published_ns=1_025_000_000),
                            received_ns=1_026_000_000)


def test_render_delay_is_fixed_after_warmup_just_above_the_sample_gap() -> None:
    timeline = MeasuredPoseTimeline(warmup_samples=40, margin_ns=5_000_000)
    _feed(timeline, 39, 40_000_000)
    assert timeline.delay_ns is None
    _feed(timeline, 2, 40_000_000, start_ns=1_000_000_000 + 39 * 40_000_000)
    assert timeline.delay_ns == 45_000_000
    assert timeline.sample_hz["pan"] == pytest.approx(25.0)


def test_faster_pose_feedback_gives_a_smaller_render_delay() -> None:
    timeline = MeasuredPoseTimeline(warmup_samples=40, margin_ns=5_000_000)
    _feed(timeline, 60, 10_000_000)
    assert timeline.delay_ns == 15_000_000


def test_pose_is_interpolated_only_when_bracketed_on_both_axes() -> None:
    timeline = MeasuredPoseTimeline(warmup_samples=3)
    _feed(timeline, 5, 40_000_000, rate=0.5)
    first_local = 1_001_000_000  # receipt - (published - measured) = measured + 1 ms
    assert timeline.pose_at(first_local + 60_000_000)[0] == pytest.approx(0.5 * 0.06, abs=1e-9)
    assert timeline.pose_at(first_local - 1) is None
    assert timeline.pose_at(first_local + 5 * 40_000_000) is None


def test_states_without_home_or_sample_times_are_ignored() -> None:
    timeline = MeasuredPoseTimeline(warmup_samples=3)
    assert not timeline.add(CamState(frame_id=1, src_ts_ms=0, state_monotonic_ns=5, pan=0.0, tilt=0.0,
                                     pan_sample_monotonic_ns=4, tilt_sample_monotonic_ns=4),
                            received_ns=10)
    assert not timeline.add(CamState(frame_id=1, src_ts_ms=0, state_monotonic_ns=5, pan=0.0, tilt=0.0,
                                     home_pan=0.0, home_tilt=0.0), received_ns=10)


def test_sim_perception_endpoint_accepts_configured_pc_lan_address() -> None:
    assert require_simulation_perception_endpoint(
        "tcp://192.168.0.1:5574", "test", "192.168.0.1"
    ) == "tcp://192.168.0.1:5574"


@pytest.mark.parametrize(
    "endpoint", ["tcp://0.0.0.0:5574", "tcp://192.168.0.5:5574", "udp://192.168.0.1:5574"]
)
def test_sim_perception_endpoint_rejects_unconfigured_or_wildcard_address(endpoint: str) -> None:
    with pytest.raises(ValueError):
        require_simulation_perception_endpoint(endpoint, "test", "192.168.0.1")
