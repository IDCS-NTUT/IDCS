from __future__ import annotations

from jetson.control_v3.clock_poller import ClockPoller
from jetson.control_v3.clock_watchdog import ClockWatchdogConfig
from jetson.control_v3.timing import ClockBounds


def _sample(at_ns: int, *, width_ns: int = 2_000_000, offset_ns: int = 100_000_000) -> ClockBounds:
    return ClockBounds(offset_ns, offset_ns + width_ns, at_ns)


def test_overwide_exchange_holds_then_requalifies_from_two_fresh_samples() -> None:
    poller = ClockPoller("tcp://127.0.0.1:1", ClockWatchdogConfig(
        max_exchange_age_ns=150_000_000,
        configured_max_drift_ppm=1000,
        max_interval_width_ns=15_000_000,
        required_samples=2,
    ))
    poller._observe_sample(_sample(1_000_000_000))
    poller._observe_sample(_sample(1_050_000_000))
    assert poller.bounds(now_ns=1_050_000_000)[1] == "verified_under_configured_policy"
    poller._observe_sample(_sample(1_100_000_000, width_ns=18_000_000))
    assert poller.bounds(now_ns=1_100_000_000)[1] == "clock_warmup"
    poller._observe_sample(_sample(1_150_000_000))
    assert poller.bounds(now_ns=1_150_000_000)[1] == "clock_warmup"
    poller._observe_sample(_sample(1_200_000_000))
    assert poller.bounds(now_ns=1_200_000_000)[1] == "verified_under_configured_policy"
    assert poller.stats()["clock_exchange_uncertainty_exceeded"] == 1
    assert poller.stats()["clock_requalifications"] == 1
    assert poller.stats()["max_exchange_width_ns"] == 18_000_000


def test_clock_drift_contradiction_remains_latched() -> None:
    poller = ClockPoller("tcp://127.0.0.1:1", ClockWatchdogConfig(
        max_exchange_age_ns=150_000_000,
        configured_max_drift_ppm=1000,
        max_interval_width_ns=15_000_000,
        required_samples=2,
    ))
    poller._observe_sample(_sample(1_000_000_000, width_ns=100))
    poller._observe_sample(_sample(1_050_000_000, width_ns=100))
    poller._observe_sample(_sample(1_100_000_000, width_ns=100, offset_ns=102_000_000))
    assert poller.bounds(now_ns=1_100_000_000)[1] == "clock_drift_policy_violated"
    poller._observe_sample(_sample(1_150_000_000))
    assert poller.bounds(now_ns=1_150_000_000)[1] == "clock_drift_policy_violated"
    assert poller.stats().get("clock_requalifications", 0) == 0
