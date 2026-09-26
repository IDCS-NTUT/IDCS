from __future__ import annotations

import pytest

from jetson.control_v3.clock_watchdog import ClockWatchdog, ClockWatchdogConfig
from jetson.control_v3.timing import ClockBounds


def _sample(time_ns: int, offset_ns: int, width_ns: int = 0) -> ClockBounds:
    return ClockBounds(
        offset_min_ns=offset_ns - width_ns // 2,
        offset_max_ns=offset_ns + width_ns // 2,
        observed_jetson_ns=time_ns,
    )


def test_no_supplied_drift_limit_never_yields_controller_bounds() -> None:
    watchdog = ClockWatchdog(ClockWatchdogConfig(max_exchange_age_ns=50_000_000))
    watchdog.observe(_sample(1_000_000_000, 100_000_000))
    watchdog.observe(_sample(1_020_000_000, 100_000_100))
    assert watchdog.bounds(jetson_now_ns=1_030_000_000) == (
        None, "drift_bound_unqualified"
    )


def test_externally_configured_limit_is_age_gated_and_violations_latch() -> None:
    watchdog = ClockWatchdog(ClockWatchdogConfig(
        max_exchange_age_ns=50_000_000,
        configured_max_drift_ppm=10.0,
    ))
    assert watchdog.observe(_sample(1_000_000_000, 100_000_000)) == "observed"
    assert watchdog.bounds(jetson_now_ns=1_010_000_000)[1] == "clock_warmup"
    assert watchdog.observe(_sample(2_000_000_000, 100_005_000)) == "observed"
    bounds, reason = watchdog.bounds(jetson_now_ns=2_020_000_000)
    assert reason == "verified_under_configured_policy"
    assert bounds is not None and bounds.max_drift_ppm == 10.0
    assert watchdog.bounds(jetson_now_ns=2_100_000_000)[1] == "clock_sample_stale"
    assert watchdog.observe(_sample(3_000_000_000, 100_030_000)) == "clock_drift_policy_violated"
    assert watchdog.bounds(jetson_now_ns=3_000_000_000)[1] == "clock_drift_policy_violated"
    assert watchdog.observe(_sample(4_000_000_000, 100_035_000)) == "clock_drift_policy_violated"
    watchdog.reset()
    assert watchdog.bounds(jetson_now_ns=4_000_000_000)[1] == "clock_warmup"


def test_nonmonotonic_exchange_and_future_query_fail_closed() -> None:
    watchdog = ClockWatchdog(ClockWatchdogConfig(
        max_exchange_age_ns=20_000_000,
        configured_max_drift_ppm=100.0,
        required_samples=1,
    ))
    watchdog.observe(_sample(100, 10))
    assert watchdog.bounds(jetson_now_ns=99)[1] == "clock_query_before_sample"
    assert watchdog.observe(_sample(100, 10)) == "clock_exchange_nonmonotonic"
    assert watchdog.bounds(jetson_now_ns=101)[0] is None


@pytest.mark.parametrize("limit", [-1.0, float("nan"), 1_000_000.0])
def test_invalid_drift_policy_rejected(limit: float) -> None:
    with pytest.raises(ValueError, match="drift"):
        ClockWatchdogConfig(max_exchange_age_ns=1, configured_max_drift_ppm=limit)
