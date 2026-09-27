from __future__ import annotations

import math
from pathlib import Path

import pytest

from common.gimbal.gray_box import load_qualified_plants
from common.gimbal.mks_servo42_rs485 import f6_level_speed_rad_s, min_f6_speed_rad_s
from tools.latency_gain_sweep import (
    Gains,
    LatencySpec,
    LoopConfig,
    Scenario,
    search_scenarios,
    simulate,
    suite_cost,
)

FIT = Path(__file__).resolve().parents[1] / "artifacts/gimbal_fit/controller_sysid_pid_range_wire_20260914"


@pytest.fixture(scope="module")
def yaw():
    return load_qualified_plants(FIT / "fit_report.json", FIT / "independent_validation_report.json")["yaw"]


STEP = Scenario("step", 4.0, lambda t: 0.06 if t >= 0.5 else 0.0, 0.5, 0.06)


def test_unquantized_zero_latency_step_converges(yaw) -> None:
    result = simulate(yaw, Gains(8), STEP, LatencySpec(0.0, 0.0), LoopConfig(quantize_f6=False))
    assert result["metrics"]["final_window_mean_abs_error_rad"] < 1e-4


def test_f6_model_leaves_a_deadband_of_half_the_slowest_speed_over_kp(yaw) -> None:
    kp = 8.0
    result = simulate(yaw, Gains(kp), STEP, LatencySpec(0.0, 0.0), LoopConfig())
    settled = result["metrics"]["final_window_mean_abs_error_rad"]
    # Requests below half the slowest measured speed (0.224 rad/s) encode as zero.
    assert 0.0 < settled <= 0.5 * min_f6_speed_rad_s() / kp


def test_gain_too_small_to_reach_one_rpm_never_moves(yaw) -> None:
    result = simulate(yaw, Gains(1.0), STEP, LatencySpec(0.0, 0.0), LoopConfig())
    assert result["metrics"]["final_window_mean_abs_error_rad"] == pytest.approx(0.06, abs=1e-6)
    assert all(command == 0.0 for command in result["commands"])


def test_applied_commands_are_measured_f6_level_speeds_within_cap(yaw) -> None:
    loop = LoopConfig(rate_limit_rad_s=0.8)
    result = simulate(yaw, Gains(8), STEP, LatencySpec(0.03), loop)
    speeds = {round(abs(f6_level_speed_rad_s(level)), 12) for level in range(0, 12)}
    assert all(round(abs(c), 12) in speeds for c in result["commands"])
    assert max(abs(c) for c in result["commands"]) <= 0.8


def test_latency_delays_the_first_response(yaw) -> None:
    loop = LoopConfig(quantize_f6=False)
    fast = simulate(yaw, Gains(8), STEP, LatencySpec(0.0, 0.0), loop)
    slow = simulate(yaw, Gains(8), STEP, LatencySpec(0.2, 0.0), loop)

    def first_move_s(result):
        moved = abs(result["error_rad"] - 0.06) > 1e-3
        return result["time_s"][(result["time_s"] >= 0.5) & moved][0]

    assert first_move_s(slow) - first_move_s(fast) == pytest.approx(0.2, abs=0.04)


def test_high_gain_is_penalised_more_at_high_latency(yaw) -> None:
    loop = LoopConfig(quantize_f6=False)
    scenarios = search_scenarios()

    def ratio(latency_s):
        high = suite_cost(yaw, Gains(16), scenarios, LatencySpec(latency_s), loop)[0]
        low = suite_cost(yaw, Gains(4), scenarios, LatencySpec(latency_s), loop)[0]
        return high / low

    assert ratio(0.2) > ratio(0.0)


def test_simulation_is_reproducible_for_a_seed(yaw) -> None:
    first = suite_cost(yaw, Gains(6, 0.5, 0.02), search_scenarios(), LatencySpec(0.06, 0.004, seed=3), LoopConfig())[0]
    second = suite_cost(yaw, Gains(6, 0.5, 0.02), search_scenarios(), LatencySpec(0.06, 0.004, seed=3), LoopConfig())[0]
    assert first == second


def test_rate_limit_is_respected(yaw) -> None:
    big = Scenario("big", 3.0, lambda t: 0.5 if t > 0.1 else 0.0)
    result = simulate(yaw, Gains(30), big, LatencySpec(0.0), LoopConfig(quantize_f6=False, rate_limit_rad_s=0.2))
    assert max(abs(c) for c in result["commands"]) <= 0.2 + 1e-12
    quantized = simulate(yaw, Gains(30), big, LatencySpec(0.0), LoopConfig(rate_limit_rad_s=0.5))
    assert max(abs(c) for c in quantized["commands"]) <= 0.5 + 1e-12  # cap on actual F6 speed


def test_rate_feedforward_reduces_ramp_tracking_error(yaw) -> None:
    from tools.feedforward_sweep import fast_scenarios
    from tools.latency_gain_sweep import FeedforwardConfig
    ramp = next(s for s in fast_scenarios() if s.name == "ramp_0p3")
    loop = LoopConfig(fps=60.0, rate_limit_rad_s=1.0, accel_limit_rad_s2=10.0, quantize_f6=False)
    plain = simulate(yaw, Gains(10), ramp, LatencySpec(0.06), loop)["metrics"]["rms_error_rad"]
    with_ff = simulate(yaw, Gains(10), ramp, LatencySpec(0.06), loop,
                       FeedforwardConfig(rate_scale=1.0, accel_sigma_rad_s2=2.0))["metrics"]["rms_error_rad"]
    assert with_ff < 0.7 * plain


def test_fast_scenarios_have_finite_acceleration() -> None:
    import numpy as np
    from tools.feedforward_sweep import fast_scenarios
    for scenario in fast_scenarios():
        t = np.arange(0.0, scenario.duration_s, 0.001)
        position = np.array([scenario.target(x) for x in t])
        acceleration = np.diff(position, 2) / 0.001 ** 2
        assert np.abs(acceleration).max() < 10.0, scenario.name


def test_unreachable_rate_cap_is_refused_when_quantizing() -> None:
    with pytest.raises(ValueError, match="slowest F6 speed"):
        LoopConfig(rate_limit_rad_s=0.2)
    LoopConfig(rate_limit_rad_s=0.2, quantize_f6=False)  # ideal actuator: any cap
