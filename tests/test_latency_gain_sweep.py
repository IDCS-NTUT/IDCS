from __future__ import annotations

import math
from pathlib import Path

import pytest

from common.gimbal.gray_box import load_qualified_plants
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


def test_f6_quantum_leaves_a_deadband_of_one_rpm_over_kp(yaw) -> None:
    kp = 8.0
    result = simulate(yaw, Gains(kp), STEP, LatencySpec(0.0, 0.0), LoopConfig())
    settled = result["metrics"]["final_window_mean_abs_error_rad"]
    one_rpm = 2.0 * math.pi / 60.0
    assert 0.0 < settled <= one_rpm / kp


def test_gain_too_small_to_reach_one_rpm_never_moves(yaw) -> None:
    result = simulate(yaw, Gains(1.0), STEP, LatencySpec(0.0, 0.0), LoopConfig())
    assert result["metrics"]["final_window_mean_abs_error_rad"] == pytest.approx(0.06, abs=1e-6)
    assert all(command == 0.0 for command in result["commands"])


def test_applied_commands_are_whole_rpm_multiples(yaw) -> None:
    result = simulate(yaw, Gains(8), STEP, LatencySpec(0.03), LoopConfig())
    one_rpm = 2.0 * math.pi / 60.0
    assert all(abs(c / one_rpm - round(c / one_rpm)) < 1e-9 for c in result["commands"])


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
    result = simulate(yaw, Gains(30), big, LatencySpec(0.0), LoopConfig(quantize_f6=False))
    assert max(abs(c) for c in result["commands"]) <= 0.2 + 1e-12


def test_one_rpm_gain_scales_only_the_one_rpm_level(yaw) -> None:
    ramp = Scenario("ramp", 3.0, lambda t: 0.1 * t)  # demands ~1 RPM
    nominal = simulate(yaw, Gains(8), ramp, LatencySpec(0.0, 0.0), LoopConfig())
    fast = simulate(yaw, Gains(8), ramp, LatencySpec(0.0, 0.0), LoopConfig(f6_one_rpm_gain=2.4))
    one_rpm = 2.0 * math.pi / 60.0
    assert {round(abs(c) / one_rpm, 9) for c in fast["commands"]} <= {0.0, 2.4}
    assert {round(abs(c) / one_rpm, 9) for c in nominal["commands"]} <= {0.0, 1.0}


def test_measured_f6_table_matches_bench_probe() -> None:
    from tools.latency_gain_sweep import f6_measured_rad_s
    assert f6_measured_rad_s(0) == 0.0
    assert f6_measured_rad_s(1) == pytest.approx(114 * 2 * math.pi / 3200)
    assert f6_measured_rad_s(-3) == pytest.approx(-228 * 2 * math.pi / 3200)
    assert f6_measured_rad_s(3) < f6_measured_rad_s(4) < f6_measured_rad_s(5)
