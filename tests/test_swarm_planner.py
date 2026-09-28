import math
import unittest

from common.control import (
    AxisPair,
    ControlConfig,
    LaserAimingControlConfig,
    PidConfig,
    SwarmEvalConfig,
    SwarmTimingConfig,
)
from common.threat_calc import (
    compute_breakthrough_time,
    compute_radial_closing_speed,
    compute_zone_feature_vector,
    estimate_time_to_engage,
)
from jetson.swarm_planner import (
    PlannerTarget,
    SwarmPlannerSettings,
    evaluate_swarm_targets,
)
from tools.benchmark_swarm_planner import _summarize


class _DummyPub:
    def __init__(self) -> None:
        self.sent = []

    def send_string(self, payload, flags=0):
        self.sent.append((payload, flags))


def _planner_settings() -> SwarmPlannerSettings:
    return SwarmPlannerSettings(
        yaw_rate_limit_rad_s=1.0,
        pitch_rate_limit_rad_s=1.0,
        yaw_accel_limit_rad_s2=2.0,
        pitch_accel_limit_rad_s2=2.0,
        max_engage_distance_m=None,
        exact_search_limit=6,
        beam_width=8,
        switch_absolute_damage_gain=0.25,
        switch_relative_improvement=0.10,
        timing=SwarmTimingConfig(),
    )


class ThreatTimingTests(unittest.TestCase):
    def test_radial_closing_speed_for_direct_approach(self) -> None:
        closing = compute_radial_closing_speed((10.0, 0.0), (-5.0, 0.0), (0.0, 0.0))
        self.assertAlmostEqual(closing, 5.0)

    def test_radial_closing_speed_for_lateral_motion(self) -> None:
        closing = compute_radial_closing_speed((10.0, 0.0), (0.0, 4.0), (0.0, 0.0))
        self.assertAlmostEqual(closing, 0.0)

    def test_breakthrough_time_is_infinite_when_not_closing(self) -> None:
        self.assertTrue(math.isinf(compute_breakthrough_time(10.0, 0.0)))

    def test_time_to_engage_penalizes_recover_low_conf_and_missing_range(self) -> None:
        baseline = estimate_time_to_engage(
            distance_m=10.0,
            yaw_error_rad=0.15,
            pitch_error_rad=0.05,
            yaw_rate_limit_rad_s=1.0,
            pitch_rate_limit_rad_s=1.0,
            yaw_accel_limit_rad_s2=2.0,
            pitch_accel_limit_rad_s2=2.0,
            current_yaw_rate_rad_s=0.0,
            current_pitch_rate_rad_s=0.0,
            tracker_mode="track",
            confidence=0.95,
            track_observations=5,
            range_source="average",
            predictive_only=False,
            base_track_lock_s=0.15,
            search_track_lock_s=0.25,
            recover_track_lock_s=0.40,
            low_conf_threshold=0.60,
            low_conf_penalty_s=0.20,
            min_track_observations=3,
            low_continuity_penalty_s=0.08,
            missing_range_penalty_s=0.10,
            predictive_penalty_s=0.20,
            effect_time_s=0.25,
            effect_distance_scale_s_per_m=0.01,
            confirm_time_s=0.10,
            confirm_distance_scale_s_per_m=0.004,
            settle_margin_s=0.05,
        )
        degraded = estimate_time_to_engage(
            distance_m=10.0,
            yaw_error_rad=0.15,
            pitch_error_rad=0.05,
            yaw_rate_limit_rad_s=1.0,
            pitch_rate_limit_rad_s=1.0,
            yaw_accel_limit_rad_s2=2.0,
            pitch_accel_limit_rad_s2=2.0,
            current_yaw_rate_rad_s=0.0,
            current_pitch_rate_rad_s=0.0,
            tracker_mode="recover",
            confidence=0.40,
            track_observations=1,
            range_source=None,
            predictive_only=True,
            base_track_lock_s=0.15,
            search_track_lock_s=0.25,
            recover_track_lock_s=0.40,
            low_conf_threshold=0.60,
            low_conf_penalty_s=0.20,
            min_track_observations=3,
            low_continuity_penalty_s=0.08,
            missing_range_penalty_s=0.10,
            predictive_penalty_s=0.20,
            effect_time_s=0.25,
            effect_distance_scale_s_per_m=0.01,
            confirm_time_s=0.10,
            confirm_distance_scale_s_per_m=0.004,
            settle_margin_s=0.05,
        )
        self.assertGreater(degraded, baseline)

    def test_time_to_engage_increases_with_distance(self) -> None:
        near_time = estimate_time_to_engage(
            distance_m=8.0,
            yaw_error_rad=0.15,
            pitch_error_rad=0.05,
            yaw_rate_limit_rad_s=1.0,
            pitch_rate_limit_rad_s=1.0,
            yaw_accel_limit_rad_s2=2.0,
            pitch_accel_limit_rad_s2=2.0,
            current_yaw_rate_rad_s=0.0,
            current_pitch_rate_rad_s=0.0,
            tracker_mode="track",
            confidence=0.95,
            track_observations=5,
            range_source="average",
            predictive_only=False,
            base_track_lock_s=0.15,
            search_track_lock_s=0.25,
            recover_track_lock_s=0.40,
            low_conf_threshold=0.60,
            low_conf_penalty_s=0.20,
            min_track_observations=3,
            low_continuity_penalty_s=0.08,
            missing_range_penalty_s=0.10,
            predictive_penalty_s=0.20,
            effect_time_s=0.25,
            effect_distance_scale_s_per_m=0.01,
            confirm_time_s=0.10,
            confirm_distance_scale_s_per_m=0.004,
            settle_margin_s=0.05,
        )
        far_time = estimate_time_to_engage(
            distance_m=28.0,
            yaw_error_rad=0.15,
            pitch_error_rad=0.05,
            yaw_rate_limit_rad_s=1.0,
            pitch_rate_limit_rad_s=1.0,
            yaw_accel_limit_rad_s2=2.0,
            pitch_accel_limit_rad_s2=2.0,
            current_yaw_rate_rad_s=0.0,
            current_pitch_rate_rad_s=0.0,
            tracker_mode="track",
            confidence=0.95,
            track_observations=5,
            range_source="average",
            predictive_only=False,
            base_track_lock_s=0.15,
            search_track_lock_s=0.25,
            recover_track_lock_s=0.40,
            low_conf_threshold=0.60,
            low_conf_penalty_s=0.20,
            min_track_observations=3,
            low_continuity_penalty_s=0.08,
            missing_range_penalty_s=0.10,
            predictive_penalty_s=0.20,
            effect_time_s=0.25,
            effect_distance_scale_s_per_m=0.01,
            confirm_time_s=0.10,
            confirm_distance_scale_s_per_m=0.004,
            settle_margin_s=0.05,
        )
        self.assertGreater(far_time, near_time)

    def test_zone_feature_vector_marks_nested_zones(self) -> None:
        features = compute_zone_feature_vector(
            4.0,
            {"warning": 20.0, "restricted": 10.0, "critical": 5.0},
        )
        self.assertEqual(features[:3], (1.0, 1.0, 1.0))
        self.assertGreater(features[3], 0.0)


class SwarmPlannerTests(unittest.TestCase):
    def test_benchmark_summary_reports_basic_stats(self) -> None:
        summary = _summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean_ms"], 2.5)
        self.assertAlmostEqual(summary["median_ms"], 2.5)
        self.assertAlmostEqual(summary["max_ms"], 4.0)
        self.assertGreater(summary["p95_ms"], 0.0)

    def test_two_target_order_prefers_imminent_breakthrough(self) -> None:
        decision = evaluate_swarm_targets(
            [
                PlannerTarget(
                    target_id=1,
                    box_index=0,
                    cls="drone",
                    confidence=0.6,
                    damage_weight=1.0,
                    distance_m=4.0,
                    radial_closing_speed_m_s=4.0,
                    yaw_error_rad=0.05,
                    pitch_error_rad=0.02,
                    bbox_area_norm=0.02,
                    track_observations=5,
                    range_source="average",
                    threat_level="threatening",
                    tracker_mode="track",
                ),
                PlannerTarget(
                    target_id=2,
                    box_index=1,
                    cls="drone",
                    confidence=0.95,
                    damage_weight=4.0,
                    distance_m=12.0,
                    radial_closing_speed_m_s=3.0,
                    yaw_error_rad=0.04,
                    pitch_error_rad=0.02,
                    bbox_area_norm=0.02,
                    track_observations=5,
                    range_source="average",
                    threat_level="threatening",
                    tracker_mode="track",
                ),
            ],
            _planner_settings(),
        )
        self.assertEqual(decision.chosen_target_id, 1)

    def test_three_target_order_balances_damage_and_timing(self) -> None:
        decision = evaluate_swarm_targets(
            [
                PlannerTarget(
                    target_id=1,
                    box_index=0,
                    cls="drone",
                    confidence=0.7,
                    damage_weight=1.0,
                    distance_m=3.8,
                    radial_closing_speed_m_s=4.0,
                    yaw_error_rad=0.03,
                    pitch_error_rad=0.01,
                    bbox_area_norm=0.02,
                    track_observations=4,
                    range_source="average",
                    threat_level="threatening",
                    tracker_mode="track",
                ),
                PlannerTarget(
                    target_id=2,
                    box_index=1,
                    cls="munition",
                    confidence=0.9,
                    damage_weight=5.0,
                    distance_m=5.4,
                    radial_closing_speed_m_s=4.0,
                    yaw_error_rad=0.05,
                    pitch_error_rad=0.01,
                    bbox_area_norm=0.02,
                    track_observations=4,
                    range_source="average",
                    threat_level="threatening",
                    tracker_mode="track",
                ),
                PlannerTarget(
                    target_id=3,
                    box_index=2,
                    cls="drone",
                    confidence=0.8,
                    damage_weight=1.0,
                    distance_m=18.0,
                    radial_closing_speed_m_s=4.0,
                    yaw_error_rad=0.02,
                    pitch_error_rad=0.01,
                    bbox_area_norm=0.02,
                    track_observations=4,
                    range_source="average",
                    threat_level="threatening",
                    tracker_mode="track",
                ),
            ],
            _planner_settings(),
        )
        # The damage-aware objective prioritizes the high-consequence
        # munition when its breakthrough timing is also near-term.
        self.assertEqual(decision.chosen_target_id, 2)

    def test_hysteresis_keeps_previous_target_when_gain_is_small(self) -> None:
        decision = evaluate_swarm_targets(
            [
                PlannerTarget(
                    target_id=1,
                    box_index=0,
                    cls="drone",
                    confidence=0.8,
                    damage_weight=2.0,
                    distance_m=10.0,
                    radial_closing_speed_m_s=4.0,
                    yaw_error_rad=0.04,
                    pitch_error_rad=0.02,
                    bbox_area_norm=0.02,
                    track_observations=5,
                    range_source="average",
                    threat_level="threatening",
                    tracker_mode="track",
                ),
                PlannerTarget(
                    target_id=2,
                    box_index=1,
                    cls="drone",
                    confidence=0.8,
                    damage_weight=2.0,
                    distance_m=10.0,
                    radial_closing_speed_m_s=4.0,
                    yaw_error_rad=0.04,
                    pitch_error_rad=0.02,
                    bbox_area_norm=0.02,
                    track_observations=5,
                    range_source="average",
                    threat_level="threatening",
                    tracker_mode="track",
                ),
            ],
            _planner_settings(),
            previous_target_id=2,
        )
        self.assertEqual(decision.chosen_target_id, 2)

    def test_targets_outside_engage_distance_are_ranked_but_not_selected(self) -> None:
        settings = SwarmPlannerSettings(
            yaw_rate_limit_rad_s=1.0,
            pitch_rate_limit_rad_s=1.0,
            yaw_accel_limit_rad_s2=2.0,
            pitch_accel_limit_rad_s2=2.0,
            max_engage_distance_m=10.0,
            exact_search_limit=6,
            beam_width=8,
            switch_absolute_damage_gain=0.25,
            switch_relative_improvement=0.10,
            timing=SwarmTimingConfig(),
        )
        decision = evaluate_swarm_targets(
            [
                PlannerTarget(
                    target_id=1,
                    box_index=0,
                    cls="drone",
                    confidence=0.8,
                    damage_weight=3.0,
                    distance_m=18.0,
                    radial_closing_speed_m_s=4.0,
                    yaw_error_rad=0.03,
                    pitch_error_rad=0.01,
                    bbox_area_norm=0.02,
                    track_observations=4,
                    range_source="average",
                    threat_level="threatening",
                    tracker_mode="track",
                ),
                PlannerTarget(
                    target_id=2,
                    box_index=1,
                    cls="drone",
                    confidence=0.8,
                    damage_weight=2.0,
                    distance_m=16.0,
                    radial_closing_speed_m_s=4.0,
                    yaw_error_rad=0.05,
                    pitch_error_rad=0.01,
                    bbox_area_norm=0.02,
                    track_observations=4,
                    range_source="average",
                    threat_level="threatening",
                    tracker_mode="track",
                ),
            ],
            settings,
        )
        self.assertIsNone(decision.chosen_target_id)
        self.assertEqual(len(decision.candidate_results), 2)
        self.assertFalse(any(item.engageable_now for item in decision.candidate_results))


def _close_target(target_id: int, distance_m: float, closing: float, yaw: float = 0.0) -> PlannerTarget:
    return PlannerTarget(
        target_id=target_id, box_index=target_id, cls="drone", confidence=0.8,
        damage_weight=2.0, distance_m=distance_m, radial_closing_speed_m_s=closing,
        yaw_error_rad=yaw, pitch_error_rad=0.0, bbox_area_norm=0.02,
        track_observations=4, range_source="width", threat_level="threatening",
        tracker_mode="track",
    )


class PlannerLastChanceTests(unittest.TestCase):
    def test_engages_the_best_margin_when_none_can_be_engaged_in_time(self) -> None:
        # Both break through before the estimated engagement time.
        targets = [_close_target(1, 0.6, 3.0, yaw=0.5), _close_target(2, 0.9, 3.0)]
        decision = evaluate_swarm_targets(targets, _planner_settings())
        self.assertFalse(any(item.engageable_now for item in decision.candidate_results))
        self.assertEqual(decision.chosen_target_id, 2)  # later breakthrough, no slew

    def test_keeps_the_previous_target_unless_another_is_clearly_better(self) -> None:
        targets = [_close_target(1, 0.85, 3.0), _close_target(2, 0.9, 3.0)]
        decision = evaluate_swarm_targets(targets, _planner_settings(), previous_target_id=1)
        self.assertEqual(decision.chosen_target_id, 1)


class ClosingSpeedFitTests(unittest.TestCase):
    def test_fit_rejects_range_noise_and_needs_a_window(self) -> None:
        import random
        from jetson.swarm_planner import _closing_speed_from_samples

        rng = random.Random(3)
        samples = tuple((0.1 * i, 3.0 - 1.0 * 0.1 * i + rng.uniform(-0.3, 0.3)) for i in range(10))
        self.assertAlmostEqual(_closing_speed_from_samples(samples), 1.0, delta=0.35)
        # One-step differences of the same data swing by several m/s.
        steps = [(a[1] - b[1]) / 0.1 for a, b in zip(samples, samples[1:])]
        self.assertGreater(max(steps) - min(steps), 4.0)
        self.assertIsNone(_closing_speed_from_samples(samples[:2]))
        receding = tuple((0.1 * i, 3.0 + 0.1 * i) for i in range(5))
        self.assertEqual(_closing_speed_from_samples(receding), 0.0)


class SelectionCarryOverTests(unittest.TestCase):
    def test_vanished_selection_continues_on_the_track_where_it_was(self) -> None:
        from jetson.swarm_planner import SwarmPlannerRuntime, _RuntimeTrackState

        runtime = object.__new__(SwarmPlannerRuntime)
        runtime._track_history = {
            7: _RuntimeTrackState(last_seen_time_s=1.0, last_center_uv=(410.0, 300.0), last_distance_m=2.0),
            8: _RuntimeTrackState(last_seen_time_s=1.0, last_center_uv=(900.0, 200.0), last_distance_m=2.0),
        }
        runtime._last_choice = (3, (400.0, 310.0), 0.9)
        targets = [_close_target(7, 2.0, 1.0), _close_target(8, 2.0, 1.0)]
        self.assertEqual(runtime._carry_over_previous(3, targets, 1.0), 7)
        self.assertEqual(runtime._carry_over_previous(3, targets, 1.6), 3)  # too old
        self.assertEqual(runtime._carry_over_previous(8, targets, 1.0), 8)  # still present
