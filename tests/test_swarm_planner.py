import math
import unittest

from common.control import (
    AxisPair,
    ControlConfig,
    LaserAimingControlConfig,
    SwarmEvalConfig,
    SwarmTimingConfig,
)
from common.schemas import Box, DetectionMsg
from common.threat_calc import (
    compute_breakthrough_time,
    compute_radial_closing_speed,
    compute_zone_feature_vector,
    estimate_time_to_engage,
)
from jetson.controller import ControlLoop
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
        self.assertEqual(decision.chosen_target_id, 1)

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


class SwarmControllerIntegrationTests(unittest.TestCase):
    def test_control_loop_uses_swarm_planner_selector(self) -> None:
        cfg = ControlConfig(
            mode="rate",
            loop_hz=30.0,
            fx_px=800.0,
            fy_px=820.0,
            cx_px=640.0,
            cy_px=360.0,
            aim_mode="camera_center",
            kp=AxisPair(0.0, 0.0),
            kd=AxisPair(0.0, 0.0),
            ki=AxisPair(0.0, 0.0),
            rate_limits=AxisPair(1.0, 1.0),
            accel_limits=AxisPair(2.0, 2.0),
            deadband_px=0.0,
            smooth_px_alpha=0.0,
            lost_target_timeout_ms=100,
            reinit_on_lost=True,
            target_selector="swarm_planner",
            yaw_sign=1.0,
            pitch_sign=-1.0,
            frame_size=(1280, 720),
            fov_deg=None,
            laser=LaserAimingControlConfig(
                tolerance_px=3.0,
                use_range="known_size",
                default_distance_m=25.0,
            ),
            swarm_eval=SwarmEvalConfig(enabled=True),
        )
        loop = ControlLoop(cfg, _DummyPub())
        msg = DetectionMsg(
            frame_id=1,
            src_ts_ms=1000,
            rx_ts_ms=1010,
            infer_ts_ms=1020,
            img_w=1280,
            img_h=720,
            tracker_mode="track",
            boxes=[
                Box(
                    x=0.40,
                    y=0.40,
                    w=0.08,
                    h=0.08,
                    conf=0.55,
                    cls="drone",
                    track_id=10,
                    distance_m=5.0,
                    distance_src="average",
                    threat_level="threatening",
                ),
                Box(
                    x=0.52,
                    y=0.42,
                    w=0.08,
                    h=0.08,
                    conf=0.95,
                    cls="drone",
                    track_id=20,
                    distance_m=18.0,
                    distance_src="average",
                    threat_level="threatening",
                ),
            ],
        )

        loop.update_detection(msg)

        self.assertEqual(msg.target_track_id, 10)
        self.assertEqual(msg.target_idx, 0)
        self.assertIsNotNone(msg.swarm_expected_total_damage)
        self.assertIsNotNone(msg.boxes[0].priority_score)
        self.assertIsNotNone(msg.boxes[0].engagement_rank)
        self.assertEqual(msg.boxes[0].threat_level, "threatening")

    def test_async_learned_mode_falls_back_cleanly_without_result(self) -> None:
        cfg = ControlConfig(
            mode="rate",
            loop_hz=30.0,
            fx_px=800.0,
            fy_px=820.0,
            cx_px=640.0,
            cy_px=360.0,
            aim_mode="camera_center",
            kp=AxisPair(0.0, 0.0),
            kd=AxisPair(0.0, 0.0),
            ki=AxisPair(0.0, 0.0),
            rate_limits=AxisPair(1.0, 1.0),
            accel_limits=AxisPair(2.0, 2.0),
            deadband_px=0.0,
            smooth_px_alpha=0.0,
            lost_target_timeout_ms=100,
            reinit_on_lost=True,
            target_selector="swarm_planner",
            yaw_sign=1.0,
            pitch_sign=-1.0,
            frame_size=(1280, 720),
            fov_deg=None,
            laser=LaserAimingControlConfig(
                tolerance_px=3.0,
                use_range="known_size",
                default_distance_m=25.0,
            ),
            swarm_eval=SwarmEvalConfig(enabled=True),
        )
        loop = ControlLoop(cfg, _DummyPub())
        runtime = loop._swarm_planner
        runtime._async_enabled = True
        runtime._learned_selector = object()
        runtime._learned_tensorrt = None
        runtime._get_async_result = lambda target_ids, previous_target_id: None
        runtime._submit_async_request = lambda *args, **kwargs: None

        msg = DetectionMsg(
            frame_id=1,
            src_ts_ms=1000,
            rx_ts_ms=1010,
            infer_ts_ms=1020,
            img_w=1280,
            img_h=720,
            tracker_mode="track",
            boxes=[
                Box(
                    x=0.40,
                    y=0.40,
                    w=0.08,
                    h=0.08,
                    conf=0.55,
                    cls="drone",
                    track_id=10,
                    distance_m=5.0,
                    distance_src="average",
                    threat_level="threatening",
                ),
                Box(
                    x=0.52,
                    y=0.42,
                    w=0.08,
                    h=0.08,
                    conf=0.95,
                    cls="drone",
                    track_id=20,
                    distance_m=18.0,
                    distance_src="average",
                    threat_level="threatening",
                ),
            ],
        )

        loop.update_detection(msg)

        self.assertEqual(msg.target_track_id, 10)
        self.assertEqual(msg.target_idx, 0)

    def test_track_mode_keeps_existing_target_while_swarm_updates_background(self) -> None:
        cfg = ControlConfig(
            mode="rate",
            loop_hz=30.0,
            fx_px=800.0,
            fy_px=820.0,
            cx_px=640.0,
            cy_px=360.0,
            aim_mode="camera_center",
            kp=AxisPair(0.0, 0.0),
            kd=AxisPair(0.0, 0.0),
            ki=AxisPair(0.0, 0.0),
            rate_limits=AxisPair(1.0, 1.0),
            accel_limits=AxisPair(2.0, 2.0),
            deadband_px=0.0,
            smooth_px_alpha=0.0,
            lost_target_timeout_ms=100,
            reinit_on_lost=True,
            target_selector="swarm_planner",
            yaw_sign=1.0,
            pitch_sign=-1.0,
            frame_size=(1280, 720),
            fov_deg=None,
            laser=LaserAimingControlConfig(
                tolerance_px=3.0,
                use_range="known_size",
                default_distance_m=25.0,
            ),
            swarm_eval=SwarmEvalConfig(enabled=True),
        )
        loop = ControlLoop(cfg, _DummyPub())
        first = DetectionMsg(
            frame_id=1,
            src_ts_ms=1000,
            rx_ts_ms=1010,
            infer_ts_ms=1020,
            img_w=1280,
            img_h=720,
            tracker_mode="track",
            boxes=[
                Box(
                    x=0.35,
                    y=0.40,
                    w=0.06,
                    h=0.06,
                    conf=0.85,
                    cls="drone",
                    track_id=10,
                    distance_m=40.0,
                    distance_src="average",
                    threat_level="suspicious",
                ),
            ],
        )

        loop.update_detection(first)

        self.assertEqual(first.target_track_id, 10)
        second = DetectionMsg(
            frame_id=2,
            src_ts_ms=1033,
            rx_ts_ms=1043,
            infer_ts_ms=1053,
            img_w=1280,
            img_h=720,
            tracker_mode="track",
            boxes=[
                Box(
                    x=0.36,
                    y=0.41,
                    w=0.06,
                    h=0.06,
                    conf=0.86,
                    cls="drone",
                    track_id=10,
                    distance_m=38.0,
                    distance_src="average",
                    threat_level="suspicious",
                ),
                Box(
                    x=0.52,
                    y=0.44,
                    w=0.12,
                    h=0.12,
                    conf=0.98,
                    cls="drone",
                    track_id=20,
                    distance_m=4.0,
                    distance_src="average",
                    threat_level="threatening",
                ),
            ],
        )

        loop.update_detection(second)

        self.assertEqual(second.target_track_id, 10)
        self.assertEqual(second.target_idx, 0)
        self.assertEqual(second.boxes[1].engagement_rank, 1)
        self.assertIsNotNone(second.boxes[1].priority_score)

    def test_track_mode_releases_lock_when_existing_target_disappears(self) -> None:
        cfg = ControlConfig(
            mode="rate",
            loop_hz=30.0,
            fx_px=800.0,
            fy_px=820.0,
            cx_px=640.0,
            cy_px=360.0,
            aim_mode="camera_center",
            kp=AxisPair(0.0, 0.0),
            kd=AxisPair(0.0, 0.0),
            ki=AxisPair(0.0, 0.0),
            rate_limits=AxisPair(1.0, 1.0),
            accel_limits=AxisPair(2.0, 2.0),
            deadband_px=0.0,
            smooth_px_alpha=0.0,
            lost_target_timeout_ms=100,
            reinit_on_lost=True,
            target_selector="swarm_planner",
            yaw_sign=1.0,
            pitch_sign=-1.0,
            frame_size=(1280, 720),
            fov_deg=None,
            laser=LaserAimingControlConfig(
                tolerance_px=3.0,
                use_range="known_size",
                default_distance_m=25.0,
            ),
            swarm_eval=SwarmEvalConfig(enabled=True),
        )
        loop = ControlLoop(cfg, _DummyPub())
        first = DetectionMsg(
            frame_id=1,
            src_ts_ms=1000,
            rx_ts_ms=1010,
            infer_ts_ms=1020,
            img_w=1280,
            img_h=720,
            tracker_mode="track",
            boxes=[
                Box(
                    x=0.35,
                    y=0.40,
                    w=0.06,
                    h=0.06,
                    conf=0.85,
                    cls="drone",
                    track_id=10,
                    distance_m=40.0,
                    distance_src="average",
                    threat_level="suspicious",
                ),
            ],
        )
        second = DetectionMsg(
            frame_id=2,
            src_ts_ms=1033,
            rx_ts_ms=1043,
            infer_ts_ms=1053,
            img_w=1280,
            img_h=720,
            tracker_mode="track",
            boxes=[
                Box(
                    x=0.52,
                    y=0.44,
                    w=0.12,
                    h=0.12,
                    conf=0.98,
                    cls="drone",
                    track_id=20,
                    distance_m=4.0,
                    distance_src="average",
                    threat_level="threatening",
                ),
            ],
        )

        loop.update_detection(first)
        loop.update_detection(second)

        self.assertEqual(second.target_track_id, 20)
        self.assertEqual(second.target_idx, 0)

    def test_learned_encoder_marks_previous_target_features(self) -> None:
        cfg = ControlConfig(
            mode="rate",
            loop_hz=30.0,
            fx_px=800.0,
            fy_px=820.0,
            cx_px=640.0,
            cy_px=360.0,
            aim_mode="camera_center",
            kp=AxisPair(0.0, 0.0),
            kd=AxisPair(0.0, 0.0),
            ki=AxisPair(0.0, 0.0),
            rate_limits=AxisPair(1.0, 1.0),
            accel_limits=AxisPair(2.0, 2.0),
            deadband_px=0.0,
            smooth_px_alpha=0.0,
            lost_target_timeout_ms=100,
            reinit_on_lost=True,
            target_selector="swarm_planner",
            yaw_sign=1.0,
            pitch_sign=-1.0,
            frame_size=(1280, 720),
            fov_deg=None,
            laser=LaserAimingControlConfig(
                tolerance_px=3.0,
                use_range="known_size",
                default_distance_m=25.0,
            ),
            swarm_eval=SwarmEvalConfig(enabled=True),
        )
        loop = ControlLoop(cfg, _DummyPub())
        runtime = loop._swarm_planner
        runtime._learned_max_targets = 3
        runtime._learned_target_feature_size = 21
        runtime._learned_global_feature_size = 7
        targets = [
            PlannerTarget(
                target_id=10,
                box_index=0,
                cls="drone",
                confidence=0.9,
                damage_weight=2.0,
                distance_m=20.0,
                radial_closing_speed_m_s=2.0,
                yaw_error_rad=0.1,
                pitch_error_rad=0.0,
                bbox_area_norm=0.01,
                track_observations=3,
            ),
            PlannerTarget(
                target_id=20,
                box_index=1,
                cls="drone",
                confidence=0.8,
                damage_weight=2.5,
                distance_m=18.0,
                radial_closing_speed_m_s=2.5,
                yaw_error_rad=-0.1,
                pitch_error_rad=0.0,
                bbox_area_norm=0.02,
                track_observations=4,
            ),
        ]
        candidate_results = runtime._build_model_candidate_results(
            targets,
            current_yaw_rate_rad_s=0.0,
            current_pitch_rate_rad_s=0.0,
        )

        target_features, _, _ = runtime._encode_model_inputs(
            targets,
            candidate_results,
            previous_target_id=20,
        )

        self.assertEqual(target_features.shape, (1, 3, 21))
        self.assertEqual(float(target_features[0, 0, 19]), 0.0)
        self.assertEqual(float(target_features[0, 1, 19]), 1.0)
        self.assertEqual(float(target_features[0, 0, 20]), 1.0)
        self.assertEqual(float(target_features[0, 1, 20]), 1.0)


if __name__ == "__main__":
    unittest.main()
