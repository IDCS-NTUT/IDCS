import math
import unittest
from types import SimpleNamespace

from common.perception import (
    NormalizedBoxV2,
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
    TargetSelectionV2,
)
from pc.sim_camera import SimCamera


class SimCameraStateTests(unittest.TestCase):
    def test_camera_projection_uses_explicit_sim_fov(self) -> None:
        cam = SimCamera(
            width=1280,
            height=720,
            renderer_name="cpu",
            camera={"fov_y_deg": 60.0},
        )

        model = cam.get_camera_model_info()

        self.assertAlmostEqual(model["fov_y_deg"], 60.0)
        self.assertAlmostEqual(model["fov_x_deg"], 91.4928445, places=6)
        self.assertAlmostEqual(model["fx_px"], 623.5382907, places=6)
        self.assertAlmostEqual(model["fy_px"], 623.5382907, places=6)

    def test_camera_projection_supports_independent_axis_fov(self) -> None:
        cam = SimCamera(
            width=1280,
            height=720,
            renderer_name="cpu",
            camera={"fov_x_deg": 135.0, "fov_y_deg": 73.0},
        )

        model = cam.get_camera_model_info()

        self.assertAlmostEqual(model["fov_x_deg"], 135.0)
        self.assertAlmostEqual(model["fov_y_deg"], 73.0)
        self.assertAlmostEqual(model["fx_px"], 265.0966799, places=6)
        self.assertAlmostEqual(model["fy_px"], 486.5120777, places=6)

    def test_ground_truth_snapshot_guarantees_selected_visible_target(self) -> None:
        cam = SimCamera(
            width=1280,
            height=720,
            renderer_name="cpu",
            camera={"fov_x_deg": 135.0, "fov_y_deg": 73.0},
            scene={
                "mode": "static_targets",
                "targets": [{"sprite": "drone", "width": 0.35, "ground": [0.0, -1.0], "ground_y": 0.9}],
                "buildings": [],
                "cubes": [],
            },
        )
        cam.next_frame()

        snapshot = cam.build_ground_truth_snapshot(123, 456_000_000)

        self.assertEqual(snapshot.frame.frame_id, 123)
        self.assertEqual(snapshot.frame.source_clock_domain, "pc_monotonic")
        self.assertIs(snapshot.frame.source_identity_verified, True)
        self.assertEqual(len(snapshot.tracks), 1)
        self.assertEqual(snapshot.tracks[0].class_id, "drone")
        self.assertEqual(snapshot.tracks[0].missed_frames, 0)
        self.assertIsNotNone(snapshot.selection)
        self.assertEqual(snapshot.selection.track_id, snapshot.tracks[0].track_id)
        self.assertEqual(snapshot.selection.policy, "sim_ground_truth")
        self.assertEqual(len(snapshot.assessments), 1)
        self.assertAlmostEqual(snapshot.assessments[0].distance_m, 1.0, places=2)

    def _assert_centre_almost_equal(
        self,
        actual: tuple[float, float, float],
        expected: tuple[float, float, float],
        places: int = 6,
    ) -> None:
        for idx in range(3):
            self.assertAlmostEqual(actual[idx], expected[idx], places=places, msg=f"coord[{idx}]")

    def _single_target_centre(
        self, cam: SimCamera, frame_id: int
    ) -> tuple[float, float, float]:
        targets = cam._describe_billboards(frame_id)
        self.assertEqual(len(targets), 1)
        centre = targets[0]["centre"]
        self.assertEqual(len(centre), 3)
        return (
            float(centre[0]),
            float(centre[1]),
            float(centre[2]),
        )

    def _planner_eval_scene(self, **overrides):
        planner_eval = {
            "seed": 11,
            "max_active_targets": 1,
            "spawn_interval_s": [100.0, 100.0],
            "spawn_distance_m": [10.0, 10.0],
            "spawn_arc_deg": [0.0, 0.0],
            "altitude_m": [2.0, 2.0],
            "speed_m_s": [1.0, 1.0],
            "engage_dwell_s": 1.0,
            "match_radius_px": 80.0,
            "breach_zone": "critical",
        }
        planner_eval.update(overrides)
        return {
            "mode": "planner_eval",
            "defended_asset": {
                "id": "asset_0",
                "position_world": [0.0, 0.0, 0.0],
            },
            "threat_eval_zones": {
                "enabled": True,
                "zones": {
                    "warning": {"type": "circle", "radius_m": 5.0},
                    "restricted": {"type": "circle", "radius_m": 3.0},
                    "critical": {"type": "circle", "radius_m": 1.0},
                },
            },
            "planner_eval": planner_eval,
        }

    def _perception_snapshot(
        self,
        *,
        frame_id: int,
        box_center: tuple[float, float] = (160.0, 120.0),
        img_size: tuple[int, int] = (320, 240),
    ) -> PerceptionSnapshotV2:
        img_w, img_h = img_size
        box_w = 0.1
        box_h = 0.1
        center_u, center_v = box_center
        return PerceptionSnapshotV2(
            sequence=frame_id,
            frame=PerceptionFrameV2(
                frame_id=frame_id,
                source_time_ns=frame_id * 100_000_000,
                observed_time_ns=frame_id * 100_000_000 + 2,
                source_clock_domain="test.monotonic",
                observation_clock_domain="test.monotonic",
                width=img_w,
                height=img_h,
            ),
            tracks=(
                PerceptionTrackV2(
                    track_id=4,
                    box=NormalizedBoxV2(
                        x=max(0.0, min(1.0 - box_w, (center_u / img_w) - box_w * 0.5)),
                        y=max(0.0, min(1.0 - box_h, (center_v / img_h) - box_h * 0.5)),
                        w=box_w,
                        h=box_h,
                    ),
                    class_id="drone",
                    confidence=0.95,
                    missed_frames=0,
                ),
            ),
            selection=TargetSelectionV2(
                track_id=4,
                source_frame_id=frame_id,
                applied_frame_id=frame_id,
                selected_time_ns=frame_id * 100_000_000 + 3,
                selection_clock_domain="test.monotonic",
                policy="test",
            ),
        )

    def test_apply_cam_state_wraps_and_clamps_pose(self) -> None:
        cam = SimCamera(width=320, height=240, renderer_name="cpu", debug=False)

        cam.apply_cam_state(
            pan=(4.0 * math.pi) + 0.25,
            tilt=math.radians(120.0),
            pan_rate=0.4,
            tilt_rate=-0.2,
        )

        pose = cam.get_pose()
        self.assertAlmostEqual(float(pose["pan"]), 0.25, places=6)
        self.assertAlmostEqual(float(pose["tilt"]), math.radians(80.0), places=6)
        self.assertAlmostEqual(float(pose["pan_rate"]), 0.4, places=6)
        self.assertAlmostEqual(float(pose["tilt_rate"]), -0.2, places=6)

    def test_apply_cam_state_defaults_missing_rates_to_zero(self) -> None:
        cam = SimCamera(width=320, height=240, renderer_name="cpu", debug=False)

        cam.apply_cam_state(pan=-0.5, tilt=-math.radians(120.0))

        pose = cam.get_pose()
        self.assertAlmostEqual(float(pose["pan"]), -0.5, places=6)
        self.assertAlmostEqual(float(pose["tilt"]), -math.radians(80.0), places=6)
        self.assertEqual(float(pose["pan_rate"]), 0.0)
        self.assertEqual(float(pose["tilt_rate"]), 0.0)

    def test_planner_eval_spawns_deterministically_and_moves_toward_asset(self) -> None:
        scene = self._planner_eval_scene(
            max_active_targets=2,
            spawn_interval_s=[1.0, 1.0],
        )
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=1.0,
        )

        first_frame = cam._describe_billboards(1)
        self.assertEqual(len(first_frame), 1)
        projected = cam._project_planner_eval_targets(1)
        self.assertEqual(len(projected), 1)
        self.assertGreaterEqual(projected[0][1][0], 0.0)
        self.assertLessEqual(projected[0][1][0], 319.0)
        self.assertGreaterEqual(projected[0][1][1], 0.0)
        self.assertLessEqual(projected[0][1][1], 239.0)
        first_centre = first_frame[0]["centre"]
        first_distance = math.hypot(float(first_centre[0]), float(first_centre[2]))
        self.assertGreater(first_distance, 1.0)

        second_frame = cam._describe_billboards(2)
        self.assertEqual(len(second_frame), 2)
        moved_first = next(item for item in second_frame if item["target_id"] == 1)
        moved_centre = moved_first["centre"]
        moved_distance = math.hypot(float(moved_centre[0]), float(moved_centre[2]))
        self.assertLess(moved_distance, first_distance)
        self.assertLess(float(moved_centre[1]), float(first_centre[1]))
        self.assertEqual(cam.get_planner_eval_stats()["spawned"], 2)

    def test_planner_eval_flies_toward_configured_asset_height(self) -> None:
        scene = self._planner_eval_scene(
            altitude_m=[4.0, 4.0],
            speed_m_s=[1.0, 1.0],
        )
        threat_eval = SimpleNamespace(
            enabled=True,
            asset_world=(0.0, 3.0, 0.0),
            zone_radii={"critical": 1.0},
        )
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            threat_eval=threat_eval,
            fps_hz=1.0,
        )
        self.assertIsNotNone(cam._planner_eval)  # type: ignore[attr-defined]
        first_frame = cam._planner_eval.describe_targets(1, spawn_camera=None)  # type: ignore[union-attr]
        second_frame = cam._planner_eval.describe_targets(2, spawn_camera=None)  # type: ignore[union-attr]

        first_y = float(first_frame[0]["centre"][1])
        moved_y = float(second_frame[0]["centre"][1])
        self.assertLess(moved_y, first_y)
        self.assertGreater(moved_y, 3.8)

    def test_planner_eval_aim_dwell_removes_only_matched_target(self) -> None:
        scene = self._planner_eval_scene(
            max_active_targets=2,
            spawn_interval_s=[1.0, 1.0],
            engage_dwell_s=0.5,
            match_radius_px=40.0,
        )
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=2.0,
        )
        cam._describe_billboards(3)
        projected = cam._project_planner_eval_targets(3)
        self.assertGreaterEqual(len(projected), 2)
        target_id, target_uv = projected[0]

        cam.apply_perception_feedback(
            self._perception_snapshot(
                frame_id=3,
                box_center=target_uv,
            )
        )

        remaining_ids = {
            int(item["target_id"])
            for item in cam._describe_billboards(3)
        }
        self.assertNotIn(target_id, remaining_ids)
        self.assertEqual(len(remaining_ids), 1)
        self.assertEqual(cam.get_planner_eval_stats()["eliminated"], 1)

    def test_planner_eval_scores_the_laser_hit_point_not_the_image_centre(self) -> None:
        from common.control import LaserMountConfig

        def camera(offset_up_m: float) -> SimCamera:
            cam = SimCamera(
                width=320, height=240, renderer_name="cpu", debug=False,
                scene=self._planner_eval_scene(), fps_hz=2.0,
                laser_mount=LaserMountConfig.from_raw_config(
                    {"laser": {"offset_m": {"x": 0.0, "y": offset_up_m, "z": 0.0}}}),
            )
            cam._describe_billboards(1)
            return cam

        # On the camera axis the laser hits the image centre at any depth.
        axial = camera(0.0)
        target_id, target_uv = axial._project_planner_eval_targets(1)[0]
        self.assertAlmostEqual(
            axial._laser_miss_px(1, target_id),
            math.hypot(target_uv[0] - 159.5, target_uv[1] - 119.5), places=3)

        # 2 m below the camera: the hit point drops by fy * 2 / depth pixels.
        below = camera(-2.0)
        target = below._planner_eval.active[0]
        depth = float(target.position[2]) * -1.0  # camera looks down -Z
        fy_px = 239 / 2 / math.tan(math.radians(60.0) / 2)
        axial_miss_v = target_uv[1] - 119.5
        miss = below._laser_miss_px(1, target_id)
        self.assertAlmostEqual(
            miss, math.hypot(target_uv[0] - 159.5, axial_miss_v - fy_px * 2.0 / depth), places=2)

    def test_planner_eval_feedback_never_runs_the_scenario_ahead_of_rendering(self) -> None:
        scene = self._planner_eval_scene(spawn_interval_s=[1.0, 1.0], max_active_targets=3)
        cam = SimCamera(width=320, height=240, renderer_name="cpu", debug=False,
                        scene=scene, fps_hz=2.0)
        cam._describe_billboards(1)
        # A transport frame id (Unix-microsecond epoch) must not be taken as
        # the simulator's frame: it would spawn years of targets.
        cam.apply_perception_feedback(self._perception_snapshot(
            frame_id=1_790_566_083_246_952, box_center=(160.0, 120.0)))
        self.assertLessEqual(cam.get_planner_eval_stats()["spawned"], 2)

    def test_planner_eval_invalid_or_false_feedback_does_not_remove_target(self) -> None:
        scene = self._planner_eval_scene(
            engage_dwell_s=0.5,
            match_radius_px=20.0,
        )
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=2.0,
        )
        cam._describe_billboards(1)

        cam.apply_perception_feedback(
            self._perception_snapshot(
                frame_id=1,
                box_center=(319.0, 239.0),
            )
        )
        cam.apply_perception_feedback(
            self._perception_snapshot(
                frame_id=2,
                box_center=(319.0, 239.0),
            )
        )

        self.assertEqual(len(cam._describe_billboards(2)), 1)
        self.assertEqual(cam.get_planner_eval_stats()["eliminated"], 0)

    def test_planner_eval_breach_zone_removes_target_and_counts_breach(self) -> None:
        scene = self._planner_eval_scene(
            spawn_distance_m=[2.0, 2.0],
            speed_m_s=[20.0, 20.0],
            breach_zone="critical",
        )
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=1.0,
        )

        self.assertEqual(len(cam._describe_billboards(1)), 1)
        self.assertEqual(len(cam._describe_billboards(2)), 0)
        stats = cam.get_planner_eval_stats()
        self.assertEqual(stats["breached"], 1)
        self.assertEqual(stats["active"], 0)

    def test_static_targets_mode_uses_configured_targets(self) -> None:
        scene = {
            "mode": "static_targets",
            "planner_eval": {
                "max_active_targets": 1,
            },
            "targets": [
                {
                    "sprite": "drone",
                    "ground": [1.0, -4.0],
                    "ground_y": 2.0,
                    "width": 0.4,
                }
            ],
        }
        cam = SimCamera(320, 240, fps=30.0, scene=scene)

        self.assertFalse(cam.planner_eval_enabled())
        targets = cam._describe_billboards(1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["sprite"], "drone")

    def test_path_movement_follows_points_and_wraps_to_first(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "ground": [999.0, -999.0],
                    "ground_y": 50.0,
                    "height": 10.0,
                    "movement": {
                        "type": "path",
                        "speed_m_s": 1.0,
                        "points": [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                    },
                }
            ]
        }
        cam = SimCamera(
            width=320, height=240, renderer_name="cpu", debug=False, scene=scene, fps_hz=1.0
        )

        self._assert_centre_almost_equal(self._single_target_centre(cam, 1), (0.0, 0.0, 0.0))
        self._assert_centre_almost_equal(self._single_target_centre(cam, 2), (1.0, 0.0, 0.0))
        self._assert_centre_almost_equal(self._single_target_centre(cam, 3), (1.0, 1.0, 0.0))
        self._assert_centre_almost_equal(self._single_target_centre(cam, 4), (0.0, 1.0, 0.0))
        self._assert_centre_almost_equal(self._single_target_centre(cam, 5), (0.0, 0.0, 0.0))

    def test_path_movement_interpolates_straight_line_in_3d(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "movement": {
                        "type": "path",
                        "speed_m_s": 1.0,
                        "points": [
                            [0.0, 0.0, 0.0],
                            [0.0, 1.0, math.sqrt(3.0)],
                        ],
                    },
                }
            ]
        }
        cam = SimCamera(
            width=320, height=240, renderer_name="cpu", debug=False, scene=scene, fps_hz=1.0
        )

        centre_frame_2 = self._single_target_centre(cam, 2)
        self.assertAlmostEqual(centre_frame_2[0], 0.0, places=6)
        self.assertAlmostEqual(centre_frame_2[1], 0.5, places=6)
        self.assertAlmostEqual(centre_frame_2[2], math.sqrt(3.0) * 0.5, places=6)
        self._assert_centre_almost_equal(
            self._single_target_centre(cam, 3), (0.0, 1.0, math.sqrt(3.0))
        )

    def test_invalid_path_points_fall_back_to_static_target(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "person",
                    "width": 1.0,
                    "height": 2.0,
                    "ground": [3.0, -4.0],
                    "ground_y": 2.0,
                    "movement": {
                        "type": "path",
                        "speed_m_s": 1.0,
                        "points": [[0.0, 0.0, 0.0]],
                    },
                }
            ]
        }
        cam = SimCamera(width=320, height=240, renderer_name="cpu", debug=False, scene=scene)
        expected_centre = (3.0, 3.0, -4.0)
        self.assertEqual(self._single_target_centre(cam, 1), expected_centre)
        self.assertEqual(self._single_target_centre(cam, 25), expected_centre)

    def test_non_positive_or_invalid_path_speed_disables_motion(self) -> None:
        for speed_value in (0.0, -1.0, "fast"):
            with self.subTest(speed_value=speed_value):
                scene = {
                    "targets": [
                        {
                            "sprite": "drone",
                            "width": 0.4,
                            "height": 0.4,
                            "centre": [9.0, 8.0, 7.0],
                            "movement": {
                                "type": "path",
                                "speed_m_s": speed_value,
                                "points": [
                                    [0.0, 0.0, 0.0],
                                    [2.0, 0.0, 0.0],
                                ],
                            },
                        }
                    ]
                }
                cam = SimCamera(
                    width=320,
                    height=240,
                    renderer_name="cpu",
                    debug=False,
                    scene=scene,
                    fps_hz=1.0,
                )
                expected_centre = (9.0, 8.0, 7.0)
                self.assertEqual(self._single_target_centre(cam, 1), expected_centre)
                self.assertEqual(self._single_target_centre(cam, 10), expected_centre)

    def test_dynamic_path_accelerates_from_first_waypoint(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "movement": {
                        "type": "path",
                        "speed_m_s": 2.0,
                        "points": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                        "dynamics": {
                            "enabled": True,
                            "max_accel_m_s2": 1.0,
                            "max_decel_m_s2": 2.0,
                            "arrival_radius_m": 0.1,
                        },
                    },
                }
            ]
        }
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=1.0,
        )

        self._assert_centre_almost_equal(self._single_target_centre(cam, 1), (0.0, 0.0, 0.0))
        self._assert_centre_almost_equal(self._single_target_centre(cam, 2), (1.0, 0.0, 0.0))
        self._assert_centre_almost_equal(self._single_target_centre(cam, 3), (3.0, 0.0, 0.0))

    def test_dynamic_path_carries_velocity_through_waypoint_turn(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "movement": {
                        "type": "path",
                        "speed_m_s": 2.0,
                        "points": [
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 2.0, 0.0],
                        ],
                        "dynamics": {
                            "enabled": True,
                            "max_accel_m_s2": 2.0,
                            "max_decel_m_s2": 2.0,
                            "arrival_radius_m": 0.2,
                        },
                    },
                }
            ]
        }
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=2.0,
        )

        centre = self._single_target_centre(cam, 5)
        state = cam._billboard_path_states[0]  # type: ignore[attr-defined]
        velocity = state["velocity"]

        self.assertGreater(centre[1], 0.0)
        self.assertGreater(float(velocity[0]), 0.1)
        self.assertGreater(float(velocity[1]), 0.1)
        self.assertEqual(int(state["waypoint_idx"]), 2)

    def test_dynamic_path_decelerates_near_waypoint(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "movement": {
                        "type": "path",
                        "speed_m_s": 4.0,
                        "points": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                        "dynamics": {
                            "enabled": True,
                            "max_accel_m_s2": 8.0,
                            "max_decel_m_s2": 8.0,
                            "arrival_radius_m": 0.1,
                        },
                    },
                }
            ]
        }
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=4.0,
        )

        self._single_target_centre(cam, 3)
        cruise_speed = float(cam._billboard_path_states[0]["velocity"][0])  # type: ignore[attr-defined]
        self._single_target_centre(cam, 6)
        near_waypoint_speed = float(cam._billboard_path_states[0]["velocity"][0])  # type: ignore[attr-defined]

        self.assertAlmostEqual(cruise_speed, 4.0, places=6)
        self.assertLess(abs(near_waypoint_speed), cruise_speed)

    def test_invalid_dynamic_path_config_uses_existing_path_movement(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "movement": {
                        "type": "path",
                        "speed_m_s": 1.0,
                        "points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        "dynamics": {
                            "enabled": True,
                            "max_accel_m_s2": "fast",
                            "arrival_radius_m": 0.1,
                        },
                    },
                }
            ]
        }
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=1.0,
        )

        self._assert_centre_almost_equal(self._single_target_centre(cam, 2), (1.0, 0.0, 0.0))
        self._assert_centre_almost_equal(self._single_target_centre(cam, 3), (2.0, 0.0, 0.0))

    def test_dynamic_path_backward_frame_resets_deterministically(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "movement": {
                        "type": "path",
                        "speed_m_s": 2.0,
                        "points": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                        "dynamics": {
                            "enabled": True,
                            "max_accel_m_s2": 1.0,
                            "max_decel_m_s2": 2.0,
                            "arrival_radius_m": 0.1,
                        },
                    },
                }
            ]
        }
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=1.0,
        )
        fresh = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=1.0,
        )

        self._single_target_centre(cam, 5)
        reset_centre = self._single_target_centre(cam, 3)
        fresh_centre = self._single_target_centre(fresh, 3)

        self._assert_centre_almost_equal(reset_centre, fresh_centre)

    def test_circle_movement_without_dynamics_remains_exact(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "centre": [0.0, 0.0, 0.0],
                    "movement": {
                        "type": "circle",
                        "radius": 2.0,
                        "speed": 0.5,
                        "phase": 0.0,
                    },
                }
            ]
        }
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=2.0,
        )

        expected_frame_2 = (
            (math.cos(1.0) - 1.0) * 2.0,
            0.0,
            math.sin(1.0) * 2.0,
        )
        self._assert_centre_almost_equal(self._single_target_centre(cam, 2), expected_frame_2)

    def test_dynamic_circle_uses_shared_motion_filter(self) -> None:
        legacy_scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "centre": [0.0, 0.0, 0.0],
                    "movement": {
                        "type": "circle",
                        "radius": 2.0,
                        "speed": 0.5,
                        "phase": 0.0,
                    },
                }
            ]
        }
        dynamic_scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "centre": [0.0, 0.0, 0.0],
                    "movement": {
                        "type": "circle",
                        "radius": 2.0,
                        "speed": 0.5,
                        "phase": 0.0,
                        "dynamics": {
                            "enabled": True,
                            "max_accel_m_s2": 1.0,
                            "max_decel_m_s2": 1.0,
                            "arrival_radius_m": 0.1,
                        },
                    },
                }
            ]
        }
        legacy = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=legacy_scene,
            fps_hz=2.0,
        )
        dynamic = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=dynamic_scene,
            fps_hz=2.0,
        )

        legacy_centre = self._single_target_centre(legacy, 2)
        dynamic_centre = self._single_target_centre(dynamic, 2)
        legacy_distance = math.hypot(legacy_centre[0], legacy_centre[2])
        dynamic_distance = math.hypot(dynamic_centre[0], dynamic_centre[2])
        state = dynamic._billboard_motion_states[0]  # type: ignore[attr-defined]

        self.assertGreater(dynamic_distance, 0.0)
        self.assertLess(dynamic_distance, legacy_distance)
        self.assertGreater(abs(float(state["velocity"][0])), 0.0)
        self.assertGreater(abs(float(state["velocity"][2])), 0.0)
        self.assertNotIn("waypoint_idx", state)

    def test_invalid_dynamic_circle_config_uses_existing_circle_movement(self) -> None:
        legacy_scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "centre": [0.0, 0.0, 0.0],
                    "movement": {
                        "type": "circle",
                        "radius": 2.0,
                        "speed": 0.5,
                        "phase": 0.0,
                    },
                }
            ]
        }
        invalid_dynamic_scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "width": 0.4,
                    "height": 0.4,
                    "centre": [0.0, 0.0, 0.0],
                    "movement": {
                        "type": "circle",
                        "radius": 2.0,
                        "speed": 0.5,
                        "phase": 0.0,
                        "dynamics": {
                            "enabled": True,
                            "max_accel_m_s2": "fast",
                        },
                    },
                }
            ]
        }
        legacy = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=legacy_scene,
            fps_hz=2.0,
        )
        invalid_dynamic = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=invalid_dynamic_scene,
            fps_hz=2.0,
        )

        self._assert_centre_almost_equal(
            self._single_target_centre(invalid_dynamic, 2),
            self._single_target_centre(legacy, 2),
        )

    def test_scene_building_material_fields_are_preserved(self) -> None:
        scene = {
            "buildings": [
                {
                    "base_centre": [1.0, 2.0],
                    "footprint": [6.0, 4.0],
                    "height": 8.0,
                    "albedo_map": "textures/building/concrete_wall_diffuse.png",
                    "normal_map": "textures/building/concrete_wall_normal_gl.png",
                    "metallic": 0.2,
                    "roughness": 0.7,
                    "uv_scale": [3.0, 2.0],
                }
            ]
        }
        cam = SimCamera(width=320, height=240, renderer_name="cpu", debug=False, scene=scene)

        buildings = cam._describe_buildings()

        self.assertEqual(len(buildings), 1)
        self.assertEqual(buildings[0]["albedo_map"], scene["buildings"][0]["albedo_map"])
        self.assertEqual(buildings[0]["normal_map"], scene["buildings"][0]["normal_map"])
        self.assertEqual(buildings[0]["metallic"], 0.2)
        self.assertEqual(buildings[0]["roughness"], 0.7)
        self.assertEqual(buildings[0]["uv_scale"], [3.0, 2.0])


if __name__ == "__main__":
    unittest.main()


def test_repeat_request_for_current_frame_does_not_replay_dynamic_motion() -> None:
    from pc.sim_camera import SimCamera

    gen = SimCamera(width=64, height=36, fps_hz=60.0, scene={
        "mode": "static_targets", "buildings": [], "cubes": [],
        "targets": [{"sprite": "drone", "width": 0.35, "movement": {
            "type": "path", "speed_m_s": 0.3,
            "points": [[0.0, 1.0, -0.9], [0.1, 1.0, -0.9], [0.1, 1.1, -0.9]],
            "dynamics": {"enabled": True, "max_accel_m_s2": 1.5, "max_decel_m_s2": 1.5,
                         "arrival_radius_m": 0.01}}}]})
    steps = []
    original = gen._integrate_dynamic_path_step
    gen._integrate_dynamic_path_step = lambda state, *a, **k: (steps.append(1), original(state, *a, **k))
    for frame in range(1, 201):
        gen.next_frame()
        gen.build_ground_truth_snapshot(frame, 0)  # second request for the same frame
    assert len(steps) == 199  # one integration step per new frame, never a replay
