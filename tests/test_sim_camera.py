import math
import unittest
from types import SimpleNamespace

from common.schemas import Box, DetectionMsg
from pc.sim_camera import SimCamera


class SimCameraStateTests(unittest.TestCase):
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

    def _detection_msg(
        self,
        *,
        frame_id: int,
        laser_on_target: bool | None,
        box_center: tuple[float, float] = (160.0, 120.0),
        img_size: tuple[int, int] = (320, 240),
    ) -> DetectionMsg:
        img_w, img_h = img_size
        box_w = 0.1
        box_h = 0.1
        center_u, center_v = box_center
        return DetectionMsg(
            frame_id=frame_id,
            src_ts_ms=frame_id * 100,
            rx_ts_ms=frame_id * 100 + 1,
            infer_ts_ms=frame_id * 100 + 2,
            img_w=img_w,
            img_h=img_h,
            boxes=[
                Box(
                    x=(center_u / img_w) - box_w * 0.5,
                    y=(center_v / img_h) - box_h * 0.5,
                    w=box_w,
                    h=box_h,
                    cls="drone",
                    conf=0.95,
                )
            ],
            target_idx=0,
            laser_on_target=laser_on_target,
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

    def test_apply_control_rates_preserves_passthrough_without_accel_limits(self) -> None:
        cam = SimCamera(width=320, height=240, renderer_name="cpu", debug=False)

        cam.apply_control_rates(1.0, -0.5, 1.0)

        pose = cam.get_pose()
        self.assertAlmostEqual(float(pose["pan_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(pose["tilt_rate"]), -0.5, places=6)
        self.assertAlmostEqual(float(pose["pan"]), 1.0, places=6)
        self.assertAlmostEqual(float(pose["tilt"]), -0.5, places=6)

    def test_apply_control_rates_uses_accel_intent_for_pose_update(self) -> None:
        cam = SimCamera(width=320, height=240, renderer_name="cpu", debug=False)

        cam.apply_control_rates(
            1.0,
            -1.0,
            1.0,
            pan_accel_rad_s2=0.5,
            tilt_accel_rad_s2=0.25,
        )

        pose = cam.get_pose()
        self.assertAlmostEqual(float(pose["pan_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(pose["tilt_rate"]), -0.25, places=6)
        self.assertAlmostEqual(float(pose["pan"]), 0.25, places=6)
        self.assertAlmostEqual(float(pose["tilt"]), -0.125, places=6)

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

    def test_planner_eval_preserves_render_profile(self) -> None:
        scene = self._planner_eval_scene(
            render_profile="yolo_drone_high_contrast",
            metallic=0.0,
            roughness=0.9,
            uv_scale=[1.0, 1.0],
        )
        cam = SimCamera(width=320, height=240, renderer_name="cpu", debug=False, scene=scene)

        targets = cam._describe_billboards(1)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["render_profile"], "yolo_drone_high_contrast")
        self.assertEqual(targets[0]["metallic"], 0.0)
        self.assertEqual(targets[0]["roughness"], 0.9)
        self.assertEqual(targets[0]["uv_scale"], [1.0, 1.0])

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

    def test_planner_eval_tilted_camera_respects_altitude_bounds(self) -> None:
        scene = self._planner_eval_scene(altitude_m=[1.5, 4.0])
        cam = SimCamera(
            width=320,
            height=240,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            fps_hz=30.0,
        )
        cam.apply_cam_state(pan=0.0, tilt=1.0)

        cam._project_planner_eval_targets(1)

        self.assertIsNotNone(cam._planner_eval)  # type: ignore[attr-defined]
        target = cam._planner_eval.active[0]  # type: ignore[union-attr]
        self.assertGreaterEqual(float(target.position[1]), 1.5)
        self.assertLessEqual(float(target.position[1]), 4.0)
        planar_distance = math.hypot(
            float(target.position[0]), float(target.position[2])
        )
        self.assertAlmostEqual(planar_distance, 10.0, places=5)

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

        cam.apply_detection_feedback(
            self._detection_msg(
                frame_id=3,
                laser_on_target=True,
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

        cam.apply_detection_feedback(
            self._detection_msg(
                frame_id=1,
                laser_on_target=True,
                box_center=(319.0, 239.0),
            )
        )
        cam.apply_detection_feedback(
            self._detection_msg(
                frame_id=2,
                laser_on_target=False,
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

    def test_scene_target_render_profiles_are_preserved(self) -> None:
        scene = {
            "targets": [
                {
                    "sprite": "drone",
                    "ground": [1.0, -4.0],
                    "ground_y": 2.0,
                    "width": 0.5,
                    "render_profile": "yolo_drone_high_contrast",
                    "metallic": 0.0,
                    "roughness": 0.9,
                }
            ],
            "meshes": [
                {
                    "asset": "meshes/drone.stl",
                    "sprite": "drone",
                    "centre": [0.0, 2.0, -3.0],
                    "scale": 0.8,
                    "render_profile": "yolo_drone_mesh",
                    "albedo_map": "textures/drone_albedo.png",
                }
            ],
        }
        cam = SimCamera(width=320, height=240, renderer_name="cpu", debug=False, scene=scene)

        targets = cam._describe_billboards(1)
        meshes = cam._describe_meshes()

        self.assertEqual(targets[0]["render_profile"], "yolo_drone_high_contrast")
        self.assertEqual(targets[0]["metallic"], 0.0)
        self.assertEqual(targets[0]["roughness"], 0.9)
        self.assertEqual(meshes[0]["render_profile"], "yolo_drone_mesh")
        self.assertEqual(meshes[0]["albedo_map"], "textures/drone_albedo.png")


if __name__ == "__main__":
    unittest.main()
