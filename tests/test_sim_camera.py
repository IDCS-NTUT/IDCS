import math
import unittest
from types import SimpleNamespace

from pc.sim_camera import SimCamera
from pc.renderers._common import build_camera


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

    def test_evaluation_spawning_is_deterministic_with_seed(self) -> None:
        evaluation = {
            "enabled": True,
            "seed": 123,
            "max_active_targets": 3,
            "spawn_interval_s": 0.0,
            "spawn_radius_m": 14.0,
            "target_speed_m_s": 1.0,
            "classes": [{"sprite": "drone", "width": 0.4, "ground_y": 2.0}],
        }
        first = SimCamera(
            width=640,
            height=360,
            renderer_name="cpu",
            debug=False,
            evaluation=evaluation,
            fps_hz=10.0,
        )
        second = SimCamera(
            width=640,
            height=360,
            renderer_name="cpu",
            debug=False,
            evaluation=evaluation,
            fps_hz=10.0,
        )

        first_targets = first._evaluation.describe_targets(1)  # type: ignore[union-attr]
        second_targets = second._evaluation.describe_targets(1)  # type: ignore[union-attr]

        self.assertEqual(
            [target["centre"] for target in first_targets],
            [target["centre"] for target in second_targets],
        )

    def test_evaluation_targets_spawn_inside_initial_camera_view(self) -> None:
        evaluation = {
            "enabled": True,
            "seed": 4,
            "max_active_targets": 4,
            "spawn_interval_s": 0.0,
            "spawn_radius_m": 18.0,
            "target_speed_m_s": 1.0,
            "classes": [{"sprite": "drone", "width": 0.4, "ground_y": 2.0}],
        }
        cam = SimCamera(
            width=640,
            height=360,
            renderer_name="cpu",
            debug=False,
            evaluation=evaluation,
            fps_hz=10.0,
        )

        world = cam.describe_world(1)
        camera = build_camera(world["camera"], context=cam, width=cam.width, height=cam.height)
        self.assertIsNotNone(camera)
        targets = [obj for obj in world["objects"] if obj.get("type") == "target"]
        self.assertGreaterEqual(len(targets), 1)
        for target in targets:
            self.assertIn("size", target)
            self.assertNotIn("width", target)
            self.assertNotIn("height", target)
            projected = self._project(camera, target["centre"], cam.width, cam.height)
            self.assertIsNotNone(projected)
            assert projected is not None
            self.assertGreaterEqual(projected[0], 0.0)
            self.assertLessEqual(projected[0], cam.width - 1)
            self.assertGreaterEqual(projected[1], 0.0)
            self.assertLessEqual(projected[1], cam.height - 1)

    def test_evaluation_targets_move_toward_defended_asset(self) -> None:
        evaluation = {
            "enabled": True,
            "seed": 8,
            "max_active_targets": 1,
            "spawn_interval_s": 999.0,
            "spawn_radius_m": 16.0,
            "target_speed_m_s": 2.0,
            "classes": [{"sprite": "drone", "width": 0.4, "ground_y": 2.0}],
        }
        threat_eval = {
            "defended_asset": {"position_world": [0.0, 0.0]},
            "zones": {"critical": {"type": "circle", "radius_m": 0.5}},
        }
        cam = SimCamera(
            width=640,
            height=360,
            renderer_name="cpu",
            debug=False,
            evaluation=evaluation,
            threat_eval=threat_eval,
            fps_hz=1.0,
        )

        start = cam._evaluation.describe_targets(1)[0]["centre"]  # type: ignore[union-attr]
        later = cam._evaluation.describe_targets(3)[0]["centre"]  # type: ignore[union-attr]
        start_distance = math.hypot(float(start[0]), float(start[2]))
        later_distance = math.hypot(float(later[0]), float(later[2]))

        self.assertLess(later_distance, start_distance)

    def test_evaluation_lock_dwell_removes_target_after_duration(self) -> None:
        evaluation = {
            "enabled": True,
            "seed": 11,
            "max_active_targets": 1,
            "spawn_interval_s": 999.0,
            "spawn_radius_m": 16.0,
            "target_speed_m_s": 0.1,
            "lock_dwell_s": 2.0,
            "lock_tolerance_px": 25.0,
            "classes": [{"sprite": "drone", "width": 0.4, "ground_y": 2.0}],
        }
        cam = SimCamera(
            width=640,
            height=360,
            renderer_name="cpu",
            debug=False,
            evaluation=evaluation,
            fps_hz=1.0,
        )
        first_target = cam._evaluation.describe_targets(1)[0]  # type: ignore[union-attr]
        world = cam.describe_world(1)
        camera = build_camera(world["camera"], context=cam, width=cam.width, height=cam.height)
        uv = self._project(camera, first_target["centre"], cam.width, cam.height)
        self.assertIsNotNone(uv)
        assert uv is not None

        cmd = SimpleNamespace(
            target_ok=True,
            target_uv=uv,
            err_uv=(0.0, 0.0),
            laser_on_target=None,
        )
        cam.apply_evaluation_control(cmd)

        self.assertEqual(len(cam._evaluation.describe_targets(2)), 1)  # type: ignore[union-attr]
        self.assertEqual(len(cam._evaluation.describe_targets(3)), 0)  # type: ignore[union-attr]
        metrics = cam.evaluation_metrics()
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics["neutralized"], 1.0)

    def test_evaluation_replaces_static_targets_but_preserves_buildings(self) -> None:
        scene = {
            "buildings": [
                {
                    "base_centre": [4.0, -12.0],
                    "footprint": [2.0, 3.0],
                    "height": 5.0,
                    "color": [10, 20, 30],
                }
            ],
            "targets": [
                {
                    "sprite": "person",
                    "height": 1.7,
                    "ground": [99.0, -99.0],
                }
            ],
        }
        evaluation = {
            "enabled": True,
            "seed": 5,
            "max_active_targets": 1,
            "spawn_interval_s": 999.0,
            "classes": [{"sprite": "drone", "width": 0.4, "ground_y": 2.0}],
        }
        cam = SimCamera(
            width=640,
            height=360,
            renderer_name="cpu",
            debug=False,
            scene=scene,
            evaluation=evaluation,
            fps_hz=10.0,
        )

        objects = cam.describe_world(1)["objects"]
        buildings = [obj for obj in objects if obj.get("type") == "building"]
        targets = [obj for obj in objects if obj.get("type") == "target"]

        self.assertEqual(len(buildings), 1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["sprite"], "drone")

    def _project(self, camera, point, width: int, height: int):
        if camera is None:
            return None
        import numpy as np

        rel = np.asarray(point, dtype=np.float32) - camera["position"]
        x = float(np.dot(rel, camera["right"]))
        y = float(np.dot(rel, camera["up"]))
        z = float(np.dot(rel, camera["forward"]))
        if z <= 0.05:
            return None
        f = 1.0 / math.tan(math.radians(float(camera["fov_y"])) * 0.5)
        x_ndc = (x / z) * (f / float(camera["aspect"]))
        y_ndc = (y / z) * f
        return (
            (x_ndc + 1.0) * 0.5 * (width - 1),
            (1.0 - (y_ndc + 1.0) * 0.5) * (height - 1),
        )


if __name__ == "__main__":
    unittest.main()
