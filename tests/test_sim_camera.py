import math
import unittest

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
