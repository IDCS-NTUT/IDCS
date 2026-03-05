import math
import unittest

from pc.sim_camera import SimCamera


class SimCameraStateTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
