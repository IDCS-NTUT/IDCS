import math
import unittest

import numpy as np

from common.control import AxisPair, ControlConfig, LaserAimingControlConfig, PidConfig
from tools import benchmark_3d_trajectory as benchmark


def _control_config() -> ControlConfig:
    return ControlConfig(
        mode="rate",
        loop_hz=50.0,
        fx_px=500.0,
        fy_px=500.0,
        cx_px=640.0,
        cy_px=360.0,
        aim_mode="camera_center",
        deadband_px=0.0,
        smooth_px_alpha=1.0,
        lost_target_timeout_ms=100,
        reinit_on_lost=True,
        target_selector="largest_area",
        yaw_sign=1.0,
        pitch_sign=1.0,
        frame_size=(1280, 720),
        fov_deg=None,
        laser=LaserAimingControlConfig(
            tolerance_px=20.0,
            use_range="infinite",
            default_distance_m=10.0,
        ),
        pid=PidConfig(
            kp=AxisPair(2.0, 2.0),
            ki=AxisPair(0.1, 0.1),
            kd=AxisPair(0.05, 0.05),
            rate_limits=AxisPair(2.0, 2.0),
            accel_limits=AxisPair(20.0, 20.0),
        ),
        gimbal_accel_limits=AxisPair(20.0, 20.0),
    )


class TrajectoryBenchmarkTests(unittest.TestCase):
    def test_coordinates_convert_to_expected_axes(self):
        coords = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        yaw, pitch = benchmark.coordinates_to_axes(coords)
        self.assertAlmostEqual(0.0, yaw[0])
        self.assertAlmostEqual(math.pi / 2.0, yaw[1])
        self.assertAlmostEqual(math.pi / 4.0, pitch[2])

    def test_spline_interpolates_each_knot(self):
        knots = np.asarray([0.0, 1.0, 2.0])
        points = np.asarray([[3.0, 0.0, 0.0], [3.0, 1.0, 0.5], [3.0, 0.0, 1.0]])
        actual = benchmark.cubic_hermite_spline(knots, points, knots)
        np.testing.assert_allclose(points, actual, atol=1e-12)

    def test_pid_simulation_reduces_static_pointing_error(self):
        times = np.arange(0.0, 3.0, 0.02)
        yaw_ref = np.full(len(times), 0.2)
        pitch_ref = np.full(len(times), -0.1)
        plants = {
            "yaw": benchmark.AxisPlant(8.0, 8.0, 4.0, 0.0, "synthetic"),
            "pitch": benchmark.AxisPlant(8.0, 8.0, 4.0, 0.0, "synthetic"),
        }
        sim = benchmark.simulate_pid_benchmark(times, yaw_ref, pitch_ref, plants, _control_config(), 0.0, 0.0)
        summary = benchmark.build_summary(sim, plants, 0.0, 0.0)
        self.assertLess(abs(sim["error_yaw_rad"][-1]), 0.03)
        self.assertLess(abs(sim["error_pitch_rad"][-1]), 0.03)
        self.assertLess(summary["pointing_error"]["final_rad"], 0.05)


if __name__ == "__main__":
    unittest.main()
