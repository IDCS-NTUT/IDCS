import math
import unittest

from common.gimbal.mks_servo42_rs485 import MksServo42Axis, SpeedCommandDither


class SpeedCommandDitherTests(unittest.TestCase):
    def test_static_encoder_reports_zero_below_one_rpm(self):
        self.assertEqual(0.0, MksServo42Axis.quantized_speed_rad_s(0.101, 1.0))

    def test_dither_preserves_sub_rpm_average_rate(self):
        dither = SpeedCommandDither(gear_ratio=1.0)
        requested = 0.05
        outputs = [dither.quantize(requested) for _ in range(200)]
        self.assertTrue(any(value == 0.0 for value in outputs))
        self.assertTrue(any(value > 0.0 for value in outputs))
        self.assertAlmostEqual(requested, sum(outputs) / len(outputs), delta=0.001)

    def test_direction_change_clears_fractional_residual(self):
        dither = SpeedCommandDither(gear_ratio=1.0)
        dither.quantize(0.08)
        output = dither.quantize(-0.08)
        self.assertLessEqual(output, 0.0)
        self.assertLess(abs(output), 2.0 * math.pi / 60.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
