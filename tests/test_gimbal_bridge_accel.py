import math
import unittest

from jetson import gimbal_bridge


class GimbalBridgeAccelTests(unittest.TestCase):
    def test_rate_slew_limit_uses_physical_accel(self) -> None:
        limited = gimbal_bridge._limit_rate_by_accel(
            desired_rate=1.0,
            previous_rate=0.2,
            accel_rad_s2=2.0,
            dt_s=0.1,
        )

        self.assertAlmostEqual(limited, 0.4)

    def test_rate_slew_limit_passthrough_without_accel(self) -> None:
        limited = gimbal_bridge._limit_rate_by_accel(
            desired_rate=-0.7,
            previous_rate=0.2,
            accel_rad_s2=None,
            dt_s=0.1,
        )

        self.assertAlmostEqual(limited, -0.7)

    def test_physical_accel_maps_to_mks_byte(self) -> None:
        byte = gimbal_bridge._mks_accel_byte_from_physical(3.5)

        self.assertEqual(byte, 10)

    def test_requested_accel_clamps_to_configured_limit_before_byte_mapping(self) -> None:
        requested, effective = gimbal_bridge._clamp_requested_accel(
            requested=9.0,
            configured_limit=3.5,
        )

        self.assertAlmostEqual(requested, 9.0)
        self.assertAlmostEqual(effective, 3.5)
        self.assertEqual(gimbal_bridge._mks_accel_byte_from_physical(effective), 10)

    def test_invalid_requested_accel_defaults_to_configured_limit(self) -> None:
        requested, effective = gimbal_bridge._clamp_requested_accel(
            requested=math.nan,
            configured_limit=2.8,
        )

        self.assertAlmostEqual(requested, 2.8)
        self.assertAlmostEqual(effective, 2.8)

    def test_physical_accel_mapping_returns_zero_for_invalid_input(self) -> None:
        for value in (None, 0.0, -1.0, math.nan):
            with self.subTest(value=value):
                byte = gimbal_bridge._mks_accel_byte_from_physical(value)
                self.assertEqual(byte, 0)

    def test_serial_targets_read_physical_gimbal_accel_limits(self) -> None:
        targets, _ = gimbal_bridge._build_serial_targets(
            {
                "gimbal": {
                    "pitch_motor_a_addr": 2,
                    "pitch_motor_b_addr": 3,
                    "yaw_accel_limit_rad_s2": 4.2,
                    "pitch_accel_limit_rad_s2": 2.8,
                },
            }
        )

        self.assertAlmostEqual(targets["yaw_accel_limit_rad_s2"], 4.2)
        self.assertAlmostEqual(targets["pitch_accel_limit_rad_s2"], 2.8)


if __name__ == "__main__":
    unittest.main()
