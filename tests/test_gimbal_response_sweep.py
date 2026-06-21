import argparse
import unittest
from argparse import Namespace

from jetson.tools import gimbal_response_sweep as sweep


class GimbalResponseSweepTests(unittest.TestCase):
    def test_parse_rates_accepts_positive_csv(self) -> None:
        self.assertEqual(sweep._parse_rates("0.1, 0.5,1"), [0.1, 0.5, 1.0])

    def test_parse_rates_rejects_zero_and_negative_values(self) -> None:
        for value in ("0", "-0.1", "0.1,-0.2"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    sweep._parse_rates(value)

    def test_parse_accel_bytes_accepts_decimal_and_hex(self) -> None:
        self.assertEqual(sweep._parse_accel_bytes("1,10,0xFF"), [1, 10, 255])

    def test_parse_accel_bytes_rejects_out_of_range(self) -> None:
        for value in ("-1", "256"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    sweep._parse_accel_bytes(value)

    def test_direction_order_balances_positive_then_negative(self) -> None:
        self.assertEqual(sweep._directions("both"), [1, -1])
        self.assertEqual(sweep._directions("positive"), [1])
        self.assertEqual(sweep._directions("negative"), [-1])

    def test_payload_text_preserves_address_and_exact_bytes(self) -> None:
        self.assertEqual(
            sweep._payload_text([(2, (0x80, 0x0A, 0x05)), (3, (0x00, 0x0A, 0x05))]),
            "2:800A05;3:000A05",
        )

    def test_validation_rejects_nonpositive_step_duration(self) -> None:
        args = Namespace(
            repeat=1,
            sample_hz=50.0,
            pre_roll_s=0.5,
            step_s=0.0,
            post_roll_s=1.0,
            rest_s=0.5,
            settle_rate_rad_s=0.03,
            settle_hold_s=0.25,
            settle_timeout_s=5.0,
            reply_drain_s=0.5,
        )
        self.assertEqual(sweep._validate_args(args), "--step-s must be > 0")


if __name__ == "__main__":
    unittest.main()
