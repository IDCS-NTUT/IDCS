import argparse
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from jetson.tools import gimbal_response_sweep as sweep


def _base_args(**overrides):
    values = {
        "repeat": 1,
        "sample_hz": 50.0,
        "pre_roll_s": 0.5,
        "step_s": 1.0,
        "post_roll_s": 1.0,
        "rest_s": 0.5,
        "settle_rate_rad_s": 0.03,
        "settle_hold_s": 0.25,
        "settle_timeout_s": 5.0,
        "reply_drain_s": 0.5,
        "profile": "step",
        "profile_duration_s": 4.0,
        "profile_segment_s": 0.5,
        "seed": 7,
        "zero_hold_s": 0.25,
        "chirp_start_hz": 0.1,
        "chirp_end_hz": 1.0,
        "sine_freqs": [0.2, 0.5],
        "fail_if_no_exclusive": False,
        "assume_exclusive": True,
        "rates": [0.5],
        "accel_bytes": [1],
        "directions": "both",
        "operator_note": "",
    }
    values.update(overrides)
    return Namespace(**values)


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
        args = _base_args(step_s=0.0)
        self.assertEqual(sweep._validate_args(args), "--step-s must be > 0")

    def test_step_profile_preserves_legacy_phase_shape(self) -> None:
        args = _base_args(profile="step", step_s=1.25, rest_s=0.75)

        segments = sweep._build_command_profile(args, rate=0.5, direction=-1, seed=1)

        self.assertEqual([seg.phase for seg in segments], ["pre", "step", "post", "rest"])
        step = segments[1]
        self.assertEqual(step.rate_rad_s, -0.5)
        self.assertEqual(step.duration_s, 1.25)
        self.assertEqual(step.requested_rate_source, "step")

    def test_prbs_profile_is_deterministic_for_same_seed(self) -> None:
        args = _base_args(profile="prbs", profile_duration_s=2.0, profile_segment_s=0.25)

        first = sweep._build_command_profile(args, rate=0.4, direction=1, seed=123)
        second = sweep._build_command_profile(args, rate=0.4, direction=1, seed=123)

        self.assertEqual(
            [(seg.phase, seg.rate_rad_s, seg.duration_s) for seg in first],
            [(seg.phase, seg.rate_rad_s, seg.duration_s) for seg in second],
        )
        prbs_rates = [seg.rate_rad_s for seg in first if seg.phase == "prbs"]
        self.assertTrue(prbs_rates)
        self.assertTrue(all(abs(rate) == 0.4 for rate in prbs_rates))

    def test_random_step_profile_changes_with_seed(self) -> None:
        args = _base_args(profile="random-step", profile_duration_s=3.0, profile_segment_s=0.5)

        first = sweep._build_command_profile(args, rate=0.6, direction=1, seed=10)
        second = sweep._build_command_profile(args, rate=0.6, direction=1, seed=11)

        self.assertNotEqual(
            [seg.rate_rad_s for seg in first if seg.phase == "random_step"],
            [seg.rate_rad_s for seg in second if seg.phase == "random_step"],
        )

    def test_chirp_and_sine_profiles_stay_within_rate_bounds(self) -> None:
        for profile in ("chirp", "sine"):
            with self.subTest(profile=profile):
                args = _base_args(profile=profile, profile_duration_s=2.0, profile_segment_s=0.1)
                segments = sweep._build_command_profile(args, rate=0.7, direction=1, seed=99)
                active_rates = [
                    seg.rate_rad_s
                    for seg in segments
                    if seg.phase in {"chirp", "sine"}
                ]
                self.assertTrue(active_rates)
                self.assertLessEqual(max(abs(rate) for rate in active_rates), 0.7 + 1e-9)

    def test_validation_rejects_generated_profile_without_duration(self) -> None:
        args = _base_args(profile="prbs", profile_duration_s=0.0)
        self.assertEqual(
            sweep._validate_args(args),
            "--profile-duration-s must be > 0 for generated profiles",
        )

    def test_validation_rejects_fail_if_no_exclusive_without_assumption(self) -> None:
        args = _base_args(fail_if_no_exclusive=True, assume_exclusive=False)
        self.assertEqual(
            sweep._validate_args(args),
            "--fail-if-no-exclusive requires --assume-exclusive",
        )

    def test_v2_csv_fields_append_without_removing_legacy_fields(self) -> None:
        self.assertEqual(sweep.CSV_FIELDS[: len(sweep.BASE_CSV_FIELDS)], sweep.BASE_CSV_FIELDS)
        for field in ("profile", "reply_latency_ms", "reply_bytes_hex", "valid_encoder"):
            self.assertIn(field, sweep.CSV_FIELDS)

    def test_manifest_data_includes_v2_profile_and_quality_metadata(self) -> None:
        args = _base_args(profile="prbs", operator_note="bench run")
        quality = sweep._empty_quality()
        quality["samples"] = 3
        quality["reply_latency_ms_values"].extend([1.0, 3.0])

        with patch.object(sweep, "_git_metadata", return_value={"commit": "abc", "dirty": False}):
            manifest = sweep._manifest_data(
                args=args,
                config_paths=[Path("configs/control.yaml")],
                cfg={"control": {}},
                axis_configs=[],
                output_csv=Path("logs/out.csv"),
                quality=quality,
                status="complete",
            )

        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["profile"]["name"], "prbs")
        self.assertEqual(manifest["operator_note"], "bench run")
        self.assertEqual(manifest["quality"]["samples"], 3)
        self.assertEqual(manifest["quality"]["reply_latency_ms"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
