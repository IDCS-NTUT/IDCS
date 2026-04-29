#!/usr/bin/env python3
"""Validation script for threat evaluation calculations.

Tests distance computation, zone membership, and distance rate calculations
against expected values from threat evaluation configuration.

Run from project root:
    python -m tests.validate_threat_calculations
"""

import sys
import math
from pathlib import Path

# Add repo to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from common.threat_calc import (
    compute_distance_to_asset,
    compute_distance_rate,
    get_zone_id_for_distance,
    get_zone_membership,
    parse_zone_config,
    validate_asset_position,
)


def test_distance_computation():
    """Test basic distance calculations."""
    print("\n=== TEST: Distance Computation ===")

    asset_pos = (0.0, 0.0)
    test_cases = [
        ((0.0, 0.0), 0.0, "at asset"),
        ((5.0, 0.0), 5.0, "5m east"),
        ((0.0, 5.0), 5.0, "5m north"),
        ((-5.0, 0.0), 5.0, "5m west"),
        ((0.0, -5.0), 5.0, "5m south"),
        ((3.0, 4.0), 5.0, "3-4-5 triangle"),
        ((10.0, 10.0), math.sqrt(200), "diagonal"),
    ]

    all_pass = True
    for target_pos, expected_dist, description in test_cases:
        try:
            computed = compute_distance_to_asset(target_pos, asset_pos)
            passed = abs(computed - expected_dist) < 1e-6
            status = "✓ PASS" if passed else "✗ FAIL"
            print(
                f"{status} | {description:20s} | computed={computed:7.2f}m, "
                f"expected={expected_dist:7.2f}m"
            )
            if not passed:
                all_pass = False
        except Exception as e:
            print(f"✗ FAIL | {description:20s} | Exception: {e}")
            all_pass = False

    return all_pass


def test_zone_membership():
    """Test zone assignment based on distances."""
    print("\n=== TEST: Zone Membership (Radii: critical=5m, restricted=10m, warning=20m) ===")

    zone_radii = {"critical": 5.0, "restricted": 10.0, "warning": 20.0}

    test_cases = [
        (0.0, {"critical": True, "restricted": True, "warning": True}, "at asset center"),
        (2.5, {"critical": True, "restricted": True, "warning": True}, "2.5m in critical"),
        (5.0, {"critical": True, "restricted": True, "warning": True}, "exactly at critical radius"),
        (7.5, {"critical": False, "restricted": True, "warning": True}, "7.5m in restricted"),
        (10.0, {"critical": False, "restricted": True, "warning": True}, "exactly at restricted radius"),
        (15.0, {"critical": False, "restricted": False, "warning": True}, "15m in warning"),
        (20.0, {"critical": False, "restricted": False, "warning": True}, "exactly at warning radius"),
        (25.0, {"critical": False, "restricted": False, "warning": False}, "25m outside all zones"),
    ]

    all_pass = True
    for distance, expected_membership, description in test_cases:
        try:
            computed = get_zone_membership(distance, zone_radii)
            passed = computed == expected_membership
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} | {description:30s} | distance={distance:5.1f}m")
            if not passed:
                print(
                    f"       Expected: {expected_membership}\n"
                    f"       Got:      {computed}"
                )
                all_pass = False
        except Exception as e:
            print(f"✗ FAIL | {description:30s} | Exception: {e}")
            all_pass = False

    return all_pass


def test_zone_id_assignment():
    """Test innermost zone ID assignment."""
    print("\n=== TEST: Zone ID Assignment (innermost zone wins) ===")

    zone_radii = {"critical": 5.0, "restricted": 10.0, "warning": 20.0}

    test_cases = [
        (0.0, "critical", "at center"),
        (5.0, "critical", "at critical radius"),
        (7.5, "restricted", "in restricted zone"),
        (10.0, "restricted", "at restricted radius"),
        (15.0, "warning", "in warning zone"),
        (20.0, "warning", "at warning radius"),
        (25.0, "normal", "outside all zones"),
    ]

    all_pass = True
    for distance, expected_zone, description in test_cases:
        try:
            computed = get_zone_id_for_distance(distance, zone_radii)
            passed = computed == expected_zone
            status = "✓ PASS" if passed else "✗ FAIL"
            print(
                f"{status} | {description:25s} | distance={distance:5.1f}m "
                f"→ zone={computed:12s} (expected={expected_zone})"
            )
            if not passed:
                all_pass = False
        except Exception as e:
            print(f"✗ FAIL | {description:25s} | Exception: {e}")
            all_pass = False

    return all_pass


def test_distance_rate():
    """Test distance rate (velocity towards/away) calculation."""
    print("\n=== TEST: Distance Rate Calculation ===")

    test_cases = [
        # (current, previous, dt, expected_rate, description)
        (10.0, 15.0, 1.0, -5.0, "approaching at 5 m/s"),
        (20.0, 10.0, 1.0, 10.0, "receding at 10 m/s"),
        (10.0, 10.0, 1.0, 0.0, "no change"),
        (9.5, 10.0, 0.1, -5.0, "approaching at 5 m/s, short dt"),
        (10.5, 10.0, 0.1, 5.0, "receding at 5 m/s, short dt"),
    ]

    all_pass = True
    for current, previous, dt, expected_rate, description in test_cases:
        try:
            computed = compute_distance_rate(current, previous, dt)
            passed = abs(computed - expected_rate) < 1e-6
            status = "✓ PASS" if passed else "✗ FAIL"
            sign = "approaching" if computed < 0 else "receding" if computed > 0 else "stationary"
            print(
                f"{status} | {description:35s} | rate={computed:7.2f} m/s ({sign})"
            )
            if not passed:
                all_pass = False
        except Exception as e:
            print(f"✗ FAIL | {description:35s} | Exception: {e}")
            all_pass = False

    return all_pass


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n=== TEST: Error Handling ===")

    all_pass = True

    # Test invalid distance (negative)
    try:
        get_zone_id_for_distance(-1.0, {"critical": 5.0})
        print("✗ FAIL | Negative distance should raise ValueError")
        all_pass = False
    except ValueError:
        print("✓ PASS | Negative distance raises ValueError")

    # Test invalid distance (infinity)
    try:
        get_zone_id_for_distance(float("inf"), {"critical": 5.0})
        print("✗ FAIL | Infinite distance should raise ValueError")
        all_pass = False
    except ValueError:
        print("✓ PASS | Infinite distance raises ValueError")

    # Test invalid dt for rate calculation
    try:
        compute_distance_rate(10.0, 10.0, 0.0)
        print("✗ FAIL | Zero dt should raise ValueError")
        all_pass = False
    except ValueError:
        print("✓ PASS | Zero dt raises ValueError")

    # Test asset position validation
    try:
        validate_asset_position((0.0, 0.0))
        print("✓ PASS | Valid asset position accepted")
    except ValueError as e:
        print(f"✗ FAIL | Valid position rejected: {e}")
        all_pass = False

    try:
        validate_asset_position((float("inf"), 0.0))
        print("✗ FAIL | Infinite coordinate should raise ValueError")
        all_pass = False
    except ValueError:
        print("✓ PASS | Infinite coordinate raises ValueError")

    return all_pass


def test_zone_config_parsing():
    """Test zone configuration parsing."""
    print("\n=== TEST: Zone Config Parsing ===")

    # Valid config
    try:
        config = {
            "critical": {"type": "circle", "radius_m": 5.0},
            "restricted": {"type": "circle", "radius_m": 10.0},
            "warning": {"type": "circle", "radius_m": 20.0},
        }
        radii = parse_zone_config(config)
        expected = {"critical": 5.0, "restricted": 10.0, "warning": 20.0}
        if radii == expected:
            print("✓ PASS | Valid config parsed correctly")
            return True
        else:
            print(f"✗ FAIL | Config mismatch. Got {radii}, expected {expected}")
            return False
    except Exception as e:
        print(f"✗ FAIL | Valid config raised: {e}")
        return False


def test_example_threat_scenario():
    """Test a realistic threat scenario with multiple targets."""
    print("\n=== TEST: Example Threat Scenario ===")
    print("Scenario: Defended asset at origin, multiple targets approaching\n")

    asset_pos = (0.0, 0.0)
    zone_radii = {"critical": 5.0, "restricted": 10.0, "warning": 20.0}

    targets = [
        {"id": 1, "name": "Drone Direct Approach", "positions": [(-20, 0), (-15, 0), (-10, 0), (-5, 0), (-2, 0)]},
        {"id": 2, "name": "Person Lateral Pass", "positions": [(0, -25), (0, -15), (0, -10), (0, -5), (0, 5)]},
        {"id": 3, "name": "Hovering Threat", "positions": [(-6, 0), (-6, 0), (-6, 0), (-6, 0), (-6, 0)]},
    ]

    all_pass = True
    for target in targets:
        print(f"\nTarget {target['id']}: {target['name']}")
        print(f"{'Frame':<6} {'Pos (x,y)':<15} {'Distance':<12} {'Zone':<12} {'Approach?':<10}")
        print("-" * 60)

        prev_distance = None
        for frame, pos in enumerate(target["positions"]):
            try:
                distance = compute_distance_to_asset(pos, asset_pos)
                zone_id = get_zone_id_for_distance(distance, zone_radii)

                approach_str = ""
                if prev_distance is not None:
                    rate = compute_distance_rate(distance, prev_distance, 0.033)  # 30 fps
                    approach_str = "Yes" if rate < -0.1 else "No" if rate > 0.1 else "Static"

                print(
                    f"{frame:<6} {str(pos):<15} {distance:>10.2f}m {zone_id:>12s} {approach_str:>10s}"
                )
                prev_distance = distance
            except Exception as e:
                print(f"{frame:<6} {str(pos):<15} ERROR: {e}")
                all_pass = False

    return all_pass


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("THREAT EVALUATION CALCULATION VALIDATION")
    print("=" * 70)

    results = {
        "Distance Computation": test_distance_computation(),
        "Zone Membership": test_zone_membership(),
        "Zone ID Assignment": test_zone_id_assignment(),
        "Distance Rate": test_distance_rate(),
        "Error Handling": test_error_handling(),
        "Zone Config Parsing": test_zone_config_parsing(),
        "Example Scenario": test_example_threat_scenario(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {test_name}")

    all_passed = all(results.values())
    print("=" * 70)
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
