#!/usr/bin/env python3
"""Validation script for threat evaluation calculations.

Tests distance computation, zone membership, velocity, bounding box metrics,
and time tracking against expected values from threat evaluation configuration.

Run from project root:
    python tests/validate_threat_calculations.py
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
    compute_velocity,
    compute_bbox_center,
    validate_bbox,
    validate_confidence,
    get_zone_id_for_distance,
    get_zone_membership,
    parse_zone_config,
    validate_asset_position,
    TargetThreatState,
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


def test_bbox_operations():
    """Test bounding box center and validation."""
    print("\n=== TEST: Bounding Box Operations ===")

    all_pass = True

    # Test center computation
    test_cases = [
        (0, 0, 100, 100, 50, 50, "box at origin"),
        (100, 100, 200, 200, 200, 200, "offset box"),
        (10.5, 20.5, 30.0, 40.0, 25.5, 40.5, "decimal coordinates"),
    ]

    for x, y, w, h, expected_cx, expected_cy, desc in test_cases:
        try:
            cx, cy = compute_bbox_center(x, y, w, h)
            if abs(cx - expected_cx) < 1e-6 and abs(cy - expected_cy) < 1e-6:
                print(f"✓ PASS | Bbox center {desc}: ({cx:.1f}, {cy:.1f})")
            else:
                print(
                    f"✗ FAIL | Bbox center {desc}: got ({cx:.1f}, {cy:.1f}), "
                    f"expected ({expected_cx:.1f}, {expected_cy:.1f})"
                )
                all_pass = False
        except Exception as e:
            print(f"✗ FAIL | Bbox center {desc}: {e}")
            all_pass = False

    # Test bbox validation
    try:
        validate_bbox(10, 20, 100, 50)
        print("✓ PASS | Valid bbox accepted")
    except ValueError as e:
        print(f"✗ FAIL | Valid bbox rejected: {e}")
        all_pass = False

    try:
        validate_bbox(10, 20, -100, 50)
        print("✗ FAIL | Negative width should raise ValueError")
        all_pass = False
    except ValueError:
        print("✓ PASS | Negative width raises ValueError")

    return all_pass


def test_confidence():
    """Test confidence validation and normalization."""
    print("\n=== TEST: Confidence Validation ===")

    test_cases = [
        (0.5, 0.5, "valid confidence"),
        (0.0, 0.0, "zero confidence"),
        (1.0, 1.0, "max confidence"),
        (1.5, 1.0, "over-range clamped to 1.0"),
        (-0.5, 0.0, "negative clamped to 0.0"),
    ]

    all_pass = True
    for conf_in, expected, desc in test_cases:
        try:
            result = validate_confidence(conf_in)
            if abs(result - expected) < 1e-6:
                print(f"✓ PASS | {desc}: {conf_in} → {result}")
            else:
                print(
                    f"✗ FAIL | {desc}: {conf_in} → {result}, expected {expected}"
                )
                all_pass = False
        except Exception as e:
            print(f"✗ FAIL | {desc}: {e}")
            all_pass = False

    return all_pass


def test_velocity_computation():
    """Test 2D velocity calculation."""
    print("\n=== TEST: Velocity Computation ===")

    test_cases = [
        # (curr, prev, dt, expected_vx, expected_vy, description)
        ((10, 0), (0, 0), 1.0, 10.0, 0.0, "moving east at 10 units/s"),
        ((0, 10), (0, 0), 1.0, 0.0, 10.0, "moving north at 10 units/s"),
        ((3, 4), (0, 0), 1.0, 3.0, 4.0, "diagonal motion"),
        ((0, 0), (0, 0), 1.0, 0.0, 0.0, "stationary"),
        ((5, 5), (0, 0), 0.5, 10.0, 10.0, "scaled by shorter dt"),
    ]

    all_pass = True
    for curr, prev, dt, exp_vx, exp_vy, desc in test_cases:
        try:
            vx, vy = compute_velocity(curr, prev, dt)
            if abs(vx - exp_vx) < 1e-6 and abs(vy - exp_vy) < 1e-6:
                print(
                    f"✓ PASS | {desc}: velocity=({vx:.2f}, {vy:.2f}) units/s"
                )
            else:
                print(
                    f"✗ FAIL | {desc}: got ({vx:.2f}, {vy:.2f}), "
                    f"expected ({exp_vx:.2f}, {exp_vy:.2f})"
                )
                all_pass = False
        except Exception as e:
            print(f"✗ FAIL | {desc}: {e}")
            all_pass = False

    return all_pass


def test_target_threat_state():
    """Test TargetThreatState tracking over multiple frames."""
    print("\n=== TEST: TargetThreatState Multi-Frame Tracking ===")

    asset_xy = (0.0, 0.0)
    zone_radii = {"critical": 5.0, "restricted": 10.0, "warning": 20.0}

    state = TargetThreatState(
        target_id=1,
        zone_radii=zone_radii,
        asset_xy=asset_xy,
    )

    # Simulate drone approaching asset over 5 frames (0.033s each)
    frames = [
        {"pos": (-20, 0), "time": 0.0, "desc": "warning zone, entering"},
        {"pos": (-15, 0), "time": 0.033, "desc": "warning zone, approaching"},
        {"pos": (-10, 0), "time": 0.066, "desc": "restricted zone, entering"},
        {"pos": (-5, 0), "time": 0.099, "desc": "critical zone, entering"},
        {"pos": (-3, 0), "time": 0.132, "desc": "critical zone, deep threat"},
    ]

    all_pass = True
    prev_zone = None

    print("\nFrame-by-frame trajectory:")
    print(
        f"{'Frame':<6} {'Pos':<12} {'Distance':<10} {'Zone':<12} "
        f"{'Vel(m/s)':<12} {'Dwell(s)':<10}"
    )
    print("-" * 70)

    for frame_idx, frame_data in enumerate(frames):
        try:
            metrics = state.update(
                current_xy=frame_data["pos"],
                current_time=frame_data["time"],
                confidence=0.95,
                bbox_x=0,
                bbox_y=0,
                bbox_width=50,
                bbox_height=100,
            )

            zone = metrics["zone_id"]
            distance = metrics["distance_to_asset"]
            velocity_mag = math.sqrt(
                metrics["velocity_x"] ** 2 + metrics["velocity_y"] ** 2
            )
            dwell = metrics["time_inside_zone"]

            vel_str = f"{velocity_mag:.2f}" if frame_idx > 0 else "—"
            dwell_str = f"{dwell:.3f}" if dwell > 0 else "0.000"

            print(
                f"{frame_idx:<6} {str(frame_data['pos']):<12} "
                f"{distance:>8.2f}m {zone:>12s} {vel_str:>11s} m/s {dwell_str:>9s}"
            )

            # Verify zone transitions
            if prev_zone is not None and zone != prev_zone:
                print(
                    f"       → Zone transition: {prev_zone} → {zone} "
                    f"at t={frame_data['time']:.3f}s"
                )

            # Verify approaching behavior (distance decreasing, rate negative)
            if frame_idx > 0:
                expected_approach = distance < (-20 + frame_idx * 5)
                actual_rate = metrics["distance_rate_to_asset"]
                if expected_approach and actual_rate < -0.1:
                    print(f"       ✓ Approaching confirmed (rate={actual_rate:.2f} m/s)")
                elif expected_approach:
                    print(
                        f"       ✗ Should be approaching but rate={actual_rate:.2f} m/s"
                    )
                    all_pass = False

            prev_zone = zone

        except Exception as e:
            print(f"✗ FAIL | Frame {frame_idx}: {e}")
            all_pass = False

    # Verify metrics in final state
    final_metrics = state.update(
        current_xy=(-3, 0),
        current_time=0.2,
        confidence=0.95,
        bbox_x=0,
        bbox_y=0,
        bbox_width=50,
        bbox_height=100,
    )

    # Check that we can retrieve accumulated zone times
    try:
        total_critical = state.get_total_zone_time("critical")
        print(
            f"\n✓ Total time in critical zone: {total_critical:.3f}s "
            f"(dwell={final_metrics['time_inside_zone']:.3f}s)"
        )
    except Exception as e:
        print(f"✗ FAIL | get_total_zone_time: {e}")
        all_pass = False

    return all_pass


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
        "Bounding Box Operations": test_bbox_operations(),
        "Confidence Validation": test_confidence(),
        "Velocity Computation": test_velocity_computation(),
        "Target Threat State": test_target_threat_state(),
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
