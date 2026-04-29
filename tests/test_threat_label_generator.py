"""Tests for threat label generator.

Validates rule-based classification against known scenarios.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jetson.threat_label_generator import (
    ThreatLabelGenerator,
    BENIGN,
    SUSPICIOUS,
    THREATENING,
    classify_trajectory,
)


def test_critical_zone():
    """Test: inside critical zone always triggers THREATENING."""
    print("\n=== TEST: Critical Zone ===")

    gen = ThreatLabelGenerator()

    test_cases = [
        ({"zone_id": "critical", "distance_rate_to_asset": 0.0}, "stationary in critical"),
        (
            {"zone_id": "critical", "distance_rate_to_asset": 1.0},
            "moving away from critical",
        ),
        (
            {"zone_id": "critical", "distance_rate_to_asset": -5.0},
            "approaching while in critical",
        ),
    ]

    all_pass = True
    for metrics, description in test_cases:
        result = gen.classify_frame(metrics)
        if result == THREATENING:
            print(f"✓ PASS | {description}: THREATENING")
        else:
            print(f"✗ FAIL | {description}: got {result}, expected THREATENING")
            all_pass = False

    return all_pass


def test_restricted_zone():
    """Test: restricted zone with approach triggers THREATENING."""
    print("\n=== TEST: Restricted Zone ===")

    gen = ThreatLabelGenerator()

    test_cases = [
        (
            {"zone_id": "restricted", "distance_rate_to_asset": -0.5},
            "approaching in restricted",
            THREATENING,
        ),
        (
            {"zone_id": "restricted", "distance_rate_to_asset": 0.5},
            "moving away in restricted",
            BENIGN,
        ),
        (
            {"zone_id": "restricted", "distance_rate_to_asset": 0.0},
            "stationary in restricted",
            BENIGN,
        ),
    ]

    all_pass = True
    for metrics, description, expected in test_cases:
        result = gen.classify_frame(metrics)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        actual_name = gen.get_class_name(result)
        expected_name = gen.get_class_name(expected)
        print(f"{status} | {description}: {actual_name} (expected {expected_name})")
        if result != expected:
            all_pass = False

    return all_pass


def test_warning_zone():
    """Test: inside warning zone triggers SUSPICIOUS."""
    print("\n=== TEST: Warning Zone ===")

    gen = ThreatLabelGenerator()

    test_cases = [
        ({"zone_id": "warning", "distance_rate_to_asset": -1.0}, "approaching in warning"),
        ({"zone_id": "warning", "distance_rate_to_asset": 0.0}, "stationary in warning"),
        ({"zone_id": "warning", "distance_rate_to_asset": 1.0}, "moving away in warning"),
    ]

    all_pass = True
    for metrics, description in test_cases:
        result = gen.classify_frame(metrics)
        if result == SUSPICIOUS:
            print(f"✓ PASS | {description}: SUSPICIOUS")
        else:
            print(
                f"✗ FAIL | {description}: got {gen.get_class_name(result)}, expected suspicious"
            )
            all_pass = False

    return all_pass


def test_fast_approach():
    """Test: fast closing velocity triggers THREATENING."""
    print("\n=== TEST: Fast Approach ===")

    gen = ThreatLabelGenerator(approach_speed_threshold_m_s=-1.0)

    test_cases = [
        (
            {"zone_id": "normal", "distance_rate_to_asset": -1.5},
            "fast approach from normal",
            THREATENING,
        ),
        (
            {"zone_id": "normal", "distance_rate_to_asset": -0.5},
            "slow approach from normal",
            BENIGN,
        ),
        (
            {"zone_id": "normal", "distance_rate_to_asset": 0.5},
            "moving away from normal",
            BENIGN,
        ),
    ]

    all_pass = True
    for metrics, description, expected in test_cases:
        result = gen.classify_frame(metrics)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        actual_name = gen.get_class_name(result)
        expected_name = gen.get_class_name(expected)
        print(f"{status} | {description}: {actual_name} (expected {expected_name})")
        if result != expected:
            all_pass = False

    return all_pass


def test_loitering():
    """Test: low velocity + high dwell time triggers SUSPICIOUS."""
    print("\n=== TEST: Loitering Detection ===")

    gen = ThreatLabelGenerator(
        loiter_dwell_threshold_s=2.0, loiter_velocity_threshold_m_s=0.5
    )

    test_cases = [
        (
            {
                "zone_id": "normal",
                "distance_rate_to_asset": 0.0,
                "time_inside_zone": 3.0,
                "velocity_x": 0.1,
                "velocity_y": 0.1,
            },
            "loitering (low vel, high dwell)",
            SUSPICIOUS,
        ),
        (
            {
                "zone_id": "normal",
                "distance_rate_to_asset": 0.0,
                "time_inside_zone": 1.0,
                "velocity_x": 0.1,
                "velocity_y": 0.1,
            },
            "not loitering (dwell too short)",
            BENIGN,
        ),
        (
            {
                "zone_id": "normal",
                "distance_rate_to_asset": 0.0,
                "time_inside_zone": 3.0,
                "velocity_x": 1.0,
                "velocity_y": 1.0,
            },
            "not loitering (velocity too high)",
            BENIGN,
        ),
    ]

    all_pass = True
    for metrics, description, expected in test_cases:
        result = gen.classify_frame(metrics)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        actual_name = gen.get_class_name(result)
        expected_name = gen.get_class_name(expected)
        print(f"{status} | {description}: {actual_name} (expected {expected_name})")
        if result != expected:
            all_pass = False

    return all_pass


def test_benign_default():
    """Test: default case is BENIGN."""
    print("\n=== TEST: Benign Default ===")

    gen = ThreatLabelGenerator()

    test_cases = [
        (
            {"zone_id": "normal", "distance_rate_to_asset": 0.5},
            "moving away from asset",
        ),
        ({}, "empty metrics (all defaults)"),
        (
            {
                "zone_id": "normal",
                "distance_rate_to_asset": 0.0,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
                "time_inside_zone": 0.0,
            },
            "stationary outside zones",
        ),
    ]

    all_pass = True
    for metrics, description in test_cases:
        result = gen.classify_frame(metrics)
        if result == BENIGN:
            print(f"✓ PASS | {description}: BENIGN")
        else:
            print(f"✗ FAIL | {description}: got {gen.get_class_name(result)}, expected benign")
            all_pass = False

    return all_pass


def test_window_aggregation():
    """Test: window classification with majority and max_threat aggregation."""
    print("\n=== TEST: Window Aggregation ===")

    gen = ThreatLabelGenerator()

    # Mixed window: 10 benign, 4 suspicious, 2 threatening
    window = [
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "normal", "distance_rate_to_asset": 0.5},  # BENIGN
        {"zone_id": "warning", "distance_rate_to_asset": 0.5},  # SUSPICIOUS
        {"zone_id": "warning", "distance_rate_to_asset": 0.5},  # SUSPICIOUS
        {"zone_id": "warning", "distance_rate_to_asset": 0.5},  # SUSPICIOUS
        {"zone_id": "warning", "distance_rate_to_asset": 0.5},  # SUSPICIOUS
        {"zone_id": "critical", "distance_rate_to_asset": 0.0},  # THREATENING
        {"zone_id": "critical", "distance_rate_to_asset": 0.0},  # THREATENING
    ]

    all_pass = True

    # Test max_threat aggregation (pessimistic)
    result_max = gen.classify_window(window, aggregation="max_threat")
    if result_max == THREATENING:
        print(f"✓ PASS | max_threat aggregation: THREATENING (correct pessimistic)")
    else:
        print(
            f"✗ FAIL | max_threat aggregation: got {gen.get_class_name(result_max)}, "
            f"expected threatening"
        )
        all_pass = False

    # Test majority aggregation
    result_maj = gen.classify_window(window, aggregation="majority")
    if result_maj == BENIGN:
        print(f"✓ PASS | majority aggregation: BENIGN (10/16 frames)")
    else:
        print(
            f"✗ FAIL | majority aggregation: got {gen.get_class_name(result_maj)}, expected benign"
        )
        all_pass = False

    return all_pass


def test_trajectory_classification():
    """Test: classify complete trajectory."""
    print("\n=== TEST: Trajectory Classification ===")

    # Simulate approaching trajectory
    trajectory = [
        {"zone_id": "warning", "distance_rate_to_asset": -0.5},  # SUSPICIOUS
        {"zone_id": "warning", "distance_rate_to_asset": -0.8},  # SUSPICIOUS
        {"zone_id": "restricted", "distance_rate_to_asset": -1.0},  # THREATENING
        {"zone_id": "critical", "distance_rate_to_asset": -1.5},  # THREATENING
        {"zone_id": "critical", "distance_rate_to_asset": 0.0},  # THREATENING
    ]

    labels = classify_trajectory(trajectory)
    expected = [SUSPICIOUS, SUSPICIOUS, THREATENING, THREATENING, THREATENING]

    all_pass = True
    if labels == expected:
        print(f"✓ PASS | Approaching trajectory classified correctly")
        print(f"        Labels: {[{0: 'B', 1: 'S', 2: 'T'}[l] for l in labels]}")
    else:
        print(f"✗ FAIL | Approaching trajectory misclassified")
        print(
            f"        Expected: {[{0: 'B', 1: 'S', 2: 'T'}[l] for l in expected]}\n"
            f"        Got:      {[{0: 'B', 1: 'S', 2: 'T'}[l] for l in labels]}"
        )
        all_pass = False

    return all_pass


def main():
    """Run all tests."""
    print("=" * 70)
    print("THREAT LABEL GENERATOR TESTS")
    print("=" * 70)

    results = {
        "Critical Zone": test_critical_zone(),
        "Restricted Zone": test_restricted_zone(),
        "Warning Zone": test_warning_zone(),
        "Fast Approach": test_fast_approach(),
        "Loitering Detection": test_loitering(),
        "Benign Default": test_benign_default(),
        "Window Aggregation": test_window_aggregation(),
        "Trajectory Classification": test_trajectory_classification(),
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
