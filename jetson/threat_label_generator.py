"""Rule-based threat label generator for training data.

Generates ground-truth threat class labels from target metrics using deterministic rules.
Used during dataset creation to label simulated scenarios.

Threat classes:
- 0: BENIGN - normal movement, not approaching, outside warning zone
- 1: SUSPICIOUS - loitering, near boundary, uncertain behavior
- 2: THREATENING - in critical zone, fast approach, restricted zone entry
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

__all__ = ["ThreatLabelGenerator", "ThreatClass"]

ThreatClass = Literal[0, 1, 2]
BENIGN = 0
SUSPICIOUS = 1
THREATENING = 2


class ThreatLabelGenerator:
    """Generates threat labels from target metrics using rule-based classification.

    Rules are applied in priority order:
    1. Inside critical zone → THREATENING
    2. Inside restricted zone AND approaching → THREATENING
    3. Fast closing approach → THREATENING
    4. Inside warning zone → SUSPICIOUS
    5. Loitering near boundary → SUSPICIOUS
    6. Else → BENIGN

    Attributes:
        approach_speed_threshold_m_s: Threshold for "fast approach" (negative distance_rate)
        loiter_dwell_threshold_s: Minimum dwell time to flag loitering
        loiter_velocity_threshold_m_s: Maximum velocity to flag as loitering
    """

    def __init__(
        self,
        approach_speed_threshold_m_s: float = -1.0,
        loiter_dwell_threshold_s: float = 2.0,
        loiter_velocity_threshold_m_s: float = 0.5,
    ):
        """Initialize label generator.

        Args:
            approach_speed_threshold_m_s: Threshold for "fast approach" behavior.
                Negative distance rates more extreme than this trigger threatening.
                Default: -1.0 m/s (approaching at >1 m/s)
            loiter_dwell_threshold_s: Minimum dwell time to flag loitering.
                Default: 2.0 seconds
            loiter_velocity_threshold_m_s: Maximum velocity to flag as loitering.
                Default: 0.5 m/s
        """
        self.approach_speed_threshold = approach_speed_threshold_m_s
        self.loiter_dwell_threshold = loiter_dwell_threshold_s
        self.loiter_velocity_threshold = loiter_velocity_threshold_m_s

    def classify_frame(self, metrics: Dict[str, float]) -> ThreatClass:
        """Classify a single frame based on target metrics.

        Args:
            metrics: Dict with keys:
                - zone_id: str ("critical", "restricted", "warning", "normal")
                - distance_to_asset: float (meters)
                - distance_rate_to_asset: float (m/s, negative=approaching)
                - time_inside_zone: float (seconds)
                - velocity_x, velocity_y: float (units/second)

        Returns:
            Threat class: BENIGN (0), SUSPICIOUS (1), or THREATENING (2)
        """
        zone_id = metrics.get("zone_id", "normal")
        distance_rate = metrics.get("distance_rate_to_asset", 0.0)
        dwell_time = metrics.get("time_inside_zone", 0.0)
        velocity_x = metrics.get("velocity_x", 0.0)
        velocity_y = metrics.get("velocity_y", 0.0)

        # Rule 1: Inside critical zone → THREATENING
        if zone_id == "critical":
            return THREATENING

        # Rule 2: Inside restricted zone AND approaching → THREATENING
        if zone_id == "restricted" and distance_rate < 0:
            return THREATENING

        # Rule 3: Fast closing approach → THREATENING
        if distance_rate < self.approach_speed_threshold:
            return THREATENING

        # Rule 4: Inside warning zone → SUSPICIOUS
        if zone_id == "warning":
            return SUSPICIOUS

        # Rule 5: Loitering near boundary → SUSPICIOUS
        # Low velocity + elevated dwell time = loitering
        velocity_mag = (velocity_x**2 + velocity_y**2) ** 0.5
        if (
            dwell_time > self.loiter_dwell_threshold
            and velocity_mag < self.loiter_velocity_threshold
        ):
            return SUSPICIOUS

        # Rule 6: Else → BENIGN
        return BENIGN

    def classify_window(
        self,
        metrics_window: list[Dict[str, float]],
        aggregation: Literal["majority", "max_threat"] = "max_threat",
    ) -> ThreatClass:
        """Classify a window of frames (16 frames = 1 training sample).

        Args:
            metrics_window: List of per-frame metrics dicts
            aggregation: How to combine per-frame labels:
                - "majority": most common class in window
                - "max_threat": highest threat level in window (default)

        Returns:
            Threat class for the entire window
        """
        if not metrics_window:
            return BENIGN

        # Classify each frame
        frame_labels = [self.classify_frame(m) for m in metrics_window]

        if aggregation == "max_threat":
            # Return highest threat in window (pessimistic)
            return max(frame_labels)  # type: ignore
        else:  # majority
            # Return most common class
            counts = {BENIGN: 0, SUSPICIOUS: 0, THREATENING: 0}
            for label in frame_labels:
                counts[label] += 1
            return max(counts.items(), key=lambda x: x[1])[0]

    def get_class_name(self, threat_class: ThreatClass) -> str:
        """Get human-readable class name.

        Args:
            threat_class: Class ID (0, 1, or 2)

        Returns:
            String: "benign", "suspicious", or "threatening"
        """
        names = {BENIGN: "benign", SUSPICIOUS: "suspicious", THREATENING: "threatening"}
        return names.get(threat_class, "unknown")


def classify_trajectory(
    trajectory: list[Dict[str, float]],
    label_gen: Optional[ThreatLabelGenerator] = None,
) -> list[ThreatClass]:
    """Classify a complete trajectory frame-by-frame.

    Convenience function to classify all frames in a track.

    Args:
        trajectory: List of per-frame metrics dicts
        label_gen: Label generator instance (default: new instance with defaults)

    Returns:
        List of threat classes, one per frame
    """
    if label_gen is None:
        label_gen = ThreatLabelGenerator()

    return [label_gen.classify_frame(frame) for frame in trajectory]
