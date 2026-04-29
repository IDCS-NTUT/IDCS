#!/usr/bin/env python3
"""Threat evaluation integration for server.

Manages threat calculation pipeline:
1. Extract metrics from detections
2. Maintain temporal windows per track
3. Run ML inference
4. Update boxes with threat scores
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from jetson.threat_inference import ThreatInferenceEngine, ThreatMetricsWindow
from jetson.threat_label_generator import BENIGN, SUSPICIOUS, THREATENING, ThreatLabelGenerator

__all__ = ["ThreatEvaluator"]

logger = logging.getLogger(__name__)


class ThreatEvaluator:
    """Manages threat evaluation for all tracked targets.

    Maintains:
    - Per-track threat state (for metrics calculation)
    - Per-track metrics windows (for temporal features)
    - Threat model inference engine
    """

    def __init__(
        self,
        model_engine: Optional[ThreatInferenceEngine],
        defended_asset_xy: Tuple[float, float],
        threat_zones_config: Optional[Dict[str, Any]] = None,
        enable_rule_based: bool = True,
    ):
        """Initialize threat evaluator.

        Args:
            model_engine: ONNX inference engine (can be None to disable)
            defended_asset_xy: Asset position in world coordinates
            threat_zones_config: Zone configuration from system.yaml
            enable_rule_based: Whether to also compute rule-based threat scores
        """
        self.model_engine = model_engine
        self.defended_asset_xy = defended_asset_xy
        self.threat_zones_config = threat_zones_config or {}
        self.enable_rule_based = enable_rule_based
        self.label_generator = ThreatLabelGenerator()

        # Per-track state
        self.track_history: Dict[int, Dict[str, Any]] = {}
        self.track_threat_states = self.track_history
        self.track_metrics_windows: Dict[int, ThreatMetricsWindow] = {}
        self.track_last_scores: Dict[int, Dict[str, Any]] = {}
        self.last_frame_time = 0.0

        logger.info(
            f"Threat evaluator initialized with asset at {defended_asset_xy}, "
            f"model_enabled={model_engine is not None}, rule_based={enable_rule_based}"
        )

    def update(
        self,
        boxes: List[Any],
        frame_w: int,
        frame_h: int,
        current_time_s: Optional[float] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Update threat scores for all boxes.

        Args:
            boxes: List of Box objects with track_id, x, y, w, h, conf
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
            current_time_s: Current time (uses time.time() if None)

        Returns:
            Dictionary mapping track_id to threat scores:
            {
                track_id: {
                    "threat_level": "benign|suspicious|threatening",
                    "threat_confidence": float,
                    "threat_score_benign": float,
                    "threat_score_suspicious": float,
                    "threat_score_threatening": float,
                }
            }
        """
        if current_time_s is None:
            current_time_s = time.time()

        # Calculate dt from last update
        dt = current_time_s - self.last_frame_time
        if dt <= 0:
            dt = 1.0 / 30.0  # Default to ~30fps
        self.last_frame_time = current_time_s

        threat_scores = {}

        for box in boxes:
            if box.track_id is None:
                continue

            track_id = int(box.track_id)

            # Get or create per-track buffers
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    "prev_center_xy": None,
                    "prev_distance_m": None,
                    "zone_entry_time_s": None,
                    "current_zone": None,
                    "last_update_time_s": None,
                }
                self.track_metrics_windows[track_id] = ThreatMetricsWindow(max_window_size=16)

            history = self.track_history[track_id]
            window = self.track_metrics_windows[track_id]

            center_x_px = box.x * frame_w + (box.w * frame_w) / 2.0
            center_y_px = box.y * frame_h + (box.h * frame_h) / 2.0
            distance_m = float(box.distance_m) if box.distance_m is not None else 0.0

            velocity_x = 0.0
            velocity_y = 0.0
            if history["prev_center_xy"] is not None and history["last_update_time_s"] is not None:
                prev_x, prev_y = history["prev_center_xy"]
                if dt > 0:
                    velocity_x = (center_x_px - prev_x) / dt / max(frame_w, 1)
                    velocity_y = (center_y_px - prev_y) / dt / max(frame_h, 1)

            distance_rate = 0.0
            if history["prev_distance_m"] is not None and dt > 0:
                distance_rate = (distance_m - float(history["prev_distance_m"])) / dt

            zone_id = "normal"
            if distance_m <= 5.0:
                zone_id = "critical"
            elif distance_m <= 10.0:
                zone_id = "restricted"
            elif distance_m <= 20.0:
                zone_id = "warning"

            if history["current_zone"] != zone_id:
                history["current_zone"] = zone_id
                history["zone_entry_time_s"] = current_time_s

            dwell_time = 0.0
            if history["zone_entry_time_s"] is not None:
                dwell_time = max(0.0, current_time_s - float(history["zone_entry_time_s"]))

            frame_metrics = {
                "center_x": min(1.0, max(0.0, center_x_px / max(frame_w, 1))),
                "center_y": min(1.0, max(0.0, center_y_px / max(frame_h, 1))),
                "bbox_w": box.w,
                "bbox_h": box.h,
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "confidence": box.conf,
                "distance": min(1.0, distance_m / 30.0),
                "distance_rate": distance_rate / 5.0,
                "zone_level": {"critical": 0.0, "restricted": 1.0, "warning": 2.0, "normal": 3.0}[zone_id] / 3.0,
                "dwell_time": min(1.0, dwell_time / 10.0),
            }
            window.add_frame_metrics(frame_metrics)

            threat_level = "benign"
            threat_confidence = float(box.conf)
            threat_scores_arr = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            if self.model_engine and window.is_full():
                features = window.get_features()
                if features is not None:
                    try:
                        class_id, confidence, probs = self.model_engine.predict(features)
                        threat_level = {0: "benign", 1: "suspicious", 2: "threatening"}[class_id]
                        threat_confidence = float(confidence)
                        threat_scores_arr = probs.astype(np.float32)
                    except Exception as e:
                        logger.warning(f"Inference error for track {track_id}: {e}")
            elif self.enable_rule_based:
                threat_class = self.label_generator.classify_frame(
                    {
                        "zone_id": zone_id,
                        "distance_rate_to_asset": distance_rate,
                        "time_inside_zone": dwell_time,
                        "velocity_x": velocity_x,
                        "velocity_y": velocity_y,
                    }
                )
                threat_level = {BENIGN: "benign", SUSPICIOUS: "suspicious", THREATENING: "threatening"}[threat_class]
                threat_scores_arr = np.array(
                    [1.0, 0.0, 0.0] if threat_class == BENIGN else [0.0, 1.0, 0.0] if threat_class == SUSPICIOUS else [0.0, 0.0, 1.0],
                    dtype=np.float32,
                )

            threat_scores[track_id] = {
                "threat_level": threat_level,
                "threat_confidence": threat_confidence,
                "threat_score_benign": float(threat_scores_arr[0]),
                "threat_score_suspicious": float(threat_scores_arr[1]),
                "threat_score_threatening": float(threat_scores_arr[2]),
            }

            history["prev_center_xy"] = (center_x_px, center_y_px)
            history["prev_distance_m"] = distance_m
            history["last_update_time_s"] = current_time_s

        # Clean up old tracks
        active_track_ids = {box.track_id for box in boxes if box.track_id is not None}
        for track_id in list(self.track_history.keys()):
            if track_id not in active_track_ids:
                del self.track_history[track_id]
                del self.track_metrics_windows[track_id]
                self.track_last_scores.pop(track_id, None)

        return threat_scores

    def apply_threat_scores(
        self,
        boxes: List[Any],
        threat_scores: Dict[int, Dict[str, Any]],
    ) -> None:
        """Apply threat scores to boxes.

        Args:
            boxes: List of Box objects to update
            threat_scores: Dictionary from update() method
        """
        for box in boxes:
            if box.track_id is None:
                continue

            track_id = int(box.track_id)
            if track_id in threat_scores:
                scores = threat_scores[track_id]
                box.threat_level = scores["threat_level"]
                box.threat_confidence = scores["threat_confidence"]
                box.threat_score_benign = scores["threat_score_benign"]
                box.threat_score_suspicious = scores["threat_score_suspicious"]
                box.threat_score_threatening = scores["threat_score_threatening"]
                self.track_last_scores[track_id] = scores

    def reset_track(self, track_id: int) -> None:
        """Reset state for a track.

        Args:
            track_id: Track ID to reset
        """
        if track_id in self.track_history:
            del self.track_history[track_id]
            del self.track_metrics_windows[track_id]
            self.track_last_scores.pop(track_id, None)

    def clear_all(self) -> None:
        """Clear all track states."""
        self.track_history.clear()
        self.track_metrics_windows.clear()
        self.track_last_scores.clear()

    def get_threat_status_summary(self) -> Dict[str, Any]:
        """Get summary of current threat status.

        Returns:
            Dictionary with threat statistics
        """
        if not self.track_history:
            return {
                "total_tracks": 0,
                "threatening": 0,
                "suspicious": 0,
                "benign": 0,
            }

        threatening_count = 0
        suspicious_count = 0
        benign_count = 0

        for scores in self.track_last_scores.values():
            level = scores.get("threat_level", "benign")
            if level == "threatening":
                threatening_count += 1
            elif level == "suspicious":
                suspicious_count += 1
            else:
                benign_count += 1

        return {
            "total_tracks": len(self.track_history),
            "threatening": threatening_count,
            "suspicious": suspicious_count,
            "benign": benign_count,
        }
