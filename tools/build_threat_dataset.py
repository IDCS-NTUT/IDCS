#!/usr/bin/env python3
"""Build threat evaluation training dataset from simulation logs.

Processes simulation logs to create feature windows and labels for training
the threat classification model.

Usage:
    python tools/build_threat_dataset.py \
      --sim_logs logs/*.csv \
      --output_dir dataset/ \
      --config configs/dataset.yaml

Output:
    dataset/train.npz - Training split (70%)
    dataset/val.npz - Validation split (15%)
    dataset/test.npz - Test split (15%)
    dataset/metadata.csv - Sample metadata
    dataset/statistics.json - Dataset statistics
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Add repo to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from common.threat_calc import TargetThreatState
from jetson.threat_label_generator import ThreatLabelGenerator

logger = logging.getLogger(__name__)


class ThreatDatasetBuilder:
    """Builds threat classification dataset from simulated target trajectories.

    Process:
    1. Load tracked target metrics per frame
    2. Group by track_id and create temporal windows
    3. Generate labels using rule-based classifier
    4. Normalize features
    5. Create train/val/test splits
    """

    def __init__(
        self,
        config: Dict[str, Any],
        label_generator: Optional[ThreatLabelGenerator] = None,
    ):
        """Initialize dataset builder.

        Args:
            config: Configuration dict from dataset.yaml
            label_generator: ThreatLabelGenerator instance (default: new with defaults)
        """
        self.config = config
        self.label_gen = label_generator or ThreatLabelGenerator()

        # Extract config parameters
        self.window_size = config["window"]["size"]
        self.window_stride = config["window"]["stride"]
        self.min_track_length = config["window"]["min_track_length"]
        self.min_confidence = config["filtering"]["min_confidence"]
        self.max_missing_frames = config["filtering"]["max_missing_frames"]

        self.split_ratios = config["split"]
        self.normalization = config["normalization"]
        self.output_dir = Path(config["output"]["directory"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dataset storage
        self.samples: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = {}

        logger.info(
            f"Dataset builder initialized: window_size={self.window_size}, "
            f"stride={self.window_stride}"
        )

    def add_trajectory(
        self,
        track_id: int,
        scenario_id: str,
        trajectory: List[Dict[str, float]],
    ) -> int:
        """Process a complete trajectory and create sliding windows.

        Args:
            track_id: Unique target identifier
            scenario_id: Scenario name/ID
            trajectory: List of per-frame metrics dicts

        Returns:
            Number of samples created from this trajectory
        """
        if len(trajectory) < self.min_track_length:
            logger.debug(f"Track {track_id}: too short ({len(trajectory)} < {self.min_track_length})")
            return 0

        num_samples_before = len(self.samples)

        # Create sliding windows
        for start_idx in range(0, len(trajectory) - self.window_size + 1, self.window_stride):
            window = trajectory[start_idx : start_idx + self.window_size]

            # Check window validity
            if len(window) < self.window_size:
                break

            missing_run = 0
            max_missing_run = 0
            for frame in window:
                if frame.get("missing", False):
                    missing_run += 1
                    max_missing_run = max(max_missing_run, missing_run)
                else:
                    missing_run = 0

            if max_missing_run > self.max_missing_frames:
                logger.debug(
                    f"Track {track_id} window {start_idx}: excessive missing frames "
                    f"({max_missing_run} > {self.max_missing_frames})"
                )
                continue

            confidences = [f.get("confidence", 1.0) for f in window]
            avg_confidence = np.mean(confidences)

            if avg_confidence < self.min_confidence:
                logger.debug(
                    f"Track {track_id} window {start_idx}: low confidence "
                    f"({avg_confidence:.2f} < {self.min_confidence})"
                )
                continue

            # Generate label
            label = self.label_gen.classify_window(window, aggregation="max_threat")

            # Create sample
            sample = {
                "track_id": track_id,
                "scenario_id": scenario_id,
                "frame_start": start_idx,
                "frame_end": start_idx + self.window_size - 1,
                "confidence": avg_confidence,
                "label": label,
                "window": window,  # Keep raw window for normalization
            }

            self.samples.append(sample)

        num_created = len(self.samples) - num_samples_before
        logger.debug(f"Track {track_id}: created {num_created} samples")
        return num_created

    def normalize_features(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        """Extract and normalize feature windows.

        Args:
            samples: List of sample dicts

        Returns:
            Array of shape [num_samples, window_size, num_features]
        """
        num_features = 11
        X = np.zeros((len(samples), self.window_size, num_features), dtype=np.float32)

        for sample_idx, sample in enumerate(samples):
            window = sample["window"]

            for frame_idx, frame_metrics in enumerate(window):
                # Extract and normalize features
                features = self._extract_normalized_features(frame_metrics)
                X[sample_idx, frame_idx, :] = features

        return X

    def _extract_normalized_features(self, frame_metrics: Dict[str, float]) -> np.ndarray:
        """Extract and normalize 11 features from frame metrics.

        Features (in order):
        0. center_x_norm - bbox center X normalized to [0, 1]
        1. center_y_norm - bbox center Y normalized to [0, 1]
        2. bbox_w_norm - bbox width normalized
        3. bbox_h_norm - bbox height normalized
        4. velocity_x_norm - velocity X normalized
        5. velocity_y_norm - velocity Y normalized
        6. confidence - detection confidence (no normalization)
        7. distance_to_asset_norm - distance normalized by max_distance
        8. distance_rate_to_asset_norm - rate normalized by max velocity
        9. zone_level - {0=normal, 1=warning, 2=restricted, 3=critical}
        10. time_inside_zone_norm - dwell time normalized

        Args:
            frame_metrics: Per-frame metrics dict

        Returns:
            Array of 11 normalized features
        """
        norm = self.normalization
        features = np.zeros(11, dtype=np.float32)

        # 0-1: Bbox center (normalized to image)
        center_x = frame_metrics.get("center_x", 0.0)
        center_y = frame_metrics.get("center_y", 0.0)
        features[0] = np.clip(center_x / norm["image_width"], 0.0, 1.0)
        features[1] = np.clip(center_y / norm["image_height"], 0.0, 1.0)

        # 2-3: Bbox dimensions (normalized to image)
        bbox_w = frame_metrics.get("bbox_width", 0.0)
        bbox_h = frame_metrics.get("bbox_height", 0.0)
        features[2] = np.clip(bbox_w / norm["image_width"], 0.0, 1.0)
        features[3] = np.clip(bbox_h / norm["image_height"], 0.0, 1.0)

        # 4-5: Velocity (normalized by max_velocity)
        vel_x = frame_metrics.get("velocity_x", 0.0)
        vel_y = frame_metrics.get("velocity_y", 0.0)
        features[4] = np.clip(vel_x / norm["max_velocity_m_s"], -1.0, 1.0)
        features[5] = np.clip(vel_y / norm["max_velocity_m_s"], -1.0, 1.0)

        # 6: Confidence (already in [0, 1])
        features[6] = np.clip(frame_metrics.get("confidence", 1.0), 0.0, 1.0)

        # 7: Distance to asset (normalized by max_distance)
        distance = frame_metrics.get("distance_to_asset", 0.0)
        features[7] = np.clip(distance / norm["max_distance_m"], 0.0, 1.0)

        # 8: Distance rate (normalized by max_velocity)
        dist_rate = frame_metrics.get("distance_rate_to_asset", 0.0)
        features[8] = np.clip(dist_rate / norm["max_velocity_m_s"], -1.0, 1.0)

        # 9: Zone level (categorical: 0-3)
        zone_id = frame_metrics.get("zone_id", "normal")
        zone_map = {"normal": 0, "warning": 1, "restricted": 2, "critical": 3}
        features[9] = float(zone_map.get(zone_id, 0)) / 3.0  # normalize to [0, 1]

        # 10: Time in zone (normalized by max_time)
        dwell = frame_metrics.get("time_inside_zone", 0.0)
        features[10] = np.clip(dwell / norm["max_time_in_zone_s"], 0.0, 1.0)

        return features

    def create_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create train/val/test splits and return normalized feature arrays.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if not self.samples:
            raise ValueError("No samples to split")

        # Shuffle samples
        shuffled_indices = np.random.permutation(len(self.samples))
        shuffled_samples = [self.samples[i] for i in shuffled_indices]

        # Calculate split sizes
        n_total = len(shuffled_samples)
        n_train = int(n_total * self.split_ratios["train"])
        n_val = int(n_total * self.split_ratios["val"])

        train_samples = shuffled_samples[:n_train]
        val_samples = shuffled_samples[n_train : n_train + n_val]
        test_samples = shuffled_samples[n_train + n_val :]

        logger.info(
            f"Dataset splits: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
        )

        # Normalize features
        X_train = self.normalize_features(train_samples)
        X_val = self.normalize_features(val_samples)
        X_test = self.normalize_features(test_samples)

        # Extract labels
        y_train = np.array([s["label"] for s in train_samples], dtype=np.int32)
        y_val = np.array([s["label"] for s in val_samples], dtype=np.int32)
        y_test = np.array([s["label"] for s in test_samples], dtype=np.int32)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def save_datasets(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Save datasets to NPZ files.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
        """
        # Save NPZ files
        train_path = self.output_dir / "train.npz"
        val_path = self.output_dir / "val.npz"
        test_path = self.output_dir / "test.npz"

        np.savez_compressed(train_path, X=X_train, y=y_train)
        np.savez_compressed(val_path, X=X_val, y=y_val)
        np.savez_compressed(test_path, X=X_test, y=y_test)

        logger.info(f"Saved datasets: {train_path}, {val_path}, {test_path}")

        # Save metadata
        self._save_metadata()
        self._save_statistics(y_train, y_val, y_test)

    def _save_metadata(self) -> None:
        """Save sample metadata to CSV."""
        metadata_path = self.output_dir / "metadata.csv"

        with open(metadata_path, "w") as f:
            f.write("sample_id,track_id,scenario_id,frame_start,frame_end,label,confidence\n")
            for idx, sample in enumerate(self.samples):
                f.write(
                    f"{idx},{sample['track_id']},{sample['scenario_id']},"
                    f"{sample['frame_start']},{sample['frame_end']},{sample['label']},".rstrip()
                    + f"{sample['confidence']:.3f}\n"
                )

        logger.info(f"Saved metadata to {metadata_path}")

    def _save_statistics(
        self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
    ) -> None:
        """Save dataset statistics to JSON."""
        class_names = ["benign", "suspicious", "threatening"]

        def compute_stats(y):
            unique, counts = np.unique(y, return_counts=True)
            return {
                class_names[label]: int(count) for label, count in zip(unique, counts)
            }

        stats = {
            "total_samples": len(self.samples),
            "window_size": self.window_size,
            "num_features": 11,
            "train": {
                "samples": len(y_train),
                "class_distribution": compute_stats(y_train),
            },
            "val": {"samples": len(y_val), "class_distribution": compute_stats(y_val)},
            "test": {
                "samples": len(y_test),
                "class_distribution": compute_stats(y_test),
            },
        }

        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_path}")

        # Print summary
        print("\n=== DATASET STATISTICS ===")
        print(f"Total samples: {stats['total_samples']}")
        for split in ["train", "val", "test"]:
            print(f"\n{split.upper()}:")
            print(f"  Samples: {stats[split]['samples']}")
            for cls, count in stats[split]["class_distribution"].items():
                pct = 100.0 * count / stats[split]["samples"]
                print(f"    {cls}: {count} ({pct:.1f}%)")


def _zone_id_from_distance(distance: float) -> str:
    """Map target distance to a named threat zone."""
    if distance <= 5.0:
        return "critical"
    if distance <= 10.0:
        return "restricted"
    if distance <= 20.0:
        return "warning"
    return "normal"


def _make_frame(
    x: float,
    y: float,
    step_idx: int,
    prev_distance: Optional[float],
    confidence_bias: float,
    rng: np.random.Generator,
    frame_dt: float = 0.033,
    missing_prob: float = 0.0,
) -> Dict[str, float]:
    """Create one synthetic frame with coupled geometry and threat metrics."""
    distance = float(np.sqrt(x**2 + y**2))
    distance_rate = 0.0 if prev_distance is None else (distance - prev_distance) / frame_dt
    zone_id = _zone_id_from_distance(distance)

    # Simulate larger boxes at closer ranges with a bit of jitter.
    distance_scale = max(distance, 2.0)
    bbox_height = np.clip(900.0 / distance_scale + rng.normal(0.0, 6.0), 30.0, 220.0)
    bbox_width = np.clip(bbox_height * (0.48 + rng.normal(0.0, 0.04)), 18.0, 130.0)

    center_x = np.clip(960.0 + x * 22.0 + rng.normal(0.0, 14.0), 0.0, 1919.0)
    center_y = np.clip(540.0 - y * 18.0 + rng.normal(0.0, 10.0), 0.0, 1079.0)

    confidence = np.clip(
        confidence_bias
        - 0.012 * (zone_id == "critical")
        + rng.normal(0.0, 0.05),
        0.15,
        0.99,
    )

    missing = bool(rng.random() < missing_prob)
    if missing:
        confidence = min(confidence, 0.18)

    return {
        "center_x": center_x,
        "center_y": center_y,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "confidence": confidence,
        "distance_to_asset": distance,
        "distance_rate_to_asset": distance_rate,
        "zone_id": zone_id,
        "time_inside_zone": step_idx * frame_dt,
        "missing": missing,
    }


def _populate_velocities_and_dwell(
    trajectory: List[Dict[str, float]],
    frame_dt: float = 0.033,
) -> None:
    """Back-fill velocity and zone dwell fields for a generated trajectory."""
    prev_center: Optional[Tuple[float, float]] = None
    current_zone: Optional[str] = None
    zone_entry_idx = 0

    for idx, frame in enumerate(trajectory):
        center = (float(frame["center_x"]), float(frame["center_y"]))
        if prev_center is not None:
            frame["velocity_x"] = (center[0] - prev_center[0]) / 1920.0 / frame_dt
            frame["velocity_y"] = (center[1] - prev_center[1]) / 1080.0 / frame_dt
        else:
            frame["velocity_x"] = 0.0
            frame["velocity_y"] = 0.0
        prev_center = center

        zone_id = str(frame["zone_id"])
        if zone_id != current_zone:
            current_zone = zone_id
            zone_entry_idx = idx
            frame["time_inside_zone"] = 0.0
        else:
            frame["time_inside_zone"] = (idx - zone_entry_idx) * frame_dt


def _generate_synthetic_trajectory(
    scenario_type: int,
    rng: np.random.Generator,
    length: int = 24,
) -> List[Dict[str, float]]:
    """Generate one synthetic trajectory from a richer scenario catalog."""
    trajectory: List[Dict[str, float]] = []
    prev_distance: Optional[float] = None

    for i in range(length):
        t = i / max(length - 1, 1)

        if scenario_type == 0:
            # Fast direct approach into critical zone.
            x = 24.0 - 22.5 * t + rng.normal(0.0, 0.18)
            y = rng.normal(0.0, 0.35)
            frame = _make_frame(x, y, i, prev_distance, 0.92, rng)
        elif scenario_type == 1:
            # Diagonal accelerating approach with lateral drift.
            x = 22.0 - 18.0 * (t**1.4) + rng.normal(0.0, 0.2)
            y = 8.0 - 10.5 * t + rng.normal(0.0, 0.25)
            frame = _make_frame(x, y, i, prev_distance, 0.88, rng)
        elif scenario_type == 2:
            # Loiter around the restricted-warning boundary.
            theta = 2.0 * np.pi * t * 1.6
            radius = 11.5 + 1.4 * np.sin(theta * 1.5) + rng.normal(0.0, 0.18)
            x = radius * np.cos(theta)
            y = 0.6 * radius * np.sin(theta)
            frame = _make_frame(x, y, i, prev_distance, 0.84, rng, missing_prob=0.04)
        elif scenario_type == 3:
            # Tangential flyby that briefly enters warning space but does not close much.
            x = -26.0 + 52.0 * t + rng.normal(0.0, 0.25)
            y = 13.0 + 1.4 * np.sin(2.0 * np.pi * t) + rng.normal(0.0, 0.18)
            frame = _make_frame(x, y, i, prev_distance, 0.9, rng)
        elif scenario_type == 4:
            # Clear benign retreat.
            x = 7.0 + 19.0 * t + rng.normal(0.0, 0.2)
            y = -3.0 + 0.5 * np.sin(2.0 * np.pi * t) + rng.normal(0.0, 0.15)
            frame = _make_frame(x, y, i, prev_distance, 0.94, rng)
        elif scenario_type == 5:
            # Zig-zag approach with aggressive lateral motion.
            x = 21.0 - 15.5 * t + rng.normal(0.0, 0.2)
            y = 5.5 * np.sin(3.0 * np.pi * t) + rng.normal(0.0, 0.22)
            frame = _make_frame(x, y, i, prev_distance, 0.86, rng, missing_prob=0.06)
        elif scenario_type == 6:
            # Slow ingress then hover in restricted zone.
            x = 17.0 - 8.5 * min(t * 1.3, 1.0) + rng.normal(0.0, 0.16)
            y = 1.4 * np.sin(5.0 * np.pi * t) + rng.normal(0.0, 0.12)
            frame = _make_frame(x, y, i, prev_distance, 0.83, rng)
        else:
            # Intermittent detections near a decision boundary.
            x = 12.5 - 4.0 * t + rng.normal(0.0, 0.28)
            y = 7.0 - 6.0 * t + 1.0 * np.sin(4.0 * np.pi * t) + rng.normal(0.0, 0.22)
            frame = _make_frame(x, y, i, prev_distance, 0.74, rng, missing_prob=0.12)
            if i in (6, 7, 15):
                frame["missing"] = True
                frame["confidence"] = 0.12

        trajectory.append(frame)
        prev_distance = float(frame["distance_to_asset"])

    _populate_velocities_and_dwell(trajectory)
    return trajectory


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build threat evaluation dataset from simulation logs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to dataset config file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/"),
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Approximate number of samples to generate (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["output"]["directory"] = str(args.output_dir)

    # Create builder
    builder = ThreatDatasetBuilder(config)

    logger.info(f"Building dataset with ~{args.num_samples} samples...")

    # For now, create synthetic training data for demonstration.
    # In production, this would load from actual simulation logs.
    rng = np.random.default_rng(args.seed)
    scenario_catalog_size = 8

    # Generate synthetic trajectories across a richer scenario catalog.
    for scenario_idx in range(args.num_samples // scenario_catalog_size):
        for track_idx in range(scenario_catalog_size):
            scenario_type = track_idx % scenario_catalog_size
            trajectory = _generate_synthetic_trajectory(
                scenario_type=scenario_type,
                rng=rng,
                length=24,
            )

            builder.add_trajectory(
                track_id=scenario_idx * scenario_catalog_size + track_idx,
                scenario_id=f"scenario_{scenario_idx:03d}_type_{scenario_type}",
                trajectory=trajectory,
            )

    logger.info(f"Created {len(builder.samples)} samples")

    # Create splits
    X_train, y_train, X_val, y_val, X_test, y_test = builder.create_splits()

    # Save datasets
    builder.save_datasets(X_train, y_train, X_val, y_val, X_test, y_test)

    logger.info("Dataset creation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
