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

    # For now, create synthetic training data for demonstration
    # In production, this would load from actual simulation logs
    np.random.seed(args.seed)

    asset_xy = (0.0, 0.0)
    zone_radii = {"critical": 5.0, "restricted": 10.0, "warning": 20.0}

    # Generate synthetic trajectories
    for scenario_idx in range(args.num_samples // 8):
        for track_idx in range(8):
            trajectory = []
            scenario_type = track_idx % 3

            if scenario_type == 0:
                # Approaching trajectory
                for i in range(20):
                    x = -20 + i
                    y = np.random.normal(0, 0.5)
                    trajectory.append({
                        "center_x": 100 + i * 5,
                        "center_y": 100,
                        "bbox_width": 50,
                        "bbox_height": 100,
                        "velocity_x": 1.0 + np.random.normal(0, 0.1),
                        "velocity_y": np.random.normal(0, 0.1),
                        "confidence": 0.9 + np.random.normal(0, 0.05),
                        "distance_to_asset": np.sqrt(x**2 + y**2),
                        "distance_rate_to_asset": -1.0 + np.random.normal(0, 0.2),
                        "zone_id": "critical" if np.sqrt(x**2 + y**2) <= 5 else (
                            "restricted" if np.sqrt(x**2 + y**2) <= 10 else (
                                "warning" if np.sqrt(x**2 + y**2) <= 20 else "normal"
                            )
                        ),
                        "time_inside_zone": min(i * 0.033, 5.0),
                    })
            elif scenario_type == 1:
                # Loitering trajectory
                for i in range(20):
                    x = np.random.normal(12, 1)
                    y = np.random.normal(0, 1)
                    trajectory.append({
                        "center_x": 100,
                        "center_y": 100 + i,
                        "bbox_width": 50,
                        "bbox_height": 100,
                        "velocity_x": np.random.normal(0, 0.05),
                        "velocity_y": np.random.normal(0, 0.05),
                        "confidence": 0.85 + np.random.normal(0, 0.1),
                        "distance_to_asset": np.sqrt(x**2 + y**2),
                        "distance_rate_to_asset": np.random.normal(0, 0.1),
                        "zone_id": "warning",
                        "time_inside_zone": min(i * 0.033, 10.0),
                    })
            else:
                # Benign trajectory (receding)
                for i in range(20):
                    x = 25 - i * 0.5
                    y = np.random.normal(0, 0.5)
                    trajectory.append({
                        "center_x": 100 - i * 2,
                        "center_y": 100,
                        "bbox_width": 50,
                        "bbox_height": 100,
                        "velocity_x": -0.5 + np.random.normal(0, 0.1),
                        "velocity_y": np.random.normal(0, 0.1),
                        "confidence": 0.92 + np.random.normal(0, 0.05),
                        "distance_to_asset": np.sqrt(x**2 + y**2),
                        "distance_rate_to_asset": 0.5 + np.random.normal(0, 0.1),
                        "zone_id": "normal",
                        "time_inside_zone": 0.0,
                    })

            builder.add_trajectory(
                track_id=scenario_idx * 8 + track_idx,
                scenario_id=f"scenario_{scenario_idx:03d}",
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
