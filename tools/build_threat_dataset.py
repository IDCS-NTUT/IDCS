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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# Add repo to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

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

    FEATURE_NAMES = [
        "center_x_norm",
        "center_y_norm",
        "bbox_w_norm",
        "bbox_h_norm",
        "velocity_x_norm",
        "velocity_y_norm",
        "confidence",
        "distance_to_asset_norm",
        "distance_rate_to_asset_norm",
        "zone_level_norm",
        "time_inside_zone_norm",
    ]

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

        # Split / audit settings
        self.group_by_scenario = bool(self.split_ratios.get("group_by_scenario", True))
        self.split_seed = int(self.split_ratios.get("seed", 42))
        self.heldout_ratio = float(self.split_ratios.get("heldout", 0.0))

        leakage_cfg = config.get("leakage", {})
        self.leakage_enabled = bool(leakage_cfg.get("enabled", True))
        self.leakage_bins = int(leakage_cfg.get("bins", 32))
        self.leakage_warn_threshold = float(leakage_cfg.get("suspect_accuracy_threshold", 0.98))
        self.leakage_enforce_clean = bool(leakage_cfg.get("enforce_no_suspect_features", False))

        # Dataset storage
        self.samples: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = {}
        self.split_samples: Dict[str, List[Dict[str, Any]]] = {}
        self.leakage_report: Dict[str, Any] = {}

        logger.info(
            f"Dataset builder initialized: window_size={self.window_size}, "
            f"stride={self.window_stride}"
        )

    def _scenario_group_split(self) -> Dict[str, List[Dict[str, Any]]]:
        """Split full samples by scenario to prevent sample leakage across splits."""
        scenarios: Dict[str, List[Dict[str, Any]]] = {}
        for sample in self.samples:
            scenarios.setdefault(sample["scenario_id"], []).append(sample)

        scenario_ids = np.array(sorted(scenarios.keys()))
        if scenario_ids.size == 0:
            raise ValueError("No scenarios found while creating splits")

        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(scenario_ids)

        n_scenarios = int(scenario_ids.size)
        n_heldout = int(round(n_scenarios * self.heldout_ratio))
        n_train = int(round(n_scenarios * float(self.split_ratios["train"])))
        n_val = int(round(n_scenarios * float(self.split_ratios["val"])))

        # Ensure at least one scenario remains for test when possible.
        n_used = n_heldout + n_train + n_val
        if n_used >= n_scenarios and n_scenarios >= 3:
            overflow = n_used - (n_scenarios - 1)
            n_train = max(1, n_train - overflow)

        heldout_ids = set(scenario_ids[:n_heldout].tolist())
        train_start = n_heldout
        val_start = train_start + n_train
        test_start = val_start + n_val

        train_ids = set(scenario_ids[train_start:val_start].tolist())
        val_ids = set(scenario_ids[val_start:test_start].tolist())
        test_ids = set(scenario_ids[test_start:].tolist())

        if not test_ids and val_ids:
            moved = next(iter(val_ids))
            val_ids.remove(moved)
            test_ids.add(moved)

        splits = {"train": [], "val": [], "test": [], "heldout": []}
        for sid, sid_samples in scenarios.items():
            if sid in heldout_ids:
                splits["heldout"].extend(sid_samples)
            elif sid in train_ids:
                splits["train"].extend(sid_samples)
            elif sid in val_ids:
                splits["val"].extend(sid_samples)
            else:
                splits["test"].extend(sid_samples)

        return splits

    def _sample_level_split(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback random sample-level split (legacy behavior)."""
        shuffled_indices = np.random.default_rng(self.split_seed).permutation(len(self.samples))
        shuffled_samples = [self.samples[i] for i in shuffled_indices]

        n_total = len(shuffled_samples)
        n_heldout = int(round(n_total * self.heldout_ratio))
        n_train = int(round((n_total - n_heldout) * float(self.split_ratios["train"])))
        n_val = int(round((n_total - n_heldout) * float(self.split_ratios["val"])))

        heldout_samples = shuffled_samples[:n_heldout]
        train_samples = shuffled_samples[n_heldout : n_heldout + n_train]
        val_samples = shuffled_samples[n_heldout + n_train : n_heldout + n_train + n_val]
        test_samples = shuffled_samples[n_heldout + n_train + n_val :]

        return {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
            "heldout": heldout_samples,
        }

    def _mark_sample_splits(self, splits: Dict[str, List[Dict[str, Any]]]) -> None:
        """Annotate each sample with split label for metadata export."""
        for split_name, split_samples in splits.items():
            for sample in split_samples:
                sample["split"] = split_name

    def _single_feature_oracle_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_bins: int,
    ) -> float:
        """Estimate leakage risk by best bin-majority classifier using one feature."""
        if X.size == 0 or y.size == 0:
            return 0.0
        x_min = float(np.min(X))
        x_max = float(np.max(X))
        if abs(x_max - x_min) < 1e-9:
            return max(float(np.mean(y == cls)) for cls in np.unique(y))

        bins = np.linspace(x_min, x_max, num_bins + 1)
        bin_ids = np.digitize(X, bins[1:-1], right=False)

        pred = np.zeros_like(y)
        for b in range(num_bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            labels, counts = np.unique(y[mask], return_counts=True)
            pred[mask] = labels[np.argmax(counts)]
        return float(np.mean(pred == y))

    def _run_leakage_audit(self, train_samples: Sequence[Dict[str, Any]]) -> None:
        """Check if any single feature can almost perfectly recover labels."""
        if not self.leakage_enabled:
            return
        if not train_samples:
            return

        X_train = self.normalize_features(list(train_samples))
        y_train = np.array([s["label"] for s in train_samples], dtype=np.int32)

        # Collapse temporal axis so audit can detect leaked per-frame signals.
        X_mean = X_train.mean(axis=1)
        feature_scores: Dict[str, float] = {}
        suspects: List[Dict[str, Any]] = []

        for idx, name in enumerate(self.FEATURE_NAMES):
            score = self._single_feature_oracle_accuracy(
                X=X_mean[:, idx],
                y=y_train,
                num_bins=self.leakage_bins,
            )
            feature_scores[name] = score
            if score >= self.leakage_warn_threshold:
                suspects.append({"feature": name, "oracle_accuracy": round(score, 6)})

        self.leakage_report = {
            "enabled": True,
            "threshold": self.leakage_warn_threshold,
            "num_bins": self.leakage_bins,
            "feature_oracle_accuracy": {
                k: round(v, 6) for k, v in sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
            },
            "suspects": suspects,
        }

        report_path = self.output_dir / "leakage_report.json"
        with open(report_path, "w") as f:
            json.dump(self.leakage_report, f, indent=2)

        if suspects:
            logger.warning("Leakage audit suspects found: %s", suspects)
            if self.leakage_enforce_clean:
                raise ValueError(
                    "Leakage audit failed because suspect features exceeded threshold; "
                    "see leakage_report.json for details"
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

        split_sum = (
            float(self.split_ratios.get("train", 0.0))
            + float(self.split_ratios.get("val", 0.0))
            + float(self.split_ratios.get("test", 0.0))
        )
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(f"split.train+val+test must equal 1.0, got {split_sum:.6f}")

        splits = self._scenario_group_split() if self.group_by_scenario else self._sample_level_split()
        self._mark_sample_splits(splits)

        train_samples = splits["train"]
        val_samples = splits["val"]
        test_samples = splits["test"]
        heldout_samples = splits["heldout"]

        self.split_samples = splits

        self._run_leakage_audit(train_samples)

        logger.info(
            "Dataset splits: train=%d, val=%d, test=%d, heldout=%d (scenario_group=%s)",
            len(train_samples),
            len(val_samples),
            len(test_samples),
            len(heldout_samples),
            self.group_by_scenario,
        )

        # Normalize features
        X_train = self.normalize_features(train_samples)
        X_val = self.normalize_features(val_samples)
        X_test = self.normalize_features(test_samples)
        X_heldout = self.normalize_features(heldout_samples) if heldout_samples else np.zeros((0, self.window_size, 11), dtype=np.float32)

        # Extract labels
        y_train = np.array([s["label"] for s in train_samples], dtype=np.int32)
        y_val = np.array([s["label"] for s in val_samples], dtype=np.int32)
        y_test = np.array([s["label"] for s in test_samples], dtype=np.int32)
        y_heldout = np.array([s["label"] for s in heldout_samples], dtype=np.int32)

        self.statistics["split_counts"] = {
            "train": int(len(train_samples)),
            "val": int(len(val_samples)),
            "test": int(len(test_samples)),
            "heldout": int(len(heldout_samples)),
        }
        self.statistics["scenario_counts"] = {
            "train": len({s["scenario_id"] for s in train_samples}),
            "val": len({s["scenario_id"] for s in val_samples}),
            "test": len({s["scenario_id"] for s in test_samples}),
            "heldout": len({s["scenario_id"] for s in heldout_samples}),
        }
        self.statistics["heldout_arrays"] = {"X": X_heldout, "y": y_heldout}

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

        heldout = self.statistics.get("heldout_arrays")
        heldout_path = self.output_dir / "heldout_test.npz"
        if isinstance(heldout, dict) and len(heldout.get("y", [])) > 0:
            np.savez_compressed(heldout_path, X=heldout["X"], y=heldout["y"])
            logger.info("Saved held-out scenario dataset: %s", heldout_path)

        logger.info(f"Saved datasets: {train_path}, {val_path}, {test_path}")

        # Save metadata
        self._save_metadata()
        self._save_statistics(y_train, y_val, y_test)

    def _save_metadata(self) -> None:
        """Save sample metadata to CSV."""
        metadata_path = self.output_dir / "metadata.csv"

        with open(metadata_path, "w") as f:
            f.write("sample_id,track_id,scenario_id,frame_start,frame_end,label,confidence,split\n")
            for idx, sample in enumerate(self.samples):
                f.write(
                    f"{idx},{sample['track_id']},{sample['scenario_id']},"
                    f"{sample['frame_start']},{sample['frame_end']},{sample['label']},".rstrip()
                    + f"{sample['confidence']:.3f},{sample.get('split', 'unassigned')}\n"
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
            "feature_names": self.FEATURE_NAMES,
            "train": {
                "samples": len(y_train),
                "class_distribution": compute_stats(y_train),
            },
            "val": {"samples": len(y_val), "class_distribution": compute_stats(y_val)},
            "test": {
                "samples": len(y_test),
                "class_distribution": compute_stats(y_test),
            },
            "split_counts": self.statistics.get("split_counts", {}),
            "scenario_counts": self.statistics.get("scenario_counts", {}),
        }

        heldout = self.statistics.get("heldout_arrays")
        if isinstance(heldout, dict):
            y_heldout = heldout.get("y")
            if isinstance(y_heldout, np.ndarray) and y_heldout.size > 0:
                stats["heldout"] = {
                    "samples": int(y_heldout.size),
                    "class_distribution": compute_stats(y_heldout),
                }

        if self.leakage_report:
            stats["leakage_audit"] = {
                "threshold": self.leakage_report.get("threshold"),
                "suspect_feature_count": len(self.leakage_report.get("suspects", [])),
            }

        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_path}")

        # Print summary
        print("\n=== DATASET STATISTICS ===")
        print(f"Total samples: {stats['total_samples']}")
        split_names = ["train", "val", "test"]
        if "heldout" in stats:
            split_names.append("heldout")
        for split in split_names:
            print(f"\n{split.upper()}:")
            print(f"  Samples: {stats[split]['samples']}")
            for cls, count in stats[split]["class_distribution"].items():
                denom = max(1, stats[split]["samples"])
                pct = 100.0 * count / denom
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
