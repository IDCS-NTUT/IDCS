#!/usr/bin/env python3
"""Train threat evaluation model.

Loads dataset, trains MLP model, and saves checkpoint.

Usage:
    python tools/train_threat_model.py \
      --dataset_dir dataset/ \
      --output_dir models/ \
      --epochs 100 \
      --batch_size 32
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add repo to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jetson.threat_model import ThreatClassifier, create_threat_model

logger = logging.getLogger(__name__)


CLASS_NAMES = ["benign", "suspicious", "threatening"]


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    matrix = _confusion_matrix(y_true, y_pred, num_classes=len(CLASS_NAMES))
    per_class: Dict[str, Dict[str, float]] = {}

    for idx, class_name in enumerate(CLASS_NAMES):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        support = int(matrix[idx, :].sum())
        per_class[class_name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    # Safety-centric metrics.
    benign_idx = 0
    threat_idx = 2
    false_threat_rate = _safe_div(matrix[benign_idx, threat_idx], matrix[benign_idx, :].sum())
    missed_threat_rate = _safe_div(
        matrix[threat_idx, :].sum() - matrix[threat_idx, threat_idx],
        matrix[threat_idx, :].sum(),
    )

    return {
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "safety": {
            "false_threat_rate": round(false_threat_rate, 6),
            "missed_threat_rate": round(missed_threat_rate, 6),
        },
    }


def _apply_stress_profile(X: np.ndarray, profile: str, seed: int) -> np.ndarray:
    """Apply synthetic stressors to assess robustness under degraded observations."""
    rng = np.random.default_rng(seed)
    stressed = np.array(X, copy=True)

    if profile == "noise":
        noise = rng.normal(0.0, 0.035, size=stressed.shape)
        stressed = np.clip(stressed + noise.astype(np.float32), -1.0, 1.0)
        return stressed

    if profile == "occlusion":
        # Zero-out bbox width/height for random frames to emulate occluded detections.
        occlusion_mask = rng.random((stressed.shape[0], stressed.shape[1])) < 0.25
        stressed[:, :, 2][occlusion_mask] = 0.0
        stressed[:, :, 3][occlusion_mask] = 0.0
        stressed[:, :, 6] = np.clip(stressed[:, :, 6] - 0.15, 0.0, 1.0)
        return stressed

    if profile == "variation":
        scale = rng.uniform(0.85, 1.15, size=(stressed.shape[0], 1, stressed.shape[2])).astype(np.float32)
        stressed = stressed * scale
        stressed[:, :, 0:4] = np.clip(stressed[:, :, 0:4], 0.0, 1.0)
        stressed[:, :, 4:6] = np.clip(stressed[:, :, 4:6], -1.0, 1.0)
        stressed[:, :, 6:] = np.clip(stressed[:, :, 6:], 0.0, 1.0)
        return stressed

    if profile == "combined":
        return _apply_stress_profile(
            _apply_stress_profile(
                _apply_stress_profile(stressed, "noise", seed + 11),
                "occlusion",
                seed + 17,
            ),
            "variation",
            seed + 23,
        )

    raise ValueError(f"Unknown stress profile: {profile}")


class ThreatModelTrainer:
    """Trainer for threat evaluation model."""

    def __init__(
        self,
        model: ThreatClassifier,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        """Initialize trainer.

        Args:
            model: ThreatClassifier instance
            device: Compute device
            learning_rate: Adam learning rate
            weight_decay: L2 regularization weight
        """
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(
            model.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_history = {"loss": [], "val_loss": [], "val_accuracy": []}
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training DataLoader

        Returns:
            Average training loss
        """
        self.model.train_mode()
        total_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Flatten features: [batch, 16, 11] -> [batch, 176]
            if X_batch.dim() == 3:
                batch_size = X_batch.size(0)
                X_batch = X_batch.reshape(batch_size, -1)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model.forward(X_batch)
            loss = self.criterion(logits, y_batch)

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate on validation/test set.

        Args:
            dataloader: Validation DataLoader

        Returns:
            Tuple of (loss, accuracy)
        """
        if len(dataloader) == 0:
            return 0.0, 0.0

        self.model.eval_mode()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Flatten features
                if X_batch.dim() == 3:
                    batch_size = X_batch.size(0)
                    X_batch = X_batch.reshape(batch_size, -1)

                logits = self.model.forward(X_batch)
                loss = self.criterion(logits, y_batch)
                preds = torch.argmax(logits, dim=1)

                total_loss += loss.item()
                total_correct += (preds == y_batch).sum().item()
                total_samples += y_batch.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def evaluate_detailed(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate model and return losses, predictions, and detailed metrics."""
        self.model.eval_mode()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        y_true_batches: List[np.ndarray] = []
        y_pred_batches: List[np.ndarray] = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if X_batch.dim() == 3:
                    batch_size = X_batch.size(0)
                    X_batch = X_batch.reshape(batch_size, -1)

                logits = self.model.forward(X_batch)
                loss = self.criterion(logits, y_batch)
                preds = torch.argmax(logits, dim=1)

                total_loss += loss.item()
                total_correct += (preds == y_batch).sum().item()
                total_samples += y_batch.size(0)

                y_true_batches.append(y_batch.detach().cpu().numpy())
                y_pred_batches.append(preds.detach().cpu().numpy())

        if total_samples == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "num_samples": 0,
                "metrics": _classification_report(np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            }

        y_true = np.concatenate(y_true_batches)
        y_pred = np.concatenate(y_pred_batches)
        avg_loss = total_loss / max(1, len(dataloader))
        accuracy = total_correct / total_samples

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "num_samples": int(total_samples),
            "metrics": _classification_report(y_true, y_pred),
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 10,
        log_interval: int = 10,
    ) -> Dict[str, Any]:
        """Train model.

        Args:
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            log_interval: Logging interval

        Returns:
            Training history dict
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        patience_counter = 0
        has_validation = len(val_dataloader) > 0
        if not has_validation:
            logger.warning("Validation split is empty; training will proceed without early stopping")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader)
            self.train_history["loss"].append(train_loss)

            # Validate
            if has_validation:
                val_loss, val_accuracy = self.evaluate(val_dataloader)
            else:
                val_loss, val_accuracy = train_loss, 0.0
            self.train_history["val_loss"].append(val_loss)
            self.train_history["val_accuracy"].append(val_accuracy)

            # Logging
            if (epoch + 1) % log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.4f}"
                )

            # Early stopping
            if has_validation:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(
                            f"Early stopping at epoch {epoch+1} "
                            f"(best: epoch {self.best_epoch}, loss: {self.best_val_loss:.4f})"
                        )
                        break
            else:
                self.best_val_loss = train_loss
                self.best_epoch = epoch + 1

        logger.info(f"Training complete. Best epoch: {self.best_epoch}")
        return self.train_history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train threat evaluation model")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("dataset/"),
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/"),
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--skip_stress_tests",
        action="store_true",
        help="Skip robustness stress evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stress tests",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Select device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU for training")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training")

    # Load datasets
    logger.info(f"Loading datasets from {args.dataset_dir}...")

    train_data = np.load(args.dataset_dir / "train.npz")
    val_data = np.load(args.dataset_dir / "val.npz")
    test_data = np.load(args.dataset_dir / "test.npz")

    heldout_path = args.dataset_dir / "heldout_test.npz"
    heldout_data = np.load(heldout_path) if heldout_path.exists() else None

    X_train = torch.from_numpy(train_data["X"]).float()
    y_train = torch.from_numpy(train_data["y"]).long()
    X_val = torch.from_numpy(val_data["X"]).float()
    y_val = torch.from_numpy(val_data["y"]).long()
    X_test = torch.from_numpy(test_data["X"]).float()
    y_test = torch.from_numpy(test_data["y"]).long()

    if heldout_data is not None:
        X_heldout = torch.from_numpy(heldout_data["X"]).float()
        y_heldout = torch.from_numpy(heldout_data["y"]).long()
    else:
        X_heldout = None
        y_heldout = None

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    if X_heldout is not None:
        logger.info("Held-out: %s", tuple(X_heldout.shape))

    if X_train.dim() == 3:
        input_size = int(X_train.shape[1] * X_train.shape[2])
    elif X_train.dim() == 2:
        input_size = int(X_train.shape[1])
    else:
        raise ValueError(f"Unsupported training tensor shape: {tuple(X_train.shape)}")

    logger.info("Resolved threat model input size to %d features", input_size)

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    heldout_loader = None
    if X_heldout is not None and y_heldout is not None and len(y_heldout) > 0:
        heldout_loader = DataLoader(
            TensorDataset(X_heldout, y_heldout),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # Create model
    logger.info("Creating threat model...")
    model = create_threat_model(
        input_size=input_size,
        hidden_sizes=[64, 32],
        output_size=3,
        device=device,
    )

    # Create trainer
    trainer = ThreatModelTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Train
    history = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        log_interval=10,
    )

    # Save model
    model_path = args.output_dir / "threat_model.pt"
    logger.info(f"Saving model to {model_path}...")
    model.save(str(model_path))

    # Save history
    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete!")
    logger.info(f"Model saved to {model_path}")
    logger.info(f"History saved to {history_path}")

    # Evaluate on all available splits
    logger.info("\n=== FINAL EVALUATION ===")
    model.eval_mode()

    val_report = trainer.evaluate_detailed(val_loader)
    test_report = trainer.evaluate_detailed(test_loader)

    logger.info("Validation Loss: %.4f", val_report["loss"])
    logger.info("Validation Accuracy: %.4f", val_report["accuracy"])
    logger.info("Test Loss: %.4f", test_report["loss"])
    logger.info("Test Accuracy: %.4f", test_report["accuracy"])

    heldout_report = None
    if heldout_loader is not None:
        heldout_report = trainer.evaluate_detailed(heldout_loader)
        logger.info("Held-out Loss: %.4f", heldout_report["loss"])
        logger.info("Held-out Accuracy: %.4f", heldout_report["accuracy"])

    logger.info("Confusion Matrix (test): %s", test_report["metrics"]["confusion_matrix"])
    logger.info("Safety (test): %s", test_report["metrics"]["safety"])

    stress_reports: Dict[str, Dict[str, Any]] = {}
    if not args.skip_stress_tests:
        for i, profile in enumerate(["noise", "occlusion", "variation", "combined"]):
            stressed_X_test = _apply_stress_profile(test_data["X"], profile=profile, seed=args.seed + (i * 101))
            stressed_loader = DataLoader(
                TensorDataset(torch.from_numpy(stressed_X_test).float(), y_test),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            stress_reports[f"test_{profile}"] = trainer.evaluate_detailed(stressed_loader)

            if heldout_data is not None and heldout_loader is not None:
                stressed_X_heldout = _apply_stress_profile(
                    heldout_data["X"],
                    profile=profile,
                    seed=args.seed + 5000 + (i * 101),
                )
                stressed_heldout_loader = DataLoader(
                    TensorDataset(torch.from_numpy(stressed_X_heldout).float(), y_heldout),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                stress_reports[f"heldout_{profile}"] = trainer.evaluate_detailed(stressed_heldout_loader)

        for name, report in stress_reports.items():
            logger.info(
                "Stress %s | acc=%.4f | missed_threat_rate=%.4f | false_threat_rate=%.4f",
                name,
                report["accuracy"],
                report["metrics"]["safety"]["missed_threat_rate"],
                report["metrics"]["safety"]["false_threat_rate"],
            )

    eval_report = {
        "validation": val_report,
        "test": test_report,
        "heldout": heldout_report,
        "stress": stress_reports,
        "class_names": CLASS_NAMES,
    }
    eval_report_path = args.output_dir / "evaluation_report.json"
    with open(eval_report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved evaluation report to %s", eval_report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
