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
from typing import Any, Dict, Optional, Tuple

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

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader)
            self.train_history["loss"].append(train_loss)

            # Validate
            val_loss, val_accuracy = self.evaluate(val_dataloader)
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

    X_train = torch.from_numpy(train_data["X"]).float()
    y_train = torch.from_numpy(train_data["y"]).long()
    X_val = torch.from_numpy(val_data["X"]).float()
    y_val = torch.from_numpy(val_data["y"]).long()

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

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

    # Create model
    logger.info("Creating threat model...")
    model = create_threat_model(
        input_size=176,
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

    # Evaluate on validation set
    logger.info("\n=== FINAL EVALUATION ===")
    model.eval_mode()
    val_loss, val_accuracy = trainer.evaluate(val_loader)
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
