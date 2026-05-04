#!/usr/bin/env python3
"""Train a learned swarm target-selection policy."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jetson.swarm_policy_model import (
    SwarmPolicySelector,
    THREAT_CLASS_NAMES,
    create_swarm_policy_model,
)

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SwarmPolicyTrainer:
    """Trainer for the learned swarm target-selection policy."""

    def __init__(
        self,
        model: SwarmPolicySelector,
        device: torch.device,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        regret_loss_weight: float = 0.35,
        regret_sample_weight: float = 0.15,
        value_loss_weight: float = 0.5,
        threat_class_loss_weight: float = 0.5,
    ) -> None:
        self.model = model
        self.device = device
        self.regret_loss_weight = max(0.0, float(regret_loss_weight))
        self.regret_sample_weight = max(0.0, float(regret_sample_weight))
        self.value_loss_weight = max(0.0, float(value_loss_weight))
        self.threat_class_loss_weight = max(0.0, float(threat_class_loss_weight))
        self.optimizer = optim.Adam(
            self.model.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "val_loss": [],
            "val_action_accuracy": [],
            "val_mean_regret": [],
            "val_threat_class_accuracy": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train_mode()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            (
                target_features,
                global_features,
                target_mask,
                labels,
                regrets,
                oracle_total_damage,
                threat_class_targets,
            ) = batch
            target_features = target_features.to(self.device)
            global_features = global_features.to(self.device)
            target_mask = target_mask.to(self.device)
            labels = labels.to(self.device)
            regrets = regrets.to(self.device)
            oracle_total_damage = oracle_total_damage.to(self.device)
            threat_class_targets = threat_class_targets.to(self.device)

            self.optimizer.zero_grad()
            logits, value_preds, class_logits = self.model.forward(
                target_features,
                global_features,
                target_mask,
            )
            loss = self._compute_training_loss(
                logits,
                value_preds,
                class_logits,
                target_mask,
                labels,
                regrets,
                oracle_total_damage,
                threat_class_targets,
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _compute_training_loss(
        self,
        logits: torch.Tensor,
        value_preds: torch.Tensor,
        class_logits: Optional[torch.Tensor],
        target_mask: torch.Tensor,
        labels: torch.Tensor,
        regrets: torch.Tensor,
        oracle_total_damage: torch.Tensor,
        threat_class_targets: torch.Tensor,
    ) -> torch.Tensor:
        per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
        safe_regrets = torch.nan_to_num(regrets, nan=0.0, posinf=0.0, neginf=0.0)
        max_regret = torch.max(safe_regrets, dim=1).values
        sample_weights = 1.0 + self.regret_sample_weight * max_regret
        weighted_ce = (per_sample_ce * sample_weights).mean()

        loss = weighted_ce

        if self.regret_loss_weight > 0.0:
            probs = F.softmax(logits, dim=1)
            expected_regret = torch.sum(probs * safe_regrets, dim=1)
            regret_denom = torch.clamp(oracle_total_damage + max_regret, min=1.0)
            normalized_regret = expected_regret / regret_denom
            loss = loss + self.regret_loss_weight * normalized_regret.mean()

        if self.value_loss_weight > 0.0:
            target_values = oracle_total_damage.unsqueeze(1) + safe_regrets
            valid_mask = target_mask.bool()
            if valid_mask.any():
                valid_value_preds = value_preds[valid_mask]
                valid_target_values = target_values[valid_mask]
                value_loss = F.mse_loss(valid_value_preds, valid_target_values)
                loss = loss + self.value_loss_weight * value_loss

        if (
            self.threat_class_loss_weight > 0.0
            and class_logits is not None
            and class_logits.numel() > 0
        ):
            valid_class_mask = threat_class_targets >= 0
            if valid_class_mask.any():
                class_loss = F.cross_entropy(
                    class_logits[valid_class_mask],
                    threat_class_targets[valid_class_mask],
                    weight=torch.tensor(
                        [1.0, 1.15, 1.35],
                        dtype=class_logits.dtype,
                        device=class_logits.device,
                    )[: class_logits.shape[-1]],
                )
                loss = loss + self.threat_class_loss_weight * class_loss

        return loss

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval_mode()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_top2 = 0
        total_regret = 0.0
        total_pred_total_damage = 0.0
        total_oracle_total_damage = 0.0
        total_valid_targets = 0
        total_valid_class_targets = 0
        total_correct_class_targets = 0

        with torch.no_grad():
            for batch in dataloader:
                (
                    target_features,
                    global_features,
                    target_mask,
                    labels,
                    regrets,
                    oracle_total_damage,
                    threat_class_targets,
                ) = batch
                target_features = target_features.to(self.device)
                global_features = global_features.to(self.device)
                target_mask = target_mask.to(self.device)
                labels = labels.to(self.device)
                regrets = regrets.to(self.device)
                oracle_total_damage = oracle_total_damage.to(self.device)
                threat_class_targets = threat_class_targets.to(self.device)

                logits, value_preds, class_logits = self.model.forward(
                    target_features,
                    global_features,
                    target_mask,
                )
                loss = self._compute_training_loss(
                    logits,
                    value_preds,
                    class_logits,
                    target_mask,
                    labels,
                    regrets,
                    oracle_total_damage,
                    threat_class_targets,
                )
                preds = torch.argmax(logits, dim=1)
                top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).indices

                total_loss += loss.item()
                total_samples += labels.size(0)
                total_correct += (preds == labels).sum().item()
                total_top2 += (top2 == labels.unsqueeze(1)).any(dim=1).sum().item()
                total_valid_targets += target_mask.sum(dim=1).sum().item()
                if class_logits is not None:
                    valid_class_mask = threat_class_targets >= 0
                    if valid_class_mask.any():
                        class_preds = torch.argmax(class_logits, dim=-1)
                        total_correct_class_targets += (
                            class_preds[valid_class_mask] == threat_class_targets[valid_class_mask]
                        ).sum().item()
                        total_valid_class_targets += int(valid_class_mask.sum().item())

                regrets_np = regrets.cpu().numpy()
                preds_np = preds.cpu().numpy()
                oracle_np = oracle_total_damage.cpu().numpy()
                for row_idx, pred_idx in enumerate(preds_np):
                    regret = regrets_np[row_idx, pred_idx]
                    if np.isnan(regret):
                        regret = 0.0
                    total_regret += float(regret)
                    total_pred_total_damage += float(oracle_np[row_idx] + regret)
                    total_oracle_total_damage += float(oracle_np[row_idx])

        if total_samples == 0:
            return {
                "loss": 0.0,
                "action_accuracy": 0.0,
                "top2_accuracy": 0.0,
                "mean_regret": 0.0,
                "mean_predicted_total_damage": 0.0,
                "mean_oracle_total_damage": 0.0,
                "mean_valid_targets": 0.0,
                "threat_class_accuracy": 0.0,
                "num_samples": 0,
            }

        return {
            "loss": float(total_loss / max(1, len(dataloader))),
            "action_accuracy": float(total_correct / total_samples),
            "top2_accuracy": float(total_top2 / total_samples),
            "mean_regret": float(total_regret / total_samples),
            "mean_predicted_total_damage": float(total_pred_total_damage / total_samples),
            "mean_oracle_total_damage": float(total_oracle_total_damage / total_samples),
            "mean_valid_targets": float(total_valid_targets / total_samples),
            "threat_class_accuracy": (
                float(total_correct_class_targets / total_valid_class_targets)
                if total_valid_class_targets > 0
                else 0.0
            ),
            "num_samples": int(total_samples),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        num_epochs: int,
        early_stopping_patience: int,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        logger.info("Starting swarm policy training for %d epochs", num_epochs)
        patience_counter = 0
        has_validation = len(val_loader) > 0
        if not has_validation:
            logger.warning("Validation split is empty; training will proceed without early stopping")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            self.history["loss"].append(float(train_loss))

            if has_validation:
                val_report = self.evaluate(val_loader)
                val_loss = float(val_report["loss"])
                self.history["val_loss"].append(val_loss)
                self.history["val_action_accuracy"].append(float(val_report["action_accuracy"]))
                self.history["val_mean_regret"].append(float(val_report["mean_regret"]))
                self.history["val_threat_class_accuracy"].append(
                    float(val_report["threat_class_accuracy"])
                )
            else:
                val_report = {
                    "loss": train_loss,
                    "action_accuracy": 0.0,
                    "mean_regret": 0.0,
                }
                val_loss = float(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_action_accuracy"].append(0.0)
                self.history["val_mean_regret"].append(0.0)
                self.history["val_threat_class_accuracy"].append(0.0)

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                logger.info(
                    "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_regret=%.4f | val_cls_acc=%.4f",
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    val_loss,
                    float(val_report["action_accuracy"]),
                    float(val_report["mean_regret"]),
                    float(val_report["threat_class_accuracy"]),
                )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if has_validation and patience_counter >= early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (best epoch=%d, val_loss=%.4f)",
                        epoch + 1,
                        self.best_epoch,
                        self.best_val_loss,
                    )
                    break

        if self.best_state_dict is not None:
            self.model.model.load_state_dict(self.best_state_dict)

        return self.history


def _build_dataloader(npz_path: Path, batch_size: int, shuffle: bool) -> DataLoader:
    data = np.load(npz_path)
    dataset = TensorDataset(
        torch.from_numpy(data["target_features"]).float(),
        torch.from_numpy(data["global_features"]).float(),
        torch.from_numpy(data["target_mask"]).bool(),
        torch.from_numpy(data["oracle_action_index"]).long(),
        torch.from_numpy(data["oracle_regret_by_action"]).float(),
        torch.from_numpy(data["oracle_total_damage"]).float(),
        torch.from_numpy(data["threat_class_targets"]).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_dataset_metadata(dataset_dir: Path) -> Dict[str, Any]:
    stats_path = dataset_dir / "statistics.json"
    if not stats_path.exists():
        return {}
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read dataset metadata from %s", stats_path)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/swarm_model.yaml"))
    parser.add_argument("--dataset_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--regret_loss_weight", type=float, default=None)
    parser.add_argument("--regret_sample_weight", type=float, default=None)
    parser.add_argument("--value_loss_weight", type=float, default=None)
    parser.add_argument("--threat_class_loss_weight", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--context_size", type=int, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = _load_config(args.config)
    dataset_dir = Path(args.dataset_dir or config["dataset"]["directory"])
    output_dir = Path(args.output_dir or config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    hidden_size = int(args.hidden_size or model_cfg.get("hidden_size", 96))
    context_size = int(args.context_size or model_cfg.get("context_size", 64))
    batch_size = int(args.batch_size or training_cfg.get("batch_size", 64))
    epochs = int(args.epochs or training_cfg.get("epochs", 80))
    learning_rate = float(args.learning_rate or training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(args.weight_decay or training_cfg.get("weight_decay", 1e-4))
    patience = int(
        args.early_stopping_patience
        or training_cfg.get("early_stopping_patience", 12)
    )
    regret_loss_weight = float(
        args.regret_loss_weight
        if args.regret_loss_weight is not None
        else training_cfg.get("regret_loss_weight", 0.35)
    )
    regret_sample_weight = float(
        args.regret_sample_weight
        if args.regret_sample_weight is not None
        else training_cfg.get("regret_sample_weight", 0.15)
    )
    value_loss_weight = float(
        args.value_loss_weight
        if args.value_loss_weight is not None
        else training_cfg.get("value_loss_weight", 0.5)
    )
    threat_class_loss_weight = float(
        args.threat_class_loss_weight
        if args.threat_class_loss_weight is not None
        else training_cfg.get("threat_class_loss_weight", 0.3)
    )
    seed = int(args.seed if args.seed is not None else training_cfg.get("seed", 42))

    _set_seed(seed)

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU for swarm policy training")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for swarm policy training")

    logger.info("Loading swarm datasets from %s", dataset_dir)
    train_loader = _build_dataloader(dataset_dir / "train.npz", batch_size=batch_size, shuffle=True)
    val_loader = _build_dataloader(dataset_dir / "val.npz", batch_size=batch_size, shuffle=False)
    test_loader = _build_dataloader(dataset_dir / "test.npz", batch_size=batch_size, shuffle=False)
    dataset_metadata = _load_dataset_metadata(dataset_dir)
    heldout_path = dataset_dir / "heldout.npz"
    heldout_loader = (
        _build_dataloader(heldout_path, batch_size=batch_size, shuffle=False)
        if heldout_path.exists()
        else None
    )

    train_batch = next(iter(train_loader))
    target_feature_size = int(train_batch[0].shape[-1])
    global_feature_size = int(train_batch[1].shape[-1])
    max_targets = int(train_batch[0].shape[1])
    logger.info(
        "Resolved swarm model input sizes: targets=%d features, globals=%d features, max_targets=%d",
        target_feature_size,
        global_feature_size,
        max_targets,
    )

    model = create_swarm_policy_model(
        target_feature_size=target_feature_size,
        global_feature_size=global_feature_size,
        hidden_size=hidden_size,
        context_size=context_size,
        num_threat_classes=len(THREAT_CLASS_NAMES),
        device=device,
    )
    trainer = SwarmPolicyTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        regret_loss_weight=regret_loss_weight,
        regret_sample_weight=regret_sample_weight,
        value_loss_weight=value_loss_weight,
        threat_class_loss_weight=threat_class_loss_weight,
    )

    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=epochs,
        early_stopping_patience=patience,
        log_interval=10,
    )

    checkpoint_path = output_dir / "swarm_policy.pt"
    payload = {
        "model_state_dict": model.model.state_dict(),
        "target_feature_size": target_feature_size,
        "global_feature_size": global_feature_size,
        "hidden_size": hidden_size,
        "context_size": context_size,
        "num_threat_classes": len(THREAT_CLASS_NAMES),
        "max_targets": max_targets,
        "seed": seed,
        "dataset_dir": str(dataset_dir),
    }
    if "normalization" in dataset_metadata:
        payload["normalization"] = dataset_metadata["normalization"]
    if "feature_names" in dataset_metadata:
        payload["feature_names"] = dataset_metadata["feature_names"]
    if "global_feature_names" in dataset_metadata:
        payload["global_feature_names"] = dataset_metadata["global_feature_names"]
    if "threat_class_names" in dataset_metadata:
        payload["threat_class_names"] = dataset_metadata["threat_class_names"]
    if "max_targets_tensor" in dataset_metadata:
        payload["max_targets_tensor"] = dataset_metadata["max_targets_tensor"]
    torch.save(payload, str(checkpoint_path))
    logger.info("Saved swarm policy checkpoint to %s", checkpoint_path)

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    logger.info("Running final swarm policy evaluation")
    val_report = trainer.evaluate(val_loader)
    test_report = trainer.evaluate(test_loader)
    heldout_report = trainer.evaluate(heldout_loader) if heldout_loader is not None else None

    evaluation_report = {
        "validation": val_report,
        "test": test_report,
        "heldout": heldout_report,
        "model": {
            "target_feature_size": target_feature_size,
            "global_feature_size": global_feature_size,
            "hidden_size": hidden_size,
            "context_size": context_size,
            "num_threat_classes": len(THREAT_CLASS_NAMES),
            "max_targets": max_targets,
        },
        "training": {
            "epochs_requested": epochs,
            "best_epoch": trainer.best_epoch,
            "best_val_loss": trainer.best_val_loss,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping_patience": patience,
            "regret_loss_weight": regret_loss_weight,
            "regret_sample_weight": regret_sample_weight,
            "value_loss_weight": value_loss_weight,
            "threat_class_loss_weight": threat_class_loss_weight,
            "seed": seed,
        },
        "dataset": {
            "directory": str(dataset_dir),
            "normalization": dataset_metadata.get("normalization"),
            "feature_names": dataset_metadata.get("feature_names"),
            "global_feature_names": dataset_metadata.get("global_feature_names"),
        },
    }
    evaluation_path = output_dir / "evaluation_report.json"
    evaluation_path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")

    logger.info(
        "Validation | loss=%.4f acc=%.4f regret=%.4f cls_acc=%.4f",
        val_report["loss"],
        val_report["action_accuracy"],
        val_report["mean_regret"],
        val_report["threat_class_accuracy"],
    )
    logger.info(
        "Test       | loss=%.4f acc=%.4f regret=%.4f cls_acc=%.4f",
        test_report["loss"],
        test_report["action_accuracy"],
        test_report["mean_regret"],
        test_report["threat_class_accuracy"],
    )
    if heldout_report is not None:
        logger.info(
            "Heldout    | loss=%.4f acc=%.4f regret=%.4f cls_acc=%.4f",
            heldout_report["loss"],
            heldout_report["action_accuracy"],
            heldout_report["mean_regret"],
            heldout_report["threat_class_accuracy"],
        )
    logger.info("Saved swarm evaluation report to %s", evaluation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
