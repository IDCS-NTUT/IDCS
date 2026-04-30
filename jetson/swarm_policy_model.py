"""Learned swarm target-selection policy model.

The model scores each active target in a variable-sized set using:
- per-target structured features
- pooled context across all visible targets
- global episode/state features

It is intentionally small and permutation-invariant so it can later be used
for Jetson deployment without introducing a CNN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SwarmPolicyNetwork",
    "SwarmPolicySelector",
    "create_swarm_policy_model",
    "load_swarm_policy_checkpoint",
]


class SwarmPolicyNetwork(nn.Module):
    """Scores each target and predicts per-target damage for reranking."""

    def __init__(
        self,
        target_feature_size: int,
        global_feature_size: int,
        hidden_size: int = 64,
        context_size: int = 64,
    ) -> None:
        super().__init__()

        self.target_feature_size = int(target_feature_size)
        self.global_feature_size = int(global_feature_size)
        self.hidden_size = int(hidden_size)
        self.context_size = int(context_size)

        self.target_encoder = nn.Sequential(
            nn.Linear(self.target_feature_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_size, self.context_size),
            nn.ReLU(),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + self.context_size, self.context_size),
            nn.ReLU(),
        )
        self.shared_head = nn.Sequential(
            nn.Linear(
                self.hidden_size + self.context_size + self.target_feature_size,
                self.hidden_size,
            ),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(self.hidden_size, 1)
        self.value_head = nn.Linear(self.hidden_size, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        target_features: torch.Tensor,
        global_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return masked policy logits and value predictions.

        Returns
        -------
        logits:
            Shape ``[batch, num_targets]``. Larger is better.
        value_preds:
            Shape ``[batch, num_targets]``. Lower predicted total damage is better.
        """
        if target_features.dim() != 3:
            raise ValueError(
                f"target_features must have shape [batch, num_targets, num_features], got {tuple(target_features.shape)}"
            )
        if global_features.dim() != 2:
            raise ValueError(
                f"global_features must have shape [batch, num_global_features], got {tuple(global_features.shape)}"
            )
        if target_mask.dim() != 2:
            raise ValueError(
                f"target_mask must have shape [batch, num_targets], got {tuple(target_mask.shape)}"
            )

        batch_size, num_targets, feature_size = target_features.shape
        if feature_size != self.target_feature_size:
            raise ValueError(
                f"Expected {self.target_feature_size} target features, got {feature_size}"
            )
        if global_features.shape[0] != batch_size:
            raise ValueError("target_features and global_features batch size must match")
        if target_mask.shape != (batch_size, num_targets):
            raise ValueError("target_mask shape must match [batch, num_targets]")

        target_mask_bool = target_mask.bool()
        target_mask_float = target_mask_bool.unsqueeze(-1).to(dtype=target_features.dtype)

        encoded_targets = self.target_encoder(target_features)
        masked_targets = encoded_targets * target_mask_float
        denom = target_mask_float.sum(dim=1).clamp_min(1.0)
        pooled_mean = masked_targets.sum(dim=1) / denom

        global_context = self.global_encoder(global_features)
        fused_context = self.context_fusion(torch.cat([pooled_mean, global_context], dim=1))
        expanded_context = fused_context.unsqueeze(1).expand(-1, num_targets, -1)

        scorer_input = torch.cat([encoded_targets, expanded_context, target_features], dim=-1)
        shared = self.shared_head(scorer_input)
        logits = self.policy_head(shared).squeeze(-1)
        value_preds = self.value_head(shared).squeeze(-1)
        invalid_logit_fill = torch.finfo(logits.dtype).min
        invalid_value_fill = torch.finfo(value_preds.dtype).max
        return (
            logits.masked_fill(~target_mask_bool, invalid_logit_fill),
            value_preds.masked_fill(~target_mask_bool, invalid_value_fill),
        )


class SwarmPolicySelector:
    """Wrapper around :class:`SwarmPolicyNetwork` for training and inference."""

    def __init__(
        self,
        model: SwarmPolicyNetwork,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    def forward(
        self,
        target_features: torch.Tensor,
        global_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(target_features, global_features, target_mask)

    def predict_action(
        self,
        target_features: torch.Tensor,
        global_features: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        return_probs: bool = False,
        use_value_rerank: bool = False,
        rerank_topk: int = 2,
    ) -> torch.Tensor:
        with torch.no_grad():
            logits, value_preds = self.forward(target_features, global_features, target_mask)
            if use_value_rerank:
                return self._rerank_actions(logits, value_preds, target_mask, topk=rerank_topk)
            if return_probs:
                return F.softmax(logits, dim=1)
            return torch.argmax(logits, dim=1)

    def predict_action_numpy(
        self,
        target_features: np.ndarray,
        global_features: np.ndarray,
        target_mask: np.ndarray,
        *,
        use_value_rerank: bool = False,
        rerank_topk: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.eval_mode()
        target_tensor = torch.from_numpy(target_features).float().to(self.device)
        global_tensor = torch.from_numpy(global_features).float().to(self.device)
        mask_tensor = torch.from_numpy(target_mask.astype(np.bool_)).to(self.device)
        with torch.no_grad():
            logits, value_preds = self.forward(target_tensor, global_tensor, mask_tensor)
            probs = F.softmax(logits, dim=1)
            if use_value_rerank:
                actions = self._rerank_actions(
                    logits,
                    value_preds,
                    mask_tensor,
                    topk=rerank_topk,
                )
            else:
                actions = torch.argmax(logits, dim=1)
        return actions.cpu().numpy(), probs.cpu().numpy(), value_preds.cpu().numpy()

    def _rerank_actions(
        self,
        logits: torch.Tensor,
        value_preds: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        topk: int,
    ) -> torch.Tensor:
        topk = max(1, int(topk))
        candidate_count = min(topk, logits.shape[1])
        topk_indices = torch.topk(logits, k=candidate_count, dim=1).indices
        topk_values = torch.gather(value_preds, 1, topk_indices)
        topk_mask = torch.gather(target_mask.bool(), 1, topk_indices)
        topk_values = topk_values.masked_fill(~topk_mask, torch.finfo(topk_values.dtype).max)
        reranked_positions = torch.argmin(topk_values, dim=1)
        return topk_indices.gather(1, reranked_positions.unsqueeze(1)).squeeze(1)

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "target_feature_size": self.model.target_feature_size,
            "global_feature_size": self.model.global_feature_size,
            "hidden_size": self.model.hidden_size,
            "context_size": self.model.context_size,
        }
        torch.save(payload, str(path))


def create_swarm_policy_model(
    target_feature_size: int,
    global_feature_size: int,
    *,
    hidden_size: int = 64,
    context_size: int = 64,
    device: Optional[torch.device] = None,
) -> SwarmPolicySelector:
    model = SwarmPolicyNetwork(
        target_feature_size=target_feature_size,
        global_feature_size=global_feature_size,
        hidden_size=hidden_size,
        context_size=context_size,
    )
    return SwarmPolicySelector(model=model, device=device)


def load_swarm_policy_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[SwarmPolicySelector, Dict[str, Any]]:
    """Load checkpoint and rebuild the model with saved dimensions."""
    checkpoint = torch.load(str(checkpoint_path), map_location=device or "cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing model_state_dict")

    selector = create_swarm_policy_model(
        target_feature_size=int(checkpoint["target_feature_size"]),
        global_feature_size=int(checkpoint["global_feature_size"]),
        hidden_size=int(checkpoint.get("hidden_size", 64)),
        context_size=int(checkpoint.get("context_size", 64)),
        device=device,
    )
    selector.model.load_state_dict(checkpoint["model_state_dict"])
    selector.eval_mode()
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }
    return selector, metadata
