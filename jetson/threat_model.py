"""Threat evaluation neural network model.

Implements a simple feedforward MLP for threat classification:
- Input: 176 flattened features (16 frames × 11 features)
- Hidden: 64 -> 32 neurons with ReLU activation
- Output: 3 class logits (benign, suspicious, threatening)

Architecture chosen for:
- Jetson efficiency (small, fast)
- ONNX exportability
- Explainability (no recurrence or attention)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ThreatModelMLP",
    "ThreatClassifier",
]


class ThreatModelMLP(nn.Module):
    """Simple feedforward MLP for threat classification.

    Architecture:
    - Input: [batch_size, 176] (flattened 16x11 features)
    - Dense: 176 -> 64 + ReLU
    - Dense: 64 -> 32 + ReLU
    - Dense: 32 -> 3 (logits)
    """

    def __init__(
        self,
        input_size: int = 176,
        hidden_sizes: list[int] = None,
        output_size: int = 3,
        activation: Literal["relu"] = "relu",
    ):
        """Initialize MLP.

        Args:
            input_size: Number of input features (default 176 = 16*11)
            hidden_sizes: List of hidden layer sizes (default [64, 32])
            output_size: Number of output classes (default 3)
            activation: Activation function name (default "relu")
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer (no activation, logits)
        layers.append(nn.Linear(prev_size, output_size))

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, 176]

        Returns:
            Logits tensor [batch_size, 3]
        """
        return self.layers(x)

    def predict(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """Predict threat class.

        Args:
            x: Input tensor [batch_size, 176] or [16, 11]
            return_probs: If True, return softmax probabilities

        Returns:
            Class predictions or probabilities
        """
        # Reshape if needed (handle both flattened and windowed input)
        if x.dim() == 3:  # [batch, 16, 11]
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)
        elif x.dim() == 2 and x.size(1) != self.input_size:  # [16, 11]
            x = x.reshape(1, -1)

        with torch.no_grad():
            logits = self.forward(x)
            if return_probs:
                return F.softmax(logits, dim=1)
            else:
                return torch.argmax(logits, dim=1)

    def get_class_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities for all classes.

        Args:
            x: Input tensor [batch_size, 176] or [16, 11]

        Returns:
            Softmax probabilities [batch_size, 3]
        """
        if x.dim() == 3:
            batch_size = x.size(0)
            x = x.reshape(batch_size, -1)
        elif x.dim() == 2 and x.size(1) != self.input_size:
            x = x.reshape(1, -1)

        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class ThreatClassifier:
    """Wrapper for threat model with training/inference utilities.

    Manages model training, evaluation, and inference workflows.
    """

    def __init__(
        self,
        model: Optional[ThreatModelMLP] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize classifier.

        Args:
            model: ThreatModelMLP instance (default: new with defaults)
            device: Device to use (default: auto-detect)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model or ThreatModelMLP()
        self.model.to(device)

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (training)."""
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class (inference, no grad)."""
        return self.model.predict(x, return_probs=False)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probabilities (inference, no grad)."""
        return self.model.predict(x, return_probs=True)

    def get_class_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax scores."""
        return self.model.get_class_scores(x)

    def state_dict(self) -> Dict:
        """Get model state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load model state dict."""
        self.model.load_state_dict(state_dict)

    def to(self, device: torch.device) -> None:
        """Move model to device."""
        self.device = device
        self.model.to(device)

    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model from file."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)


def create_threat_model(
    input_size: int = 176,
    hidden_sizes: list[int] = None,
    output_size: int = 3,
    device: Optional[torch.device] = None,
) -> ThreatClassifier:
    """Factory function to create threat model.

    Args:
        input_size: Number of features
        hidden_sizes: Hidden layer sizes
        output_size: Number of classes
        device: Compute device

    Returns:
        ThreatClassifier instance
    """
    model = ThreatModelMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
    )
    return ThreatClassifier(model=model, device=device)
