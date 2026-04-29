"""Threat model inference wrapper using ONNX Runtime.

Provides a simple interface for loading and running the threat classification
model in production on Jetson.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["ThreatInferenceEngine", "ThreatLevel"]

logger = logging.getLogger(__name__)

# Class mappings
ThreatLevel = {
    0: "benign",
    1: "suspicious",
    2: "threatening",
}

THREAT_CLASS_NAMES = ["benign", "suspicious", "threatening"]


class ThreatInferenceEngine:
    """ONNX-based threat classification inference engine."""

    def __init__(self, model_path: str | Path, use_gpu: bool = False):
        """Initialize threat inference engine.

        Args:
            model_path: Path to ONNX model file
            use_gpu: Whether to use GPU (if available)

        Raises:
            ImportError: If ONNX Runtime not available
            FileNotFoundError: If model file not found
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX Runtime required for threat inference. "
                "Install with: pip install onnxruntime"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Select execution provider
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        logger.info(f"Loading ONNX model from {model_path}...")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"Model loaded. Input: {self.input_name}, Output: {self.output_name}")

        self.model_path = model_path

    def predict(self, features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Predict threat class and confidence.

        Args:
            features: Feature array of shape [176] or [batch, 176]
                     Can also be [16, 11] or [batch, 16, 11] which will be flattened

        Returns:
            Tuple of (class_id, confidence, probabilities)
            - class_id: 0 (benign), 1 (suspicious), 2 (threatening)
            - confidence: Probability of predicted class
            - probabilities: Array of shape [3] with probabilities for each class

        Raises:
            ValueError: If input shape invalid
        """
        # Handle different input shapes
        if features.ndim == 1 and features.shape[0] == 176:
            # Already flattened, single sample
            features = features.reshape(1, 176)
        elif features.ndim == 2 and features.shape[1] == 176:
            # Already flattened, batch
            pass
        elif features.ndim == 2 and features.shape == (16, 11):
            # Single windowed sample, flatten
            features = features.reshape(1, 176)
        elif features.ndim == 3 and features.shape[1:] == (16, 11):
            # Batch of windowed samples, flatten
            batch_size = features.shape[0]
            features = features.reshape(batch_size, 176)
        else:
            raise ValueError(
                f"Invalid feature shape: {features.shape}. "
                f"Expected [176], [batch, 176], [16, 11], or [batch, 16, 11]"
            )

        # Ensure float32
        features = features.astype(np.float32)

        # Inference
        logits = self.session.run(None, {self.input_name: features})[0]

        # Compute probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Return first sample (or only sample)
        class_id = int(np.argmax(logits[0]))
        confidence = float(probabilities[0, class_id])
        class_probs = probabilities[0].astype(np.float32)

        return class_id, confidence, class_probs

    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict threat classes for batch.

        Args:
            features: Feature array of shape [batch, 176] or [batch, 16, 11]

        Returns:
            Tuple of (class_ids, confidences, all_probabilities)
            - class_ids: Array of predicted class IDs
            - confidences: Array of confidence scores
            - all_probabilities: Array of shape [batch, 3] with all probabilities
        """
        # Flatten if needed
        if features.ndim == 3 and features.shape[1:] == (16, 11):
            batch_size = features.shape[0]
            features = features.reshape(batch_size, 176)

        features = features.astype(np.float32)

        # Inference
        logits = self.session.run(None, {self.input_name: features})[0]

        # Compute probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Extract predictions
        class_ids = np.argmax(logits, axis=1)
        confidences = probabilities[np.arange(len(class_ids)), class_ids]

        return class_ids, confidences, probabilities

    def predict_threaten_probability(self, features: np.ndarray) -> float:
        """Get probability of threatening class.

        Args:
            features: Feature array [176] or [batch, 176] or [16, 11]

        Returns:
            Probability of threatening class (0-1)
        """
        class_id, _, probs = self.predict(features)
        return float(probs[2])  # Index 2 is "threatening"


class ThreatMetricsWindow:
    """Rolling window buffer for threat metrics.

    Maintains a sliding window of threat metrics for each tracked target,
    enabling temporal aggregation for model inference.
    """

    def __init__(self, max_window_size: int = 16):
        """Initialize metrics window.

        Args:
            max_window_size: Maximum number of frames to buffer
        """
        self.max_window_size = max_window_size
        self.metrics_buffer: List[Dict[str, float]] = []

    def add_frame_metrics(self, metrics: Dict[str, float]) -> None:
        """Add metrics from one frame.

        Args:
            metrics: Dictionary with keys:
                - center_x, center_y: Normalized bbox center [0, 1]
                - bbox_w, bbox_h: Normalized bbox size [0, 1]
                - velocity_x, velocity_y: Pixels/second
                - confidence: Detection confidence [0, 1]
                - distance: Distance to asset in meters
                - distance_rate: Rate of distance change m/s
                - zone_level: Zone level (0=critical, 1=restricted, 2=warning, 3=none)
                - dwell_time: Time in current zone in seconds
        """
        self.metrics_buffer.append(metrics)
        if len(self.metrics_buffer) > self.max_window_size:
            self.metrics_buffer.pop(0)

    def get_features(self) -> Optional[np.ndarray]:
        """Get feature window for inference.

        Returns:
            Array of shape [window_size, 11] with features, or None if window empty.
            Features in order:
            [center_x, center_y, bbox_w, bbox_h, velocity_x, velocity_y,
             confidence, distance, distance_rate, zone_level, dwell_time]
        """
        if not self.metrics_buffer:
            return None

        features_list = []
        for metrics in self.metrics_buffer:
            frame_features = [
                metrics.get("center_x", 0.0),
                metrics.get("center_y", 0.0),
                metrics.get("bbox_w", 0.0),
                metrics.get("bbox_h", 0.0),
                metrics.get("velocity_x", 0.0),
                metrics.get("velocity_y", 0.0),
                metrics.get("confidence", 0.0),
                metrics.get("distance", 0.0),
                metrics.get("distance_rate", 0.0),
                metrics.get("zone_level", 0.0),
                metrics.get("dwell_time", 0.0),
            ]
            features_list.append(frame_features)

        features = np.array(features_list, dtype=np.float32)
        return features

    def clear(self) -> None:
        """Clear buffer."""
        self.metrics_buffer.clear()

    def size(self) -> int:
        """Get current buffer size."""
        return len(self.metrics_buffer)

    def is_full(self) -> bool:
        """Check if buffer is at max size."""
        return len(self.metrics_buffer) >= self.max_window_size


def create_inference_engine(
    model_path: Optional[str | Path] = None,
    use_gpu: bool = False,
) -> Optional[ThreatInferenceEngine]:
    """Factory function to create threat inference engine.

    Args:
        model_path: Path to ONNX model. If None, tries default location.
        use_gpu: Whether to use GPU

    Returns:
        ThreatInferenceEngine or None if model not found
    """
    if model_path is None:
        # Try default location
        model_path = Path(__file__).resolve().parents[1] / "models" / "threat_model.onnx"

    if not Path(model_path).exists():
        logger.warning(f"Threat model not found at {model_path}, threat inference disabled")
        return None

    try:
        return ThreatInferenceEngine(model_path, use_gpu=use_gpu)
    except Exception as e:
        logger.error(f"Failed to create threat inference engine: {e}")
        return None
