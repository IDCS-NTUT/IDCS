#!/usr/bin/env python3
"""Export threat model to ONNX format.

Converts PyTorch model checkpoint to ONNX for deployment.

Usage:
    python tools/export_threat_onnx.py \
      --model_path models/threat_model.pt \
      --output_path models/threat_model.onnx
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add repo to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jetson.threat_model import create_threat_model

logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: Path,
    output_path: Path,
    input_shape: tuple = None,
    input_size: int = None,
    opset_version: int = 12,
    verbose: bool = False,
) -> bool:
    """Export PyTorch model to ONNX.

    Args:
        model_path: Path to saved PyTorch model
        output_path: Output path for ONNX model
        input_shape: Input tensor shape (batch_size, num_features) - auto-detected if not provided
        input_size: Number of input features - auto-detected from checkpoint if not provided
        opset_version: ONNX opset version
        verbose: Enable verbose logging

    Returns:
        True if export successful
    """
    device = torch.device("cpu")

    # Auto-detect input size from checkpoint
    if input_size is None:
        logger.info(f"Auto-detecting input size from checkpoint...")
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
        # The first layer weight has shape [output_size, input_size]
        first_layer_weight_shape = checkpoint["model.layers.0.weight"].shape
        input_size = first_layer_weight_shape[1]
        logger.info(f"Detected input size: {input_size} features")

    if input_shape is None:
        input_shape = (1, input_size)

    # Load model with detected input size
    logger.info(f"Loading model from {model_path}...")
    model = create_threat_model(
        input_size=input_size,
        hidden_sizes=[64, 32],
        output_size=3,
        device=device,
    )
    model.load(str(model_path))
    model.eval_mode()

    # Create dummy input
    logger.info(f"Creating dummy input with shape {input_shape}...")
    dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)

    # Export to ONNX
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    try:
        torch.onnx.export(
            model.model,
            dummy_input,
            str(output_path),
            input_names=["features"],
            output_names=["threat_logits"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "threat_logits": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False

    # Verify output file exists
    if not output_path.exists():
        logger.error(f"Output file not created: {output_path}")
        return False

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Export successful! Model size: {output_size_mb:.2f} MB")

    # Validate ONNX model
    logger.info("Validating ONNX model...")
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
    except ImportError:
        logger.warning("ONNX package not available for validation")
    except Exception as e:
        logger.warning(f"ONNX validation failed: {e}")

    return True


def test_onnx_inference(
    model_path_pt: Path,
    model_path_onnx: Path,
    input_size: int = None,
    num_samples: int = 5,
) -> bool:
    """Test ONNX model inference and compare with PyTorch.

    Args:
        model_path_pt: Path to PyTorch model
        model_path_onnx: Path to ONNX model
        input_size: Number of input features - auto-detected if not provided
        num_samples: Number of test samples

    Returns:
        True if outputs match within tolerance
    """
    logger.info("Testing ONNX model inference...")

    device = torch.device("cpu")

    # Auto-detect input size if not provided
    if input_size is None:
        checkpoint = torch.load(str(model_path_pt), map_location=device, weights_only=True)
        first_layer_weight_shape = checkpoint["model.layers.0.weight"].shape
        input_size = first_layer_weight_shape[1]

    # Load PyTorch model
    model_pt = create_threat_model(input_size=input_size, device=device)
    model_pt.load(str(model_path_pt))
    model_pt.eval_mode()

    # Load ONNX model
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path_onnx), providers=["CPUExecutionProvider"])
    except ImportError:
        logger.warning("ONNX Runtime not available, skipping inference test")
        return True

    # Generate test samples
    test_input = np.random.randn(num_samples, input_size).astype(np.float32)
    test_tensor = torch.from_numpy(test_input).float().to(device)

    # PyTorch inference
    with torch.no_grad():
        pt_output = model_pt.forward(test_tensor).cpu().numpy()

    # ONNX inference
    onnx_output = session.run(None, {"features": test_input})[0]

    # Compare
    max_diff = np.max(np.abs(pt_output - onnx_output))
    logger.info(f"Max output difference: {max_diff:.6f}")

    if max_diff < 1e-4:
        logger.info("✓ ONNX and PyTorch outputs match")
        return True
    else:
        logger.warning(f"⚠ Outputs differ by {max_diff:.6f} (threshold: 1e-4)")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export threat model to ONNX")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/threat_model.pt"),
        help="Path to saved PyTorch model",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("models/threat_model.onnx"),
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=12,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test ONNX inference after export",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="Number of input features (auto-detected from checkpoint if not provided)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Verify input file
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return 1

    # Create output directory
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export
    success = export_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        input_size=args.input_size,
        opset_version=args.opset_version,
        verbose=args.verbose,
    )

    if not success:
        return 1

    # Test inference
    if args.test:
        test_success = test_onnx_inference(
            model_path_pt=args.model_path,
            model_path_onnx=args.output_path,
            input_size=args.input_size,
        )
        if not test_success:
            logger.warning("Inference test had issues but export completed")

    logger.info("Export complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
