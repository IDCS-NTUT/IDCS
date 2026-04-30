#!/usr/bin/env python3
"""Export a trained swarm policy checkpoint to ONNX.

Supports exporting either:
- the policy logits only
- the multitask outputs (policy logits + value predictions)
- a DLA-oriented static policy-only graph with fixed target count

The exported graph uses three inputs:
- target_features: [batch, num_targets, target_feature_size]
- global_features: [batch, global_feature_size]
- target_mask: [batch, num_targets]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jetson.swarm_policy_model import load_swarm_policy_checkpoint

logger = logging.getLogger(__name__)


class _PolicyOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        target_features: torch.Tensor,
        global_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.model(target_features, global_features, target_mask)
        return logits


class _PolicyValueWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        target_features: torch.Tensor,
        global_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(target_features, global_features, target_mask)


class _StaticPolicyOnlyWrapper(torch.nn.Module):
    """Static-shape policy-only export path for cleaner TensorRT / DLA graphs.

    This wrapper removes the target-mask and dynamic expansion logic from the
    exported graph. It assumes a fixed number of target slots and scores every
    slot as valid. Callers should zero-pad unused target slots consistently.
    """

    def __init__(self, model: torch.nn.Module, num_targets: int) -> None:
        super().__init__()
        self.model = model
        self.num_targets = int(num_targets)

    def forward(
        self,
        target_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        encoded_targets = self.model.target_encoder(target_features)
        pooled_mean = encoded_targets.mean(dim=1)
        global_context = self.model.global_encoder(global_features)
        fused_context = self.model.context_fusion(
            torch.cat([pooled_mean, global_context], dim=1)
        )

        logits = []
        for index in range(self.num_targets):
            scorer_input = torch.cat(
                [
                    encoded_targets[:, index, :],
                    fused_context,
                    target_features[:, index, :],
                ],
                dim=1,
            )
            shared = self.model.shared_head(scorer_input)
            logits.append(self.model.policy_head(shared))
        return torch.cat(logits, dim=1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to swarm_policy.pt checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to output ONNX model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Dummy export batch size",
    )
    parser.add_argument(
        "--num_targets",
        type=int,
        default=None,
        help="Number of target slots. Defaults to checkpoint max_targets/max_targets_tensor.",
    )
    parser.add_argument(
        "--export_mode",
        choices=("policy_only", "policy_value", "dla_static_policy_only"),
        default="policy_value",
        help="Whether to export only policy logits or both logits and value predictions.",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="Enable dynamic batch axis in the ONNX export.",
    )
    parser.add_argument(
        "--dynamic_targets",
        action="store_true",
        help="Enable dynamic num_targets axis in the ONNX export.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def _make_dynamic_axes(
    output_names: Sequence[str],
    *,
    dynamic_batch: bool,
    dynamic_targets: bool,
) -> dict[str, dict[int, str]]:
    dynamic_axes: dict[str, dict[int, str]] = {}
    if dynamic_batch:
        dynamic_axes["target_features"] = {0: "batch"}
        dynamic_axes["global_features"] = {0: "batch"}
        dynamic_axes["target_mask"] = {0: "batch"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}
    if dynamic_targets:
        dynamic_axes.setdefault("target_features", {})[1] = "num_targets"
        dynamic_axes.setdefault("target_mask", {})[1] = "num_targets"
        for name in output_names:
            dynamic_axes.setdefault(name, {})[1] = "num_targets"
    return dynamic_axes


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    selector, metadata = load_swarm_policy_checkpoint(args.model_path, device=torch.device("cpu"))
    selector.eval_mode()
    model = selector.model

    target_feature_size = int(metadata["target_feature_size"])
    global_feature_size = int(metadata["global_feature_size"])
    num_targets = int(
        args.num_targets
        or metadata.get("max_targets_tensor")
        or metadata.get("max_targets")
        or 8
    )

    target_features = torch.zeros(
        (args.batch_size, num_targets, target_feature_size), dtype=torch.float32
    )
    global_features = torch.zeros(
        (args.batch_size, global_feature_size), dtype=torch.float32
    )
    target_mask = torch.ones((args.batch_size, num_targets), dtype=torch.bool)

    if args.export_mode == "policy_only":
        export_model: torch.nn.Module = _PolicyOnlyWrapper(model)
        export_args = (target_features, global_features, target_mask)
        input_names = ["target_features", "global_features", "target_mask"]
        output_names = ["policy_logits"]
    elif args.export_mode == "dla_static_policy_only":
        export_model = _StaticPolicyOnlyWrapper(model, num_targets=num_targets)
        export_args = (target_features, global_features)
        input_names = ["target_features", "global_features"]
        output_names = ["policy_logits"]
    else:
        export_model = _PolicyValueWrapper(model)
        export_args = (target_features, global_features, target_mask)
        input_names = ["target_features", "global_features", "target_mask"]
        output_names = ["policy_logits", "value_predictions"]

    dynamic_axes = {}
    if args.export_mode != "dla_static_policy_only":
        dynamic_axes = _make_dynamic_axes(
            output_names,
            dynamic_batch=args.dynamic_batch,
            dynamic_targets=args.dynamic_targets,
        )
    elif args.dynamic_batch or args.dynamic_targets:
        logger.warning(
            "Ignoring --dynamic_batch/--dynamic_targets for dla_static_policy_only export"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Exporting swarm policy ONNX to %s | mode=%s | target_features=%d | global_features=%d | num_targets=%d",
        args.output_path,
        args.export_mode,
        target_feature_size,
        global_feature_size,
        num_targets,
    )

    torch.onnx.export(
        export_model,
        export_args,
        str(args.output_path),
        input_names=input_names,
        output_names=list(output_names),
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        opset_version=args.opset_version,
        do_constant_folding=True,
    )

    logger.info("Swarm ONNX export complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
