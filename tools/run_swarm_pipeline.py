#!/usr/bin/env python3
"""Run swarm dataset generation, training, and policy benchmarking as one pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return payload


def _run_step(command: List[str]) -> None:
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/swarm_dataset.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/swarm_model.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--benchmark-output", type=Path, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--dataset-seed", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--regret-loss-weight", type=float, default=None)
    parser.add_argument("--regret-sample-weight", type=float, default=None)
    parser.add_argument("--value-loss-weight", type=float, default=None)
    parser.add_argument("--threat-class-loss-weight", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--split", default="heldout")
    parser.add_argument(
        "--policies",
        default="swarm_planner,learned_model,learned_model_rerank,learned_value_only,max_conf,closest_breakthrough,highest_damage,largest_area",
    )
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    dataset_cfg = _load_yaml(_REPO_ROOT / args.dataset_config)
    model_cfg = _load_yaml(_REPO_ROOT / args.model_config)

    dataset_dir = Path(
        args.dataset_dir
        or dataset_cfg.get("output", {}).get("directory", "artifacts/swarm/datasets/default")
    )
    model_dir = Path(
        args.model_dir
        or model_cfg.get("output", {}).get("directory", "artifacts/swarm/models/default")
    )
    benchmark_output = Path(
        args.benchmark_output or model_dir / f"policy_report_{args.split}.json"
    )

    python_exe = sys.executable
    if not python_exe:
        raise RuntimeError("Unable to resolve Python executable for pipeline subprocesses")

    if not args.skip_dataset:
        dataset_command = [
            python_exe,
            str(_REPO_ROOT / "tools" / "build_swarm_dataset.py"),
            "--config",
            str(args.dataset_config),
            "--output_dir",
            str(dataset_dir),
        ]
        if args.num_episodes is not None:
            dataset_command.extend(["--num_episodes", str(args.num_episodes)])
        if args.dataset_seed is not None:
            dataset_command.extend(["--seed", str(args.dataset_seed)])
        _run_step(dataset_command)

    checkpoint_path = model_dir / "swarm_policy.pt"

    if not args.skip_train:
        train_command = [
            python_exe,
            str(_REPO_ROOT / "tools" / "train_swarm_model.py"),
            "--config",
            str(args.model_config),
            "--dataset_dir",
            str(dataset_dir),
            "--output_dir",
            str(model_dir),
        ]
        if args.epochs is not None:
            train_command.extend(["--epochs", str(args.epochs)])
        if args.batch_size is not None:
            train_command.extend(["--batch_size", str(args.batch_size)])
        if args.learning_rate is not None:
            train_command.extend(["--learning_rate", str(args.learning_rate)])
        if args.weight_decay is not None:
            train_command.extend(["--weight_decay", str(args.weight_decay)])
        if args.early_stopping_patience is not None:
            train_command.extend(
                ["--early_stopping_patience", str(args.early_stopping_patience)]
            )
        if args.regret_loss_weight is not None:
            train_command.extend(["--regret_loss_weight", str(args.regret_loss_weight)])
        if args.regret_sample_weight is not None:
            train_command.extend(
                ["--regret_sample_weight", str(args.regret_sample_weight)]
            )
        if args.value_loss_weight is not None:
            train_command.extend(["--value_loss_weight", str(args.value_loss_weight)])
        if args.threat_class_loss_weight is not None:
            train_command.extend(
                ["--threat_class_loss_weight", str(args.threat_class_loss_weight)]
            )
        if args.hidden_size is not None:
            train_command.extend(["--hidden_size", str(args.hidden_size)])
        if args.context_size is not None:
            train_command.extend(["--context_size", str(args.context_size)])
        if args.train_seed is not None:
            train_command.extend(["--seed", str(args.train_seed)])
        if args.gpu:
            train_command.append("--gpu")
        if args.verbose:
            train_command.append("--verbose")
        _run_step(train_command)

    if not args.skip_benchmark:
        benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        benchmark_command = [
            python_exe,
            str(_REPO_ROOT / "tools" / "evaluate_swarm_policy.py"),
            "--dataset_dir",
            str(dataset_dir),
            "--config",
            str(args.dataset_config),
            "--split",
            str(args.split),
            "--policies",
            str(args.policies),
            "--model_path",
            str(checkpoint_path),
            "--output",
            str(benchmark_output),
        ]
        _run_step(benchmark_command)

    logger.info("Pipeline finished.")
    logger.info("Dataset directory: %s", dataset_dir)
    logger.info("Model directory: %s", model_dir)
    if not args.skip_benchmark:
        logger.info("Benchmark report: %s", benchmark_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
