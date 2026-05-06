#!/usr/bin/env python3
"""Benchmark swarm planner latency on synthetic episode states."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from jetson.swarm_planner import evaluate_swarm_targets
from tools.evaluate_swarm_policy import _load_episodes, _load_settings, _rebuild_target

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="artifacts/swarm/datasets/default")
    parser.add_argument("--config", default="configs/swarm_dataset.yaml")
    parser.add_argument("--split", default="heldout")
    parser.add_argument(
        "--warmup",
        type=int,
        default=32,
        help="Number of initial states to evaluate before timing.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to re-evaluate each state during timing.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _collect_states(episodes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for episode in episodes:
        targets = tuple(_rebuild_target(target) for target in episode["initial_targets"])
        states.append(
            {
                "episode_id": int(episode["episode_id"]),
                "scenario_family": str(episode["scenario_family"]),
                "num_targets": len(targets),
                "targets": targets,
            }
        )
    return states


def _percentile(values_ms: Sequence[float], q: float) -> float:
    if not values_ms:
        return 0.0
    ordered = sorted(float(v) for v in values_ms)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(values_ms: Sequence[float]) -> Dict[str, Any]:
    if not values_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "decisions_per_s": 0.0,
        }
    mean_ms = float(statistics.fmean(values_ms))
    return {
        "count": int(len(values_ms)),
        "mean_ms": mean_ms,
        "median_ms": float(statistics.median(values_ms)),
        "p95_ms": float(_percentile(values_ms, 0.95)),
        "max_ms": float(max(values_ms)),
        "decisions_per_s": float(1000.0 / mean_ms) if mean_ms > 0.0 else 0.0,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset_dir = Path(args.dataset_dir)
    config_path = Path(args.config)
    settings = _load_settings(config_path)
    episodes = _load_episodes(dataset_dir, args.split)
    states = _collect_states(episodes)
    if not states:
        raise ValueError(f"No episode states found in split '{args.split}'")

    warmup_count = max(0, min(int(args.warmup), len(states)))
    repeat = max(1, int(args.repeat))

    for state in states[:warmup_count]:
        evaluate_swarm_targets(state["targets"], settings)

    per_decision_ms: List[float] = []
    by_target_count: Dict[int, List[float]] = {}
    by_family: Dict[str, List[float]] = {}

    logger.info(
        "Benchmarking swarm planner on %d states | split=%s | repeat=%d | warmup=%d",
        len(states),
        args.split,
        repeat,
        warmup_count,
    )

    for state in states:
        for _ in range(repeat):
            start_ns = time.perf_counter_ns()
            evaluate_swarm_targets(state["targets"], settings)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            per_decision_ms.append(elapsed_ms)
            by_target_count.setdefault(int(state["num_targets"]), []).append(elapsed_ms)
            by_family.setdefault(str(state["scenario_family"]), []).append(elapsed_ms)

    report = {
        "dataset_dir": str(dataset_dir),
        "config": str(config_path),
        "split": str(args.split),
        "repeat": repeat,
        "warmup": warmup_count,
        "overall": _summarize(per_decision_ms),
        "by_target_count": {
            str(target_count): _summarize(values)
            for target_count, values in sorted(by_target_count.items())
        },
        "by_family": {
            family: _summarize(values)
            for family, values in sorted(by_family.items())
        },
    }

    logger.info(
        "Overall planner latency | mean=%.4f ms | median=%.4f ms | p95=%.4f ms | max=%.4f ms | %.1f decisions/s",
        report["overall"]["mean_ms"],
        report["overall"]["median_ms"],
        report["overall"]["p95_ms"],
        report["overall"]["max_ms"],
        report["overall"]["decisions_per_s"],
    )
    for target_count, summary in report["by_target_count"].items():
        logger.info(
            "Targets=%s | mean=%.4f ms | p95=%.4f ms | %.1f decisions/s",
            target_count,
            summary["mean_ms"],
            summary["p95_ms"],
            summary["decisions_per_s"],
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved planner benchmark report to %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
