#!/usr/bin/env python3
"""Evaluate swarm target-selection policies on synthetic episodes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from common.control import SwarmTimingConfig
from jetson.swarm_planner import (
    PlannerTarget,
    SwarmPlannerSettings,
    advance_planner_state,
    evaluate_swarm_targets,
)

logger = logging.getLogger(__name__)


def _load_settings(config_path: Path) -> SwarmPlannerSettings:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    planner_cfg = config["planner"]
    timing_cfg = planner_cfg.get("timing", {})
    timing = SwarmTimingConfig(
        base_track_lock_s=float(timing_cfg.get("base_track_lock_s", 0.15)),
        search_track_lock_s=float(timing_cfg.get("search_track_lock_s", 0.25)),
        recover_track_lock_s=float(timing_cfg.get("recover_track_lock_s", 0.40)),
        low_conf_threshold=float(timing_cfg.get("low_conf_threshold", 0.60)),
        low_conf_penalty_s=float(timing_cfg.get("low_conf_penalty_s", 0.20)),
        min_track_observations=int(timing_cfg.get("min_track_observations", 3)),
        low_continuity_penalty_s=float(timing_cfg.get("low_continuity_penalty_s", 0.08)),
        missing_range_penalty_s=float(timing_cfg.get("missing_range_penalty_s", 0.10)),
        predictive_penalty_s=float(timing_cfg.get("predictive_penalty_s", 0.20)),
        effect_time_s=float(timing_cfg.get("effect_time_s", 0.25)),
        confirm_time_s=float(timing_cfg.get("confirm_time_s", 0.10)),
        settle_margin_s=float(timing_cfg.get("settle_margin_s", 0.05)),
    )
    return SwarmPlannerSettings(
        yaw_rate_limit_rad_s=float(planner_cfg["yaw_rate_limit_rad_s"]),
        pitch_rate_limit_rad_s=float(planner_cfg["pitch_rate_limit_rad_s"]),
        yaw_accel_limit_rad_s2=float(planner_cfg["yaw_accel_limit_rad_s2"]),
        pitch_accel_limit_rad_s2=float(planner_cfg["pitch_accel_limit_rad_s2"]),
        exact_search_limit=int(planner_cfg.get("exact_search_limit", 6)),
        beam_width=int(planner_cfg.get("beam_width", 8)),
        switch_absolute_damage_gain=float(
            planner_cfg.get("switch_absolute_damage_gain", 0.25)
        ),
        switch_relative_improvement=float(
            planner_cfg.get("switch_relative_improvement", 0.10)
        ),
        timing=timing,
    )


def _load_episodes(dataset_dir: Path, split: str) -> Sequence[Mapping[str, Any]]:
    payload = json.loads((dataset_dir / f"episodes_{split}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected episode list in episodes_{split}.json")
    return payload


def _rebuild_target(target: Mapping[str, Any]) -> PlannerTarget:
    return PlannerTarget(
        target_id=int(target["target_id"]),
        box_index=int(target["box_index"]),
        cls=str(target["cls"]),
        confidence=float(target["confidence"]),
        damage_weight=float(target["damage_weight"]),
        distance_m=float(target["distance_m"]),
        radial_closing_speed_m_s=float(target["radial_closing_speed_m_s"]),
        yaw_error_rad=float(target["yaw_error_rad"]),
        pitch_error_rad=float(target["pitch_error_rad"]),
        bbox_area_norm=float(target["bbox_area_norm"]),
        track_observations=int(target["track_observations"]),
        range_source=target.get("range_source"),
        threat_level=target.get("threat_level"),
        tracker_mode=target.get("tracker_mode"),
        predictive_only=bool(target.get("predictive_only", False)),
    )


def _choose_policy_target(
    policy: str,
    state: Sequence[PlannerTarget],
    settings: SwarmPlannerSettings,
) -> int:
    if policy == "swarm_planner":
        decision = evaluate_swarm_targets(state, settings)
        if decision.chosen_target_id is None:
            raise RuntimeError("swarm_planner failed to choose a target")
        return int(decision.chosen_target_id)
    if policy == "max_conf":
        return max(state, key=lambda target: (target.confidence, -target.breakthrough_time_s())).target_id
    if policy == "closest_breakthrough":
        return min(state, key=lambda target: (target.breakthrough_time_s(), -target.damage_weight)).target_id
    if policy == "highest_damage":
        return max(state, key=lambda target: (target.damage_weight, -target.breakthrough_time_s())).target_id
    if policy == "largest_area":
        return max(state, key=lambda target: (target.bbox_area_norm, target.confidence)).target_id
    raise ValueError(f"Unsupported policy: {policy}")


def _evaluate_policy_on_episodes(
    episodes: Sequence[Mapping[str, Any]],
    policy: str,
    settings: SwarmPlannerSettings,
) -> Dict[str, Any]:
    total_damage = 0.0
    total_possible_damage = 0.0
    total_oracle_damage = 0.0
    zero_breakthroughs = 0
    per_family: Dict[str, Dict[str, float]] = {}

    for episode in episodes:
        state = tuple(_rebuild_target(target) for target in episode["initial_targets"])
        episode_damage = 0.0
        total_possible_damage += float(episode["total_possible_damage"])
        total_oracle_damage += float(episode.get("oracle_episode_damage", 0.0))

        while state:
            chosen_target_id = _choose_policy_target(policy, state, settings)
            _, immediate_damage, state = advance_planner_state(
                state,
                settings,
                target_id=chosen_target_id,
            )
            episode_damage += immediate_damage

        if episode_damage <= 1e-9:
            zero_breakthroughs += 1

        family = str(episode["scenario_family"])
        family_metrics = per_family.setdefault(
            family,
            {"episodes": 0.0, "damage": 0.0, "possible_damage": 0.0, "oracle_damage": 0.0},
        )
        family_metrics["episodes"] += 1.0
        family_metrics["damage"] += episode_damage
        family_metrics["possible_damage"] += float(episode["total_possible_damage"])
        family_metrics["oracle_damage"] += float(episode.get("oracle_episode_damage", 0.0))

        total_damage += episode_damage

    report = {
        "policy": policy,
        "episodes": len(episodes),
        "total_damage": total_damage,
        "total_possible_damage": total_possible_damage,
        "normalized_total_damage": (
            total_damage / total_possible_damage if total_possible_damage > 0 else 0.0
        ),
        "mean_episode_damage": total_damage / max(1, len(episodes)),
        "zero_breakthrough_rate": zero_breakthroughs / max(1, len(episodes)),
        "mean_oracle_regret": (total_damage - total_oracle_damage) / max(1, len(episodes)),
        "per_family": {},
    }
    for family, metrics in sorted(per_family.items()):
        report["per_family"][family] = {
            "episodes": int(metrics["episodes"]),
            "normalized_total_damage": (
                metrics["damage"] / metrics["possible_damage"]
                if metrics["possible_damage"] > 0
                else 0.0
            ),
            "mean_oracle_regret": (
                (metrics["damage"] - metrics["oracle_damage"]) / max(metrics["episodes"], 1.0)
            ),
        }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="artifacts/swarm/datasets/default")
    parser.add_argument("--config", default="configs/swarm_dataset.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--policies",
        default="swarm_planner,max_conf,closest_breakthrough,highest_damage,largest_area",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset_dir = Path(args.dataset_dir)
    settings = _load_settings(Path(args.config))
    episodes = _load_episodes(dataset_dir, args.split)
    policies = [policy.strip() for policy in str(args.policies).split(",") if policy.strip()]

    report = {
        "split": args.split,
        "dataset_dir": str(dataset_dir),
        "policies": {
            policy: _evaluate_policy_on_episodes(episodes, policy, settings)
            for policy in policies
        },
    }

    output_path = (
        Path(args.output)
        if args.output is not None
        else dataset_dir / f"policy_report_{args.split}.json"
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote policy report to %s", output_path)
    logger.info(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
