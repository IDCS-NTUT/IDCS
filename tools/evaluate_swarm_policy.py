#!/usr/bin/env python3
"""Evaluate swarm target-selection policies on synthetic episodes."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from common.control import SwarmTimingConfig
from common.threat_calc import compute_zone_feature_vector, parse_zone_config
from jetson.swarm_planner import (
    PlannerTarget,
    SwarmPlannerSettings,
    advance_planner_state,
    evaluate_swarm_targets,
)
try:
    from jetson.swarm_policy_model import load_swarm_policy_checkpoint
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_swarm_policy_checkpoint = None  # type: ignore[assignment]

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
        effect_distance_scale_s_per_m=float(
            timing_cfg.get("effect_distance_scale_s_per_m", 0.01)
        ),
        confirm_time_s=float(timing_cfg.get("confirm_time_s", 0.10)),
        confirm_distance_scale_s_per_m=float(
            timing_cfg.get("confirm_distance_scale_s_per_m", 0.004)
        ),
        settle_margin_s=float(timing_cfg.get("settle_margin_s", 0.05)),
    )
    return SwarmPlannerSettings(
        yaw_rate_limit_rad_s=float(planner_cfg["yaw_rate_limit_rad_s"]),
        pitch_rate_limit_rad_s=float(planner_cfg["pitch_rate_limit_rad_s"]),
        yaw_accel_limit_rad_s2=float(planner_cfg["yaw_accel_limit_rad_s2"]),
        pitch_accel_limit_rad_s2=float(planner_cfg["pitch_accel_limit_rad_s2"]),
        max_engage_distance_m=(
            None
            if planner_cfg.get("max_engage_distance_m") is None
            else float(planner_cfg["max_engage_distance_m"])
        ),
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


def _load_dataset_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_zone_radii(config: Mapping[str, Any]) -> Dict[str, float]:
    threat_eval_cfg = config.get("threat_eval", {}) or {}
    if not isinstance(threat_eval_cfg, Mapping):
        return {}
    zones_cfg = threat_eval_cfg.get("zones", {}) or {}
    if not isinstance(zones_cfg, Mapping) or not zones_cfg:
        return {}
    return parse_zone_config(dict(zones_cfg))


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
    *,
    learned_selector: Any = None,
    normalization: Optional[Mapping[str, float]] = None,
    max_targets_tensor: Optional[int] = None,
    zone_radii: Optional[Mapping[str, float]] = None,
) -> Optional[int]:
    if policy == "swarm_planner":
        decision = evaluate_swarm_targets(state, settings)
        return None if decision.chosen_target_id is None else int(decision.chosen_target_id)
    if policy == "learned_model":
        if learned_selector is None or normalization is None or max_targets_tensor is None:
            raise ValueError("learned_model policy requires model_path and dataset normalization")
        action_index = _predict_learned_model_action(
            state,
            settings,
            learned_selector=learned_selector,
            normalization=normalization,
            max_targets_tensor=max_targets_tensor,
            zone_radii=zone_radii or {},
            use_value_rerank=False,
            rerank_topk=1,
        )
        return int(state[action_index].target_id)
    if policy == "learned_model_rerank":
        if learned_selector is None or normalization is None or max_targets_tensor is None:
            raise ValueError("learned_model_rerank policy requires model_path and dataset normalization")
        action_index = _predict_learned_model_action(
            state,
            settings,
            learned_selector=learned_selector,
            normalization=normalization,
            max_targets_tensor=max_targets_tensor,
            zone_radii=zone_radii or {},
            use_value_rerank=True,
            rerank_topk=2,
        )
        return int(state[action_index].target_id)
    if policy == "learned_value_only":
        if learned_selector is None or normalization is None or max_targets_tensor is None:
            raise ValueError("learned_value_only policy requires model_path and dataset normalization")
        action_index = _predict_learned_model_action(
            state,
            settings,
            learned_selector=learned_selector,
            normalization=normalization,
            max_targets_tensor=max_targets_tensor,
            zone_radii=zone_radii or {},
            use_value_rerank=True,
            rerank_topk=max_targets_tensor,
        )
        return int(state[action_index].target_id)
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
    *,
    learned_selector: Any = None,
    normalization: Optional[Mapping[str, float]] = None,
    max_targets_tensor: Optional[int] = None,
    zone_radii: Optional[Mapping[str, float]] = None,
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
            chosen_target_id = _choose_policy_target(
                policy,
                state,
                settings,
                learned_selector=learned_selector,
                normalization=normalization,
                max_targets_tensor=max_targets_tensor,
                zone_radii=zone_radii,
            )
            if chosen_target_id is None:
                _, immediate_damage, state = _advance_wait_state(state, settings)
            else:
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


def _advance_wait_state(
    state: Sequence[PlannerTarget],
    settings: SwarmPlannerSettings,
) -> Tuple[float, float, Tuple[PlannerTarget, ...]]:
    """Advance the episode without engagement until the next relevant event.

    This is used when the planner chooses no target because nothing is
    engageable right now under the hardware range limit or timing constraints.
    """
    event_times: List[float] = []
    max_engage_distance_m = settings.max_engage_distance_m

    for target in state:
        breakthrough_time_s = target.breakthrough_time_s()
        if math.isfinite(breakthrough_time_s) and breakthrough_time_s > 0.0:
            event_times.append(float(breakthrough_time_s))
        if (
            max_engage_distance_m is not None
            and math.isfinite(target.distance_m)
            and target.radial_closing_speed_m_s > 0.0
            and target.distance_m > max_engage_distance_m
        ):
            time_to_range_s = (
                float(target.distance_m) - float(max_engage_distance_m)
            ) / float(target.radial_closing_speed_m_s)
            if math.isfinite(time_to_range_s) and time_to_range_s > 0.0:
                event_times.append(time_to_range_s)

    if not event_times:
        return 0.0, 0.0, tuple()

    elapsed_s = max(min(event_times), 1e-6)
    damage = 0.0
    survivors: List[PlannerTarget] = []
    for target in state:
        breakthrough_time_s = target.breakthrough_time_s()
        if breakthrough_time_s <= elapsed_s + 1e-9:
            damage += float(target.damage_weight)
            continue
        survivors.append(
            PlannerTarget(
                target_id=target.target_id,
                box_index=target.box_index,
                cls=target.cls,
                confidence=target.confidence,
                damage_weight=target.damage_weight,
                distance_m=max(
                    0.0,
                    float(target.distance_m)
                    - float(target.radial_closing_speed_m_s) * elapsed_s,
                ),
                radial_closing_speed_m_s=target.radial_closing_speed_m_s,
                yaw_error_rad=target.yaw_error_rad,
                pitch_error_rad=target.pitch_error_rad,
                bbox_area_norm=target.bbox_area_norm,
                track_observations=target.track_observations,
                range_source=target.range_source,
                threat_level=target.threat_level,
                tracker_mode=target.tracker_mode,
                predictive_only=target.predictive_only,
            )
        )
    return elapsed_s, damage, tuple(survivors)


def _predict_learned_model_action(
    state: Sequence[PlannerTarget],
    settings: SwarmPlannerSettings,
    *,
    learned_selector: Any,
    normalization: Mapping[str, float],
    max_targets_tensor: int,
    zone_radii: Mapping[str, float],
    use_value_rerank: bool,
    rerank_topk: int,
) -> int:
    target_features, global_features, target_mask = _encode_model_inputs(
        state,
        settings,
        normalization=normalization,
        max_targets_tensor=max_targets_tensor,
        target_feature_size=int(learned_selector.model.target_feature_size),
        global_feature_size=int(learned_selector.model.global_feature_size),
        zone_radii=zone_radii,
    )
    actions, _, _ = learned_selector.predict_action_numpy(
        target_features,
        global_features,
        target_mask,
        use_value_rerank=use_value_rerank,
        rerank_topk=rerank_topk,
    )
    return int(actions[0])


def _encode_model_inputs(
    state: Sequence[PlannerTarget],
    settings: SwarmPlannerSettings,
    *,
    normalization: Mapping[str, float],
    max_targets_tensor: int,
    target_feature_size: int,
    global_feature_size: int,
    zone_radii: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(state) > max_targets_tensor:
        raise ValueError(
            f"State contains {len(state)} targets but model supports max_targets_tensor={max_targets_tensor}"
        )

    decision = evaluate_swarm_targets(state, settings)
    result_by_target = {result.target_id: result for result in decision.candidate_results}

    target_features = np.zeros((1, max_targets_tensor, target_feature_size), dtype=np.float32)
    target_mask = np.zeros((1, max_targets_tensor), dtype=bool)
    breakthroughs: List[float] = []
    engage_times: List[float] = []
    total_damage_weight = 0.0
    warning_count = 0.0
    restricted_count = 0.0
    critical_count = 0.0

    for idx, target in enumerate(state):
        result = result_by_target[target.target_id]
        target_mask[0, idx] = True
        breakthroughs.append(
            _clip_norm(result.breakthrough_time_s, float(normalization["max_breakthrough_time_s"]))
        )
        engage_times.append(
            _clip_norm(result.time_to_engage_s, float(normalization["max_time_to_engage_s"]))
        )
        total_damage_weight += float(target.damage_weight)
        zone_features = compute_zone_feature_vector(float(target.distance_m), dict(zone_radii))
        warning_count += zone_features[0]
        restricted_count += zone_features[1]
        critical_count += zone_features[2]
        feature_vector = [
            _clip_norm(target.distance_m, float(normalization["max_distance_m"])),
            _clip_norm(
                target.radial_closing_speed_m_s,
                float(normalization["max_closing_speed_m_s"]),
            ),
            _clip_norm(
                result.breakthrough_time_s,
                float(normalization["max_breakthrough_time_s"]),
            ),
            _clip_norm(
                result.time_to_engage_s,
                float(normalization["max_time_to_engage_s"]),
            ),
            _clip_norm(
                target.damage_weight,
                float(normalization["max_damage_weight"]),
            ),
            float(target.confidence),
            _signed_norm(target.yaw_error_rad, float(normalization["max_angle_rad"])),
            _signed_norm(target.pitch_error_rad, float(normalization["max_angle_rad"])),
            float(target.bbox_area_norm),
            _clip_norm(
                target.track_observations,
                float(normalization["max_track_observations"]),
            ),
            zone_features[0],
            zone_features[1],
            zone_features[2],
            zone_features[3],
        ]
        if target_feature_size > len(feature_vector):
            raise ValueError(
                f"Model expects {target_feature_size} target features but evaluator provides only {len(feature_vector)}"
            )
        target_features[0, idx, :] = np.array(
            feature_vector[:target_feature_size],
            dtype=np.float32,
        )

    global_vector = [
        len(state) / max(1, max_targets_tensor),
        min(breakthroughs) if breakthroughs else 0.0,
        float(np.mean(engage_times)) if engage_times else 0.0,
        _clip_norm(
            total_damage_weight,
            float(normalization["max_damage_weight"]) * max_targets_tensor,
        ),
        warning_count / max(1, len(state)),
        restricted_count / max(1, len(state)),
        critical_count / max(1, len(state)),
    ]
    if global_feature_size > len(global_vector):
        raise ValueError(
            f"Model expects {global_feature_size} global features but evaluator provides only {len(global_vector)}"
        )
    global_features = np.array(
        [[value for value in global_vector[:global_feature_size]]],
        dtype=np.float32,
    )
    return target_features, global_features, target_mask


def _clip_norm(value: float, max_value: float) -> float:
    if value is None or not math.isfinite(float(value)):
        return 1.0
    return float(np.clip(float(value) / max(max_value, 1e-6), 0.0, 1.0))


def _signed_norm(value: float, max_abs: float) -> float:
    return float(np.clip(float(value) / max(max_abs, 1e-6), -1.0, 1.0))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="artifacts/swarm/datasets/default")
    parser.add_argument("--config", default="configs/swarm_dataset.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--policies",
        default="swarm_planner,learned_model,learned_model_rerank,learned_value_only,max_conf,closest_breakthrough,highest_damage,largest_area",
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset_dir = Path(args.dataset_dir)
    settings = _load_settings(Path(args.config))
    dataset_cfg = _load_dataset_config(Path(args.config))
    zone_radii = _load_zone_radii(dataset_cfg)
    episodes = _load_episodes(dataset_dir, args.split)
    policies = [policy.strip() for policy in str(args.policies).split(",") if policy.strip()]

    learned_selector = None
    model_metadata = None
    if any(policy.startswith("learned_") for policy in policies):
        if load_swarm_policy_checkpoint is None:
            raise RuntimeError(
                "learned_model policy requires torch and jetson.swarm_policy_model dependencies"
            )
        if args.model_path is None:
            raise ValueError("--model_path is required when evaluating learned_model")
        learned_selector, model_metadata = load_swarm_policy_checkpoint(Path(args.model_path))

    normalization = dataset_cfg["normalization"]
    if model_metadata is not None and "normalization" in model_metadata:
        normalization = model_metadata["normalization"]
    max_targets_tensor = int(dataset_cfg["episodes"]["max_targets_tensor"])
    if model_metadata is not None:
        if "max_targets_tensor" in model_metadata:
            max_targets_tensor = int(model_metadata["max_targets_tensor"])
        elif "max_targets" in model_metadata:
            max_targets_tensor = int(model_metadata["max_targets"])

    report = {
        "split": args.split,
        "dataset_dir": str(dataset_dir),
        "policies": {
            policy: _evaluate_policy_on_episodes(
                episodes,
                policy,
                settings,
                learned_selector=learned_selector,
                normalization=normalization,
                max_targets_tensor=max_targets_tensor,
                zone_radii=zone_radii,
            )
            for policy in policies
        },
    }
    if model_metadata is not None:
        report["model"] = model_metadata

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
