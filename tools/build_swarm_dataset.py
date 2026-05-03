#!/usr/bin/env python3
"""Build an episode-based dataset for swarm engagement planning."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "distance_norm",
    "closing_speed_norm",
    "breakthrough_time_norm",
    "time_to_engage_norm",
    "damage_weight_norm",
    "confidence",
    "yaw_error_norm",
    "pitch_error_norm",
    "bbox_area_norm",
    "track_observations_norm",
    "in_warning_zone",
    "in_restricted_zone",
    "in_critical_zone",
    "zone_progress_norm",
    "track_age_norm",
    "confidence_mean_recent",
    "confidence_min_recent",
    "closing_speed_mean_recent_norm",
    "closing_speed_std_recent_norm",
]

GLOBAL_FEATURE_NAMES = [
    "num_targets_norm",
    "min_breakthrough_norm",
    "mean_time_to_engage_norm",
    "total_damage_weight_norm",
    "warning_fraction",
    "restricted_fraction",
    "critical_fraction",
]

THREAT_CLASS_NAMES = ["benign", "suspicious", "threatening"]
THREAT_CLASS_TO_INDEX = {
    name: index for index, name in enumerate(THREAT_CLASS_NAMES)
}


@dataclass(frozen=True)
class SyntheticEpisode:
    episode_id: int
    scenario_family: str
    initial_targets: Tuple[PlannerTarget, ...]


class SwarmDatasetBuilder:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.output_dir = Path(config["output"]["directory"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        split_cfg = config["split"]
        self.split_seed = int(split_cfg.get("seed", 42))
        self.split_ratios = {
            "train": float(split_cfg.get("train", 0.6)),
            "val": float(split_cfg.get("val", 0.2)),
            "test": float(split_cfg.get("test", 0.1)),
            "heldout": float(split_cfg.get("heldout", 0.1)),
        }

        episodes_cfg = config["episodes"]
        self.num_episodes = int(episodes_cfg["num_episodes"])
        self.min_targets = int(episodes_cfg["min_targets"])
        self.max_targets = int(episodes_cfg["max_targets"])
        self.max_targets_tensor = int(episodes_cfg.get("max_targets_tensor", self.max_targets))
        if self.max_targets_tensor < self.max_targets:
            raise ValueError("episodes.max_targets_tensor must be >= episodes.max_targets")

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
        self.settings = SwarmPlannerSettings(
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

        self.norm = config["normalization"]
        threat_eval_cfg = config.get("threat_eval", {}) or {}
        if not isinstance(threat_eval_cfg, Mapping):
            raise ValueError("threat_eval must be a mapping when provided")
        zones_cfg = threat_eval_cfg.get("zones", {}) or {}
        if not isinstance(zones_cfg, Mapping):
            raise ValueError("threat_eval.zones must be a mapping when provided")
        self.zone_radii = parse_zone_config(dict(zones_cfg)) if zones_cfg else {}
        self.damage_by_class = {
            str(name): float(value)
            for name, value in (config.get("damage_by_class", {}) or {}).items()
        }
        self.families = (
            "direct_rush",
            "staggered_arrivals",
            "decoy_distractors",
            "high_damage_slow",
            "crossing_targets",
            "overload",
        )
        raw_weights = episodes_cfg.get("scenario_weights", {}) or {}
        if not isinstance(raw_weights, Mapping):
            raise ValueError("episodes.scenario_weights must be a mapping when provided")
        self.scenario_weights = {
            family: float(raw_weights.get(family, 1.0))
            for family in self.families
        }
        for family, weight in self.scenario_weights.items():
            if weight <= 0.0:
                raise ValueError(f"episodes.scenario_weights.{family} must be > 0")

    def build(self, *, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        family_assignments = self._allocate_family_assignments(rng)
        episodes = [
            self._generate_episode(
                episode_id=i,
                rng=rng,
                family=family_assignments[i],
            )
            for i in range(self.num_episodes)
        ]
        split_episodes = self._split_episodes(episodes)
        family_counts = {
            family: int(sum(1 for episode in episodes if episode.scenario_family == family))
            for family in self.families
        }

        stats: Dict[str, Any] = {
            "num_episodes": len(episodes),
            "feature_names": FEATURE_NAMES,
            "global_feature_names": GLOBAL_FEATURE_NAMES,
            "threat_class_names": THREAT_CLASS_NAMES,
            "normalization": dict(self.norm),
            "threat_eval_zone_radii": dict(self.zone_radii),
            "max_targets_tensor": self.max_targets_tensor,
            "scenario_weights": dict(self.scenario_weights),
            "family_counts": family_counts,
            "splits": {},
        }

        for split_name, split_items in split_episodes.items():
            step_records, episode_records = self._build_split_records(split_items)
            arrays = self._encode_split(step_records)
            np.savez(self.output_dir / f"{split_name}.npz", **arrays)
            (self.output_dir / f"records_{split_name}.json").write_text(
                json.dumps(step_records, indent=2),
                encoding="utf-8",
            )
            (self.output_dir / f"episodes_{split_name}.json").write_text(
                json.dumps(episode_records, indent=2),
                encoding="utf-8",
            )
            stats["splits"][split_name] = {
                "episodes": len(split_items),
                "decision_steps": len(step_records),
                "scenario_families": sorted({item.scenario_family for item in split_items}),
            }

        (self.output_dir / "statistics.json").write_text(
            json.dumps(stats, indent=2),
            encoding="utf-8",
        )
        return stats

    def _generate_episode(
        self,
        *,
        episode_id: int,
        rng: np.random.Generator,
        family: str,
    ) -> SyntheticEpisode:
        if family == "direct_rush":
            targets = self._scenario_direct_rush(episode_id, rng)
        elif family == "staggered_arrivals":
            targets = self._scenario_staggered_arrivals(episode_id, rng)
        elif family == "decoy_distractors":
            targets = self._scenario_decoy_distractors(episode_id, rng)
        elif family == "high_damage_slow":
            targets = self._scenario_high_damage_slow(episode_id, rng)
        elif family == "crossing_targets":
            targets = self._scenario_crossing_targets(episode_id, rng)
        else:
            targets = self._scenario_overload(episode_id, rng)
        return SyntheticEpisode(
            episode_id=episode_id,
            scenario_family=family,
            initial_targets=tuple(targets),
        )

    def _allocate_family_assignments(
        self,
        rng: np.random.Generator,
    ) -> List[str]:
        if self.num_episodes <= 0:
            return []

        base_assignments: List[str] = []
        if self.num_episodes >= len(self.families):
            base_assignments.extend(self.families)

        remaining = self.num_episodes - len(base_assignments)
        if remaining > 0:
            weights = np.array(
                [self.scenario_weights[family] for family in self.families],
                dtype=np.float64,
            )
            probs = weights / weights.sum()
            sampled = rng.choice(self.families, size=remaining, replace=True, p=probs)
            base_assignments.extend(str(family) for family in sampled.tolist())

        rng.shuffle(base_assignments)
        return base_assignments

    def _scenario_direct_rush(
        self, episode_id: int, rng: np.random.Generator
    ) -> List[PlannerTarget]:
        count = int(rng.integers(max(2, self.min_targets), min(4, self.max_targets) + 1))
        return [
            self._make_target(
                episode_id,
                rng=rng,
                index=i,
                cls="drone",
                distance_m=float(rng.uniform(12.0, 32.0)),
                closing_speed=float(rng.uniform(4.0, 9.0)),
                yaw_error_rad=float(rng.uniform(-0.35, 0.35)),
                pitch_error_rad=float(rng.uniform(-0.12, 0.12)),
                confidence=float(rng.uniform(0.70, 0.98)),
                damage_weight=float(rng.uniform(1.8, 3.5)),
                bbox_area_norm=float(rng.uniform(0.010, 0.045)),
                track_observations=int(rng.integers(2, 7)),
                range_source="average",
                threat_level="threatening",
                tracker_mode="track",
            )
            for i in range(count)
        ]

    def _scenario_staggered_arrivals(
        self, episode_id: int, rng: np.random.Generator
    ) -> List[PlannerTarget]:
        count = int(rng.integers(max(3, self.min_targets), min(6, self.max_targets) + 1))
        targets: List[PlannerTarget] = []
        for i in range(count):
            targets.append(
                self._make_target(
                    episode_id,
                    rng=rng,
                    index=i,
                    cls="drone",
                    distance_m=float(rng.uniform(18.0 + i * 4.0, 40.0 + i * 6.0)),
                    closing_speed=float(rng.uniform(2.0, 6.0)),
                    yaw_error_rad=float(rng.uniform(-0.45, 0.45)),
                    pitch_error_rad=float(rng.uniform(-0.16, 0.16)),
                    confidence=float(rng.uniform(0.60, 0.94)),
                    damage_weight=float(rng.uniform(1.5, 3.0)),
                    bbox_area_norm=float(rng.uniform(0.008, 0.030)),
                    track_observations=int(rng.integers(1, 6)),
                    range_source="height" if i % 2 == 0 else "average",
                    threat_level="suspicious" if i % 2 == 0 else "threatening",
                    tracker_mode="search" if i == 0 else "track",
                )
            )
        return targets

    def _scenario_decoy_distractors(
        self, episode_id: int, rng: np.random.Generator
    ) -> List[PlannerTarget]:
        count = int(rng.integers(max(4, self.min_targets), min(6, self.max_targets) + 1))
        targets: List[PlannerTarget] = []
        for i in range(count - 1):
            targets.append(
                self._make_target(
                    episode_id,
                    rng=rng,
                    index=i,
                    cls="decoy",
                    distance_m=float(rng.uniform(10.0, 20.0)),
                    closing_speed=float(rng.uniform(3.0, 6.0)),
                    yaw_error_rad=float(rng.uniform(-0.55, 0.55)),
                    pitch_error_rad=float(rng.uniform(-0.18, 0.18)),
                    confidence=float(rng.uniform(0.55, 0.88)),
                    damage_weight=float(rng.uniform(0.6, 1.4)),
                    bbox_area_norm=float(rng.uniform(0.010, 0.035)),
                    track_observations=int(rng.integers(1, 4)),
                    range_source="average",
                    threat_level="benign" if i % 2 == 0 else "suspicious",
                    tracker_mode="recover" if i == 0 else "track",
                )
            )
        targets.append(
            self._make_target(
                episode_id,
                rng=rng,
                index=count - 1,
                cls="munition",
                distance_m=float(rng.uniform(20.0, 30.0)),
                closing_speed=float(rng.uniform(2.0, 4.0)),
                yaw_error_rad=float(rng.uniform(-0.15, 0.15)),
                pitch_error_rad=float(rng.uniform(-0.08, 0.08)),
                confidence=float(rng.uniform(0.80, 0.99)),
                damage_weight=float(rng.uniform(4.0, 7.0)),
                bbox_area_norm=float(rng.uniform(0.006, 0.018)),
                track_observations=int(rng.integers(4, 9)),
                range_source="average",
                threat_level="threatening",
                tracker_mode="track",
            )
        )
        return targets

    def _scenario_high_damage_slow(
        self, episode_id: int, rng: np.random.Generator
    ) -> List[PlannerTarget]:
        count = int(rng.integers(max(2, self.min_targets), min(4, self.max_targets) + 1))
        targets: List[PlannerTarget] = [
            self._make_target(
                episode_id,
                rng=rng,
                index=0,
                cls="munition",
                distance_m=float(rng.uniform(25.0, 42.0)),
                closing_speed=float(rng.uniform(1.4, 2.8)),
                yaw_error_rad=float(rng.uniform(-0.22, 0.22)),
                pitch_error_rad=float(rng.uniform(-0.08, 0.08)),
                confidence=float(rng.uniform(0.82, 0.99)),
                damage_weight=float(rng.uniform(5.0, 8.0)),
                bbox_area_norm=float(rng.uniform(0.006, 0.015)),
                track_observations=int(rng.integers(5, 10)),
                range_source="average",
                threat_level="threatening",
                tracker_mode="track",
            )
        ]
        for i in range(1, count):
            targets.append(
                self._make_target(
                    episode_id,
                    rng=rng,
                    index=i,
                    cls="drone",
                    distance_m=float(rng.uniform(12.0, 26.0)),
                    closing_speed=float(rng.uniform(3.0, 6.0)),
                    yaw_error_rad=float(rng.uniform(-0.45, 0.45)),
                    pitch_error_rad=float(rng.uniform(-0.14, 0.14)),
                    confidence=float(rng.uniform(0.60, 0.90)),
                    damage_weight=float(rng.uniform(1.2, 2.5)),
                    bbox_area_norm=float(rng.uniform(0.010, 0.035)),
                    track_observations=int(rng.integers(1, 5)),
                    range_source="height",
                    threat_level="suspicious",
                    tracker_mode="search",
                )
            )
        return targets

    def _scenario_crossing_targets(
        self, episode_id: int, rng: np.random.Generator
    ) -> List[PlannerTarget]:
        count = int(rng.integers(max(3, self.min_targets), min(6, self.max_targets) + 1))
        angles = np.linspace(-0.6, 0.6, count)
        rng.shuffle(angles)
        targets: List[PlannerTarget] = []
        for i in range(count):
            targets.append(
                self._make_target(
                    episode_id,
                    rng=rng,
                    index=i,
                    cls="loitering_drone" if i % 2 == 0 else "drone",
                    distance_m=float(rng.uniform(16.0, 36.0)),
                    closing_speed=float(rng.uniform(2.5, 5.5)),
                    yaw_error_rad=float(angles[i]),
                    pitch_error_rad=float(rng.uniform(-0.20, 0.20)),
                    confidence=float(rng.uniform(0.55, 0.93)),
                    damage_weight=float(rng.uniform(2.0, 4.5)),
                    bbox_area_norm=float(rng.uniform(0.008, 0.028)),
                    track_observations=int(rng.integers(1, 6)),
                    range_source="width" if i % 2 else "average",
                    threat_level="threatening" if i % 2 else ("benign" if i % 3 == 0 else "suspicious"),
                    tracker_mode="slew" if i == 0 else "track",
                )
            )
        return targets

    def _scenario_overload(
        self, episode_id: int, rng: np.random.Generator
    ) -> List[PlannerTarget]:
        count = int(rng.integers(max(6, self.min_targets), self.max_targets + 1))
        return [
            self._make_target(
                episode_id,
                rng=rng,
                index=i,
                cls="drone" if i % 3 else "munition",
                distance_m=float(rng.uniform(10.0, 28.0)),
                closing_speed=float(rng.uniform(4.0, 10.0)),
                yaw_error_rad=float(rng.uniform(-0.70, 0.70)),
                pitch_error_rad=float(rng.uniform(-0.22, 0.22)),
                confidence=float(rng.uniform(0.50, 0.90)),
                damage_weight=float(rng.uniform(1.0, 6.0)),
                bbox_area_norm=float(rng.uniform(0.008, 0.030)),
                track_observations=int(rng.integers(1, 4)),
                range_source="average" if i % 2 else None,
                threat_level="threatening",
                tracker_mode="recover" if i % 4 == 0 else "track",
            )
            for i in range(count)
        ]

    def _make_target(
        self,
        episode_id: int,
        *,
        rng: np.random.Generator,
        index: int,
        cls: str,
        distance_m: float,
        closing_speed: float,
        yaw_error_rad: float,
        pitch_error_rad: float,
        confidence: float,
        damage_weight: float,
        bbox_area_norm: float,
        track_observations: int,
        range_source: str | None,
        threat_level: str,
        tracker_mode: str,
    ) -> PlannerTarget:
        track_age_s = float(track_observations) * float(rng.uniform(0.12, 0.28))
        confidence_mean_recent = float(
            np.clip(confidence - rng.uniform(0.0, 0.06), 0.0, 1.0)
        )
        confidence_min_recent = float(
            np.clip(
                min(confidence, confidence_mean_recent) - rng.uniform(0.0, 0.10),
                0.0,
                1.0,
            )
        )
        speed_variability_scale = 0.05
        if tracker_mode in {"search", "slew"}:
            speed_variability_scale = 0.12
        elif tracker_mode == "recover":
            speed_variability_scale = 0.18
        closing_speed_std_recent_m_s = float(
            max(0.0, closing_speed * rng.uniform(speed_variability_scale * 0.5, speed_variability_scale))
        )
        return PlannerTarget(
            target_id=episode_id * 100 + index + 1,
            box_index=index,
            cls=cls,
            confidence=confidence,
            damage_weight=damage_weight,
            distance_m=distance_m,
            radial_closing_speed_m_s=closing_speed,
            yaw_error_rad=yaw_error_rad,
            pitch_error_rad=pitch_error_rad,
            bbox_area_norm=bbox_area_norm,
            track_observations=track_observations,
            track_age_s=track_age_s,
            confidence_mean_recent=confidence_mean_recent,
            confidence_min_recent=confidence_min_recent,
            closing_speed_mean_recent_m_s=closing_speed,
            closing_speed_std_recent_m_s=closing_speed_std_recent_m_s,
            range_source=range_source,
            threat_level=threat_level,
            tracker_mode=tracker_mode,
            predictive_only=False,
        )

    def _split_episodes(
        self, episodes: Sequence[SyntheticEpisode]
    ) -> Dict[str, List[SyntheticEpisode]]:
        grouped: Dict[str, List[SyntheticEpisode]] = {}
        for episode in episodes:
            grouped.setdefault(episode.scenario_family, []).append(episode)

        rng = np.random.default_rng(self.split_seed)
        splits = {"train": [], "val": [], "test": [], "heldout": []}
        for items in grouped.values():
            items = list(items)
            rng.shuffle(items)
            count = len(items)
            n_train = int(round(count * self.split_ratios["train"]))
            n_val = int(round(count * self.split_ratios["val"]))
            n_test = int(round(count * self.split_ratios["test"]))
            n_heldout = max(0, count - n_train - n_val - n_test)
            splits["train"].extend(items[:n_train])
            splits["val"].extend(items[n_train : n_train + n_val])
            splits["test"].extend(items[n_train + n_val : n_train + n_val + n_test])
            splits["heldout"].extend(items[n_train + n_val + n_test : n_train + n_val + n_test + n_heldout])
        return splits

    def _build_split_records(
        self, episodes: Sequence[SyntheticEpisode]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        step_records: List[Dict[str, Any]] = []
        episode_records: List[Dict[str, Any]] = []

        for episode in episodes:
            state = tuple(episode.initial_targets)
            total_possible_damage = sum(target.damage_weight for target in state)
            step_index = 0
            oracle_episode_damage = 0.0

            while state:
                decision = evaluate_swarm_targets(state, self.settings)
                if decision.chosen_target_id is None:
                    break
                result_by_target = {
                    result.target_id: result for result in decision.candidate_results
                }
                current_index = {
                    target.target_id: idx for idx, target in enumerate(state)
                }
                chosen_index = current_index[decision.chosen_target_id]
                oracle_result = result_by_target[decision.chosen_target_id]

                regrets = np.full(self.max_targets_tensor, np.nan, dtype=np.float32)
                for target in state:
                    idx = current_index[target.target_id]
                    regrets[idx] = (
                        result_by_target[target.target_id].expected_total_damage
                        - oracle_result.expected_total_damage
                    )

                step_records.append(
                    {
                        "episode_id": episode.episode_id,
                        "scenario_family": episode.scenario_family,
                        "step_index": step_index,
                        "targets": [self._target_to_record(target, result_by_target[target.target_id]) for target in state],
                        "oracle_action_index": chosen_index,
                        "oracle_order": list(oracle_result.order),
                        "oracle_total_damage": oracle_result.expected_total_damage,
                        "oracle_regret_by_action": [
                            None if math.isnan(value) else float(value) for value in regrets
                        ],
                        "total_possible_damage": total_possible_damage,
                    }
                )
                if step_index == 0:
                    oracle_episode_damage = oracle_result.expected_total_damage

                _, _, state = advance_planner_state(
                    state,
                    self.settings,
                    target_id=decision.chosen_target_id,
                )
                step_index += 1

            episode_records.append(
                {
                    "episode_id": episode.episode_id,
                    "scenario_family": episode.scenario_family,
                    "total_possible_damage": total_possible_damage,
                    "oracle_episode_damage": oracle_episode_damage,
                    "initial_targets": [
                        self._target_to_record(target, None) for target in episode.initial_targets
                    ],
                }
            )

        return step_records, episode_records

    def _target_to_record(
        self,
        target: PlannerTarget,
        result: Any,
    ) -> Dict[str, Any]:
        breakthrough_time = target.breakthrough_time_s()
        return {
            "target_id": target.target_id,
            "box_index": target.box_index,
            "cls": target.cls,
            "confidence": target.confidence,
            "damage_weight": target.damage_weight,
            "distance_m": target.distance_m,
            "radial_closing_speed_m_s": target.radial_closing_speed_m_s,
            "yaw_error_rad": target.yaw_error_rad,
            "pitch_error_rad": target.pitch_error_rad,
            "bbox_area_norm": target.bbox_area_norm,
            "track_observations": target.track_observations,
            "track_age_s": target.track_age_s,
            "confidence_mean_recent": target.confidence_mean_recent,
            "confidence_min_recent": target.confidence_min_recent,
            "closing_speed_mean_recent_m_s": target.closing_speed_mean_recent_m_s,
            "closing_speed_std_recent_m_s": target.closing_speed_std_recent_m_s,
            "range_source": target.range_source,
            "threat_level": target.threat_level,
            "tracker_mode": target.tracker_mode,
            "predictive_only": target.predictive_only,
            "breakthrough_time_s": None if not math.isfinite(breakthrough_time) else breakthrough_time,
            "time_to_engage_s": None if result is None else result.time_to_engage_s,
            "expected_total_damage_if_selected": None if result is None else result.expected_total_damage,
            "engageable_now": None if result is None else result.engageable_now,
        }

    def _encode_split(self, step_records: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
        n_steps = len(step_records)
        target_features = np.zeros(
            (n_steps, self.max_targets_tensor, len(FEATURE_NAMES)), dtype=np.float32
        )
        target_mask = np.zeros((n_steps, self.max_targets_tensor), dtype=bool)
        global_features = np.zeros((n_steps, len(GLOBAL_FEATURE_NAMES)), dtype=np.float32)
        track_ids = np.full((n_steps, self.max_targets_tensor), -1, dtype=np.int32)
        oracle_action_index = np.full((n_steps,), -1, dtype=np.int32)
        oracle_order = np.full((n_steps, self.max_targets_tensor), -1, dtype=np.int32)
        oracle_total_damage = np.zeros((n_steps,), dtype=np.float32)
        oracle_regret_by_action = np.full(
            (n_steps, self.max_targets_tensor), np.nan, dtype=np.float32
        )
        threat_class_targets = np.full(
            (n_steps, self.max_targets_tensor), -1, dtype=np.int64
        )

        for row_idx, step in enumerate(step_records):
            targets = step["targets"]
            if len(targets) > self.max_targets_tensor:
                raise ValueError("Encountered more active targets than max_targets_tensor")

            breakthroughs = []
            time_to_engage_values = []
            total_damage_weight = 0.0
            warning_count = 0.0
            restricted_count = 0.0
            critical_count = 0.0
            for col_idx, target in enumerate(targets):
                breakthroughs.append(
                    self._clip_norm(
                        target["breakthrough_time_s"],
                        self.norm["max_breakthrough_time_s"],
                    )
                )
                time_to_engage_values.append(
                    self._clip_norm(
                        target["time_to_engage_s"],
                        self.norm["max_time_to_engage_s"],
                    )
                )
                total_damage_weight += float(target["damage_weight"])
                target_mask[row_idx, col_idx] = True
                track_ids[row_idx, col_idx] = int(target["target_id"])
                threat_class_targets[row_idx, col_idx] = self._threat_class_index(
                    target.get("threat_level")
                )
                oracle_regret_by_action[row_idx, col_idx] = (
                    np.nan
                    if step["oracle_regret_by_action"][col_idx] is None
                    else float(step["oracle_regret_by_action"][col_idx])
                )
                zone_features = compute_zone_feature_vector(
                    float(target["distance_m"]),
                    self.zone_radii,
                )
                warning_count += zone_features[0]
                restricted_count += zone_features[1]
                critical_count += zone_features[2]
                target_features[row_idx, col_idx, :] = np.array(
                    [
                        self._clip_norm(target["distance_m"], self.norm["max_distance_m"]),
                        self._clip_norm(
                            target["radial_closing_speed_m_s"],
                            self.norm["max_closing_speed_m_s"],
                        ),
                        self._clip_norm(
                            target["breakthrough_time_s"],
                            self.norm["max_breakthrough_time_s"],
                        ),
                        self._clip_norm(
                            target["time_to_engage_s"],
                            self.norm["max_time_to_engage_s"],
                        ),
                        self._clip_norm(
                            target["damage_weight"],
                            self.norm["max_damage_weight"],
                        ),
                        float(target["confidence"]),
                        self._signed_norm(target["yaw_error_rad"], self.norm["max_angle_rad"]),
                        self._signed_norm(target["pitch_error_rad"], self.norm["max_angle_rad"]),
                        float(target["bbox_area_norm"]),
                        self._clip_norm(
                            target["track_observations"],
                            self.norm["max_track_observations"],
                        ),
                        zone_features[0],
                        zone_features[1],
                        zone_features[2],
                        zone_features[3],
                        self._clip_norm(
                            target.get("track_age_s"),
                            self.norm["max_track_age_s"],
                        ),
                        float(
                            target.get("confidence_mean_recent", target["confidence"])
                        ),
                        float(
                            target.get("confidence_min_recent", target["confidence"])
                        ),
                        self._clip_norm(
                            target.get(
                                "closing_speed_mean_recent_m_s",
                                target["radial_closing_speed_m_s"],
                            ),
                            self.norm["max_closing_speed_m_s"],
                        ),
                        self._clip_norm(
                            target.get("closing_speed_std_recent_m_s"),
                            self.norm["max_closing_speed_m_s"],
                        ),
                    ],
                    dtype=np.float32,
                )

            global_features[row_idx, :] = np.array(
                [
                    len(targets) / max(1, self.max_targets_tensor),
                    min(breakthroughs) if breakthroughs else 0.0,
                    float(np.mean(time_to_engage_values)) if time_to_engage_values else 0.0,
                    self._clip_norm(total_damage_weight, self.norm["max_damage_weight"] * self.max_targets_tensor),
                    warning_count / max(1, len(targets)),
                    restricted_count / max(1, len(targets)),
                    critical_count / max(1, len(targets)),
                ],
                dtype=np.float32,
            )
            oracle_action_index[row_idx] = int(step["oracle_action_index"])
            oracle_total_damage[row_idx] = float(step["oracle_total_damage"])
            for col_idx, target_id in enumerate(step["oracle_order"][: self.max_targets_tensor]):
                oracle_order[row_idx, col_idx] = int(target_id)

        return {
            "target_features": target_features,
            "target_mask": target_mask,
            "global_features": global_features,
            "track_ids": track_ids,
            "oracle_action_index": oracle_action_index,
            "oracle_order": oracle_order,
            "oracle_total_damage": oracle_total_damage,
            "oracle_regret_by_action": oracle_regret_by_action,
            "threat_class_targets": threat_class_targets,
        }

    def _clip_norm(self, value: float | None, max_value: float) -> float:
        if value is None or not math.isfinite(float(value)):
            return 1.0
        return float(np.clip(float(value) / max(max_value, 1e-6), 0.0, 1.0))

    def _threat_class_index(self, threat_level: Any) -> int:
        if threat_level is None:
            return -1
        return int(THREAT_CLASS_TO_INDEX.get(str(threat_level).strip().lower(), -1))

    def _signed_norm(self, value: float, max_abs: float) -> float:
        return float(np.clip(float(value) / max(max_abs, 1e-6), -1.0, 1.0))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swarm_dataset.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with Path(args.config).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if args.output_dir is not None:
        config["output"]["directory"] = args.output_dir
    if args.num_episodes is not None:
        config["episodes"]["num_episodes"] = int(args.num_episodes)

    builder = SwarmDatasetBuilder(config)
    seed = int(args.seed if args.seed is not None else config["split"].get("seed", 42))
    stats = builder.build(seed=seed)
    logger.info("Built swarm dataset in %s", builder.output_dir)
    logger.info(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
