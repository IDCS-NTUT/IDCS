"""Swarm engagement planning utilities.

This module provides:
1. A pure target-order planner that minimizes deterministic breakthrough damage.
2. A runtime adapter that converts live detections into planner targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from common.control import (
    AxisPair,
    ControlConfig,
    SwarmEvalConfig,
    SwarmTimingConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.schemas import Box, CamState, DetectionMsg
from common.threat_calc import compute_breakthrough_time, estimate_time_to_engage

_EPS = 1e-9

__all__ = [
    "PlannerTarget",
    "PlannerCandidateResult",
    "PlannerDecision",
    "SwarmPlannerSettings",
    "evaluate_swarm_targets",
    "advance_planner_state",
    "SwarmPlannerRuntime",
]


@dataclass(frozen=True)
class SwarmPlannerSettings:
    """Numeric planner settings detached from the full control config."""

    yaw_rate_limit_rad_s: float
    pitch_rate_limit_rad_s: float
    yaw_accel_limit_rad_s2: float
    pitch_accel_limit_rad_s2: float
    exact_search_limit: int
    beam_width: int
    switch_absolute_damage_gain: float
    switch_relative_improvement: float
    timing: SwarmTimingConfig

    @classmethod
    def from_control_config(cls, config: ControlConfig) -> "SwarmPlannerSettings":
        swarm_cfg = config.swarm_eval
        return cls(
            yaw_rate_limit_rad_s=abs(float(config.rate_limits.yaw)),
            pitch_rate_limit_rad_s=abs(float(config.rate_limits.pitch)),
            yaw_accel_limit_rad_s2=abs(float(config.accel_limits.yaw)),
            pitch_accel_limit_rad_s2=abs(float(config.accel_limits.pitch)),
            exact_search_limit=int(swarm_cfg.exact_search_limit),
            beam_width=int(swarm_cfg.beam_width),
            switch_absolute_damage_gain=float(swarm_cfg.switch_absolute_damage_gain),
            switch_relative_improvement=float(swarm_cfg.switch_relative_improvement),
            timing=swarm_cfg.timing,
        )


@dataclass(frozen=True)
class PlannerTarget:
    """Planner-facing target state for one candidate hostile."""

    target_id: int
    box_index: int
    cls: str
    confidence: float
    damage_weight: float
    distance_m: float
    radial_closing_speed_m_s: float
    yaw_error_rad: float
    pitch_error_rad: float
    bbox_area_norm: float
    track_observations: int
    range_source: Optional[str] = None
    threat_level: Optional[str] = None
    tracker_mode: Optional[str] = None
    predictive_only: bool = False

    def breakthrough_time_s(self) -> float:
        return compute_breakthrough_time(self.distance_m, self.radial_closing_speed_m_s)


@dataclass(frozen=True)
class PlannerCandidateResult:
    """Evaluation result for selecting one target next."""

    target_id: int
    box_index: int
    expected_total_damage: float
    order: Tuple[int, ...]
    breakthrough_time_s: float
    time_to_engage_s: float
    damage_weight: float
    engageable_now: bool


@dataclass(frozen=True)
class PlannerDecision:
    """Chosen target plus full candidate diagnostics."""

    chosen_target_id: Optional[int]
    chosen_box_index: Optional[int]
    expected_total_damage: float
    candidate_results: Tuple[PlannerCandidateResult, ...]


@dataclass(frozen=True)
class _PlanningState:
    targets: Tuple[PlannerTarget, ...]
    current_yaw_rate_rad_s: float = 0.0
    current_pitch_rate_rad_s: float = 0.0


@dataclass
class _RuntimeTrackState:
    """Minimal per-track history for live breakthrough estimates."""

    last_seen_time_s: float
    last_center_uv: Tuple[float, float]
    last_distance_m: Optional[float]
    consecutive_hits: int = 1
    radial_closing_speed_m_s: float = 0.0


def evaluate_swarm_targets(
    targets: Sequence[PlannerTarget],
    settings: SwarmPlannerSettings,
    *,
    previous_target_id: Optional[int] = None,
    current_yaw_rate_rad_s: float = 0.0,
    current_pitch_rate_rad_s: float = 0.0,
) -> PlannerDecision:
    """Evaluate all candidates and choose the next target with hysteresis."""
    if not targets:
        return PlannerDecision(
            chosen_target_id=None,
            chosen_box_index=None,
            expected_total_damage=0.0,
            candidate_results=tuple(),
        )

    state = _PlanningState(
        targets=tuple(targets),
        current_yaw_rate_rad_s=float(current_yaw_rate_rad_s),
        current_pitch_rate_rad_s=float(current_pitch_rate_rad_s),
    )
    candidate_results: List[PlannerCandidateResult] = []
    for target in state.targets:
        time_to_engage_s = _estimate_target_time_to_engage(target, state, settings)
        immediate_damage, next_state = _simulate_action(state, target.target_id, settings)
        future_damage, future_order = _evaluate_best_order(next_state, settings)
        total_damage = immediate_damage + future_damage
        candidate_results.append(
            PlannerCandidateResult(
                target_id=target.target_id,
                box_index=target.box_index,
                expected_total_damage=total_damage,
                order=(target.target_id,) + future_order,
                breakthrough_time_s=target.breakthrough_time_s(),
                time_to_engage_s=time_to_engage_s,
                damage_weight=target.damage_weight,
                engageable_now=target.breakthrough_time_s() > time_to_engage_s,
            )
        )

    candidate_results.sort(
        key=lambda item: (
            item.expected_total_damage,
            not item.engageable_now,
            item.breakthrough_time_s,
            item.time_to_engage_s,
            -item.damage_weight,
            item.target_id,
        )
    )
    chosen = _apply_hysteresis(candidate_results, previous_target_id, settings)
    return PlannerDecision(
        chosen_target_id=chosen.target_id,
        chosen_box_index=chosen.box_index,
        expected_total_damage=chosen.expected_total_damage,
        candidate_results=tuple(candidate_results),
    )


def advance_planner_state(
    targets: Sequence[PlannerTarget],
    settings: SwarmPlannerSettings,
    *,
    target_id: int,
    current_yaw_rate_rad_s: float = 0.0,
    current_pitch_rate_rad_s: float = 0.0,
) -> Tuple[float, float, Tuple[PlannerTarget, ...]]:
    """Advance one planner step by engaging the requested target."""
    state = _PlanningState(
        targets=tuple(targets),
        current_yaw_rate_rad_s=float(current_yaw_rate_rad_s),
        current_pitch_rate_rad_s=float(current_pitch_rate_rad_s),
    )
    chosen = next(target for target in state.targets if target.target_id == target_id)
    elapsed_s = _estimate_target_time_to_engage(chosen, state, settings)
    immediate_damage, next_state = _simulate_action(state, target_id, settings)
    return elapsed_s, immediate_damage, next_state.targets


def _apply_hysteresis(
    results: Sequence[PlannerCandidateResult],
    previous_target_id: Optional[int],
    settings: SwarmPlannerSettings,
) -> PlannerCandidateResult:
    best = results[0]
    if previous_target_id is None:
        return best

    previous = next((item for item in results if item.target_id == previous_target_id), None)
    if previous is None:
        return best
    if previous.target_id == best.target_id:
        return previous
    if not previous.engageable_now:
        return best

    improvement = previous.expected_total_damage - best.expected_total_damage
    relative = improvement / max(previous.expected_total_damage, _EPS)
    if (
        improvement >= settings.switch_absolute_damage_gain
        or relative >= settings.switch_relative_improvement
    ):
        return best
    return previous


def _evaluate_best_order(
    state: _PlanningState,
    settings: SwarmPlannerSettings,
) -> Tuple[float, Tuple[int, ...]]:
    if not state.targets:
        return 0.0, tuple()
    if len(state.targets) <= settings.exact_search_limit:
        return _evaluate_best_order_exact(state, settings)
    return _evaluate_best_order_beam(state, settings)


def _evaluate_best_order_exact(
    state: _PlanningState,
    settings: SwarmPlannerSettings,
) -> Tuple[float, Tuple[int, ...]]:
    if not state.targets:
        return 0.0, tuple()

    best_damage = math.inf
    best_order: Tuple[int, ...] = tuple()
    for target in state.targets:
        immediate_damage, next_state = _simulate_action(state, target.target_id, settings)
        future_damage, future_order = _evaluate_best_order_exact(next_state, settings)
        total_damage = immediate_damage + future_damage
        order = (target.target_id,) + future_order
        if _is_order_better(total_damage, order, best_damage, best_order):
            best_damage = total_damage
            best_order = order
    return best_damage, best_order


def _evaluate_best_order_beam(
    state: _PlanningState,
    settings: SwarmPlannerSettings,
) -> Tuple[float, Tuple[int, ...]]:
    partials: List[Tuple[float, Tuple[int, ...], _PlanningState]] = [(0.0, tuple(), state)]
    for _ in range(len(state.targets)):
        expanded: List[Tuple[float, float, Tuple[int, ...], _PlanningState]] = []
        for damage_so_far, order_so_far, partial_state in partials:
            if not partial_state.targets:
                expanded.append((damage_so_far, damage_so_far, order_so_far, partial_state))
                continue
            for target in partial_state.targets:
                immediate_damage, next_state = _simulate_action(
                    partial_state, target.target_id, settings
                )
                next_damage = damage_so_far + immediate_damage
                greedy_tail_damage, _ = _greedy_rollout(next_state, settings)
                expanded.append(
                    (
                        next_damage + greedy_tail_damage,
                        next_damage,
                        order_so_far + (target.target_id,),
                        next_state,
                    )
                )
        expanded.sort(key=lambda item: (item[0], item[1], item[2]))
        partials = [(actual_damage, order, partial_state) for _, actual_damage, order, partial_state in expanded[: settings.beam_width]]

    best_damage = math.inf
    best_order: Tuple[int, ...] = tuple()
    for damage_so_far, order_so_far, partial_state in partials:
        completion_damage, completion_order = _greedy_rollout(partial_state, settings)
        total_damage = damage_so_far + completion_damage
        order = order_so_far + completion_order
        if _is_order_better(total_damage, order, best_damage, best_order):
            best_damage = total_damage
            best_order = order
    return best_damage, best_order


def _greedy_rollout(
    state: _PlanningState,
    settings: SwarmPlannerSettings,
) -> Tuple[float, Tuple[int, ...]]:
    if not state.targets:
        return 0.0, tuple()

    total_damage = 0.0
    order: List[int] = []
    working_state = state
    while working_state.targets:
        ranked = sorted(
            working_state.targets,
            key=lambda target: (
                _immediate_action_damage(working_state, target, settings),
                target.breakthrough_time_s(),
                _estimate_target_time_to_engage(target, working_state, settings),
                -target.damage_weight,
                target.target_id,
            ),
        )
        chosen = ranked[0]
        immediate_damage, working_state = _simulate_action(
            working_state, chosen.target_id, settings
        )
        total_damage += immediate_damage
        order.append(chosen.target_id)
    return total_damage, tuple(order)


def _immediate_action_damage(
    state: _PlanningState,
    target: PlannerTarget,
    settings: SwarmPlannerSettings,
) -> float:
    immediate_damage, _ = _simulate_action(state, target.target_id, settings)
    return immediate_damage


def _estimate_target_time_to_engage(
    target: PlannerTarget,
    state: _PlanningState,
    settings: SwarmPlannerSettings,
) -> float:
    timing = settings.timing
    return estimate_time_to_engage(
        yaw_error_rad=target.yaw_error_rad,
        pitch_error_rad=target.pitch_error_rad,
        yaw_rate_limit_rad_s=settings.yaw_rate_limit_rad_s,
        pitch_rate_limit_rad_s=settings.pitch_rate_limit_rad_s,
        yaw_accel_limit_rad_s2=settings.yaw_accel_limit_rad_s2,
        pitch_accel_limit_rad_s2=settings.pitch_accel_limit_rad_s2,
        current_yaw_rate_rad_s=state.current_yaw_rate_rad_s,
        current_pitch_rate_rad_s=state.current_pitch_rate_rad_s,
        tracker_mode=target.tracker_mode,
        confidence=target.confidence,
        track_observations=target.track_observations,
        range_source=target.range_source,
        predictive_only=target.predictive_only,
        base_track_lock_s=timing.base_track_lock_s,
        search_track_lock_s=timing.search_track_lock_s,
        recover_track_lock_s=timing.recover_track_lock_s,
        low_conf_threshold=timing.low_conf_threshold,
        low_conf_penalty_s=timing.low_conf_penalty_s,
        min_track_observations=timing.min_track_observations,
        low_continuity_penalty_s=timing.low_continuity_penalty_s,
        missing_range_penalty_s=timing.missing_range_penalty_s,
        predictive_penalty_s=timing.predictive_penalty_s,
        effect_time_s=timing.effect_time_s,
        confirm_time_s=timing.confirm_time_s,
        settle_margin_s=timing.settle_margin_s,
    )


def _simulate_action(
    state: _PlanningState,
    target_id: int,
    settings: SwarmPlannerSettings,
) -> Tuple[float, _PlanningState]:
    chosen = next(target for target in state.targets if target.target_id == target_id)
    elapsed_s = _estimate_target_time_to_engage(chosen, state, settings)

    damage = 0.0
    survivors: List[PlannerTarget] = []
    for target in state.targets:
        breakthrough_time_s = target.breakthrough_time_s()
        if breakthrough_time_s <= elapsed_s + _EPS:
            damage += target.damage_weight
            continue
        if target.target_id == chosen.target_id:
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
                    target.distance_m - target.radial_closing_speed_m_s * elapsed_s,
                ),
                radial_closing_speed_m_s=target.radial_closing_speed_m_s,
                yaw_error_rad=target.yaw_error_rad - chosen.yaw_error_rad,
                pitch_error_rad=target.pitch_error_rad - chosen.pitch_error_rad,
                bbox_area_norm=target.bbox_area_norm,
                track_observations=target.track_observations,
                range_source=target.range_source,
                threat_level=target.threat_level,
                tracker_mode=target.tracker_mode,
                predictive_only=target.predictive_only,
            )
        )

    return damage, _PlanningState(tuple(survivors), 0.0, 0.0)


def _is_order_better(
    total_damage: float,
    order: Tuple[int, ...],
    best_damage: float,
    best_order: Tuple[int, ...],
) -> bool:
    if total_damage + _EPS < best_damage:
        return True
    if abs(total_damage - best_damage) <= _EPS and (not best_order or order < best_order):
        return True
    return False


class SwarmPlannerRuntime:
    """Live runtime adapter that annotates boxes and chooses one target."""

    def __init__(self, control_config: ControlConfig):
        self._control_config = control_config
        self._swarm_config: SwarmEvalConfig = control_config.swarm_eval
        self._settings = SwarmPlannerSettings.from_control_config(control_config)
        self._track_history: Dict[int, _RuntimeTrackState] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._swarm_config.enabled)

    def update_and_select(
        self,
        msg: DetectionMsg,
        *,
        current_time_s: float,
        cam_state: Optional[CamState],
        previous_target_id: Optional[int],
        candidates: Optional[Sequence[Tuple[int, Box]]] = None,
    ) -> PlannerDecision:
        enumerated = list(candidates if candidates is not None else enumerate(msg.boxes))
        active_track_ids: set[int] = set()
        planner_targets: List[PlannerTarget] = []
        current_rates = AxisPair(0.0, 0.0)
        if cam_state is not None:
            current_rates = AxisPair(
                yaw=0.0 if cam_state.pan_rate is None else float(cam_state.pan_rate),
                pitch=0.0 if cam_state.tilt_rate is None else float(cam_state.tilt_rate),
            )

        for index, box in enumerated:
            track_key = self._track_key(index, box)
            if track_key > 0:
                active_track_ids.add(track_key)

            history = self._update_track_history(
                track_key,
                box,
                msg=msg,
                current_time_s=current_time_s,
            )
            self._clear_swarm_fields(box)

            if not self._is_hostile(box):
                box.damage_weight = self._damage_weight_for_box(box)
                continue

            center_u = (box.x + box.w / 2.0) * msg.img_w
            center_v = (box.y + box.h / 2.0) * msg.img_h
            px_err = pixel_delta(
                center_u,
                center_v,
                self._control_config.cx_px,
                self._control_config.cy_px,
                self._control_config,
                apply_deadband=False,
            )
            ang_err = angular_error_from_pixel_delta(px_err, self._control_config)
            planner_targets.append(
                PlannerTarget(
                    target_id=track_key,
                    box_index=index,
                    cls=box.cls,
                    confidence=float(box.conf),
                    damage_weight=self._damage_weight_for_box(box),
                    distance_m=self._distance_for_box(box),
                    radial_closing_speed_m_s=history.radial_closing_speed_m_s,
                    yaw_error_rad=ang_err.yaw,
                    pitch_error_rad=ang_err.pitch,
                    bbox_area_norm=float(box.w * box.h),
                    track_observations=history.consecutive_hits,
                    range_source=box.distance_src,
                    threat_level=box.threat_level,
                    tracker_mode=msg.tracker_mode,
                    predictive_only=False,
                )
            )

        self._track_history = {
            key: value for key, value in self._track_history.items() if key in active_track_ids
        }

        decision = evaluate_swarm_targets(
            planner_targets,
            self._settings,
            previous_target_id=previous_target_id,
            current_yaw_rate_rad_s=current_rates.yaw,
            current_pitch_rate_rad_s=current_rates.pitch,
        )
        self._annotate_boxes(msg, decision)
        return decision

    def _track_key(self, index: int, box: Box) -> int:
        if box.track_id is not None:
            return int(box.track_id)
        return -(index + 1)

    def _is_hostile(self, box: Box) -> bool:
        if box.threat_level is None:
            return True
        return box.threat_level in self._swarm_config.hostile_levels

    def _damage_weight_for_box(self, box: Box) -> float:
        if box.damage_weight is not None:
            return float(box.damage_weight)
        if box.cls in self._swarm_config.damage_by_class:
            return float(self._swarm_config.damage_by_class[box.cls])
        return float(self._swarm_config.default_damage_weight)

    def _distance_for_box(self, box: Box) -> float:
        if box.distance_m is not None and math.isfinite(float(box.distance_m)):
            return max(0.0, float(box.distance_m))
        return math.inf

    def _update_track_history(
        self,
        track_key: int,
        box: Box,
        *,
        msg: DetectionMsg,
        current_time_s: float,
    ) -> _RuntimeTrackState:
        center_u = (box.x + box.w / 2.0) * msg.img_w
        center_v = (box.y + box.h / 2.0) * msg.img_h
        distance_m = None
        if box.distance_m is not None and math.isfinite(float(box.distance_m)):
            distance_m = float(box.distance_m)

        prev = self._track_history.get(track_key)
        if prev is None:
            state = _RuntimeTrackState(
                last_seen_time_s=current_time_s,
                last_center_uv=(center_u, center_v),
                last_distance_m=distance_m,
                consecutive_hits=1,
                radial_closing_speed_m_s=self._default_closing_speed_for_box(box),
            )
            self._track_history[track_key] = state
            return state

        dt = max(current_time_s - prev.last_seen_time_s, _EPS)
        closing_speed = prev.radial_closing_speed_m_s
        if distance_m is not None and prev.last_distance_m is not None:
            closing_speed = max(0.0, (prev.last_distance_m - distance_m) / dt)
        elif closing_speed <= 0.0:
            closing_speed = self._default_closing_speed_for_box(box)

        consecutive_hits = prev.consecutive_hits + 1
        state = _RuntimeTrackState(
            last_seen_time_s=current_time_s,
            last_center_uv=(center_u, center_v),
            last_distance_m=distance_m,
            consecutive_hits=consecutive_hits,
            radial_closing_speed_m_s=closing_speed,
        )
        self._track_history[track_key] = state
        return state

    def _default_closing_speed_for_box(self, box: Box) -> float:
        if box.threat_level == "threatening":
            return 3.0
        if box.threat_level == "suspicious":
            return 1.5
        return 0.0

    def _annotate_boxes(self, msg: DetectionMsg, decision: PlannerDecision) -> None:
        result_by_index = {result.box_index: result for result in decision.candidate_results}
        damages = [result.expected_total_damage for result in decision.candidate_results]
        min_damage = min(damages) if damages else 0.0
        max_damage = max(damages) if damages else 0.0

        for rank, result in enumerate(decision.candidate_results, start=1):
            box = msg.boxes[result.box_index]
            box.breakthrough_time_s = result.breakthrough_time_s
            box.time_to_engage_s = result.time_to_engage_s
            box.damage_weight = result.damage_weight
            box.engageable_now = result.engageable_now
            box.expected_damage_if_ignored = result.damage_weight if math.isfinite(result.breakthrough_time_s) else 0.0
            box.expected_total_damage_if_selected = result.expected_total_damage
            box.engagement_rank = rank
            if max_damage - min_damage <= _EPS:
                box.priority_score = 1.0
            else:
                box.priority_score = 1.0 - (
                    (result.expected_total_damage - min_damage) / (max_damage - min_damage)
                )

        for index, box in enumerate(msg.boxes):
            if index in result_by_index:
                continue
            if box.damage_weight is None:
                box.damage_weight = self._damage_weight_for_box(box)

        msg.swarm_expected_total_damage = (
            decision.expected_total_damage if decision.chosen_target_id is not None else None
        )

    def _clear_swarm_fields(self, box: Box) -> None:
        box.priority_score = None
        box.engagement_rank = None
        box.breakthrough_time_s = None
        box.time_to_engage_s = None
        box.engageable_now = None
        box.expected_damage_if_ignored = None
        box.expected_total_damage_if_selected = None
