"""Swarm engagement planning utilities.

This module provides:
1. A pure target-order planner that minimizes deterministic breakthrough damage.
2. A runtime adapter that converts live detections into planner targets.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.control import (
    AxisPair,
    ControlConfig,
    SwarmFeatureNormalizationConfig,
    SwarmEvalConfig,
    SwarmTimingConfig,
    angular_error_from_pixel_delta,
    pixel_delta,
)
from common.schemas import Box, CamState, DetectionMsg
from common.threat_calc import (
    compute_breakthrough_time,
    compute_zone_feature_vector,
    estimate_time_to_engage,
    get_zone_id_for_distance,
)
try:
    from jetson.swarm_policy_model import THREAT_CLASS_NAMES, load_swarm_policy_checkpoint
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    THREAT_CLASS_NAMES = ("benign", "suspicious", "threatening")
    load_swarm_policy_checkpoint = None  # type: ignore[assignment]
try:
    from jetson.swarm_policy_trt import (
        SwarmPolicyTensorRTEngine,
        TensorRTRuntimeUnavailableError,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SwarmPolicyTensorRTEngine = None  # type: ignore[assignment]
    TensorRTRuntimeUnavailableError = RuntimeError  # type: ignore[assignment]

_EPS = 1e-9
_LOG = logging.getLogger("jetson.swarm_planner")

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


@dataclass(frozen=True)
class _AsyncInferenceRequest:
    request_id: int
    target_ids: Tuple[int, ...]
    target_features: np.ndarray
    global_features: np.ndarray
    target_mask: np.ndarray


@dataclass(frozen=True)
class _AsyncInferenceResult:
    request_id: int
    completed_at_s: float
    target_ids: Tuple[int, ...]
    chosen_target_id: int
    class_predictions: Dict[int, Tuple[str, float, np.ndarray]]


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
        self._learned_selector = None
        self._learned_tensorrt = None
        self._latest_model_class_predictions: Dict[int, Tuple[str, float, np.ndarray]] = {}
        self._learned_max_targets = int(self._swarm_config.learned_model.max_targets_tensor)
        self._learned_target_feature_size = 10
        self._learned_global_feature_size = 4
        self._learned_normalization = self._swarm_config.learned_model.normalization
        self._async_enabled = bool(self._swarm_config.learned_model.async_worker)
        self._async_max_result_age_s = (
            float(self._swarm_config.learned_model.max_result_age_ms) / 1000.0
        )
        self._async_lock = threading.Lock()
        self._async_event = threading.Event()
        self._async_shutdown = False
        self._async_pending_request: Optional[_AsyncInferenceRequest] = None
        self._async_latest_result: Optional[_AsyncInferenceResult] = None
        self._async_request_counter = 0
        self._async_worker_thread: Optional[threading.Thread] = None
        self._load_learned_model()
        self._start_async_worker_if_needed()

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
        self._latest_model_class_predictions = {}
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
            damage_weight = self._damage_weight_for_box(box)
            distance_m = self._distance_for_box(box)
            threat_level = self._classify_live_threat_level(
                distance_m=distance_m,
                radial_closing_speed_m_s=history.radial_closing_speed_m_s,
                yaw_error_rad=ang_err.yaw,
                pitch_error_rad=ang_err.pitch,
                confidence=float(box.conf),
                track_observations=history.consecutive_hits,
                range_source=box.distance_src,
                tracker_mode=msg.tracker_mode,
                predictive_only=False,
                current_yaw_rate_rad_s=current_rates.yaw,
                current_pitch_rate_rad_s=current_rates.pitch,
            )
            self._apply_rule_based_threat_annotation(box, threat_level)
            box.damage_weight = damage_weight

            if self._is_excluded_target_class(box):
                continue

            include_for_ranking = self._is_hostile(box)
            if self._learned_selector is not None or self._learned_tensorrt is not None:
                include_for_ranking = True
            if not include_for_ranking:
                continue

            planner_targets.append(
                PlannerTarget(
                    target_id=track_key,
                    box_index=index,
                    cls=box.cls,
                    confidence=float(box.conf),
                    damage_weight=damage_weight,
                    distance_m=distance_m,
                    radial_closing_speed_m_s=history.radial_closing_speed_m_s,
                    yaw_error_rad=ang_err.yaw,
                    pitch_error_rad=ang_err.pitch,
                    bbox_area_norm=float(box.w * box.h),
                    track_observations=history.consecutive_hits,
                    range_source=box.distance_src,
                    threat_level=threat_level,
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
        if (self._learned_selector is not None or self._learned_tensorrt is not None) and planner_targets:
            if self._async_enabled:
                decision = self._apply_learned_selection_async(
                    planner_targets,
                    decision,
                    previous_target_id=previous_target_id,
                )
            else:
                decision = self._apply_learned_selection(
                    planner_targets,
                    decision,
                    previous_target_id=previous_target_id,
                )
        self._annotate_boxes(msg, decision)
        return decision

    def _start_async_worker_if_needed(self) -> None:
        if not self._async_enabled:
            return
        if self._learned_selector is None and self._learned_tensorrt is None:
            return
        if self._async_worker_thread is not None:
            return
        self._async_worker_thread = threading.Thread(
            target=self._async_worker_loop,
            name="swarm-learned-worker",
            daemon=True,
        )
        self._async_worker_thread.start()

    def _async_worker_loop(self) -> None:
        while True:
            self._async_event.wait()
            self._async_event.clear()
            if self._async_shutdown:
                return
            while True:
                with self._async_lock:
                    request = self._async_pending_request
                    self._async_pending_request = None
                if request is None:
                    break
                try:
                    chosen_target_id, class_predictions = self._run_learned_inference(
                        request.target_ids,
                        request.target_features,
                        request.global_features,
                        request.target_mask,
                    )
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    _LOG.warning(
                        "Swarm learned selector failed during async inference: %s; keeping planner fallback",
                        exc,
                    )
                    continue
                result = _AsyncInferenceResult(
                    request_id=request.request_id,
                    completed_at_s=time.monotonic(),
                    target_ids=request.target_ids,
                    chosen_target_id=chosen_target_id,
                    class_predictions=class_predictions,
                )
                with self._async_lock:
                    self._async_latest_result = result

    def _load_learned_model(self) -> None:
        learned_cfg = self._swarm_config.learned_model
        if not learned_cfg.enabled:
            return
        if not learned_cfg.model_path:
            if not learned_cfg.fallback_to_planner:
                raise ValueError(
                    "swarm learned model enabled without swarm_eval.learned_model.model_path"
                )
            _LOG.warning(
                "swarm learned model enabled without swarm_eval.learned_model.model_path; using planner fallback"
            )
            return

        model_path = Path(learned_cfg.model_path).expanduser()
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[1] / model_path

        backend = learned_cfg.backend
        if backend == "auto":
            backend = "tensorrt" if model_path.suffix.lower() == ".engine" else "torch"

        try:
            if backend == "tensorrt":
                if SwarmPolicyTensorRTEngine is None:
                    raise TensorRTRuntimeUnavailableError(
                        "TensorRT runtime module is unavailable"
                    )
                engine = SwarmPolicyTensorRTEngine(model_path)
                self._learned_tensorrt = engine
                if engine.max_targets > 0:
                    self._learned_max_targets = int(engine.max_targets)
                self._learned_target_feature_size = int(engine.target_feature_size)
                self._learned_global_feature_size = int(engine.global_feature_size)
                _LOG.info("Loaded swarm TensorRT engine from %s", model_path)
                return
            if load_swarm_policy_checkpoint is None:
                raise RuntimeError("Torch runtime is unavailable for swarm learned model")
            selector, metadata = load_swarm_policy_checkpoint(model_path)
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            if not learned_cfg.fallback_to_planner:
                raise
            _LOG.warning(
                "Failed to load swarm learned model from %s: %s; using planner fallback",
                model_path,
                exc,
            )
            return

        self._learned_selector = selector
        if "normalization" in metadata and isinstance(metadata["normalization"], dict):
            self._learned_normalization = self._normalization_from_mapping(
                metadata["normalization"]
            )
        if "max_targets_tensor" in metadata:
            self._learned_max_targets = max(1, int(metadata["max_targets_tensor"]))
        elif "max_targets" in metadata:
            self._learned_max_targets = max(1, int(metadata["max_targets"]))
        if "target_feature_size" in metadata:
            self._learned_target_feature_size = max(1, int(metadata["target_feature_size"]))
        if "global_feature_size" in metadata:
            self._learned_global_feature_size = max(1, int(metadata["global_feature_size"]))
        _LOG.info("Loaded swarm learned model from %s", model_path)

    def _normalization_from_mapping(
        self,
        mapping: Dict[str, float],
    ) -> SwarmFeatureNormalizationConfig:
        return SwarmFeatureNormalizationConfig(
            max_distance_m=float(
                mapping.get(
                    "max_distance_m", self._swarm_config.learned_model.normalization.max_distance_m
                )
            ),
            max_closing_speed_m_s=float(
                mapping.get(
                    "max_closing_speed_m_s",
                    self._swarm_config.learned_model.normalization.max_closing_speed_m_s,
                )
            ),
            max_breakthrough_time_s=float(
                mapping.get(
                    "max_breakthrough_time_s",
                    self._swarm_config.learned_model.normalization.max_breakthrough_time_s,
                )
            ),
            max_time_to_engage_s=float(
                mapping.get(
                    "max_time_to_engage_s",
                    self._swarm_config.learned_model.normalization.max_time_to_engage_s,
                )
            ),
            max_damage_weight=float(
                mapping.get(
                    "max_damage_weight",
                    self._swarm_config.learned_model.normalization.max_damage_weight,
                )
            ),
            max_angle_rad=float(
                mapping.get(
                    "max_angle_rad", self._swarm_config.learned_model.normalization.max_angle_rad
                )
            ),
            max_track_observations=float(
                mapping.get(
                    "max_track_observations",
                    self._swarm_config.learned_model.normalization.max_track_observations,
                )
            ),
        )

    def _apply_learned_selection(
        self,
        planner_targets: Sequence[PlannerTarget],
        planner_decision: PlannerDecision,
        *,
        previous_target_id: Optional[int],
    ) -> PlannerDecision:
        try:
            chosen_target_id, model_class_predictions = self._predict_learned_outputs(
                planner_targets,
                planner_decision.candidate_results,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            if not self._swarm_config.learned_model.fallback_to_planner:
                raise
            _LOG.warning(
                "Swarm learned selector failed during inference: %s; using planner fallback",
                exc,
            )
            return planner_decision
        self._latest_model_class_predictions = model_class_predictions

        result_by_target = {
            result.target_id: result for result in planner_decision.candidate_results
        }
        learned_result = result_by_target.get(chosen_target_id)
        if learned_result is None:
            return planner_decision

        chosen_result = self._apply_learned_hysteresis(
            learned_result,
            planner_decision.candidate_results,
            previous_target_id=previous_target_id,
        )
        return PlannerDecision(
            chosen_target_id=chosen_result.target_id,
            chosen_box_index=chosen_result.box_index,
            expected_total_damage=chosen_result.expected_total_damage,
            candidate_results=planner_decision.candidate_results,
        )

    def _apply_learned_selection_async(
        self,
        planner_targets: Sequence[PlannerTarget],
        planner_decision: PlannerDecision,
        *,
        previous_target_id: Optional[int],
    ) -> PlannerDecision:
        target_features, global_features, target_mask = self._encode_model_inputs(
            planner_targets,
            planner_decision.candidate_results,
        )
        target_ids = tuple(int(target.target_id) for target in planner_targets)
        self._submit_async_request(
            target_ids,
            target_features,
            global_features,
            target_mask,
        )
        latest_result = self._get_async_result(target_ids)
        if latest_result is None:
            return planner_decision

        self._latest_model_class_predictions = latest_result.class_predictions
        result_by_target = {
            result.target_id: result for result in planner_decision.candidate_results
        }
        learned_result = result_by_target.get(latest_result.chosen_target_id)
        if learned_result is None:
            return planner_decision

        chosen_result = self._apply_learned_hysteresis(
            learned_result,
            planner_decision.candidate_results,
            previous_target_id=previous_target_id,
        )
        return PlannerDecision(
            chosen_target_id=chosen_result.target_id,
            chosen_box_index=chosen_result.box_index,
            expected_total_damage=chosen_result.expected_total_damage,
            candidate_results=planner_decision.candidate_results,
        )

    def _submit_async_request(
        self,
        target_ids: Tuple[int, ...],
        target_features: np.ndarray,
        global_features: np.ndarray,
        target_mask: np.ndarray,
    ) -> None:
        with self._async_lock:
            self._async_request_counter += 1
            self._async_pending_request = _AsyncInferenceRequest(
                request_id=self._async_request_counter,
                target_ids=target_ids,
                target_features=np.array(target_features, copy=True),
                global_features=np.array(global_features, copy=True),
                target_mask=np.array(target_mask, copy=True),
            )
        self._async_event.set()

    def _get_async_result(
        self,
        target_ids: Tuple[int, ...],
    ) -> Optional[_AsyncInferenceResult]:
        with self._async_lock:
            result = self._async_latest_result
        if result is None:
            return None
        if result.target_ids != target_ids:
            return None
        if (time.monotonic() - result.completed_at_s) > self._async_max_result_age_s:
            return None
        return result

    def _predict_learned_outputs(
        self,
        planner_targets: Sequence[PlannerTarget],
        candidate_results: Sequence[PlannerCandidateResult],
    ) -> Tuple[int, Dict[int, Tuple[str, float, np.ndarray]]]:
        if self._learned_selector is None and self._learned_tensorrt is None:
            raise RuntimeError("Learned selector is not loaded")
        target_features, global_features, target_mask = self._encode_model_inputs(
            planner_targets,
            candidate_results,
        )
        return self._run_learned_inference(
            tuple(int(target.target_id) for target in planner_targets),
            target_features,
            global_features,
            target_mask,
        )

    def _run_learned_inference(
        self,
        target_ids: Tuple[int, ...],
        target_features: np.ndarray,
        global_features: np.ndarray,
        target_mask: np.ndarray,
    ) -> Tuple[int, Dict[int, Tuple[str, float, np.ndarray]]]:
        class_predictions: Dict[int, Tuple[str, float, np.ndarray]] = {}
        if self._learned_tensorrt is not None:
            outputs = self._learned_tensorrt.predict(
                target_features,
                global_features,
                target_mask,
            )
            logits = outputs[self._learned_tensorrt.policy_output_name]
            if (
                self._learned_tensorrt.threat_class_output_name is not None
                and self._learned_tensorrt.threat_class_output_name in outputs
            ):
                class_predictions = self._decode_class_predictions(
                    target_ids,
                    outputs[self._learned_tensorrt.threat_class_output_name],
                )
            action_index = int(np.argmax(logits[0]))
        else:
            logits, _, class_probs = self._learned_selector.predict_outputs_numpy(
                target_features,
                global_features,
                target_mask,
            )
            if class_probs is not None:
                class_predictions = self._decode_class_predictions_from_probs(
                    target_ids,
                    class_probs,
                )
            action_index = int(np.argmax(logits[0]))
        if action_index < 0 or action_index >= len(target_ids):
            raise ValueError(f"Learned selector returned invalid action index {action_index}")
        return int(target_ids[action_index]), class_predictions

    def _decode_class_predictions(
        self,
        target_ids: Sequence[int],
        class_logits: np.ndarray,
    ) -> Dict[int, Tuple[str, float, np.ndarray]]:
        shifted = class_logits - np.max(class_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        class_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return self._decode_class_predictions_from_probs(target_ids, class_probs)

    def _decode_class_predictions_from_probs(
        self,
        target_ids: Sequence[int],
        class_probs: np.ndarray,
    ) -> Dict[int, Tuple[str, float, np.ndarray]]:
        predictions: Dict[int, Tuple[str, float, np.ndarray]] = {}
        if class_probs.ndim != 3 or class_probs.shape[0] == 0:
            return predictions
        for index, target_id in enumerate(target_ids):
            probs = class_probs[0, index]
            class_id = int(np.argmax(probs))
            if class_id >= len(THREAT_CLASS_NAMES):
                continue
            predictions[int(target_id)] = (
                str(THREAT_CLASS_NAMES[class_id]),
                float(probs[class_id]),
                probs.astype(np.float32, copy=False),
            )
        return predictions

    def _encode_model_inputs(
        self,
        planner_targets: Sequence[PlannerTarget],
        candidate_results: Sequence[PlannerCandidateResult],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(planner_targets) > self._learned_max_targets:
            raise ValueError(
                f"Runtime state has {len(planner_targets)} targets but model supports {self._learned_max_targets}"
            )

        norm = self._learned_normalization
        result_by_target = {result.target_id: result for result in candidate_results}
        target_features = np.zeros(
            (1, self._learned_max_targets, self._learned_target_feature_size),
            dtype=np.float32,
        )
        target_mask = np.zeros((1, self._learned_max_targets), dtype=bool)
        breakthroughs: List[float] = []
        engage_times: List[float] = []
        total_damage_weight = 0.0
        warning_count = 0.0
        restricted_count = 0.0
        critical_count = 0.0

        for idx, target in enumerate(planner_targets):
            result = result_by_target[target.target_id]
            target_mask[0, idx] = True
            breakthroughs.append(
                self._clip_norm(result.breakthrough_time_s, norm.max_breakthrough_time_s)
            )
            engage_times.append(
                self._clip_norm(result.time_to_engage_s, norm.max_time_to_engage_s)
            )
            total_damage_weight += float(target.damage_weight)
            zone_features = compute_zone_feature_vector(
                float(target.distance_m),
                self._control_config.threat_eval.zone_radii,
            )
            warning_count += zone_features[0]
            restricted_count += zone_features[1]
            critical_count += zone_features[2]
            feature_vector = [
                self._clip_norm(target.distance_m, norm.max_distance_m),
                self._clip_norm(
                    target.radial_closing_speed_m_s, norm.max_closing_speed_m_s
                ),
                self._clip_norm(
                    result.breakthrough_time_s, norm.max_breakthrough_time_s
                ),
                self._clip_norm(result.time_to_engage_s, norm.max_time_to_engage_s),
                self._clip_norm(target.damage_weight, norm.max_damage_weight),
                float(target.confidence),
                self._signed_norm(target.yaw_error_rad, norm.max_angle_rad),
                self._signed_norm(target.pitch_error_rad, norm.max_angle_rad),
                float(target.bbox_area_norm),
                self._clip_norm(
                    target.track_observations, norm.max_track_observations
                ),
                zone_features[0],
                zone_features[1],
                zone_features[2],
                zone_features[3],
            ]
            if self._learned_target_feature_size > len(feature_vector):
                raise ValueError(
                    f"Model expects {self._learned_target_feature_size} target features but runtime provides only {len(feature_vector)}"
                )
            target_features[0, idx, :] = np.array(
                feature_vector[: self._learned_target_feature_size],
                dtype=np.float32,
            )

        global_vector = [
            len(planner_targets) / max(1, self._learned_max_targets),
            min(breakthroughs) if breakthroughs else 0.0,
            float(np.mean(engage_times)) if engage_times else 0.0,
            self._clip_norm(
                total_damage_weight,
                norm.max_damage_weight * self._learned_max_targets,
            ),
            warning_count / max(1, len(planner_targets)),
            restricted_count / max(1, len(planner_targets)),
            critical_count / max(1, len(planner_targets)),
        ]
        if self._learned_global_feature_size > len(global_vector):
            raise ValueError(
                f"Model expects {self._learned_global_feature_size} global features but runtime provides only {len(global_vector)}"
            )
        global_features = np.array(
            [[value for value in global_vector[: self._learned_global_feature_size]]],
            dtype=np.float32,
        )
        return target_features, global_features, target_mask

    def _apply_learned_hysteresis(
        self,
        chosen: PlannerCandidateResult,
        results: Sequence[PlannerCandidateResult],
        *,
        previous_target_id: Optional[int],
    ) -> PlannerCandidateResult:
        if previous_target_id is None:
            return chosen

        previous = next((item for item in results if item.target_id == previous_target_id), None)
        if previous is None or previous.target_id == chosen.target_id:
            return chosen
        if not previous.engageable_now:
            return chosen

        improvement = previous.expected_total_damage - chosen.expected_total_damage
        relative = improvement / max(previous.expected_total_damage, _EPS)
        if (
            improvement >= self._settings.switch_absolute_damage_gain
            or relative >= self._settings.switch_relative_improvement
        ):
            return chosen
        return previous

    def _track_key(self, index: int, box: Box) -> int:
        if box.track_id is not None:
            return int(box.track_id)
        return -(index + 1)

    def _is_hostile(self, box: Box) -> bool:
        if box.threat_level is None:
            return True
        return box.threat_level in self._swarm_config.hostile_levels

    def _is_excluded_target_class(self, box: Box) -> bool:
        cls_name = str(box.cls).strip().lower()
        if not cls_name:
            return False
        return cls_name in {
            excluded.strip().lower()
            for excluded in self._swarm_config.excluded_target_classes
        }

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

    def _classify_live_threat_level(
        self,
        *,
        distance_m: float,
        radial_closing_speed_m_s: float,
        yaw_error_rad: float,
        pitch_error_rad: float,
        confidence: float,
        track_observations: int,
        range_source: Optional[str],
        tracker_mode: Optional[str],
        predictive_only: bool,
        current_yaw_rate_rad_s: float,
        current_pitch_rate_rad_s: float,
    ) -> str:
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            return "benign"

        if self._control_config.threat_eval.enabled and self._control_config.threat_eval.zone_radii:
            try:
                zone_id = get_zone_id_for_distance(
                    distance_m,
                    self._control_config.threat_eval.zone_radii,
                )
            except ValueError:
                zone_id = "normal"
            if zone_id == "critical":
                return "threatening"
            if zone_id == "restricted" and radial_closing_speed_m_s > 0.0:
                return "threatening"
            if zone_id == "warning":
                return "suspicious"

        breakthrough_time_s = compute_breakthrough_time(distance_m, radial_closing_speed_m_s)
        time_to_engage_s = estimate_time_to_engage(
            yaw_error_rad=yaw_error_rad,
            pitch_error_rad=pitch_error_rad,
            yaw_rate_limit_rad_s=self._settings.yaw_rate_limit_rad_s,
            pitch_rate_limit_rad_s=self._settings.pitch_rate_limit_rad_s,
            yaw_accel_limit_rad_s2=self._settings.yaw_accel_limit_rad_s2,
            pitch_accel_limit_rad_s2=self._settings.pitch_accel_limit_rad_s2,
            current_yaw_rate_rad_s=current_yaw_rate_rad_s,
            current_pitch_rate_rad_s=current_pitch_rate_rad_s,
            tracker_mode=tracker_mode,
            confidence=confidence,
            track_observations=track_observations,
            range_source=range_source,
            predictive_only=predictive_only,
            base_track_lock_s=self._settings.timing.base_track_lock_s,
            search_track_lock_s=self._settings.timing.search_track_lock_s,
            recover_track_lock_s=self._settings.timing.recover_track_lock_s,
            low_conf_threshold=self._settings.timing.low_conf_threshold,
            low_conf_penalty_s=self._settings.timing.low_conf_penalty_s,
            min_track_observations=self._settings.timing.min_track_observations,
            low_continuity_penalty_s=self._settings.timing.low_continuity_penalty_s,
            missing_range_penalty_s=self._settings.timing.missing_range_penalty_s,
            predictive_penalty_s=self._settings.timing.predictive_penalty_s,
            effect_time_s=self._settings.timing.effect_time_s,
            confirm_time_s=self._settings.timing.confirm_time_s,
            settle_margin_s=self._settings.timing.settle_margin_s,
        )
        if not math.isfinite(breakthrough_time_s) or radial_closing_speed_m_s <= 0.05:
            if distance_m <= 20.0:
                return "threatening" if track_observations >= 2 and confidence >= 0.85 else "suspicious"
            if distance_m <= 40.0:
                return "suspicious"
            return "benign"

        engage_margin_s = breakthrough_time_s - time_to_engage_s
        if breakthrough_time_s <= max(5.0, time_to_engage_s + 1.0) or engage_margin_s <= 1.5:
            return "threatening"
        if breakthrough_time_s <= 15.0 or radial_closing_speed_m_s >= 0.5:
            return "suspicious"
        return "benign"

    def _apply_rule_based_threat_annotation(self, box: Box, threat_level: str) -> None:
        box.threat_level = threat_level
        box.threat_confidence = float(np.clip(float(box.conf), 0.0, 1.0))
        box.threat_score_benign = 1.0 if threat_level == "benign" else 0.0
        box.threat_score_suspicious = 1.0 if threat_level == "suspicious" else 0.0
        box.threat_score_threatening = 1.0 if threat_level == "threatening" else 0.0

    def _apply_model_threat_annotation(self, box: Box, track_key: int) -> None:
        prediction = self._latest_model_class_predictions.get(track_key)
        if prediction is None:
            return
        threat_level, confidence, probs = prediction
        box.threat_level = threat_level
        box.threat_confidence = confidence
        box.threat_score_benign = float(probs[0]) if probs.shape[0] > 0 else None
        box.threat_score_suspicious = float(probs[1]) if probs.shape[0] > 1 else None
        box.threat_score_threatening = float(probs[2]) if probs.shape[0] > 2 else None

    def _annotate_boxes(self, msg: DetectionMsg, decision: PlannerDecision) -> None:
        result_by_index = {result.box_index: result for result in decision.candidate_results}
        damages = [result.expected_total_damage for result in decision.candidate_results]
        min_damage = min(damages) if damages else 0.0
        max_damage = max(damages) if damages else 0.0

        for rank, result in enumerate(decision.candidate_results, start=1):
            box = msg.boxes[result.box_index]
            self._apply_model_threat_annotation(box, self._track_key(result.box_index, box))
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
            self._apply_model_threat_annotation(box, self._track_key(index, box))
            if box.damage_weight is None:
                box.damage_weight = self._damage_weight_for_box(box)

        msg.swarm_expected_total_damage = (
            decision.expected_total_damage if decision.chosen_target_id is not None else None
        )

    def _clear_swarm_fields(self, box: Box) -> None:
        box.threat_level = None
        box.threat_confidence = None
        box.threat_score_benign = None
        box.threat_score_suspicious = None
        box.threat_score_threatening = None
        box.priority_score = None
        box.engagement_rank = None
        box.breakthrough_time_s = None
        box.time_to_engage_s = None
        box.engageable_now = None
        box.expected_damage_if_ignored = None
        box.expected_total_damage_if_selected = None

    def _clip_norm(self, value: float, max_value: float) -> float:
        if not math.isfinite(float(value)):
            return 1.0
        return float(np.clip(float(value) / max(max_value, 1e-6), 0.0, 1.0))

    def _signed_norm(self, value: float, max_abs: float) -> float:
        return float(np.clip(float(value) / max(max_abs, 1e-6), -1.0, 1.0))
