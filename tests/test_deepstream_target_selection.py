import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.camera import CameraIntrinsics
from common.config import expand_config_dirs, load_config_bundle
from common.control import ControlConfig
from common.perception import (
    PerceptionSnapshotV2,
    TargetSelectionV2,
    TrackAssessmentV2,
)
from common.synthetic_perception import load_synthetic_scenario, snapshot_at
from jetson.deepstream import async_target_selection as async_module
from jetson.deepstream.async_target_selection import (
    AsyncDeepStreamTargetSelector,
    _apply_completed_snapshot,
)
from jetson.deepstream.target_selection import (
    DeepStreamTargetSelector,
    normalize_snapshot_class_labels,
)
from jetson.swarm_planner import (
    PlannerDecision,
    PlannerFrameObservation,
    PlannerObservationResult,
    SwarmPlannerRuntime,
)

BASE = expand_config_dirs([Path("configs/base")])


def _synthetic_snapshot(frame_index: int = 2):
    scenario = load_synthetic_scenario(
        Path("tests/fixtures/synthetic_tracking_v1.json")
    )
    return snapshot_at(scenario, frame_index)


def test_person_only_frame_is_unselected_without_loading_learned_runtime():
    selector = DeepStreamTargetSelector.from_paths(
        [
            *BASE,
        ]
    )
    source = _synthetic_snapshot()
    person_tracks = tuple(
        track.model_copy(update={"class_id": "1"})
        for track in source.tracks
    )
    source = source.model_copy(update={"tracks": person_tracks})

    result = selector.select_snapshot(source, now_s=1.0)

    assert result.tracks[0].class_id == "person"
    assert result.selection is None
    assert selector._planner is None


def test_sim_scene_intrinsics_and_known_size_calibration() -> None:
    config = load_config_bundle([*BASE, Path("configs/sim/v2_scene.yaml")]).mutable_copy()
    intrinsics = CameraIntrinsics.from_raw_config(config, (1280, 720))

    # The simulator renders the physical camera contract (135 x 73 deg).
    assert intrinsics.fov_deg == pytest.approx((135.0, 73.0))
    assert config["sim"]["camera"] == {"fov_x_deg": 135.0, "fov_y_deg": 73.0}
    assert config["camera"]["known_size_ranging"]["dimension"] == "width"
    # The ranging size is the detector's box width for this mesh (calibrated
    # against simulator truth), wider than the mesh's side length.
    assert config["camera"]["known_size_ranging"]["class_sizes_m"]["drone"] == 0.52
    assert config["sim"]["scene"]["targets"][0]["width"] == 0.35


def test_label_normalization_is_available_before_async_policy_results():
    source = _synthetic_snapshot()
    tracks = tuple(
        track.model_copy(update={"class_id": "1"})
        for track in source.tracks
    )
    source = source.model_copy(update={"tracks": tracks})

    result = normalize_snapshot_class_labels(source, {"0": "drone", "1": "person"})

    assert source.tracks[0].class_id == "1"
    assert result.tracks[0].class_id == "person"


def test_v2_selection_uses_guaranteed_synthetic_track_without_model_runtime():
    paths = [
        *BASE,
    ]
    config = load_config_bundle(paths).data
    calls = []

    class DeterministicPlanner:
        def update_and_select_snapshot(self, snapshot, **kwargs):
            calls.append((snapshot, kwargs))
            ranged = snapshot.assessments[0]
            assessment = TrackAssessmentV2.model_validate({
                **ranged.model_dump(mode="json"),
                "priority_score": 0.75,
            })
            return SimpleNamespace(
                selected_track_id=41,
                assessments=(assessment,),
            )

    selector = DeepStreamTargetSelector(
        config,
        planner_factory=lambda _control: DeterministicPlanner(),
    )
    source = _synthetic_snapshot()

    result = selector.select_snapshot(source, now_s=12.5)

    assert source.selection is None
    assert result.selection is not None
    assert result.selection.track_id == 41
    assert result.selection.source_frame_id == 2
    assert result.selection.applied_frame_id == 2
    assert result.selection.selected_time_ns == 12_500_000_000
    assert result.selection.selection_clock_domain == "synthetic"
    assert result.assessments[0].track_id == 41
    assert result.assessments[0].distance_m is not None
    assert result.assessments[0].priority_score == 0.75
    assert calls[0][0].tracks[0].track_id == 41
    assert calls[0][0].tracks[0].class_id == "drone"


def test_swarm_runtime_v2_adapter_returns_immutable_track_assessments():
    runtime = object.__new__(SwarmPlannerRuntime)
    calls = []

    def deterministic_update(observation, **kwargs):
        calls.append((observation, kwargs))
        decision = PlannerDecision(
            chosen_target_id=41,
            chosen_box_index=0,
            expected_total_damage=1.25,
            candidate_results=(),
        )
        return PlannerObservationResult(
            decision=decision,
            assessments=(TrackAssessmentV2(
                track_id=41,
                threat_level="threatening",
                priority_score=0.75,
            ),),
        )

    runtime.update_and_select_observation = deterministic_update
    source = _synthetic_snapshot()

    result = runtime.update_and_select_snapshot(
        source,
        current_time_s=12.5,
        previous_target_id=None,
    )

    assert result.selected_track_id == 41
    assert result.decision.expected_total_damage == 1.25
    assert result.assessments == (
        TrackAssessmentV2(
            track_id=41,
            threat_level="threatening",
            priority_score=0.75,
        ),
    )
    assert isinstance(calls[0][0], PlannerFrameObservation)
    assert calls[0][0].tracks[0].track_id == 41
    assert calls[0][0].tracks[0].class_id == "drone"
    assert source.assessments == ()


def test_v2_selector_composes_with_real_rule_planner_on_synthetic_track():
    paths = [
        *BASE,
    ]
    config = load_config_bundle(paths).mutable_copy()
    config["swarm_eval"]["learned_model"]["enabled"] = False
    selector = DeepStreamTargetSelector(config)

    result = selector.select_snapshot(_synthetic_snapshot(), now_s=12.5)

    assert result.selection is not None
    assert result.selection.track_id == 41
    assert result.assessments[0].track_id == 41
    assert result.assessments[0].distance_m is not None
    assert result.assessments[0].distance_src == "width"
    assert result.assessments[0].threat_level == "suspicious"
    assert result.assessments[0].priority_score == 1.0


def test_async_selector_passes_one_hashed_config_snapshot_to_worker(monkeypatch):
    class FakeQueue:
        def __init__(self):
            self.values = []

        def put_nowait(self, value):
            self.values.append(value)

        def get_nowait(self):
            raise queue.Empty

        def cancel_join_thread(self):
            self.join_cancelled = True

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon
            self.alive = False

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            self.alive = False

        def terminate(self):
            self.alive = False

    class FakeContext:
        def __init__(self):
            self.process = None

        def Queue(self, maxsize):
            return FakeQueue()

        def Process(self, **kwargs):
            self.process = FakeProcess(**kwargs)
            return self.process

    context = FakeContext()
    monkeypatch.setattr(async_module.mp, "get_context", lambda _method: context)
    selector = AsyncDeepStreamTargetSelector([
        *BASE,
    ])

    assert context.process is not None
    config_snapshot, digest, _requests, _results = context.process.args
    assert isinstance(config_snapshot, dict)
    assert len(digest) == 64
    assert selector.report()["config_digest"] == digest

    selector.close()


def test_async_result_preserves_decision_source_and_names_application_frame():
    scenario = load_synthetic_scenario(
        Path("tests/fixtures/synthetic_tracking_v1.json")
    )
    source = snapshot_at(scenario, 2)
    current = snapshot_at(scenario, 4)
    completed = source.model_copy(update={
        "assessments": (TrackAssessmentV2(track_id=41, priority_score=0.75),),
        "selection": TargetSelectionV2(
            track_id=41,
            source_frame_id=2,
            applied_frame_id=2,
            selected_time_ns=1_205_000_000,
            selection_clock_domain="synthetic",
            policy="deterministic_test",
        ),
    })

    result, selected = _apply_completed_snapshot(current, completed)

    assert selected
    assert result.selection is not None
    assert result.selection.source_frame_id == 2
    assert result.selection.applied_frame_id == 4
    assert result.assessments[0].track_id == 41


def test_async_result_drops_decision_when_track_is_no_longer_present():
    scenario = load_synthetic_scenario(
        Path("tests/fixtures/synthetic_tracking_v1.json")
    )
    source = snapshot_at(scenario, 2)
    current = snapshot_at(scenario, 4).model_copy(update={"tracks": ()})
    completed = source.model_copy(update={
        "assessments": (TrackAssessmentV2(track_id=41, priority_score=0.75),),
        "selection": TargetSelectionV2(
            track_id=41,
            source_frame_id=2,
            applied_frame_id=2,
            selected_time_ns=1_205_000_000,
            selection_clock_domain="synthetic",
            policy="deterministic_test",
        ),
    })

    result, selected = _apply_completed_snapshot(current, completed)

    assert not selected
    assert result.selection is None
    assert result.assessments == ()
