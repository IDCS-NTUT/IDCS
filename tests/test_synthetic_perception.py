from pathlib import Path

import pytest
from pydantic import ValidationError

from common.perception import (
    NormalizedBoxV2,
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
    TargetSelectionV2,
)
from common.synthetic_perception import (
    SyntheticPerceptionScenario,
    SyntheticScenarioError,
    SyntheticTrackSpec,
    iter_deliveries,
    load_synthetic_scenario,
    snapshot_at,
    synthetic_scenario_digest,
    to_legacy_detection_msg,
)


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_tracking_v1.json"


def test_fixture_has_deterministic_occlusion_and_identity_switch():
    scenario = load_synthetic_scenario(FIXTURE)

    assert len(synthetic_scenario_digest(scenario)) == 64
    assert synthetic_scenario_digest(scenario) == synthetic_scenario_digest(
        load_synthetic_scenario(FIXTURE)
    )
    assert snapshot_at(scenario, 0).tracks[0].track_id == 41
    assert snapshot_at(scenario, 3).tracks == ()
    assert snapshot_at(scenario, 4).tracks[0].track_id == 41
    assert snapshot_at(scenario, 5).tracks[0].track_id == 42
    assert snapshot_at(scenario, 5).tracks[0].box.x == pytest.approx(0.4)
    assert len(synthetic_scenario_digest(scenario)) == 64


def test_delivery_faults_are_explicit_reproducible_and_detector_free():
    scenario = load_synthetic_scenario(FIXTURE)
    deliveries = iter_deliveries(scenario)

    assert [(item.snapshot.frame.frame_id, item.duplicate_index) for item in deliveries] == [
        (0, 0), (2, 0), (2, 1), (3, 0), (1, 0), (4, 0),
        (5, 0), (7, 0), (8, 0), (9, 0),
    ]
    assert all(item.snapshot.frame.source_clock_domain == "synthetic" for item in deliveries)


def test_legacy_adapter_can_supply_tracks_or_raw_detections():
    snapshot = snapshot_at(load_synthetic_scenario(FIXTURE), 2)

    tracked = to_legacy_detection_msg(snapshot)
    raw = to_legacy_detection_msg(snapshot, use_tracks=False)

    assert tracked.boxes[0].track_id == 41
    assert raw.boxes[0].track_id is None
    assert tracked.boxes[0].conf == 1.0
    assert tracked.frame_id == 2


def test_snapshot_rejects_selection_not_tied_to_present_track():
    frame = PerceptionFrameV2(
        frame_id=1,
        source_time_ns=1,
        observed_time_ns=2,
        source_clock_domain="synthetic",
        observation_clock_domain="synthetic",
        width=640,
        height=480,
    )
    with pytest.raises(ValidationError, match="identify a track"):
        PerceptionSnapshotV2(
            sequence=1,
            frame=frame,
            tracks=(),
            selection=TargetSelectionV2(
                track_id=99,
                source_frame_id=1,
                selected_time_ns=2,
                policy="test",
            ),
        )


def test_scenario_fails_when_scripted_target_leaves_frame():
    scenario = SyntheticPerceptionScenario(
        seed=0,
        frame_count=2,
        fps=10,
        width=640,
        height=480,
        targets=(SyntheticTrackSpec(
            logical_id="bad",
            track_id=1,
            class_id="drone",
            first_frame=0,
            last_frame=1,
            start_center_norm=(0.95, 0.5),
            velocity_norm_per_frame=(0.1, 0.0),
            size_norm=(0.2, 0.2),
        ),),
    )

    with pytest.raises(SyntheticScenarioError, match="leaves the frame"):
        snapshot_at(scenario, 0)


def test_normalized_box_is_strict_and_snapshot_is_frozen():
    with pytest.raises(ValidationError, match="inside the source frame"):
        NormalizedBoxV2(x=0.9, y=0.1, w=0.2, h=0.2)

    snapshot = snapshot_at(load_synthetic_scenario(FIXTURE), 0)
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.sequence = 9
