"""Deterministic current-frame range qualification under partial obstruction."""

from pathlib import Path

import pytest

from common.aiming import solve_snapshot_aiming
from common.config import expand_config_dirs, load_config_bundle
from common.control import ControlConfig, LaserMountConfig
from common.perception import NormalizedBoxV2, TargetSelectionV2
from common.synthetic_perception import load_synthetic_scenario, snapshot_at
from jetson.deepstream.async_target_selection import _apply_completed_snapshot
from jetson.deepstream.range_guard import KnownSizeRangeGuard


def _setup():
    config = load_config_bundle(expand_config_dirs([Path("configs/base")])).mutable_copy()
    source = snapshot_at(
        load_synthetic_scenario(Path("tests/fixtures/synthetic_tracking_v1.json")), 2
    )
    guard = KnownSizeRangeGuard(config, {"0": "drone", "1": "person"})
    return config, source, guard


def _frame(source, *, box=None, missed=0, session=None, frame_id=None):
    frame = source.frame.model_copy(update={
        "tracker_session": session if session is not None else source.frame.tracker_session,
        "frame_id": frame_id if frame_id is not None else source.frame.frame_id,
    })
    track = source.tracks[0].model_copy(update={
        "box": box if box is not None else source.tracks[0].box,
        "missed_frames": missed,
    })
    return source.model_copy(update={"frame": frame, "tracks": (track,), "assessments": ()})


def _distance(snapshot):
    return snapshot.assessments[0] if snapshot.assessments else None


def test_partial_obstruction_holds_then_expires_without_refreshing():
    _, source, guard = _setup()
    first = guard.apply(_frame(source), now_s=1.0)
    measured = _distance(first)
    assert measured is not None
    assert measured.distance_quality == "measured"
    assert measured.distance_age_ms == 0.0

    # A narrow yet aspect-valid box would report a much greater distance.
    partial = NormalizedBoxV2(x=0.4, y=0.4, w=0.05, h=0.1)
    held = guard.apply(_frame(source, box=partial), now_s=1.10)
    assert _distance(held).distance_m == measured.distance_m
    assert _distance(held).distance_quality == "held"
    assert _distance(held).distance_age_ms == pytest.approx(100.0)

    expired = guard.apply(_frame(source, box=partial), now_s=1.26)
    assert _distance(expired) is None
    assert _distance(guard.apply(_frame(source, box=partial), now_s=2.10)) is None
    recovered = guard.apply(_frame(source), now_s=2.11)
    assert _distance(recovered).distance_quality == "measured"
    assert _distance(recovered).distance_m == measured.distance_m


def test_edge_aspect_tracker_only_and_new_session_do_not_publish_suspect_range():
    _, source, guard = _setup()
    guard.apply(_frame(source), now_s=1.0)
    edge = NormalizedBoxV2(x=0.0, y=0.4, w=0.1, h=0.1)
    assert _distance(guard.apply(_frame(source, box=edge), now_s=1.1)).distance_quality == "held"
    aspect = NormalizedBoxV2(x=0.4, y=0.4, w=0.1, h=0.02)
    assert _distance(guard.apply(_frame(source, box=aspect), now_s=1.2)).distance_quality == "held"
    assert _distance(guard.apply(_frame(source, missed=1), now_s=1.3)) is None
    assert _distance(guard.apply(_frame(source, box=edge, session="new-session"), now_s=1.31)) is None


def test_old_worker_range_is_requalified_against_current_clipped_frame():
    _, source, guard = _setup()
    prior = guard.apply(_frame(source), now_s=1.0)
    old_assessment = prior.assessments[0].model_copy(update={"priority_score": 0.8})
    completed = prior.model_copy(update={"assessments": (old_assessment,)})
    edge = NormalizedBoxV2(x=0.0, y=0.4, w=0.1, h=0.1)
    current = _frame(source, box=edge, frame_id=3)
    applied, _ = _apply_completed_snapshot(current, completed)
    current_result = guard.apply(applied, now_s=1.10)
    assert _distance(current_result).priority_score == 0.8
    assert _distance(current_result).distance_quality == "held"
    assert _distance(guard.apply(applied, now_s=1.26)).distance_m is None


def test_parallax_uses_held_range_but_never_silent_default_after_expiry():
    config, source, guard = _setup()
    control = ControlConfig.from_raw_config(config, (source.frame.width, source.frame.height))
    mount = LaserMountConfig.from_raw_config(config)
    selection = TargetSelectionV2(
        track_id=source.tracks[0].track_id,
        source_frame_id=source.frame.frame_id,
        applied_frame_id=source.frame.frame_id,
        selected_time_ns=0,
        selection_clock_domain=source.frame.observation_clock_domain,
        policy="test",
    )
    selected = _frame(source).model_copy(update={"selection": selection})
    first = guard.apply(selected, now_s=1.0)
    measured = solve_snapshot_aiming(first, control, mount)
    assert measured.parallax_active
    assert measured.distance_source.startswith("known_size:")

    edge = NormalizedBoxV2(x=0.0, y=0.4, w=0.1, h=0.1)
    partial = selected.model_copy(update={"tracks": (
        selected.tracks[0].model_copy(update={"box": edge}),
    )})
    held = solve_snapshot_aiming(guard.apply(partial, now_s=1.1), control, mount)
    assert held.parallax_active
    assert held.distance_m == measured.distance_m
    assert held.distance_source.startswith("known_size_held:")

    expired = solve_snapshot_aiming(guard.apply(partial, now_s=1.26), control, mount)
    assert expired.distance_m is None
    assert expired.distance_source is None
    assert not expired.parallax_active
    assert expired.aim_px == (control.cx_px, control.cy_px)
