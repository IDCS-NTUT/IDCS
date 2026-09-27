from __future__ import annotations

import numpy as np

from common.perception import (
    NormalizedBoxV2,
    PerceptionFrameV2,
    PerceptionSnapshotV2,
    PerceptionTrackV2,
    TargetSelectionV2,
    TrackAssessmentV2,
)
from common.schemas import (
    CamState,
    ControlCmd,
    ControlDiagnostics,
    ControlEstimatorAxisDiagnostics,
    ControlTimingDiagnostics,
)
from pc.v2_hud import (
    V2HudRenderer,
    elevation_tape_ticks,
    heading_tape_ticks,
    resolve_hud_fov,
)


def _snapshot() -> PerceptionSnapshotV2:
    frame = PerceptionFrameV2(
        frame_id=42,
        source_time_ns=1_000_000,
        received_time_ns=2_000_000,
        observed_time_ns=2_100_000,
        source_clock_domain="pc_monotonic",
        receive_clock_domain="jetson_monotonic",
        observation_clock_domain="jetson_monotonic",
        width=1280,
        height=720,
    )
    tracks = (
        PerceptionTrackV2(
            track_id=7,
            box=NormalizedBoxV2(x=0.62, y=0.30, w=0.12, h=0.16),
            class_id="drone",
            confidence=0.91,
            age_frames=15,
            missed_frames=0,
        ),
        PerceptionTrackV2(
            track_id=8,
            box=NormalizedBoxV2(x=0.34, y=0.45, w=0.08, h=0.32),
            class_id="person",
            confidence=0.87,
            age_frames=11,
            missed_frames=0,
        ),
    )
    return PerceptionSnapshotV2(
        sequence=42,
        frame=frame,
        tracks=tracks,
        assessments=(
            TrackAssessmentV2(
                track_id=7,
                distance_m=12.4,
                distance_src="width",
                threat_level="threatening",
                engagement_rank=1,
            ),
        ),
        selection=TargetSelectionV2(
            track_id=7,
            source_frame_id=42,
            applied_frame_id=42,
            selected_time_ns=2_100_000,
            selection_clock_domain="jetson_monotonic",
            policy="synthetic_guaranteed_selection",
        ),
    )


def _command() -> ControlCmd:
    return ControlCmd(
        frame_id=42,
        src_ts_ms=1,
        cmd_ts_ms=2,
        target_ok=True,
        target_uv=(870.0, 274.0),
        err_uv=(230.0, -86.0),
        err_rad=(0.18, 0.11),
        pan_rate_cmd=0.12,
        tilt_rate_cmd=0.07,
        controller_mode="mpc",
        laser_origin_px=(640.0, 690.0),
        laser_dot_px=(870.0, 274.0),
        laser_on_target=False,
        mpc={
            "yaw": {
                "status": "solved",
                "terms": {"theta": 99.0, "effort": 88.0},
            },
        },
    )


def _diagnostics() -> ControlDiagnostics:
    def axis(feedforward: float) -> ControlEstimatorAxisDiagnostics:
        return ControlEstimatorAxisDiagnostics(
            estimator_enabled=True,
            feedforward_term_rad_s=feedforward,
            desired_rate_pre_limit_rad_s=feedforward,
            desired_rate_post_limit_rad_s=feedforward,
            final_rate_rad_s=feedforward,
        )

    return ControlDiagnostics(
        observation_sequence=42,
        intent_sequence=43,
        created_monotonic_ns=3_000_000,
        reason="tracking",
        track_id=7,
        timing=ControlTimingDiagnostics(),
        yaw=axis(0.12),
        pitch=axis(-0.04),
    )


def test_operational_hud_inventory_is_complete_and_excludes_mpc_terms() -> None:
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    before = frame.copy()
    renderer = V2HudRenderer(hfov_deg=91.5, vfov_deg=60.0, authority_label="SIM")
    report = renderer.render(
        frame,
        snapshot=_snapshot(),
        cam_state=CamState(
            frame_id=42,
            src_ts_ms=1,
            pan=0.18,
            tilt=0.11,
            pan_rate=0.02,
            tilt_rate=-0.01,
        ),
        control_cmd=_command(),
        control_diagnostics=_diagnostics(),
        cam_state_age_s=0.012,
        control_age_s=0.008,
        diagnostics_age_s=0.006,
    )

    assert set(report.elements) == {
        "center_reticle",
        "feedforward_indicator",
        "attitude",
        "tracks",
        "track_history",
        "selection",
        "range",
        "threat",
        "rank",
        "control_status",
        "parallax_status",
        "parallax_cue",
        "freshness",
    }
    assert all("mpc" not in element.lower() for element in report.elements)
    assert report.feedforward_state == "active"
    assert report.feedforward_yaw_rad_s == 0.12
    assert report.feedforward_pitch_rad_s == -0.04
    assert np.count_nonzero(frame != before) > 3000


def test_hud_keeps_safe_minimum_without_optional_telemetry() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    report = V2HudRenderer(hfov_deg=90.0, vfov_deg=60.0).render(
        frame,
        snapshot=None,
        cam_state=None,
        control_cmd=None,
    )
    assert report.elements == ("center_reticle", "feedforward_indicator")
    assert report.feedforward_state == "unavailable"
    assert np.count_nonzero(frame) > 0


def test_feedforward_indicator_exposes_stale_telemetry() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    report = V2HudRenderer(hfov_deg=90.0, vfov_deg=60.0).render(
        frame,
        snapshot=None,
        cam_state=None,
        control_cmd=None,
        control_diagnostics=_diagnostics(),
        diagnostics_age_s=0.251,
    )

    assert report.feedforward_state == "stale"
    assert report.feedforward_yaw_rad_s == 0.12
    assert report.feedforward_pitch_rad_s == -0.04


def test_simulator_fov_overrides_real_camera_and_control_fov() -> None:
    config = {
        "sim": {"camera": {"fov_y_deg": 60.0}},
        "camera": {"intrinsics": {"fov_deg": {"h": 135.0, "v": 73.0}}},
        "control": {"fov_deg": {"h": 120.0, "v": 70.0}},
    }
    hfov, vfov = resolve_hud_fov(config, (1280, 720))
    assert abs(hfov - 91.49284451967722) < 1e-9
    assert vfov == 60.0


def test_heading_ticks_slide_continuously_instead_of_relabeling_fixed_slots() -> None:
    before = {tick.value_deg: tick.position_px for tick in heading_tape_ticks(10.0, 90.0, 1280)}
    after = {tick.value_deg: tick.position_px for tick in heading_tape_ticks(10.25, 90.0, 1280)}
    common = sorted(set(before) & set(after))
    assert common
    shifts = [after[value] - before[value] for value in common]
    assert all(shift < 0.0 for shift in shifts)
    assert max(shifts) - min(shifts) < 1e-9
    assert abs(shifts[0]) > 1.0


def test_elevation_ticks_slide_continuously_with_sub_step_pitch() -> None:
    before = {tick.value_deg: tick.position_px for tick in elevation_tape_ticks(4.0, 60.0, 110, 700)}
    after = {tick.value_deg: tick.position_px for tick in elevation_tape_ticks(4.25, 60.0, 110, 700)}
    common = sorted(set(before) & set(after))
    assert common
    shifts = [after[value] - before[value] for value in common]
    assert all(shift > 0.0 for shift in shifts)
    assert max(shifts) - min(shifts) < 1e-9
    assert abs(shifts[0]) > 1.0


def test_hud_draws_full_operator_inventory_from_diagnostics_without_control_cmd() -> None:
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    diagnostics = _diagnostics().model_copy(update={
        "target_center_norm": (0.55, 0.45), "aim_reference_norm": (0.5, 0.5),
    })
    report = V2HudRenderer(hfov_deg=135.0, vfov_deg=73.0, authority_label="HIL").render(
        frame, snapshot=_snapshot(),
        cam_state=CamState(frame_id=42, src_ts_ms=1, pan=0.1, tilt=0.0),
        control_cmd=None, control_diagnostics=diagnostics,
        cam_state_age_s=0.01, diagnostics_age_s=0.02,
    )
    assert {"control_status", "parallax_status", "parallax_cue", "freshness",
            "feedforward_indicator"} <= set(report.elements)
    assert report.feedforward_state == "active"


def test_stale_diagnostics_hold_state_and_hide_the_aim_cue() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    diagnostics = _diagnostics().model_copy(update={
        "target_center_norm": (0.55, 0.45), "aim_reference_norm": (0.5, 0.5),
    })
    report = V2HudRenderer(hfov_deg=90.0, vfov_deg=60.0).render(
        frame, snapshot=None, cam_state=None, control_cmd=None,
        control_diagnostics=diagnostics, diagnostics_age_s=1.0,
    )
    assert "control_status" in report.elements and "parallax_cue" not in report.elements


def test_hud_fov_uses_independent_sim_camera_axes() -> None:
    config = {"sim": {"camera": {"fov_x_deg": 135.0, "fov_y_deg": 73.0}}}
    assert resolve_hud_fov(config, (1280, 720)) == (135.0, 73.0)
