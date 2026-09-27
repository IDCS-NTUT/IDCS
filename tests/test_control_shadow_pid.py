from __future__ import annotations

import copy
import json
from pathlib import Path

from common.schemas import ControlObservation
from jetson.control.pid import AxisPIDConfig, BasicPID
from jetson.control.shadow_pid import ShadowPIDController
from jetson.control.timing import ClockBounds
from tools.replay_control_pid import _merge, replay


FIXTURE = Path(__file__).parent / "fixtures" / "control_pid_replay_v1.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_versioned_replay_matches_independent_golden_shadow_intents() -> None:
    fixture = _fixture()
    assert replay(fixture) == fixture["golden"]
    assert all(record["mode"] == "shadow" for record in fixture["golden"])
    assert [record["reason"] for record in fixture["golden"]] == [
        "tracking", "tracking", "frame_identity_unverified", "tracking",
        "target_switch_hold", "tracking", "tracking", "capture_age_out_of_bounds",
    ]


def test_bearing_rate_field_cannot_influence_raw_pid() -> None:
    fixture = _fixture()
    altered = copy.deepcopy(fixture)
    altered["base_observation"]["target"]["bearing_rate_rad_s"] = [-999.0, 500.0]
    assert replay(altered) == fixture["golden"]


def test_source_clock_domain_and_missing_drift_bound_fail_closed() -> None:
    fixture = _fixture()
    fixture["base_observation"]["source_clock_domain"] = "unknown"
    assert replay(fixture)[0]["reason"] == "source_clock_domain_invalid"
    fixture = _fixture()
    fixture["clock_exchange"]["max_drift_ppm"] = None
    assert replay(fixture)[0]["reason"] == "no_clock_drift_bound"


def test_sequence_and_source_frame_regressions_hold() -> None:
    fixture = _fixture()
    fixture["steps"][2]["observation"]["source_identity_verified"] = True
    fixture["steps"][2]["observation"]["source_frame_id"] = 99
    assert replay(fixture)[2]["reason"] == "source_frame_regressed"
    fixture = _fixture()
    fixture["steps"][2]["observation"]["sequence"] = 2
    assert replay(fixture)[2]["reason"] == "observation_sequence_nonmonotonic"


def test_stale_gimbal_and_safety_hold_independently() -> None:
    fixture = _fixture()
    fixture["base_observation"]["gimbal"]["sample_age_ms"] = 101.0
    assert replay(fixture)[0]["reason"] == "gimbal_invalid"
    fixture = _fixture()
    fixture["base_observation"]["safety"]["sample_age_ms"] = 751.0
    assert replay(fixture)[0]["reason"] == "safety_hold"


def test_shadow_intent_expires_at_issue_and_cannot_be_live() -> None:
    fixture = _fixture()
    config = fixture["controller"]
    controller = ShadowPIDController(
        BasicPID(AxisPIDConfig(**config["yaw"]), AxisPIDConfig(**config["pitch"])),
        max_clock_sample_age_ns=config["max_clock_sample_age_ns"],
        max_capture_age_ns=config["max_capture_age_ns"],
        max_gimbal_age_ns=config["max_gimbal_age_ns"],
        max_safety_age_ns=config["max_safety_age_ns"],
    )
    observation = ControlObservation.model_validate(_merge(
        fixture["base_observation"], fixture["steps"][0]["observation"]
    ))
    result = controller.decide(observation, ClockBounds.from_exchange(**fixture["clock_exchange"]))
    assert result.intent.mode == "shadow"
    assert result.intent.valid_until_monotonic_ns == result.intent.issued_monotonic_ns


def test_explicit_feedforward_is_separate_and_shadow_only() -> None:
    fixture = _fixture()
    config = fixture["controller"]
    controllers = [ShadowPIDController(
        BasicPID(AxisPIDConfig(**config["yaw"]), AxisPIDConfig(**config["pitch"])),
        max_clock_sample_age_ns=config["max_clock_sample_age_ns"],
        max_capture_age_ns=config["max_capture_age_ns"],
        max_gimbal_age_ns=config["max_gimbal_age_ns"],
        max_safety_age_ns=config["max_safety_age_ns"],
    ) for _ in range(2)]
    clock = ClockBounds.from_exchange(**fixture["clock_exchange"])
    for step in fixture["steps"][:2]:
        observation = ControlObservation.model_validate(_merge(
            fixture["base_observation"], step["observation"]
        ))
        baseline = controllers[0].decide(observation, clock)
        combined = controllers[1].decide(
            observation, clock, feedforward_rad_s=(0.03, -0.02)
        )
    assert baseline.pid.reason == combined.pid.reason == "tracking"
    assert baseline.pid.yaw.feedforward_rad_s == 0.0
    assert combined.pid.yaw.feedforward_rad_s == 0.03
    assert combined.pid.pitch.feedforward_rad_s == -0.02
    assert combined.intent.mode == "shadow"
    assert combined.intent.valid_until_monotonic_ns == combined.intent.issued_monotonic_ns


def test_p_only_does_not_require_unmeasured_gimbal_rate_but_d_does() -> None:
    fixture = _fixture()
    raw = _merge(fixture["base_observation"], fixture["steps"][0]["observation"])
    raw["gimbal"]["yaw_rate_rad_s"] = None
    raw["gimbal"]["pitch_rate_rad_s"] = None
    observation = ControlObservation.model_validate(raw)
    config = fixture["controller"]
    clock = ClockBounds.from_exchange(**fixture["clock_exchange"])

    def controller(kd: float) -> ShadowPIDController:
        yaw = {**config["yaw"], "kd": kd}
        pitch = {**config["pitch"], "kd": kd}
        return ShadowPIDController(
            BasicPID(AxisPIDConfig(**yaw), AxisPIDConfig(**pitch)),
            max_clock_sample_age_ns=config["max_clock_sample_age_ns"],
            max_capture_age_ns=config["max_capture_age_ns"],
            max_gimbal_age_ns=config["max_gimbal_age_ns"],
            max_safety_age_ns=config["max_safety_age_ns"],
        )

    assert controller(0.0).decide(observation, clock).intent.reason == "tracking"
    assert controller(0.5).decide(observation, clock).intent.reason == "gimbal_invalid"


def test_replay_rejects_unknown_version() -> None:
    fixture = _fixture()
    fixture["schema_version"] = 2
    try:
        replay(fixture)
    except ValueError as error:
        assert "version" in str(error)
    else:
        raise AssertionError("unknown replay version accepted")
