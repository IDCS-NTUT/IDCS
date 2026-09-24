from __future__ import annotations

import json

import pytest

from common.schemas import ControlIntent, control_intent_from_json
from jetson.gimbal_bridge import (
    EncoderAnchoredRenderPredictor,
    LiveIntentGate,
    _encode_timed_speed_cmd,
    _intent_command_priority,
    _quantized_camera_rate,
    _should_forward_intent,
    _wait_for_status,
)


def test_render_predictor_integrates_command_then_reanchors_to_encoder() -> None:
    predictor = EncoderAnchoredRenderPredictor()
    predictor.anchor_pan(1.0, sample_s=10.0)
    predictor.anchor_tilt(-0.5, sample_s=10.0)
    predictor.set_command_rates(0.2, -0.1, at_s=10.0)

    predicted = predictor.pose(at_s=10.05)

    assert predicted is not None
    assert predicted.pan_rad == pytest.approx(1.01)
    assert predicted.tilt_rad == pytest.approx(-0.505)
    assert predicted.encoder_age_s == pytest.approx(0.05)

    predictor.anchor_pan(1.018, sample_s=10.1)
    predictor.anchor_tilt(-0.509, sample_s=10.1)
    corrected = predictor.pose(at_s=10.15)

    assert corrected is not None
    assert corrected.pan_rad == pytest.approx(1.028)
    assert corrected.tilt_rad == pytest.approx(-0.514)
    assert corrected.encoder_age_s == pytest.approx(0.05)
    assert corrected.pan_correction_rad == pytest.approx(-0.002)
    assert corrected.tilt_correction_rad == pytest.approx(0.001)
    assert predictor.correction_stats() == pytest.approx(
        {
            "pan_count": 1,
            "tilt_count": 1,
            "pan_mean_abs_rad": 0.002,
            "tilt_mean_abs_rad": 0.001,
            "pan_max_abs_rad": 0.002,
            "tilt_max_abs_rad": 0.001,
        }
    )


def test_render_predictor_stops_immediately_on_zero_command() -> None:
    predictor = EncoderAnchoredRenderPredictor()
    predictor.anchor_pan(0.0, sample_s=1.0)
    predictor.anchor_tilt(0.0, sample_s=1.0)
    predictor.set_command_rates(0.2, 0.1, at_s=1.0)
    predictor.set_command_rates(0.0, 0.0, at_s=1.05)

    stopped = predictor.pose(at_s=1.2)

    assert stopped is not None
    assert stopped.pan_rad == pytest.approx(0.01)
    assert stopped.tilt_rad == pytest.approx(0.005)
    assert stopped.pan_rate_rad_s == 0.0
    assert stopped.tilt_rate_rad_s == 0.0


def test_render_predictor_requires_both_encoder_axes() -> None:
    predictor = EncoderAnchoredRenderPredictor()
    predictor.anchor_pan(0.0, sample_s=1.0)

    assert predictor.pose(at_s=1.1) is None


def _intent(
    *,
    sequence: int = 1,
    observation_sequence: int = 2,
    issued_ns: int = 1_000_000_000,
    valid_until_ns: int = 1_080_000_000,
    mode: str = "live",
) -> ControlIntent:
    return ControlIntent(
        sequence=sequence,
        observation_sequence=observation_sequence,
        issued_monotonic_ns=issued_ns,
        valid_until_monotonic_ns=valid_until_ns,
        mode=mode,
        yaw_rate_rad_s=0.2,
        pitch_rate_rad_s=-0.1,
        reason="tracking",
    )


def test_control_intent_decoder_requires_versioned_intent() -> None:
    intent = _intent()
    decoded = control_intent_from_json(intent.model_dump_json())
    assert decoded == intent

    payload = json.loads(intent.model_dump_json())
    payload["type"] = "ControlCmd"
    with pytest.raises(ValueError):
        control_intent_from_json(payload)


def test_live_intent_gate_rejects_shadow_expired_and_out_of_order() -> None:
    gate = LiveIntentGate(watchdog_ns=100_000_000)
    assert gate.accept(_intent(mode="shadow"), now_ns=1_010_000_000).reason == "non_live_intent"
    assert gate.accept(
        _intent(valid_until_ns=1_005_000_000), now_ns=1_010_000_000
    ).reason == "intent_expired"

    accepted = gate.accept(_intent(), now_ns=1_010_000_000)
    duplicate = gate.accept(_intent(), now_ns=1_020_000_000)
    regressed_observation = gate.accept(
        _intent(sequence=2, observation_sequence=1), now_ns=1_020_000_000
    )

    assert accepted.accepted is True
    assert duplicate.reason == "intent_out_of_order" and duplicate.stop_required
    assert regressed_observation.reason == "observation_out_of_order"


def test_live_intent_gate_watchdog_requests_one_stop() -> None:
    gate = LiveIntentGate(watchdog_ns=100_000_000)
    assert gate.accept(_intent(), now_ns=1_010_000_000).accepted
    assert not gate.watchdog_stop_required(now_ns=1_070_000_000)
    assert gate.watchdog_stop_required(now_ns=1_081_000_000)
    # The bridge acknowledges the stop only after serial publication succeeds.
    gate.mark_stopped()
    assert not gate.watchdog_stop_required(now_ns=1_200_000_000)


def test_live_intent_gate_retries_watchdog_until_stop_publication_succeeds() -> None:
    gate = LiveIntentGate(watchdog_ns=100_000_000)
    assert gate.accept(_intent(), now_ns=1_010_000_000).accepted

    assert gate.watchdog_stop_required(now_ns=1_081_000_000)
    assert gate.watchdog_stop_required(now_ns=1_082_000_000)
    gate.mark_stopped()
    assert not gate.watchdog_stop_required(now_ns=1_083_000_000)


def test_successful_zero_intent_marks_bridge_stopped_without_extra_watchdog() -> None:
    gate = LiveIntentGate(watchdog_ns=100_000_000)
    assert gate.accept(_intent(), now_ns=1_010_000_000).accepted
    shutdown = _intent(
        sequence=2,
        observation_sequence=2,
        issued_ns=1_020_000_000,
        valid_until_ns=1_070_000_000,
    ).model_copy(
        update={
            "yaw_rate_rad_s": 0.0,
            "pitch_rate_rad_s": 0.0,
            "reason": "controller_shutdown",
        }
    )
    assert gate.accept(shutdown, now_ns=1_020_000_000).accepted

    gate.mark_command_sent(shutdown)

    assert not gate.watchdog_stop_required(now_ns=1_200_000_000)


def test_tracking_commands_coalesce_but_full_stops_remain_critical() -> None:
    assert _intent_command_priority(0.2, 0.0) == "high"
    assert _intent_command_priority(0.0, -0.1) == "high"
    assert _intent_command_priority(0.0, 0.0) == "critical"


def test_repeated_zero_intents_are_suppressed_only_after_confirmed_stop() -> None:
    zero = _intent().model_copy(
        update={
            "yaw_rate_rad_s": 0.0,
            "pitch_rate_rad_s": 0.0,
            "reason": "safety_invalid",
        }
    )

    assert _should_forward_intent(zero, was_stopped=False)
    assert not _should_forward_intent(zero, was_stopped=True)
    assert _should_forward_intent(_intent(), was_stopped=True)


def test_live_intent_gate_accepts_new_timestamp_epoch_after_restart() -> None:
    gate = LiveIntentGate(watchdog_ns=100_000_000)
    assert gate.accept(
        _intent(sequence=100, observation_sequence=100), now_ns=1_010_000_000
    ).accepted
    restarted = _intent(
        sequence=1_700_000_000_001,
        observation_sequence=1_700_000_000_001,
        issued_ns=1_020_000_000,
        valid_until_ns=1_100_000_000,
    )

    assert gate.accept(restarted, now_ns=1_020_000_000).accepted


def test_live_intent_gate_rejects_motion_disguised_as_safety_hold() -> None:
    gate = LiveIntentGate(watchdog_ns=100_000_000)
    intent = _intent().model_copy(update={"reason": "manual_active"})

    result = gate.accept(intent, now_ns=1_010_000_000)

    assert not result.accepted
    assert result.reason == "motion_reason_not_authorized"


def test_timed_f6_payload_appends_big_endian_ten_ms_runtime() -> None:
    payload = _encode_timed_speed_cmd(
        0.2,
        acc=10,
        gear_ratio=1.0,
        max_rate=0.5,
        runtime_ms=300,
    )

    assert len(payload) == 7
    assert payload[-4:] == (0x00, 0x00, 0x00, 0x1E)
    with pytest.raises(ValueError, match="positive"):
        _encode_timed_speed_cmd(
            0.0, acc=10, gear_ratio=1.0, max_rate=0.5, runtime_ms=0
        )


def test_predictor_rate_matches_bounded_quantized_firmware_payload() -> None:
    rate = _quantized_camera_rate(
        -0.5,
        motor_sign=1.0,
        gear_ratio=1.0,
        max_rate=0.2,
    )

    assert rate == pytest.approx(-2.0 * 3.141592653589793 / 60.0)


class _StatusReplies:
    def __init__(self, batches: list[list[dict[str, object]]]) -> None:
        self._batches = iter(batches)

    def recv_nowait(self) -> list[dict[str, object]]:
        return next(self._batches, [])


def test_status_wait_returns_only_missing_addresses_for_bounded_retry() -> None:
    replies = _StatusReplies(
        [[{"func": "F1", "addr": 1, "reply": {"parsed": {"status": 1}}}]]
    )

    missing = _wait_for_status(replies, [1, 2], timeout_s=0.001)  # type: ignore[arg-type]

    assert missing == {2}


def test_status_wait_rejects_explicit_fault_status() -> None:
    replies = _StatusReplies(
        [[{"func": "F1", "addr": 3, "reply": {"parsed": {"status": 0}}}]]
    )

    with pytest.raises(SystemExit, match="status query failed for addr=3"):
        _wait_for_status(replies, [3], timeout_s=0.1)  # type: ignore[arg-type]
