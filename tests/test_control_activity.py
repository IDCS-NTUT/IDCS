import json

import pytest

from common.control_activity import ActivityCache, ActivityEnvelope, IntentActivityMonitor
from common.schemas import CamState, ControlIntent
from common.pc_mode import PcModePoll, parse_poll
from jetson.pc_mode_channel import PcModeChannel
from pc.mode_agent import SimActivitySampler


def intent(*, mode="live", reason="tracking", sequence=1):
    return ControlIntent(sequence=sequence, observation_sequence=sequence, issued_monotonic_ns=1_000_000_000,
        valid_until_monotonic_ns=1_100_000_000, mode=mode, yaw_rate_rad_s=0.0,
        pitch_rate_rad_s=0.0, reason=reason)


def test_sink_monitor_requires_applied_live_unexpired_intent_including_zero_rate():
    monitor = IntentActivityMonitor("sim", session="sink")
    assert not monitor.sample(now_ns=1_000_000_000).applied
    monitor.mark_applied(intent())
    sample = monitor.sample(now_ns=1_050_000_000)
    assert sample.applied and sample.reason == "tracking"  # centred target, zero rates
    assert not monitor.sample(now_ns=1_100_000_001).applied
    assert not monitor.sample(now_ns=999_999_999).applied
    assert not monitor.sample(now_ns=1_050_000_000, ready=False).applied
    monitor.mark_applied(intent(mode="shadow"))
    assert not monitor.sample(now_ns=1_050_000_000).applied
    monitor.clear()
    assert not monitor.sample(now_ns=1_050_000_000).applied


def test_cache_rejects_duplicates_out_of_order_wrong_backend_and_retired_session():
    monitor = IntentActivityMonitor("sim", session="old")
    monitor.mark_applied(intent())
    first = ActivityEnvelope(activity=monitor.sample(now_ns=1_050_000_000))
    cache = ActivityCache("sim")
    assert cache.accept(first, 0)
    assert cache.tracking(0)
    assert not cache.accept(first, .49)
    assert not cache.fresh(.5)
    new = IntentActivityMonitor("sim", session="new")
    new.mark_applied(intent())
    assert cache.accept(ActivityEnvelope(activity=new.sample(now_ns=1_050_000_000)), .6)
    assert not cache.accept(ActivityEnvelope(activity=monitor.sample(now_ns=1_050_000_000)), .7)
    other = IntentActivityMonitor("gimbal", session="bridge")
    assert not cache.accept(ActivityEnvelope(activity=other.sample(now_ns=1)), .8)


def test_pc_sampler_never_renews_dead_sink_from_live_agent_polls():
    monitor = IntentActivityMonitor("sim", session="sink")
    monitor.mark_applied(intent())
    sample = monitor.sample(now_ns=1_020_000_000)
    state = CamState(frame_id=1, src_ts_ms=1, pan=0, tilt=0, control_activity=sample)
    sampler = SimActivitySampler()
    sampler.receive(state.model_dump_json().encode(), 1_040_000_000)
    assert sampler.sample(1_060_000_000).age_ms == pytest.approx(40)
    assert sampler.sample(1_521_000_000) is None
    sampler.receive(state.model_dump_json().encode(), 1_522_000_000)
    assert sampler.sample(1_523_000_000) is None


def test_pc_protocol_rejects_wrong_backend_and_invalid_ages():
    monitor = IntentActivityMonitor("sim", session="sink")
    monitor.mark_applied(intent())
    envelope = ActivityEnvelope(activity=monitor.sample(now_ns=1_010_000_000))
    poll = PcModePoll(frozenset(), session="agent", activity=envelope)
    assert parse_poll(poll.to_json()) == poll
    raw = json.loads(poll.to_json())
    raw["activity"]["activity"]["backend"] = "gimbal"
    with pytest.raises(ValueError):
        parse_poll(json.dumps(raw))
    raw["activity"]["activity"]["backend"] = "sim"
    for age in (-1.0, float("nan"), float("inf")):
        raw["activity"]["age_ms"] = age
        with pytest.raises(ValueError):
            parse_poll(json.dumps(raw))


def test_full_sink_to_pc_poll_to_jetson_to_pi_contract_without_remote_clock_guess():
    from common.control_activity import SimActivityRelay
    from rpi.panel_indicators import PanelIndicators

    monitor = IntentActivityMonitor("sim", session="sink")
    monitor.mark_applied(intent())
    pc = SimActivitySampler()
    activity = monitor.sample(now_ns=1_020_000_000)
    pc.receive(CamState(frame_id=1, src_ts_ms=1, pan=0, tilt=0,
        control_activity=activity).model_dump_json().encode(), 1_040_000_000)
    now_jetson = [100.0]  # arbitrarily offset; never compared against PC timestamps
    channel = PcModeChannel(clock=lambda: now_jetson[0])
    units = frozenset({"idcs-sim-streamer.service", "idcs-sim-controller.service"})
    channel.poll(PcModePoll(units, session="agent", activity=pc.sample(1_060_000_000)).to_json())
    relay = SimActivityRelay(activity=channel.activity.forward(100.02)).model_dump_json()
    pi = PanelIndicators()
    pi.on_sim_activity(relay, 300.0)  # another unrelated local clock
    assert pi.state(300.0).safety_lamp
    assert not pi.state(300.45).safety_lamp  # 60ms upstream age + 450ms Pi age
    now_jetson[0] = 100.48
    channel.poll(PcModePoll(units, session="agent", activity=pc.sample(1_060_000_000)).to_json())
    assert not channel.activity.fresh(100.48)  # same sequence cannot prolong tracking


@pytest.mark.parametrize("units", [frozenset(), frozenset({"idcs-sim-streamer.service"}),
    frozenset({"idcs-sim-streamer.service", "idcs-sim-controller.service", "idcs-hil-streamer.service"}),
    frozenset({"idcs-sim-streamer.service", "idcs-sim-controller.service",
               "idcs-sim-swarm-streamer.service", "idcs-sim-swarm-controller.service"})])
def test_pc_and_jetson_reject_no_owner_conflicting_owners_and_hil(units):
    from pc.mode_agent import PcModeWorker
    from common.pc_mode import PC_STATUS_UNITS

    monitor = IntentActivityMonitor("sim", session="sink")
    monitor.mark_applied(intent())
    envelope = ActivityEnvelope(activity=monitor.sample(now_ns=1_010_000_000))
    def run(args, **_):
        assert args == ["is-active", *PC_STATUS_UNITS]
        return 0, "\n".join("active" if unit in units else "inactive" for unit in PC_STATUS_UNITS)
    assert PcModeWorker(run=run).poll(envelope).activity is None
    channel = PcModeChannel(clock=lambda: 0)
    channel.poll(PcModePoll(units, session="agent", activity=envelope).to_json())
    assert not channel.activity.tracking(0)


@pytest.mark.parametrize("prefix", ["idcs-sim", "idcs-sim-swarm"])
def test_exactly_one_sim_owner_forwards_positive_receiver_proof(prefix):
    from pc.mode_agent import PcModeWorker
    from common.pc_mode import PC_STATUS_UNITS

    units = frozenset({f"{prefix}-streamer.service", f"{prefix}-controller.service"})
    def run(args, **_):
        assert args == ["is-active", *PC_STATUS_UNITS]
        return 0, "\n".join("active" if unit in units else "inactive" for unit in PC_STATUS_UNITS)
    monitor = IntentActivityMonitor("sim", session="sink")
    monitor.mark_applied(intent())
    envelope = ActivityEnvelope(activity=monitor.sample(now_ns=1_010_000_000))
    poll = PcModeWorker(run=run).poll(envelope)
    assert poll.activity == envelope
    channel = PcModeChannel(clock=lambda: 0)
    channel.poll(poll.to_json())
    assert channel.activity.tracking(0)


def test_slow_systemd_query_cannot_send_an_already_expired_cached_activity():
    from pc.mode_agent import PcModeWorker
    from common.pc_mode import PC_STATUS_UNITS

    monitor = IntentActivityMonitor("sim", session="sink")
    monitor.mark_applied(intent())
    sampler = SimActivitySampler()
    sampler.receive(CamState(frame_id=1, src_ts_ms=1, pan=0, tilt=0,
        control_activity=monitor.sample(now_ns=1_020_000_000)).model_dump_json().encode(), 1_040_000_000)
    now = [1_050_000_000]
    units = {"idcs-sim-streamer.service", "idcs-sim-controller.service"}
    def slow_query(args, **_):
        now[0] = 1_800_000_000
        return 0, "\n".join("active" if unit in units else "inactive" for unit in PC_STATUS_UNITS)
    assert PcModeWorker(run=slow_query).poll(lambda: sampler.sample(now[0])).activity is None
