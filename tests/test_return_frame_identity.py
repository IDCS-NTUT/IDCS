"""Exact return-video to source-frame join contracts."""

from common.return_frame_identity import ReturnFrameIdentity
from common.rtp_identity import RtpFrameKey, RtpPacketIdentity


def _header(frame_id: int = 42, session: str = "sim-a") -> dict:
    return {
        "rtp_ssrc": 123,
        "rtp_timestamp": 456,
        "frame_id": frame_id,
        "source_time_ns": 789_000_000,
        "source_clock_domain": "pc_monotonic",
        "tracker_session": session,
    }


def test_exact_return_key_and_pts_are_required() -> None:
    identity = ReturnFrameIdentity()
    assert identity.push_header(_header())
    identity.push_marker(1000, RtpFrameKey(123, 456))
    identity.push_decoded(1000)
    assert identity.displayed(999, 1.0) is None
    frame = identity.displayed(1000, 1.01)
    assert frame is not None
    assert (frame.tracker_session, frame.frame_id) == ("sim-a", 42)
    assert identity.latest(1.2) == frame
    assert identity.latest(1.27) is None
    assert identity.displayed(1000, 1.3) is None  # replayed PTS


def test_ambiguous_return_mapping_and_session_change_fail_closed() -> None:
    identity = ReturnFrameIdentity()
    identity.push_marker(1000, RtpFrameKey(123, 456))
    identity.push_decoded(1000)
    assert identity.push_header(_header())
    assert not identity.push_header(_header(session="sim-b"))
    assert identity.displayed(1000, 1.0) is None
    assert identity.latest(1.0) is None


def test_missing_header_and_source_restart_clear_identity() -> None:
    identity = ReturnFrameIdentity()
    identity.push_marker(1000, RtpFrameKey(123, 456))
    identity.push_decoded(1000)
    assert identity.displayed(1000, 1.0) is None
    assert identity.latest(1.0) is None
    assert not identity.push_header({"frame_id": 1})
    assert identity.push_header(_header())
    assert identity.latest(1.1) is None  # late sideband cannot relabel passed pixels
    identity.push_marker(1001, RtpFrameKey(123, 456))
    assert identity.displayed(1001, 1.2) is None  # reused RTP key
    identity.clear()
    assert identity.latest(1.2) is None


def test_unidentified_new_output_invalidates_prior_display() -> None:
    identity = ReturnFrameIdentity()
    identity.push_header(_header())
    identity.push_marker(1000, RtpFrameKey(123, 456))
    identity.push_decoded(1000)
    assert identity.displayed(1000, 1.0) is not None
    assert identity.displayed(1001, 1.01) is None
    assert identity.latest(1.02) is None


def test_rtp_marker_without_pts_uses_only_same_key_packet_pts() -> None:
    identity = ReturnFrameIdentity()
    key = RtpFrameKey(123, 456)
    identity.push_header(_header())
    identity.push_rtp_packet(1000, RtpPacketIdentity(key, 1, False))
    identity.push_rtp_packet(None, RtpPacketIdentity(key, 2, True))
    identity.push_decoded(1000)
    assert identity.displayed(1000, 1.0).frame_id == 42


def test_output_without_pts_clears_prior_selection():
    identity = ReturnFrameIdentity()
    identity.push_header(_header())
    identity.push_marker(1000, RtpFrameKey(123, 456))
    identity.push_decoded(1000)
    assert identity.displayed(1000, 1.0) is not None
    identity.unidentified_output()
    assert identity.latest(1.01) is None


def test_fallback_output_cannot_use_return_rtp_metadata_without_live_decode():
    identity = ReturnFrameIdentity()
    identity.push_header(_header())
    identity.push_marker(1000, RtpFrameKey(123, 456))
    assert identity.displayed(1000, 1.0) is None
    assert identity.latest(1.01) is None


def test_roi_source_status_needs_fresh_exact_video_session_and_no_control():
    from dataclasses import replace
    from common.return_frame_identity import RoiSourceLease, DisplayedSourceFrame
    lease = RoiSourceLease()
    frame = DisplayedSourceFrame('run', 1, 10, 'pc_monotonic', 1.0)
    status = dict(type='visual_roi_status_v1',source_mode='sim',control_disabled=True,
                  source_ready=True,tracker_session='run',sequence=1)
    assert lease.observe(status, 1.0)
    assert lease.live(1.4, frame)
    assert not lease.live(1.51, frame)
    assert not lease.observe(status, 1.4)  # duplicate cannot refresh freshness
    assert not lease.live(1.51, frame)
    assert not lease.live(1.1, replace(frame,tracker_session='other'))
    assert not lease.live(1.1, replace(frame,source_clock_domain='jetson_monotonic'))
    assert not lease.observe(dict(status,sequence=2,control_disabled=False),1.2)
    assert not lease.live(1.21, frame)
    assert not lease.observe(dict(status,sequence=3,source_mode='camera'),1.3)
