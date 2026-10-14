"""Return-video RTP identity must never infer source identity by order."""

from common.rtp_identity import RtpFrameKey, RtpPacketIdentity
from jetson.visual_lock.return_identity import ReturnRtpMapper, SourceIdentity


def _packet(marker: bool = True) -> RtpPacketIdentity:
    return RtpPacketIdentity(RtpFrameKey(12, 34), 5, marker)


def test_exact_encoded_pts_maps_to_verified_source_only() -> None:
    mapper = ReturnRtpMapper()
    source = SourceIdentity("session", 77, 88, "pc_monotonic")
    mapper.record_source(1000, source)
    assert mapper.encoded_packet(1000, _packet(False)) is None
    assert mapper.encoded_packet(999, _packet()) is None
    payload = mapper.encoded_packet(1000, _packet())
    assert payload["frame_id"] == 77
    assert payload["rtp_timestamp"] == 34
    assert mapper.encoded_packet(1000, _packet()) is None


def test_conflicting_pts_poisoned_and_capacity_bounded() -> None:
    mapper = ReturnRtpMapper(capacity=1)
    mapper.record_source(1000, SourceIdentity("a", 1, 10, "pc_monotonic"))
    mapper.record_source(1000, SourceIdentity("b", 2, 20, "pc_monotonic"))
    assert mapper.encoded_packet(1000, _packet()) is None
    assert mapper.ambiguous == 1
    mapper.record_source(1001, SourceIdentity("a", 3, 30, "pc_monotonic"))
    mapper.record_source(1002, SourceIdentity("a", 4, 40, "pc_monotonic"))
    assert mapper.encoded_packet(1001, _packet()) is None
    assert mapper.encoded_packet(1002, _packet())["frame_id"] == 4


def test_marker_without_pts_uses_same_rtp_key_first_packet() -> None:
    mapper = ReturnRtpMapper()
    mapper.record_source(1234, SourceIdentity("run", 9, 88, "pc_monotonic"))
    assert mapper.encoded_packet(1234, _packet(False)) is None
    assert mapper.encoded_packet(None, _packet())['frame_id'] == 9


def test_conflicting_packet_pts_never_recovers_within_rtp_frame() -> None:
    mapper = ReturnRtpMapper()
    mapper.record_source(1234, SourceIdentity("run", 9, 88, "pc_monotonic"))
    mapper.encoded_packet(1234, _packet(False))
    mapper.encoded_packet(4321, _packet(False))
    assert mapper.encoded_packet(1234, _packet()) is None
