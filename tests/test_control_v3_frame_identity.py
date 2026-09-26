from __future__ import annotations

import random

import pytest

from jetson.control_v3.frame_identity import (
    FrameIdentityJoiner,
    RtpFrameKey,
    SourceFrameHeader,
)
from common.rtp_identity import parse_rtp_identity


def _header(frame_id: int, *, ssrc: int = 7) -> SourceFrameHeader:
    return SourceFrameHeader(
        key=RtpFrameKey(ssrc, frame_id * 3000),
        frame_id=frame_id,
        source_time_ns=1_000_000_000 + frame_id * 33_333_333,
        source_clock_domain="pc_monotonic",
    )


def test_exact_join_ignores_header_and_marker_arrival_order() -> None:
    joiner = FrameIdentityJoiner()
    for frame_id in (3, 1, 2):
        joiner.push_header(_header(frame_id))
    for frame_id in (2, 3, 1):
        joiner.push_marker(decoded_pts_ns=frame_id * 1_000_000, key=_header(frame_id).key)
    for frame_id in (1, 2, 3):
        result = joiner.match(decoded_pts_ns=frame_id * 1_000_000)
        assert result.verified and result.header == _header(frame_id)


def test_dropped_video_or_header_never_shifts_identity() -> None:
    joiner = FrameIdentityJoiner()
    for frame_id in range(1, 13):
        if frame_id != 5:  # metadata path lost frame 5
            joiner.push_header(_header(frame_id))
        if frame_id not in (3, 8):  # video path lost frames 3 and 8
            joiner.push_marker(decoded_pts_ns=frame_id * 1_000_000, key=_header(frame_id).key)
    assert joiner.match(decoded_pts_ns=5_000_000).reason == "source_header_missing"
    assert joiner.match(decoded_pts_ns=3_000_000).reason == "rtp_marker_missing"
    assert joiner.match(decoded_pts_ns=6_000_000).header == _header(6)


def test_reordered_headers_markers_and_independent_losses_never_mislabel() -> None:
    rng = random.Random(34719)
    joiner = FrameIdentityJoiner(capacity=512)
    frame_ids = list(range(1, 201))
    headers = [frame_id for frame_id in frame_ids if frame_id % 13 != 0]
    markers = [frame_id for frame_id in frame_ids if frame_id % 17 != 0]
    rng.shuffle(headers)
    rng.shuffle(markers)
    for frame_id in markers:
        joiner.push_marker(decoded_pts_ns=frame_id * 1_000_000, key=_header(frame_id).key)
    for frame_id in headers:
        joiner.push_header(_header(frame_id))
    rng.shuffle(markers)
    for frame_id in markers:
        result = joiner.match(decoded_pts_ns=frame_id * 1_000_000)
        if frame_id % 13 == 0:
            assert not result.verified and result.reason == "source_header_missing"
        else:
            assert result.verified and result.header == _header(frame_id)
    assert joiner.ambiguous_headers == 0
    assert joiner.ambiguous_markers == 0


def test_conflicting_header_and_marker_poison_only_their_keys() -> None:
    joiner = FrameIdentityJoiner()
    joiner.push_header(_header(1))
    joiner.push_header(SourceFrameHeader(_header(1).key, 99, 2_000_000_000, "pc_monotonic"))
    joiner.push_marker(decoded_pts_ns=1, key=_header(1).key)
    assert joiner.match(decoded_pts_ns=1).reason == "source_header_ambiguous"
    joiner.push_header(_header(2))
    joiner.push_marker(decoded_pts_ns=2, key=_header(2).key)
    joiner.push_marker(decoded_pts_ns=2, key=_header(3).key)
    assert joiner.match(decoded_pts_ns=2).reason == "rtp_marker_ambiguous"
    assert joiner.ambiguous_headers == 1
    assert joiner.ambiguous_markers == 1


def test_reuse_is_rejected() -> None:
    joiner = FrameIdentityJoiner()
    joiner.push_header(_header(1))
    joiner.push_marker(decoded_pts_ns=10, key=_header(1).key)
    assert joiner.match(decoded_pts_ns=10).verified
    assert joiner.match(decoded_pts_ns=10).reason == "decoded_pts_reused"
    joiner.push_marker(decoded_pts_ns=20, key=_header(1).key)
    assert joiner.match(decoded_pts_ns=20).reason == "rtp_marker_missing"


def test_capacity_eviction_fails_closed() -> None:
    joiner = FrameIdentityJoiner(capacity=2)
    for frame_id in (1, 2, 3):
        joiner.push_header(_header(frame_id))
        joiner.push_marker(decoded_pts_ns=frame_id, key=_header(frame_id).key)
    assert joiner.match(decoded_pts_ns=1).reason == "rtp_marker_missing"
    assert joiner.match(decoded_pts_ns=2).header == _header(2)
    assert joiner.dropped_capacity >= 2


@pytest.mark.parametrize("ssrc,timestamp", [(-1, 0), (0, 2**32)])
def test_invalid_rtp_fields_rejected(ssrc: int, timestamp: int) -> None:
    with pytest.raises(ValueError):
        RtpFrameKey(ssrc, timestamp)


def test_rtp_header_parser_extracts_key_without_fifo_assumptions() -> None:
    header = bytes.fromhex("80e0123401020304aabbccdd")
    packet = parse_rtp_identity(header)
    assert packet.key == RtpFrameKey(0xAABBCCDD, 0x01020304)
    assert packet.sequence == 0x1234 and packet.marker
    with pytest.raises(ValueError):
        parse_rtp_identity(b"\x80")
