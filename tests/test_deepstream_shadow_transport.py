from pathlib import Path

import pytest

from common.perception import perception_snapshot_from_json
from common.schemas import detection_msg_from_json
from common.synthetic_perception import load_synthetic_scenario, snapshot_at
from jetson.deepstream import shadow_transport
from jetson.deepstream.shadow_transport import ShadowTransport


class _FakeSocket:
    def __init__(self) -> None:
        self.bound = None
        self.sent: list[str] = []

    def setsockopt(self, *_args) -> None:
        pass

    def bind(self, endpoint: str) -> None:
        self.bound = endpoint

    def send_string(self, payload: str, *, flags: int) -> None:
        assert flags == shadow_transport.zmq.NOBLOCK
        self.sent.append(payload)

    def close(self) -> None:
        pass


class _FakeContext:
    def __init__(self) -> None:
        self.sockets: list[_FakeSocket] = []

    def socket(self, _kind) -> _FakeSocket:
        socket = _FakeSocket()
        self.sockets.append(socket)
        return socket

    def term(self) -> None:
        pass


def _snapshot():
    scenario = load_synthetic_scenario(
        Path("tests/fixtures/synthetic_tracking_v1.json")
    )
    return snapshot_at(scenario, 2)


def test_transport_publishes_lossless_v2_and_explicit_legacy_adapter(monkeypatch):
    context = _FakeContext()
    monkeypatch.setattr(shadow_transport.zmq, "Context", lambda: context)
    transport = ShadowTransport(
        header_bind=None,
        result_bind="tcp://127.0.0.1:6101",
        snapshot_bind="tcp://127.0.0.1:6102",
    )

    source = _snapshot()
    assert transport.publish(source)

    legacy_socket, v2_socket = context.sockets
    assert legacy_socket.bound == "tcp://127.0.0.1:6101"
    assert v2_socket.bound == "tcp://127.0.0.1:6102"
    assert perception_snapshot_from_json(v2_socket.sent[0]) == source
    legacy = detection_msg_from_json(legacy_socket.sent[0])
    assert legacy.frame_id == source.frame.frame_id
    assert any(
        box.track_id == source.tracks[0].track_id for box in legacy.boxes
    )
    assert transport.report()["snapshot_published"] == 1
    assert transport.report()["legacy_published"] == 1


def test_transport_requires_distinct_output_endpoints():
    with pytest.raises(ValueError, match="at least one"):
        ShadowTransport(header_bind=None, result_bind=None)
    with pytest.raises(ValueError, match="distinct"):
        ShadowTransport(
            header_bind=None,
            result_bind="tcp://127.0.0.1:6101",
            snapshot_bind="tcp://127.0.0.1:6101",
        )
