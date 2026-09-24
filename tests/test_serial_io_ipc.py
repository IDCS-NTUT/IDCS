from __future__ import annotations

import threading
import uuid

import zmq

from common.serial_io import SerialCommandClient


def test_serial_command_client_sends_typed_request_and_receives_ack() -> None:
    context = zmq.Context()
    endpoint = f"inproc://serial-command-{uuid.uuid4().hex}"
    server = context.socket(zmq.REP)
    server.bind(endpoint)
    observed: list[dict[str, object]] = []

    def respond() -> None:
        observed.append(server.recv_json())
        server.send_json(
            {
                "type": "SerialCommandAck",
                "cmd_id": "test:1",
                "accepted": True,
                "queued": True,
            }
        )

    thread = threading.Thread(target=respond)
    thread.start()
    client = SerialCommandClient(endpoint, timeout_ms=100, ctx=context)
    try:
        reply = client.send_command({"cmd_id": "test:1", "type": "wrong"})
    finally:
        client.close()
        thread.join(timeout=1.0)
        server.close(linger=0)
        context.destroy(linger=0)

    assert reply is not None
    assert reply["accepted"] is True
    assert observed == [{"cmd_id": "test:1", "type": "SerialCommandRequest"}]


def test_serial_command_client_timeout_resets_req_socket() -> None:
    context = zmq.Context()
    endpoint = f"inproc://serial-command-timeout-{uuid.uuid4().hex}"
    client = SerialCommandClient(endpoint, timeout_ms=1, ctx=context)
    try:
        assert client.send_command({"cmd_id": "test:timeout"}) is None
        assert client.send_command({"cmd_id": "test:timeout-again"}) is None
    finally:
        client.close()
        context.destroy(linger=0)
