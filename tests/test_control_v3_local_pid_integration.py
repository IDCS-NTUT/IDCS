from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import zmq
import pytest

from common.schemas import CamState, ManualControlState


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.mark.parametrize("axis", ["yaw", "pitch_a"])
def test_isolated_live_pid_emits_bounded_active_axis_then_holds_on_emergency(tmp_path: Path, axis: str) -> None:
    repo = Path(__file__).resolve().parents[1]
    gimbal_port, manual_port, intent_port = (_free_port() for _ in range(3))
    context = zmq.Context()
    gimbal_pub = context.socket(zmq.PUB)
    gimbal_pub.bind(f"tcp://127.0.0.1:{gimbal_port}")
    manual_push = context.socket(zmq.PUSH)
    manual_push.connect(f"tcp://127.0.0.1:{manual_port}")
    intent_sub = context.socket(zmq.SUB)
    intent_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    intent_sub.connect(f"tcp://127.0.0.1:{intent_port}")
    trace = tmp_path / "trial.jsonl"
    process = subprocess.Popen([
        sys.executable, "-m", "jetson.control_v3.local_pid_trial",
        "--gimbal-sub", f"tcp://127.0.0.1:{gimbal_port}",
        "--manual-bind", f"tcp://127.0.0.1:{manual_port}",
        "--intent-bind", f"tcp://127.0.0.1:{intent_port}",
        "--duration-s", "5.0", "--trace", str(trace), "--axis", axis,
        "--enable-live-intent-publish", "--acknowledge-unloaded-hardware",
    ], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    received: list[dict] = []
    start = time.monotonic()
    try:
        while time.monotonic() - start < 7.0 and process.poll() is None:
            elapsed = time.monotonic() - start
            gimbal_pub.send_string(CamState(
                frame_id=int(elapsed * 50), src_ts_ms=0,
                state_monotonic_ns=time.monotonic_ns(), pan=0.0, tilt=0.0,
                pan_rate=0.0, tilt_rate=0.0,
            ).model_dump_json())
            manual_push.send_string(ManualControlState(
                src_ts_ms=0, source="isolated_test", active=False,
                emergency=elapsed >= 5.0, control_cmd_enabled=True,
                joystick_raw=(0, 0), joystick_rate_cmd=(0.0, 0.0),
            ).model_dump_json())
            while intent_sub.poll(0):
                received.append(intent_sub.recv_json())
            time.sleep(0.02)
        output, errors = process.communicate(timeout=3)
        assert process.returncode == 0, (output, errors)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
        for endpoint in (gimbal_pub, manual_push, intent_sub):
            endpoint.close(0)
        context.term()
    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    ticks = [record for record in records if record["type"] == "tick"]
    active_field = "yaw_rate_rad_s" if axis == "yaw" else "pitch_rate_rad_s"
    idle_field = "pitch_rate_rad_s" if axis == "yaw" else "yaw_rate_rad_s"
    assert any(record["intent"][active_field] > 0.0 for record in ticks)
    assert any(record["status"] == "safety_hold" for record in ticks)
    assert received and any(item[active_field] > 0.0 for item in received)
    assert all(item["mode"] == "live" for item in received)
    assert all(abs(item[active_field]) <= 0.2 for item in received)
    assert all(item[idle_field] == 0.0 for item in received)
