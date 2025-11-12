#!/usr/bin/env python3
"""Run a local DeepStream smoke test across streamer/server processes."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import zmq


def _load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"configuration at {path} is not a mapping")
    return data


def _parse_tcp_port(endpoint: str) -> int:
    if not endpoint.startswith("tcp://"):
        raise ValueError(f"unsupported endpoint (expected tcp://host:port): {endpoint}")
    _, _, rest = endpoint.partition("tcp://")
    if rest.count(":") != 1:
        raise ValueError(f"endpoint missing host or port: {endpoint}")
    host, _, port_str = rest.rpartition(":")
    if not host:
        raise ValueError(f"endpoint missing host: {endpoint}")
    try:
        port = int(port_str)
    except ValueError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"endpoint port is not numeric: {endpoint}") from exc
    if port <= 0 or port > 65535:
        raise ValueError(f"endpoint port out of range: {endpoint}")
    return port


def _spawn(cmd: List[str], *, cwd: Optional[Path] = None) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(cwd) if cwd else None)


def _terminate_processes(processes: List[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                proc.terminate()
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def run_smoke(config_path: Path, duration: float) -> None:
    cfg = _load_config(config_path)
    net_cfg = cfg.get("net") or {}
    if not isinstance(net_cfg, dict):
        raise RuntimeError("net section must be a mapping")

    results_ep = net_cfg.get("zmq_results")
    ctrl_ep = net_cfg.get("zmq_control")
    if not results_ep or not ctrl_ep:
        raise RuntimeError("config must define net.zmq_results and net.zmq_control")

    results_port = _parse_tcp_port(str(results_ep))
    ctrl_port = _parse_tcp_port(str(ctrl_ep))

    ctx = zmq.Context()
    results_sub = ctx.socket(zmq.SUB)
    ctrl_sub = ctx.socket(zmq.SUB)
    results_sub.setsockopt(zmq.RCVHWM, 10)
    ctrl_sub.setsockopt(zmq.RCVHWM, 10)
    results_sub.connect(f"tcp://127.0.0.1:{results_port}")
    ctrl_sub.connect(f"tcp://127.0.0.1:{ctrl_port}")
    results_sub.setsockopt(zmq.SUBSCRIBE, b"")
    ctrl_sub.setsockopt(zmq.SUBSCRIBE, b"")

    processes: List[subprocess.Popen] = []
    try:
        server_cmd = [
            sys.executable,
            "-m",
            "jetson.server",
            "--config",
            str(config_path),
            "--pipeline",
            "deepstream",
        ]
        server_proc = _spawn(server_cmd)
        processes.append(server_proc)

        # Give the server a moment to bind sockets and launch DeepStream.
        time.sleep(3.0)

        streamer_cmd = [
            sys.executable,
            "-m",
            "pc.streamer",
            "--config",
            str(config_path),
        ]
        streamer_proc = _spawn(streamer_cmd)
        processes.append(streamer_proc)

        poller = zmq.Poller()
        poller.register(results_sub, zmq.POLLIN)
        poller.register(ctrl_sub, zmq.POLLIN)

        result_count = 0
        control_count = 0
        deadline = time.monotonic() + duration
        last_result: Optional[Dict[str, Any]] = None
        last_control: Optional[Dict[str, Any]] = None

        while time.monotonic() < deadline:
            for proc in list(processes):
                code = proc.poll()
                if code is not None and code != 0:
                    args = proc.args
                    cmd_text = args if isinstance(args, str) else " ".join(args)
                    raise RuntimeError(f"process {cmd_text} exited with {code}")
            events = dict(poller.poll(timeout=500))
            if results_sub in events:
                payload = results_sub.recv_string(zmq.NOBLOCK)
                try:
                    last_result = json.loads(payload)
                except json.JSONDecodeError:
                    last_result = {"raw": payload}
                result_count += 1
            if ctrl_sub in events:
                payload = ctrl_sub.recv_string(zmq.NOBLOCK)
                try:
                    last_control = json.loads(payload)
                except json.JSONDecodeError:
                    last_control = {"raw": payload}
                control_count += 1
            if result_count and control_count:
                # Enough data collected; allow processes to settle briefly.
                time.sleep(2.0)
                break
        else:
            # Exhausted duration without both channels reporting.
            pass

        if result_count == 0:
            raise RuntimeError("no DetectionMsg samples received from DeepStream pipeline")
        if control_count == 0:
            raise RuntimeError("no ControlCmd samples received from control loop")

        summary = {
            "detections": result_count,
            "control": control_count,
            "last_detection": last_result,
            "last_control": last_control,
        }
        print(json.dumps(summary, indent=2))
    finally:
        _terminate_processes(processes)
        results_sub.close(0)
        ctrl_sub.close(0)
        ctx.term()


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepStream end-to-end smoke test")
    ap.add_argument("--config", default="configs/dev.yaml", type=Path)
    ap.add_argument("--duration", type=float, default=45.0)
    args = ap.parse_args()

    run_smoke(args.config, args.duration)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSmoke test interrupted", file=sys.stderr)
        raise
