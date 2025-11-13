#!/usr/bin/env python3
"""Run a local DeepStream smoke test across streamer/server processes."""

from __future__ import annotations

import argparse
import ipaddress
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import zmq


def _load_config(path: Path) -> Dict[str, Any]:
    import yaml

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


def _is_local_address(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host in {"localhost", ""}
    return addr.is_loopback or addr.is_unspecified


def _probe_return_feed(
    host: str,
    port: int,
    duration: float,
    *,
    bind_host: Optional[str] = None,
    min_packets: int = 3,
) -> Dict[str, Any]:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if min_packets <= 0:
        raise ValueError("min_packets must be positive")
    bind_addr = bind_host or host or "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((bind_addr, int(port)))
    except OSError as exc:  # pragma: no cover - depends on OS networking
        sock.close()
        raise RuntimeError(f"failed to bind return feed probe socket: {exc}") from exc
    sock.settimeout(0.5)
    packets = 0
    bytes_rx = 0
    start = time.monotonic()
    min_packets = max(1, int(min_packets))
    try:
        while time.monotonic() - start < duration:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:  # pragma: no cover - defensive logging
                raise RuntimeError(f"return feed probe failed: {exc}") from exc
            packets += 1
            bytes_rx += len(data)
            if packets >= min_packets:
                break
    finally:
        sock.close()
    elapsed = time.monotonic() - start
    if packets < min_packets:
        raise RuntimeError("no return video packets received from DeepStream")
    return {
        "packets": packets,
        "bytes": bytes_rx,
        "port": port,
        "host": host,
        "bind_host": bind_addr,
        "elapsed_s": round(elapsed, 3),
    }


def run_smoke(
    config_path: Path,
    duration: float,
    *,
    verify_return_feed: bool = True,
    return_feed_timeout: float = 5.0,
) -> None:
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

    try:
        return_port_value = net_cfg["rtp_return_port"]
    except KeyError as exc:
        raise RuntimeError("config must define net.rtp_return_port") from exc
    try:
        return_port = int(return_port_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("net.rtp_return_port must be an integer") from exc

    return_host = str(net_cfg.get("pc_ip") or "")
    if not return_host:
        raise RuntimeError("config must define net.pc_ip for return video")

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
        return_stats: Optional[Dict[str, Any]] = None

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

        source_spec = str(cfg.get("source", "sim") or "")
        if verify_return_feed and source_spec.startswith("file:"):
            print("[smoke] skipping return feed probe; file source disables return video")
            verify_return_feed = False

        if verify_return_feed:
            if not _is_local_address(return_host):
                print(
                    "[smoke] skipping return feed probe; net.pc_ip must be loopback for local test"
                )
            else:
                timeout = return_feed_timeout if return_feed_timeout > 0 else 5.0
                print(
                    f"[smoke] probing return video feed on {return_host}:{return_port}"
                )
                return_stats = _probe_return_feed(
                    return_host,
                    return_port,
                    timeout,
                    bind_host=return_host,
                    min_packets=3,
                )

        summary = {
            "detections": result_count,
            "control": control_count,
            "last_detection": last_result,
            "last_control": last_control,
        }
        if return_stats is not None:
            summary["return_feed"] = return_stats
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
    ap.add_argument(
        "--skip-return-feed-check",
        action="store_false",
        dest="verify_return_feed",
        help="Skip probing the DeepStream return video feed",
    )
    ap.add_argument(
        "--return-feed-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for return video packets before failing",
    )
    args = ap.parse_args()

    run_smoke(
        args.config,
        args.duration,
        verify_return_feed=args.verify_return_feed,
        return_feed_timeout=args.return_feed_timeout,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSmoke test interrupted", file=sys.stderr)
        raise
