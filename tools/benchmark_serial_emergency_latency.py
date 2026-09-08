"""Measure same-host emergency request-to-wire latency through serial_io_service.

The tool sends F7 only. With ``--fault-addr`` it first starts a reply-waiting
encoder query to an unused address, exercising the bounded in-flight blocking
case before each emergency request. Subscribe to SerialEmergencyTiming events
to obtain the service's actual pre-write monotonic timestamp.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import zmq

from common.serial_io import SerialCommandClient


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-endpoint", default="tcp://127.0.0.1:5571")
    parser.add_argument("--command-endpoint", default="tcp://127.0.0.1:5570")
    parser.add_argument("--reply-endpoint", default="tcp://127.0.0.1:5572")
    parser.add_argument(
        "--transport", choices=("command", "update"), default="command"
    )
    parser.add_argument("--target", default="gimbal")
    parser.add_argument("--addresses", default="1,2,3")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--budget-ms", type=float, default=25.0)
    parser.add_argument("--fault-addr", type=int, default=None)
    parser.add_argument("--fault-timeout-ms", type=int, default=8)
    parser.add_argument("--fault-retries", type=int, default=1)
    parser.add_argument("--fault-head-start-ms", type=float, default=2.0)
    parser.add_argument("--trial-settle-ms", type=float, default=25.0)
    parser.add_argument("--event-timeout-ms", type=float, default=250.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = fraction * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    alpha = index - low
    return ordered[low] * (1.0 - alpha) + ordered[high] * alpha


def main() -> int:
    args = _args()
    if args.trials <= 0 or args.budget_ms <= 0.0:
        raise ValueError("trials and budget must be positive")
    addresses = [int(value.strip(), 0) for value in args.addresses.split(",")]
    ctx = zmq.Context()
    update = ctx.socket(zmq.PUB)
    update.setsockopt(zmq.LINGER, 0)
    update.connect(args.update_endpoint)
    command_client = SerialCommandClient(
        args.command_endpoint,
        timeout_ms=max(int(args.event_timeout_ms), 1),
        ctx=ctx,
    )
    telemetry = ctx.socket(zmq.SUB)
    telemetry.setsockopt(zmq.LINGER, 0)
    telemetry.setsockopt_string(zmq.SUBSCRIBE, f"serial.telemetry.{args.target}")
    telemetry.connect(args.reply_endpoint)
    poller = zmq.Poller()
    poller.register(telemetry, zmq.POLLIN)
    time.sleep(0.35)

    samples: list[dict[str, Any]] = []
    try:
        for trial in range(1, args.trials + 1):
            if args.fault_addr is not None:
                fault_now = time.time_ns()
                update.send_json(
                    {
                        "type": "SerialUpdate",
                        "source": "tools.benchmark_serial_emergency_latency",
                        "target": args.target,
                        "commands": [
                            {
                                "cmd_id": f"fault:encoder:{trial}:{fault_now}",
                                "func": "0x31",
                                "addr": args.fault_addr,
                                "payload": [],
                                "expect_reply": True,
                                "expected_len": 6,
                                "priority": "high",
                                "timeout_ms": args.fault_timeout_ms,
                                "retry": args.fault_retries,
                                "target": args.target,
                            }
                        ],
                    }
                )
                time.sleep(max(args.fault_head_start_ms, 0.0) / 1000.0)

            request_ns = time.monotonic_ns()
            command_ids = {
                f"latency:estop:{trial}:{addr}:{request_ns}" for addr in addresses
            }
            emergency_commands = [
                {
                    "cmd_id": cmd_id,
                    "func": "F7",
                    "addr": addr,
                    "payload": [],
                    "expect_reply": False,
                    "priority": "critical",
                    "target": args.target,
                    "request_monotonic_ns": request_ns,
                    "request_host": socket.gethostname(),
                }
                for addr, cmd_id in zip(addresses, sorted(command_ids))
            ]
            if args.transport == "command":
                for command in emergency_commands:
                    ack = command_client.send_command(command)
                    if ack is None or not bool(ack.get("accepted")):
                        raise RuntimeError(
                            f"emergency command not acknowledged: {command['cmd_id']} {ack}"
                        )
            else:
                update.send_json(
                    {
                        "type": "SerialUpdate",
                        "source": "tools.benchmark_serial_emergency_latency",
                        "target": args.target,
                        "request_monotonic_ns": request_ns,
                        "request_host": socket.gethostname(),
                        "commands": emergency_commands,
                    }
                )

            trial_samples: list[dict[str, Any]] = []
            deadline = time.monotonic() + args.event_timeout_ms / 1000.0
            while command_ids and time.monotonic() < deadline:
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000.0))
                if not dict(poller.poll(remaining_ms)).get(telemetry):
                    continue
                topic_body = telemetry.recv_string()
                _topic, body = topic_body.split(" ", 1)
                event = json.loads(body)
                cmd_id = str(event.get("cmd_id", ""))
                if cmd_id not in command_ids:
                    continue
                command_ids.remove(cmd_id)
                wire_ns = event.get("timing", {}).get("wire_monotonic_ns")
                latency_ms = (
                    (int(wire_ns) - request_ns) / 1e6 if wire_ns is not None else None
                )
                trial_samples.append(
                    {
                        "trial": trial,
                        "cmd_id": cmd_id,
                        "addr": event.get("addr"),
                        "request_monotonic_ns": request_ns,
                        "wire_monotonic_ns": wire_ns,
                        "request_to_wire_ms": latency_ms,
                        "service_ingress_to_wire_ms": event.get("timing", {}).get(
                            "ingress_to_wire_ms"
                        ),
                        "service_request_to_wire_ms": event.get("timing", {}).get(
                            "request_to_wire_ms"
                        ),
                        "budget_scope": event.get("timing", {}).get("budget_scope"),
                        "service_budget_missed": event.get("timing", {}).get(
                            "budget_missed"
                        ),
                        "status": event.get("status"),
                    }
                )
            if command_ids:
                raise RuntimeError(
                    f"trial {trial} missing emergency telemetry: {sorted(command_ids)}"
                )
            samples.extend(trial_samples)
            time.sleep(max(args.trial_settle_ms, 0.0) / 1000.0)
    finally:
        command_client.close()
        telemetry.close(linger=0)
        update.close(linger=0)
        ctx.term()

    values = [
        float(sample["request_to_wire_ms"])
        for sample in samples
        if sample["request_to_wire_ms"] is not None
    ]
    report = {
        "format": "idcs.serial_emergency_latency",
        "version": 1,
        "host": socket.gethostname(),
        "clock_domain": "same_host_monotonic",
        "transport": args.transport,
        "fault_injected": args.fault_addr is not None,
        "fault_addr": args.fault_addr,
        "fault_timeout_ms": args.fault_timeout_ms,
        "fault_retries": args.fault_retries,
        "trials": args.trials,
        "commands": len(samples),
        "budget_ms": args.budget_ms,
        "request_to_wire_ms": {
            "min": min(values),
            "median": statistics.median(values),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "max": max(values),
        },
        "budget_misses": sum(value > args.budget_ms for value in values),
        "service_budget_misses": sum(
            bool(sample["service_budget_missed"]) for sample in samples
        ),
        "status_errors": sum(sample["status"] != "complete" for sample in samples),
        "samples": samples,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return (
        0
        if report["budget_misses"] == 0
        and report["service_budget_misses"] == 0
        and report["status_errors"] == 0
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
