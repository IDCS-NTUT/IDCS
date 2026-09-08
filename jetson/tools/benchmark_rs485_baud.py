"""Characterize non-motion MKS RS485 query reliability at supported baud rates.

By default this only tests the current host baud.  ``--switch-baud`` changes
the configured UART baud selector (0x8A) on each addressed motor, benchmarks
the requested rates, then restores every reachable motor to ``--restore-baud``.
It never enables, moves, zeros, or disables a motor.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.gimbal.mks_servo42_rs485 import RS485Bus


_SELECTOR_BY_BAUD = {
    9600: 1,
    19200: 2,
    25000: 3,
    38400: 4,
    57600: 5,
    115200: 6,
    256000: 7,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyCH341USB0")
    parser.add_argument("--current-baud", type=int, default=38400)
    parser.add_argument("--rates", default="38400,57600,115200,256000")
    parser.add_argument("--addresses", default="1,2,3")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=0.01)
    parser.add_argument("--switch-baud", action="store_true")
    parser.add_argument("--restore-baud", type=int, default=38400)
    parser.add_argument("--assume-exclusive", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _parse_csv(text: str) -> list[int]:
    return [int(item.strip(), 0) for item in text.split(",") if item.strip()]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1.0 - (index - lower)) + ordered[upper] * (index - lower)


def _probe_addr(port: str, baud: int, addr: int, timeout_s: float) -> bool:
    try:
        with RS485Bus(port, baud, timeout=timeout_s, max_retries=0) as bus:
            return len(bus.send_command(addr, 0xF1, (), expected_response_len=1, retries=0)) == 1
    except Exception:  # noqa: BLE001
        return False


def _discover_bauds(port: str, rates: list[int], addresses: list[int], timeout_s: float) -> dict[int, int | None]:
    found: dict[int, int | None] = {}
    for addr in addresses:
        found[addr] = next(
            (rate for rate in rates if _probe_addr(port, rate, addr, timeout_s)), None
        )
    return found


def _set_baud(port: str, baud: int, addr: int, selector: int, timeout_s: float) -> bool:
    """Send 0x8A without waiting: a motor may switch before replying."""

    try:
        with RS485Bus(port, baud, timeout=timeout_s, max_retries=0) as bus:
            bus.send_command(addr, 0x8A, (selector,), response_expected=False, retries=0)
        return True
    except Exception:  # noqa: BLE001
        return False


def _switch_all(
    port: str,
    *,
    current_by_addr: dict[int, int | None],
    target_baud: int,
    timeout_s: float,
) -> dict[int, bool]:
    selector = _SELECTOR_BY_BAUD[target_baud]
    return {
        addr: bool(current_baud is not None)
        and _set_baud(port, current_baud, addr, selector, timeout_s)
        for addr, current_baud in current_by_addr.items()
    }


def _benchmark(port: str, baud: int, addresses: list[int], samples: int, timeout_s: float) -> dict[str, Any]:
    queries = {"status_f1": (0xF1, 1), "encoder_31": (0x31, 6)}
    rows: dict[str, dict[str, Any]] = {
        name: {"attempts": 0, "successes": 0, "failures": 0, "latencies_ms": []}
        for name in queries
    }
    started = time.monotonic()
    with RS485Bus(port, baud, timeout=timeout_s, max_retries=0) as bus:
        for _sample in range(samples):
            for addr in addresses:
                for name, (func, response_len) in queries.items():
                    row = rows[name]
                    row["attempts"] += 1
                    begun = time.monotonic_ns()
                    try:
                        response = bus.send_command(
                            addr,
                            func,
                            (),
                            expected_response_len=response_len,
                            retries=0,
                        )
                        if len(response) != response_len:
                            raise RuntimeError("unexpected response length")
                    except Exception:  # noqa: BLE001
                        row["failures"] += 1
                    else:
                        row["successes"] += 1
                        row["latencies_ms"].append((time.monotonic_ns() - begun) / 1e6)
    elapsed_s = time.monotonic() - started
    summary: dict[str, Any] = {}
    for name, row in rows.items():
        values = row.pop("latencies_ms")
        attempts = row["attempts"]
        summary[name] = {
            **row,
            "failure_rate": row["failures"] / attempts if attempts else None,
            "latency_ms": {
                "min": min(values) if values else None,
                "median": statistics.median(values) if values else None,
                "p95": _percentile(values, 0.95),
                "p99": _percentile(values, 0.99),
                "max": max(values) if values else None,
            },
        }
    return {"elapsed_s": elapsed_s, "queries": summary}


def main() -> int:
    args = _args()
    rates = _parse_csv(args.rates)
    addresses = _parse_csv(args.addresses)
    if not rates or not addresses or args.samples <= 0 or args.timeout_s <= 0.0:
        raise ValueError("rates, addresses, samples, and timeout must be positive")
    unsupported = [rate for rate in set(rates + [args.current_baud, args.restore_baud]) if rate not in _SELECTOR_BY_BAUD]
    if unsupported:
        raise ValueError(f"unsupported MKS baud rate(s): {unsupported}")
    if args.switch_baud and not args.assume_exclusive:
        raise ValueError("--switch-baud requires --assume-exclusive")

    report: dict[str, Any] = {
        "format": "idcs.rs485_baud_benchmark",
        "version": 1,
        "port": args.port,
        "addresses": addresses,
        "timeout_s": args.timeout_s,
        "samples_per_command_address": args.samples,
        "switch_baud": bool(args.switch_baud),
        "requested_rates": rates,
        "results": {},
        "switch_events": [],
    }
    candidate_rates = list(dict.fromkeys(rates + [args.current_baud, args.restore_baud]))
    current = _discover_bauds(args.port, candidate_rates, addresses, args.timeout_s)
    report["initial_discovery"] = current
    active = args.current_baud
    try:
        for rate in rates:
            if rate != active:
                if not args.switch_baud:
                    report["results"][str(rate)] = {"status": "not_tested_without_switch"}
                    continue
                current = _discover_bauds(args.port, candidate_rates, addresses, args.timeout_s)
                sent = _switch_all(
                    args.port,
                    current_by_addr=current,
                    target_baud=rate,
                    timeout_s=args.timeout_s,
                )
                time.sleep(0.08)
                after = _discover_bauds(args.port, candidate_rates, addresses, args.timeout_s)
                report["switch_events"].append({"target_baud": rate, "sent": sent, "discovery": after})
                if any(found != rate for found in after.values()):
                    report["results"][str(rate)] = {"status": "unreachable_after_switch", "discovery": after}
                    active = next((found for found in after.values() if found is not None), active)
                    continue
                active = rate
            report["results"][str(rate)] = {"status": "complete", **_benchmark(args.port, rate, addresses, args.samples, args.timeout_s)}
    finally:
        if args.switch_baud:
            current = _discover_bauds(args.port, candidate_rates, addresses, args.timeout_s)
            sent = _switch_all(
                args.port,
                current_by_addr=current,
                target_baud=args.restore_baud,
                timeout_s=args.timeout_s,
            )
            time.sleep(0.08)
            report["restore"] = {
                "target_baud": args.restore_baud,
                "sent": sent,
                "discovery": _discover_bauds(args.port, candidate_rates, addresses, args.timeout_s),
            }
    output = Path(args.output or f"logs/rs485_baud_benchmark_{int(time.time())}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(value.get("status") == "complete" for value in report["results"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
