"""Read-only check of mapped PC frame age on the Jetson."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import zmq

from common.perception import perception_snapshot_from_json
from jetson.clock_sync_client import ClockSyncClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-endpoint", required=True)
    parser.add_argument("--clock-endpoint", required=True)
    parser.add_argument("--duration-s", type=float, default=6.0)
    args = parser.parse_args()
    clock = ClockSyncClient(args.clock_endpoint)
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(args.snapshot_endpoint)
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    ages_ms: list[float] = []
    uncertainties_ms: list[float] = []
    clock.start()
    try:
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline:
            if sub not in dict(poller.poll(100)):
                continue
            snapshot = perception_snapshot_from_json(sub.recv())
            now_ns = time.monotonic_ns()
            sample = clock.best_sample(now_ns=now_ns)
            if sample is None:
                continue
            if snapshot.frame.source_clock_domain not in {"pc_monotonic", "pc.monotonic"}:
                continue
            ages_ms.append((now_ns - sample.map_pc_ns(snapshot.frame.source_time_ns)) / 1e6)
            uncertainties_ms.append(sample.uncertainty_ns / 1e6)
    finally:
        clock.close()
        sub.close(0)
        context.term()
    if not ages_ms:
        return 1
    print(json.dumps({
        "samples": len(ages_ms),
        "frame_age_min_ms": min(ages_ms),
        "frame_age_mean_ms": statistics.fmean(ages_ms),
        "frame_age_max_ms": max(ages_ms),
        "clock_uncertainty_max_ms": max(uncertainties_ms),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
