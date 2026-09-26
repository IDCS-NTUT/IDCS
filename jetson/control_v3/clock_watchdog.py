"""Causal clock-age and consistency watchdog; never infers its own drift limit.

An externally justified drift limit is required before this can supply
ClockBounds to a controller. Exchange agreement is a fault detector, not
proof that an oscillator will obey the configured limit in the future.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from jetson.control_v3.timing import ClockBounds


@dataclass(frozen=True)
class ClockWatchdogConfig:
    max_exchange_age_ns: int
    configured_max_drift_ppm: float | None = None
    required_samples: int = 2
    capacity: int = 256

    def __post_init__(self) -> None:
        if self.max_exchange_age_ns <= 0 or self.required_samples < 1 or self.capacity < self.required_samples:
            raise ValueError("invalid clock watchdog limits")
        if self.configured_max_drift_ppm is not None and (
            not math.isfinite(self.configured_max_drift_ppm)
            or not 0 <= self.configured_max_drift_ppm < 1_000_000
        ):
            raise ValueError("configured drift limit must be finite and nonnegative")


class ClockWatchdog:
    """Retains exchanges and latches time/order or configured-limit faults."""

    def __init__(self, config: ClockWatchdogConfig) -> None:
        self.config = config
        self._samples: deque[ClockBounds] = deque(maxlen=config.capacity)
        self._fault: str | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._fault = None

    def observe(self, sample: ClockBounds) -> str:
        if self._fault is not None:
            return self._fault
        if self._samples and sample.observed_jetson_ns <= self._samples[-1].observed_jetson_ns:
            self._fault = "clock_exchange_nonmonotonic"
            return self._fault
        drift = self.config.configured_max_drift_ppm
        if drift is not None:
            for previous in self._samples:
                elapsed_ns = sample.observed_jetson_ns - previous.observed_jetson_ns
                expansion_ns = math.ceil(elapsed_ns * drift / 1_000_000)
                if (
                    sample.offset_min_ns > previous.offset_max_ns + expansion_ns
                    or previous.offset_min_ns > sample.offset_max_ns + expansion_ns
                ):
                    self._fault = "clock_drift_policy_violated"
                    return self._fault
        self._samples.append(sample)
        return "observed"

    def bounds(self, *, jetson_now_ns: int) -> tuple[ClockBounds | None, str]:
        if self._fault is not None:
            return None, self._fault
        if len(self._samples) < self.config.required_samples:
            return None, "clock_warmup"
        if self.config.configured_max_drift_ppm is None:
            return None, "drift_bound_unqualified"
        latest = self._samples[-1]
        if jetson_now_ns < latest.observed_jetson_ns:
            return None, "clock_query_before_sample"
        if jetson_now_ns - latest.observed_jetson_ns > self.config.max_exchange_age_ns:
            return None, "clock_sample_stale"
        return ClockBounds(
            offset_min_ns=latest.offset_min_ns,
            offset_max_ns=latest.offset_max_ns,
            observed_jetson_ns=latest.observed_jetson_ns,
            max_drift_ppm=self.config.configured_max_drift_ppm,
        ), "verified_under_configured_policy"
