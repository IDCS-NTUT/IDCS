from __future__ import annotations

import pytest

from common.clock_sync import calculate_clock_offset


def test_four_timestamp_exchange_recovers_clock_offset() -> None:
    # PC is 100 ms ahead of Jetson; transport takes 1 ms in each direction.
    sample = calculate_clock_offset(
        1_000_000_000,
        1_101_000_000,
        1_101_200_000,
        1_002_200_000,
    )
    assert sample.offset_ns == 100_000_000
    assert sample.round_trip_ns == 2_000_000
    assert sample.uncertainty_ns == 1_000_000
    assert sample.map_pc_ns(1_150_000_000) == 1_050_000_000


def test_four_timestamp_exchange_rejects_impossible_order() -> None:
    with pytest.raises(ValueError):
        calculate_clock_offset(10, 20, 19, 30)
