from __future__ import annotations

import numpy as np

from common.schemas import CamState
import pc.v2_hud as hud


def test_attitude_tapes_are_relative_to_the_mount_home(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(hud, "elevation_tape_ticks",
                        lambda pitch_deg, *a, **k: seen.setdefault("pitch", pitch_deg) and [])
    monkeypatch.setattr(hud, "heading_tape_ticks",
                        lambda yaw_deg, *a, **k: seen.setdefault("yaw", yaw_deg) and [])
    state = CamState(frame_id=1, src_ts_ms=0, pan=-2.4 + 0.1, tilt=1.125 + 0.05,
                     home_pan=-2.4, home_tilt=1.125)
    hud._draw_attitude(np.zeros((720, 1280, 3), np.uint8), state, 135.0, 73.0)
    assert abs(seen["pitch"] - np.degrees(0.05)) < 1e-6
    assert abs(seen["yaw"] - np.degrees(0.1)) < 1e-6
