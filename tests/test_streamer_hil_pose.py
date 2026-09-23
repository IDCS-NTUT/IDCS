import math

import pytest

from common.schemas import CamState
from pc.streamer import relative_hil_pose


def _cam_state(**overrides) -> CamState:
    values = {
        "frame_id": 1,
        "src_ts_ms": 1,
        "pan": 1.2,
        "tilt": -0.4,
        "home_pan": 1.0,
        "home_tilt": -0.5,
    }
    values.update(overrides)
    return CamState(**values)


def test_relative_hil_pose_uses_bridge_startup_home() -> None:
    pan, tilt = relative_hil_pose(_cam_state())

    assert pan == pytest.approx(0.2)
    assert tilt == pytest.approx(0.1)


def test_relative_hil_pose_wraps_pan_delta() -> None:
    pan, tilt = relative_hil_pose(
        _cam_state(pan=-math.pi + 0.1, home_pan=math.pi - 0.1)
    )

    assert pan == pytest.approx(0.2)
    assert tilt == pytest.approx(0.1)


@pytest.mark.parametrize("missing_field", ["home_pan", "home_tilt"])
def test_relative_hil_pose_requires_complete_home_reference(missing_field: str) -> None:
    assert relative_hil_pose(_cam_state(**{missing_field: None})) is None
