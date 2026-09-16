"""Simulation motion-mode contract shared by host-side components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SimulationMotionMode:
    name: str
    use_jetson_cam_state: bool
    moves_physical_mount: bool


def resolve_simulation_motion_mode(
    sim_config: Mapping[str, Any],
) -> SimulationMotionMode:
    """Resolve the established ``sim.use_jetson_cam_state`` mode switch.

    False selects the stable simulated-motion substitute.  True selects the
    hardware-in-loop contract: a separately authorized tuned controller may
    move the physical mount, and the simulated camera follows fresh encoder
    state.  Merely resolving this config never starts hardware authority.
    """

    use_jetson_cam_state = sim_config.get("use_jetson_cam_state", False)
    if not isinstance(use_jetson_cam_state, bool):
        raise ValueError("sim.use_jetson_cam_state must be boolean")
    if use_jetson_cam_state:
        return SimulationMotionMode(
            name="hardware_in_loop",
            use_jetson_cam_state=True,
            moves_physical_mount=True,
        )
    return SimulationMotionMode(
        name="stable_substitute",
        use_jetson_cam_state=False,
        moves_physical_mount=False,
    )
