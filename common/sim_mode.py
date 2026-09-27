"""Simulation motion-mode contract shared by host-side components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit


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


def require_simulation_loopback_endpoint(endpoint: str, name: str) -> str:
    """Accept only explicit TCP loopback endpoints for simulator actuation."""

    value = str(endpoint or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme != "tcp" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError(f"{name} must be a tcp loopback endpoint")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError(f"{name} must include a valid port")
    return value
