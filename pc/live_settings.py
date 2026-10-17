"""Safe-point changes to the simulator's display geometry only."""
from common.control import ControlConfig, LaserMountConfig


def apply_simulator_settings(cap, values, frame_size):
    control = ControlConfig.from_raw_config(values, frame_size)
    mount = LaserMountConfig.from_raw_config(values)
    scenario = getattr(getattr(cap, "gen", None), "_planner_eval", None)
    area = None
    if scenario is not None:
        area = scenario._resolve_protected_area({}, control.threat_eval)
    cap._laser_mount = mount
    if hasattr(cap, "gen"):
        cap.gen._laser_mount = mount
    if area is not None:
        scenario.asset_position, scenario.zone_radii = area
