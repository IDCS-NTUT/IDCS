"""Conservative idle proof for configuration changes; no command authority."""


def configuration_engaged(safety, last_intent_active: bool) -> bool:
    return bool(last_intent_active or not (
        safety.valid and safety.sample_age_ms is not None and safety.sample_age_ms <= 750
        and safety.manual_active and not safety.auto_allowed))


def prepare_controller_settings(values, frame_size, camera_intrinsics=None, *, policy):
    """Pure typed parsing, performed by the live-settings worker."""
    from common.control import ControlConfig, LaserMountConfig
    from jetson.control.runtime_config import ControlRuntimeConfig
    from jetson.control.video_controller import VideoControllerCore
    updated = ControlRuntimeConfig.from_config(values)
    return (updated,
            ControlConfig.from_raw_config(values, frame_size, camera_intrinsics=camera_intrinsics),
            LaserMountConfig.from_raw_config(values), values.get("gimbal", {}),
            VideoControllerCore.prepare_settings(updated, policy))


def apply_controller_settings(core, assembler, current, prepared):
    """Owner-only origin check and bounded swaps; preserve controller history."""
    updated, aim, mount, limits, coefficients = prepared
    if (updated.max_travel_rad != current.max_travel_rad and core._origin_rad is not None
            and current.local_clock != "pc"):
        for axis, origin in zip(("yaw", "pitch"), core._origin_rad):
            low, high = limits.get(axis + "_min_rad"), limits.get(axis + "_max_rad")
            if ((low is not None and origin - updated.max_travel_rad < low)
                    or (high is not None and origin + updated.max_travel_rad > high)):
                raise ValueError(f"controller.max_travel_rad exceeds bridge {axis} travel at the current origin")
    core.apply_settings(coefficients)
    assembler._config, assembler._laser_mount = aim, mount
    return updated
