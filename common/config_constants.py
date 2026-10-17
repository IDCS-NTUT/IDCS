"""Fixed policy values from configs/base at 245d103 (2026-10-17).

These are algorithm/calibration-format constants, never live tuning keys.
The swarm loader already accepts model metadata normalization; that metadata
remains authoritative when present. This is the legacy model-format fallback.
"""
from common.config import merge_config_layers

CAMERA_KNOWN_SIZE_RANGING_CLASS_ASPECT_RATIO_LIMITS = {'drone': {'min': 0.4, 'max': 2.0}, 'person': {'min': 1.2, 'max': 4.5}}
CAMERA_KNOWN_SIZE_RANGING_MIN_PIXELS = 10.0
CAMERA_KNOWN_SIZE_RANGING_EMA_ALPHA = 0.9
CAMERA_KNOWN_SIZE_RANGING_QUALITY = {'hold_s': 0.25, 'edge_margin_px': 2.0, 'jump_slack_fraction': 0.1, 'max_fractional_change_per_s': 1.0}
LASER_RENDER = {'beam_length_m': 5.0, 'color_bgr': [0, 0, 255], 'thickness_px': 2, 'hit_tolerance_px': 30.0}
GIMBAL_STEPS_PER_REV = 3200
GIMBAL_COUNTS_PER_REV = 16384
RPI_RUNTIME_CONTROL_INDICATORS = {'fire_control_off_delay_s': 1.0, 'safety_off_delay_s': 1.0, 'perception_stale_s': 0.5, 'hardware_stale_s': 0.5, 'control_stale_s': 0.5}
SWARM_EVAL_LEARNED_MODEL_NORMALIZATION = {'max_distance_m': 80.0, 'max_closing_speed_m_s': 12.0, 'max_breakthrough_time_s': 40.0, 'max_time_to_engage_s': 8.0, 'max_damage_weight': 10.0, 'max_angle_rad': 0.8, 'max_track_observations': 10, 'max_track_age_s': 5.0}

FIXED_VALUES = {
    'camera.known_size_ranging.class_aspect_ratio_limits': CAMERA_KNOWN_SIZE_RANGING_CLASS_ASPECT_RATIO_LIMITS,
    'camera.known_size_ranging.min_pixels': CAMERA_KNOWN_SIZE_RANGING_MIN_PIXELS,
    'camera.known_size_ranging.ema_alpha': CAMERA_KNOWN_SIZE_RANGING_EMA_ALPHA,
    'camera.known_size_ranging.quality': CAMERA_KNOWN_SIZE_RANGING_QUALITY,
    'laser.render': LASER_RENDER,
    'gimbal.steps_per_rev': GIMBAL_STEPS_PER_REV,
    'gimbal.counts_per_rev': GIMBAL_COUNTS_PER_REV,
    'rpi.runtime_control.indicators': RPI_RUNTIME_CONTROL_INDICATORS,
    'swarm_eval.learned_model.normalization': SWARM_EVAL_LEARNED_MODEL_NORMALIZATION,
}

def with_constants(config):
    """Materialize constants for existing typed readers; never accept overrides."""
    result = merge_config_layers(config)
    for path, value in FIXED_VALUES.items():
        parts = path.split('.')
        node = result
        for key in parts[:-1]:
            if key not in node:
                break
            node = node[key]
        else:
            node[parts[-1]] = merge_config_layers({'value': value})['value']
    return result
