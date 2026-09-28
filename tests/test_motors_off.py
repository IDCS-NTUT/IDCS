from tools.motors_off import motor_addresses
from common.config_sync import expand_config_paths, load_merged_config


def test_addresses_come_from_the_gimbal_config():
    gimbal = load_merged_config(expand_config_paths("configs/base", "configs/bench/uncoupled.yaml"))["gimbal"]
    assert motor_addresses(gimbal) == [1, 2, 3]
    assert motor_addresses({"yaw_addr": 1, "pitch_motor_a_addr": 2, "pitch_motor_b_addr": None}) == [1, 2]
