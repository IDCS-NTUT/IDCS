import argparse
import math
import unittest
from unittest.mock import patch

from jetson import camstate_devices, gimbal_bridge


class _FakeSMBus:
    def __init__(self, bus: int) -> None:
        self.bus = int(bus)
        self.write_calls = []
        self.read_byte_values = {}
        self.read_byte_calls = []
        self.read_block_values = {}
        self.read_block_calls = []
        self.closed = False

    def write_byte_data(self, addr: int, reg: int, value: int) -> None:
        self.write_calls.append((int(addr), int(reg), int(value)))

    def read_byte_data(self, addr: int, reg: int) -> int:
        key = (int(addr), int(reg))
        self.read_byte_calls.append(key)
        return int(self.read_byte_values.get(key, 0))

    def read_i2c_block_data(self, addr: int, reg: int, length: int):
        key = (int(addr), int(reg), int(length))
        self.read_block_calls.append(key)
        raw = self.read_block_values.get(key, [0] * int(length))
        return [int(v) for v in raw]

    def close(self) -> None:
        self.closed = True


def _expected_orientation(
    ax: float,
    ay: float,
    az: float,
    mx: float,
    my: float,
    mz: float,
) -> tuple[float, float]:
    pitch = math.atan2(ax, math.sqrt(ay * ay + az * az))
    roll = math.atan2(-ay, az)
    mx_aligned = my
    my_aligned = mx
    mz_aligned = -mz
    mx2 = mx_aligned * math.cos(pitch) + mz_aligned * math.sin(pitch)
    my2 = (
        mx_aligned * math.sin(roll) * math.sin(pitch)
        + my_aligned * math.cos(roll)
        - mz_aligned * math.sin(roll) * math.cos(pitch)
    )
    heading = math.atan2(my2, mx2)
    return pitch, heading


class CamstateDevicesStandaloneTests(unittest.TestCase):
    def test_sensor_cfg_defaults_match_reduced_schema(self) -> None:
        args = argparse.Namespace(publish_hz=None)
        cfg = camstate_devices._build_sensor_cfg({}, args=args)

        self.assertEqual(cfg.mpu_bus, 7)
        self.assertEqual(cfg.mpu_addr, 0x68)
        self.assertEqual(cfg.mag_addr, 0x0C)
        self.assertAlmostEqual(cfg.publish_hz, 50.0)

    def test_sensor_cfg_rejects_removed_keys(self) -> None:
        args = argparse.Namespace(publish_hz=None)
        with self.assertRaises(SystemExit) as ctx:
            camstate_devices._build_sensor_cfg({"alpha": 0.95}, args=args)
        self.assertIn("removed keys", str(ctx.exception))

    def test_reader_init_writes_powerdown_then_continuous_mag_mode(self) -> None:
        fake_mpu = _FakeSMBus(7)
        fake_mag = _FakeSMBus(7)

        def _factory(_bus: int):
            if not hasattr(_factory, "count"):
                _factory.count = 0
            _factory.count += 1
            return fake_mpu if _factory.count == 1 else fake_mag

        cfg = camstate_devices.SensorConfig(mpu_bus=7)
        with patch.object(camstate_devices, "SMBus", _factory):
            reader = camstate_devices.SensorReader(cfg)
            reader.init()

        self.assertEqual(
            fake_mpu.write_calls,
            [
                (cfg.mpu_addr, camstate_devices._PWR_MGMT_1, 0),
                (cfg.mpu_addr, camstate_devices._INT_PIN_CFG, camstate_devices._INT_PIN_CFG_BYPASS_VAL),
            ],
        )
        self.assertEqual(
            fake_mag.write_calls,
            [
                (cfg.mag_addr, camstate_devices._MAG_CNTL1, camstate_devices._MAG_POWER_DOWN),
                (cfg.mag_addr, camstate_devices._MAG_CNTL1, camstate_devices._MAG_CONTINUOUS_100HZ),
            ],
        )

    def test_reader_mag_decode_uses_st1_ready_and_reads_st2(self) -> None:
        fake_mpu = _FakeSMBus(7)
        fake_mag = _FakeSMBus(7)

        def _factory(_bus: int):
            if not hasattr(_factory, "count"):
                _factory.count = 0
            _factory.count += 1
            return fake_mpu if _factory.count == 1 else fake_mag

        cfg = camstate_devices.SensorConfig(mpu_bus=7)
        fake_mag.read_byte_values[(cfg.mag_addr, camstate_devices._MAG_ST1)] = 0x01
        fake_mag.read_byte_values[(cfg.mag_addr, camstate_devices._MAG_ST2)] = 0x00
        fake_mag.read_block_values[(cfg.mag_addr, camstate_devices._MAG_DATA, 6)] = [
            0x34,
            0x12,
            0xFE,
            0xFF,
            0x01,
            0x80,
        ]

        with patch.object(camstate_devices, "SMBus", _factory):
            reader = camstate_devices.SensorReader(cfg)
            sample = reader.read_mag()

        self.assertEqual(sample, (4660, -2, -32767))
        self.assertIn((cfg.mag_addr, camstate_devices._MAG_ST2), fake_mag.read_byte_calls)
        self.assertIn((cfg.mag_addr, camstate_devices._MAG_DATA, 6), fake_mag.read_block_calls)

    def test_orientation_matches_test_py_model(self) -> None:
        ax, ay, az = 0.41, -0.22, 0.88
        mx, my, mz = 120.0, -55.0, 34.0
        expected_pitch, expected_heading = _expected_orientation(ax, ay, az, mx, my, mz)
        pitch, heading = camstate_devices._compute_orientation(ax, ay, az, mx, my, mz)

        self.assertAlmostEqual(pitch, expected_pitch, places=8)
        self.assertAlmostEqual(heading, expected_heading, places=8)
        raw_heading = math.atan2(my, mx)
        self.assertGreater(abs(heading - raw_heading), 1e-4)


class CamstateDevicesBridgeTests(unittest.TestCase):
    def test_device_cfg_defaults_match_reduced_schema(self) -> None:
        cfg = gimbal_bridge._build_device_sensor_cfg({})
        self.assertEqual(cfg.mpu_bus, 7)
        self.assertEqual(cfg.mpu_addr, 0x68)
        self.assertEqual(cfg.mag_addr, 0x0C)

    def test_device_cfg_rejects_removed_keys(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            gimbal_bridge._build_device_sensor_cfg({"pitch_gyro_axis": "y"})
        self.assertIn("removed keys", str(ctx.exception))

    def test_reader_init_writes_powerdown_then_continuous_mag_mode(self) -> None:
        fake_mpu = _FakeSMBus(7)
        fake_mag = _FakeSMBus(7)

        def _factory(_bus: int):
            if not hasattr(_factory, "count"):
                _factory.count = 0
            _factory.count += 1
            return fake_mpu if _factory.count == 1 else fake_mag

        cfg = gimbal_bridge._DeviceSensorConfig(mpu_bus=7)
        with patch.object(gimbal_bridge, "SMBus", _factory):
            reader = gimbal_bridge._DeviceSensorReader(cfg)
            reader.init()

        self.assertEqual(
            fake_mpu.write_calls,
            [
                (cfg.mpu_addr, gimbal_bridge._SENSOR_PWR_MGMT_1, 0),
                (cfg.mpu_addr, gimbal_bridge._SENSOR_INT_PIN_CFG, gimbal_bridge._SENSOR_INT_BYPASS_VAL),
            ],
        )
        self.assertEqual(
            fake_mag.write_calls,
            [
                (cfg.mag_addr, gimbal_bridge._SENSOR_MAG_CNTL1, gimbal_bridge._SENSOR_MAG_POWER_DOWN),
                (cfg.mag_addr, gimbal_bridge._SENSOR_MAG_CNTL1, gimbal_bridge._SENSOR_MAG_CONTINUOUS_100HZ),
            ],
        )

    def test_reader_mag_decode_uses_st1_ready_and_reads_st2(self) -> None:
        fake_mpu = _FakeSMBus(7)
        fake_mag = _FakeSMBus(7)

        def _factory(_bus: int):
            if not hasattr(_factory, "count"):
                _factory.count = 0
            _factory.count += 1
            return fake_mpu if _factory.count == 1 else fake_mag

        cfg = gimbal_bridge._DeviceSensorConfig(mpu_bus=7)
        fake_mag.read_byte_values[(cfg.mag_addr, gimbal_bridge._SENSOR_MAG_ST1)] = 0x01
        fake_mag.read_byte_values[(cfg.mag_addr, gimbal_bridge._SENSOR_MAG_ST2)] = 0x00
        fake_mag.read_block_values[(cfg.mag_addr, gimbal_bridge._SENSOR_MAG_DATA, 6)] = [
            0x34,
            0x12,
            0xFE,
            0xFF,
            0x01,
            0x80,
        ]

        with patch.object(gimbal_bridge, "SMBus", _factory):
            reader = gimbal_bridge._DeviceSensorReader(cfg)
            sample = reader.read_mag()

        self.assertEqual(sample, (4660, -2, -32767))
        self.assertIn((cfg.mag_addr, gimbal_bridge._SENSOR_MAG_ST2), fake_mag.read_byte_calls)
        self.assertIn((cfg.mag_addr, gimbal_bridge._SENSOR_MAG_DATA, 6), fake_mag.read_block_calls)

    def test_orientation_matches_test_py_model(self) -> None:
        ax, ay, az = -0.37, 0.14, 0.93
        mx, my, mz = 210.0, 42.0, -77.0
        expected_pitch, expected_heading = _expected_orientation(ax, ay, az, mx, my, mz)
        pitch, heading = gimbal_bridge._compute_orientation(ax, ay, az, mx, my, mz)

        self.assertAlmostEqual(pitch, expected_pitch, places=8)
        self.assertAlmostEqual(heading, expected_heading, places=8)
        raw_heading = math.atan2(my, mx)
        self.assertGreater(abs(heading - raw_heading), 1e-4)


if __name__ == "__main__":
    unittest.main()
