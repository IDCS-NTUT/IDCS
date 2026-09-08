import unittest

from jetson.tools.gimbal_trajectory_benchmark import _safe_serial_startup


class HardwareTrajectorySafetyTests(unittest.TestCase):
    def test_trajectory_startup_excludes_encoder_zero_commands(self):
        merged = {
            "serial_io": {
                "startup": [
                    {"func": "0x81", "addr": 1},
                    {"func": "0x92", "addr": 1},
                    {"func": 0x92, "addr": 2},
                    {"func": "0x31", "addr": 1},
                ]
            }
        }

        self.assertEqual(
            [{"func": "0x81", "addr": 1}, {"func": "0x31", "addr": 1}],
            _safe_serial_startup(merged),
        )


if __name__ == "__main__":
    unittest.main()
