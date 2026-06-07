import logging
import sys
import types
import unittest


sys.modules.setdefault("smbus", types.SimpleNamespace(SMBus=object))

from rpi.manual_control import ManualSwitchIO, resolve_gpio_config  # noqa: E402


class FakeGPIO:
    BCM = "BCM"
    IN = "IN"
    OUT = "OUT"
    HIGH = 1
    LOW = 0
    PUD_UP = "PUD_UP"
    PUD_DOWN = "PUD_DOWN"
    PUD_OFF = "PUD_OFF"

    def __init__(self) -> None:
        self.setup_calls = []
        self.outputs = {}
        self.inputs = {}
        self.cleaned = False

    def setwarnings(self, _enabled):
        return None

    def setmode(self, _mode):
        return None

    def setup(self, pin, direction, pull_up_down=None, initial=None):
        self.setup_calls.append((pin, direction, pull_up_down, initial))
        if direction == self.OUT:
            self.outputs[pin] = initial
        else:
            self.inputs.setdefault(pin, self.HIGH)

    def input(self, pin):
        return self.inputs.get(pin, self.HIGH)

    def output(self, pin, level):
        self.outputs[pin] = level

    def cleanup(self):
        self.cleaned = True


class RpiGpioConfigTests(unittest.TestCase):
    def test_resolve_gpio_config_allows_custom_roles_and_removal(self):
        cfg = resolve_gpio_config(
            {
                "inputs": {"fire": 4, "extra_switch": 12},
                "outputs": {"red_light": 25},
                "input_pull": "up",
                "input_active_level": "low",
                "output_active_level": "low",
            }
        )

        self.assertEqual(cfg["inputs"], {"fire": 4, "extra_switch": 12})
        self.assertEqual(cfg["outputs"], {"red_light": 25})

    def test_resolve_gpio_config_accepts_latched_input_modes(self):
        cfg = resolve_gpio_config(
            {
                "inputs": {"control_switch": 20},
                "outputs": {},
                "input_modes": {"control_switch": "latch"},
            }
        )

        self.assertEqual(cfg["input_modes"], {"control_switch": "latch"})

    def test_manual_switch_io_uses_active_low_inputs_and_outputs(self):
        fake_gpio = FakeGPIO()
        sys.modules["RPi"] = types.SimpleNamespace(GPIO=fake_gpio)
        sys.modules["RPi.GPIO"] = fake_gpio
        fake_gpio.inputs = {
            4: fake_gpio.HIGH,
            5: fake_gpio.LOW,
            6: fake_gpio.HIGH,
            16: fake_gpio.HIGH,
            20: fake_gpio.LOW,
        }
        switch = ManualSwitchIO(
            enabled=True,
            poll_dt=0.005,
            debounce_s=0.05,
            gpio_config={
                "inputs": {
                    "fire": 4,
                    "fire_control": 5,
                    "safety": 6,
                    "emergency": 16,
                    "control_switch": 20,
                },
                "outputs": {
                    "fire_control_light": 21,
                    "safety_light": 22,
                    "green_light": 23,
                    "yellow_light": 24,
                    "red_light": 25,
                },
                "input_pull": "up",
                "input_active_level": "low",
                "output_active_level": "low",
            },
            log=logging.getLogger("test"),
        )

        self.assertTrue(switch.setup())
        state = switch.update()

        self.assertTrue(state["active"])
        self.assertTrue(state["control_cmd_enabled"])
        self.assertFalse(state["emergency"])
        self.assertEqual(fake_gpio.outputs[21], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.HIGH)

    def test_latched_control_switch_retains_state_after_button_release(self):
        fake_gpio = FakeGPIO()
        sys.modules["RPi"] = types.SimpleNamespace(GPIO=fake_gpio)
        sys.modules["RPi.GPIO"] = fake_gpio
        fake_gpio.inputs = {20: fake_gpio.HIGH}
        switch = ManualSwitchIO(
            enabled=True,
            poll_dt=0.005,
            debounce_s=0.0,
            gpio_config={
                "inputs": {"control_switch": 20},
                "outputs": {"green_light": 23},
                "input_pull": "up",
                "input_active_level": "low",
                "output_active_level": "low",
                "input_modes": {"control_switch": "latch"},
            },
            log=logging.getLogger("test"),
        )

        self.assertTrue(switch.setup())
        self.assertFalse(switch.update()["active"])

        fake_gpio.inputs[20] = fake_gpio.LOW
        self.assertTrue(switch.update()["active"])
        fake_gpio.inputs[20] = fake_gpio.HIGH
        self.assertTrue(switch.update()["active"])
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.LOW)

        fake_gpio.inputs[20] = fake_gpio.LOW
        self.assertFalse(switch.update()["active"])
        fake_gpio.inputs[20] = fake_gpio.HIGH
        self.assertFalse(switch.update()["active"])
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.HIGH)


if __name__ == "__main__":
    unittest.main()
