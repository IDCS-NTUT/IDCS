import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.modules.setdefault("smbus", types.SimpleNamespace(SMBus=object))

from rpi.manual_control import ManualSwitchIO, resolve_gpio_config  # noqa: E402
from rpi.runtime_control import _RuntimeSessionLock, _pid_is_running, build_arg_parser  # noqa: E402
from rpi.panel_indicators import IndicatorState


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
    def test_missing_gpio_cannot_select_manual(self):
        switch = ManualSwitchIO(
            enabled=False, poll_dt=0.005, debounce_s=0.05,
            log=logging.getLogger("test"),
        )
        self.assertFalse(switch.setup())
        self.assertFalse(switch.update()["active"])

    def test_default_deadzone_covers_observed_idle_joystick_bias(self):
        from rpi.manual_control import map_value_to_rate

        args = build_arg_parser().parse_args([])
        self.assertEqual(args.deadzone, 12)
        self.assertEqual(map_value_to_rate(119, deadzone=args.deadzone, max_rad_s=1.0), 0.0)
        self.assertEqual(map_value_to_rate(118, deadzone=args.deadzone, max_rad_s=1.0), 0.0)

    def test_resolve_gpio_config_allows_custom_roles_and_removal(self):
        cfg = resolve_gpio_config(
            {
                "inputs": {"fire": 4, "extra_switch": 12},
                "outputs": {"red_light": 25},
                "input_pull": "up",
                "output_active_level": "low",
            }
        )

        self.assertEqual(cfg["inputs"], {"fire": 4, "extra_switch": 12})
        self.assertEqual(cfg["outputs"], {"red_light": 25})
        self.assertEqual(cfg["input_pulls"], {"fire": "up", "extra_switch": "up"})
        self.assertEqual(cfg["input_active_levels"], {"fire": "low", "extra_switch": "low"})

    def test_resolve_gpio_config_allows_per_role_input_pulls(self):
        cfg = resolve_gpio_config(
            {
                "inputs": {"fire": 4, "emergency": 16},
                "outputs": {"red_light": 25},
                "input_pull": "up",
                "input_pulls": {"fire": "down"},
                "output_active_level": "low",
            }
        )

        self.assertEqual(cfg["input_pulls"], {"fire": "down", "emergency": "up"})
        self.assertEqual(cfg["input_active_levels"], {"fire": "high", "emergency": "low"})

    def test_resolve_gpio_config_allows_per_role_active_level_override(self):
        cfg = resolve_gpio_config(
            {
                "inputs": {"fire": 4, "emergency": 16},
                "outputs": {"red_light": 25},
                "input_pull": "up",
                "input_active_levels": {"emergency": "high"},
                "output_active_level": "low",
            }
        )

        self.assertEqual(cfg["input_pulls"], {"fire": "up", "emergency": "up"})
        self.assertEqual(cfg["input_active_levels"], {"fire": "low", "emergency": "high"})

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
                    "fire_control_lamp": 21,
                    "safety_lamp": 22,
                    "green_light": 23,
                    "yellow_light": 24,
                    "red_light": 25,
                },
                "input_pull": "up",
                "input_pulls": {"fire": "down", "emergency": "up"},
                "output_active_level": "low",
            },
            log=logging.getLogger("test"),
        )

        self.assertTrue(switch.setup())
        state = switch.update()

        self.assertTrue(state["active"])
        self.assertTrue(state["control_cmd_enabled"])
        self.assertTrue(switch.fire)
        self.assertFalse(state["emergency"])
        self.assertEqual(fake_gpio.outputs[21], fake_gpio.HIGH)
        self.assertEqual(fake_gpio.outputs[22], fake_gpio.HIGH)
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.HIGH)
        self.assertEqual(
            [call[:3] for call in fake_gpio.setup_calls if call[1] == fake_gpio.IN],
            [
                (4, fake_gpio.IN, fake_gpio.PUD_DOWN),
                (5, fake_gpio.IN, fake_gpio.PUD_UP),
                (6, fake_gpio.IN, fake_gpio.PUD_UP),
                (16, fake_gpio.IN, fake_gpio.PUD_UP),
                (20, fake_gpio.IN, fake_gpio.PUD_UP),
            ],
        )

    def test_manual_switch_io_keeps_green_light_active_when_inactive(self):
        fake_gpio = FakeGPIO()
        sys.modules["RPi"] = types.SimpleNamespace(GPIO=fake_gpio)
        sys.modules["RPi.GPIO"] = fake_gpio
        fake_gpio.inputs = {
            4: fake_gpio.HIGH,
            5: fake_gpio.HIGH,
            6: fake_gpio.HIGH,
            16: fake_gpio.HIGH,
            20: fake_gpio.HIGH,
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
                    "fire_control_lamp": 21,
                    "safety_lamp": 22,
                    "green_light": 23,
                    "yellow_light": 24,
                    "red_light": 25,
                },
                "input_pull": "up",
                "output_active_level": "low",
            },
            log=logging.getLogger("test"),
        )

        self.assertTrue(switch.setup())
        state = switch.update()

        self.assertFalse(state["active"])
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.HIGH)

    def test_green_light_is_reasserted_if_pin_level_changes_externally(self):
        fake_gpio = FakeGPIO()
        sys.modules["RPi"] = types.SimpleNamespace(GPIO=fake_gpio)
        sys.modules["RPi.GPIO"] = fake_gpio
        fake_gpio.inputs = {}
        switch = ManualSwitchIO(
            enabled=True,
            poll_dt=0.005,
            debounce_s=0.05,
            gpio_config={
                "inputs": {},
                "outputs": {"green_light": 23},
                "output_active_level": "low",
            },
            log=logging.getLogger("test"),
        )

        self.assertTrue(switch.setup())
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.LOW)

        fake_gpio.outputs[23] = fake_gpio.HIGH
        switch.update()
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.LOW)

    def test_target_lamps_are_independent_of_inputs_and_yellow_is_hardware_only(self):
        fake_gpio = FakeGPIO()
        sys.modules["RPi"] = types.SimpleNamespace(GPIO=fake_gpio)
        sys.modules["RPi.GPIO"] = fake_gpio
        fake_gpio.inputs = {5: fake_gpio.LOW, 6: fake_gpio.LOW, 12: fake_gpio.LOW,
                            13: fake_gpio.HIGH, 16: fake_gpio.LOW}
        switch = ManualSwitchIO(
            enabled=True,
            poll_dt=0.005,
            debounce_s=0.05,
            gpio_config={
                "inputs": {"control_switch": 5, "emergency": 6, "fire_control": 12,
                           "fire": 13, "safety": 16},
                "outputs": {
                    "fire_control_lamp": 22, "safety_lamp": 25,
                    "yellow_light": 23, "red_light": 24,
                },
                "input_pull": "up",
                "input_pulls": {"control_switch": "down", "fire_control": "down", "safety": "down"},
                "input_active_levels": {"emergency": "high"},
                "output_active_level": "low",
            },
            log=logging.getLogger("test"),
        )

        self.assertTrue(switch.setup())
        switch.set_indicators(IndicatorState(True, True, False))
        switch.update()
        self.assertEqual(fake_gpio.outputs[22], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.LOW)

        fake_gpio.inputs[16] = fake_gpio.HIGH
        switch.update()
        self.assertEqual(fake_gpio.outputs[22], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.LOW)
        fake_gpio.inputs[16] = fake_gpio.LOW
        switch.update()

        fake_gpio.inputs[12] = fake_gpio.HIGH
        switch.update()
        self.assertTrue(switch.control_cmd_enabled)
        self.assertEqual(fake_gpio.outputs[22], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.LOW)

        fake_gpio.inputs[16] = fake_gpio.HIGH
        switch.update()
        self.assertTrue(switch.safety)
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.HIGH)

        fake_gpio.inputs[13] = fake_gpio.LOW
        switch.update()
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.HIGH)

        switch.set_indicators(IndicatorState(True, True, True))
        switch.update()
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.LOW)

        fake_gpio.inputs[6] = fake_gpio.HIGH
        switch.update()
        self.assertTrue(switch.emergency)
        self.assertEqual(fake_gpio.outputs[22], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.LOW)
        self.assertEqual(fake_gpio.outputs[24], fake_gpio.LOW)

    def test_shared_lamp_power_allows_switch_inputs_to_read_high(self):
        class SharedLampGPIO(FakeGPIO):
            def input(self, pin):
                lamp_pin = {12: 22, 16: 25}.get(pin)
                if lamp_pin is not None:
                    return self.inputs.get(pin, self.HIGH) if self.outputs.get(lamp_pin) == self.LOW else self.LOW
                return super().input(pin)

        fake_gpio = SharedLampGPIO()
        fake_gpio.inputs = {5: fake_gpio.LOW, 6: fake_gpio.LOW, 13: fake_gpio.HIGH}
        sys.modules["RPi"] = types.SimpleNamespace(GPIO=fake_gpio)
        sys.modules["RPi.GPIO"] = fake_gpio
        switch = ManualSwitchIO(
            enabled=True, poll_dt=0.005, debounce_s=0.05,
            gpio_config={
                "inputs": {"control_switch": 5, "emergency": 6, "fire_control": 12,
                           "fire": 13, "safety": 16},
                "outputs": {"fire_control_lamp": 22, "safety_lamp": 25},
                "shared_lamp_inputs": {"fire_control": "fire_control_lamp", "safety": "safety_lamp"},
                "input_pull": "up",
                "input_pulls": {"control_switch": "down", "fire_control": "down", "safety": "down"},
                "input_active_levels": {"emergency": "high"},
                "output_active_level": "low",
            },
            log=logging.getLogger("test"),
        )

        self.assertTrue(switch.setup())
        self.assertEqual(fake_gpio.outputs[22], fake_gpio.HIGH)
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.HIGH)
        self.assertFalse(switch.update()["control_cmd_enabled"])
        self.assertFalse(switch.input_validity["fire_control"])
        switch.set_indicators(IndicatorState(True, True))
        # Newly powered contacts are withheld for one poll, then reread.
        self.assertFalse(switch.update()["control_cmd_enabled"])
        state = switch.update()
        self.assertTrue(state["control_cmd_enabled"])
        self.assertTrue(switch.safety)
        switch.set_indicators(IndicatorState())
        self.assertFalse(switch.update()["control_cmd_enabled"])
        self.assertFalse(switch.safety)
        self.assertFalse(switch.input_validity["safety"])
        # Manual mode, E-stop and fire button are not lamp-powered contacts.
        self.assertTrue(switch.input_validity["control_switch"])
        self.assertTrue(switch.input_validity["emergency"])
        self.assertTrue(switch.input_validity["fire"])
        switch.set_indicators(IndicatorState(True, True))
        self.assertFalse(switch.update()["control_cmd_enabled"])
        self.assertTrue(switch.update()["control_cmd_enabled"])
        switch.set_indicators(IndicatorState())
        switch.update()
        # A contact physically opened while dark must not reuse its old high.
        fake_gpio.inputs[12] = fake_gpio.LOW
        fake_gpio.inputs[16] = fake_gpio.LOW
        switch.set_indicators(IndicatorState(True, True))
        switch.update()
        self.assertFalse(switch.update()["control_cmd_enabled"])
        self.assertFalse(switch.safety)
        self.assertTrue(switch.input_validity["safety"])
        fake_gpio.inputs[12] = fake_gpio.HIGH
        self.assertTrue(switch.update()["control_cmd_enabled"])
        self.assertFalse(switch.safety)
        switch.cleanup()
        self.assertTrue(fake_gpio.cleaned)
        self.assertTrue(all(value == fake_gpio.HIGH for value in fake_gpio.outputs.values()))

    def test_shared_contacts_cannot_be_latched_or_include_other_roles(self):
        with self.assertRaisesRegex(ValueError, "level mode"):
            resolve_gpio_config({"inputs": {"fire_control": 12},
                "outputs": {"fire_control_lamp": 22}, "input_modes": {"fire_control": "latch"},
                "shared_lamp_inputs": {"fire_control": "fire_control_lamp"}})
        with self.assertRaisesRegex(ValueError, "fire_control/safety"):
            resolve_gpio_config({"inputs": {"fire": 13}, "outputs": {"safety_lamp": 25},
                "shared_lamp_inputs": {"fire": "safety_lamp"}})

    def test_manual_exception_powers_safety_without_target_and_emergency_removes_it(self):
        from common.config import load_config_bundle
        from rpi.panel_indicators import PanelIndicators

        class SharedGPIO(FakeGPIO):
            def input(self, pin):
                lamp = {12: 22, 16: 25}.get(pin)
                if lamp is not None and self.outputs.get(lamp) != self.LOW:
                    return self.LOW
                return super().input(pin)

        gpio = SharedGPIO()
        gpio.inputs = {5: gpio.LOW, 6: gpio.LOW, 12: gpio.HIGH, 13: gpio.HIGH, 16: gpio.HIGH}
        sys.modules["RPi"] = types.SimpleNamespace(GPIO=gpio)
        sys.modules["RPi.GPIO"] = gpio
        config = load_config_bundle([Path(__file__).resolve().parents[1] / "configs/base/panel.yaml"]).data
        switch = ManualSwitchIO(enabled=True, poll_dt=.005, debounce_s=0,
            gpio_config=config["rpi"]["gpio"], log=logging.getLogger("test"))
        lamps = PanelIndicators()
        self.assertTrue(switch.setup())
        gpio.inputs[5] = gpio.HIGH
        self.assertTrue(switch.update()["active"])
        switch.set_indicators(lamps.state(0, manual_active=switch.active))
        self.assertFalse(switch.update()["control_cmd_enabled"])
        switch.update()
        self.assertTrue(switch.safety)
        self.assertTrue(switch.input_validity["safety"])
        self.assertFalse(switch.input_validity["fire_control"])
        gpio.inputs[5] = gpio.LOW
        self.assertTrue(switch.update()["active"])  # manual latch persists
        gpio.inputs[6] = gpio.HIGH
        self.assertTrue(switch.update()["emergency"])
        switch.set_indicators(lamps.state(.1, manual_active=switch.active, emergency=switch.emergency))
        switch.update()
        self.assertFalse(switch.safety)
        self.assertFalse(switch.input_validity["safety"])
        self.assertEqual(gpio.outputs[25], gpio.HIGH)
        switch.cleanup()

    def test_emergency_can_be_configured_active_high(self):
        fake_gpio = FakeGPIO()
        sys.modules["RPi"] = types.SimpleNamespace(GPIO=fake_gpio)
        sys.modules["RPi.GPIO"] = fake_gpio
        fake_gpio.inputs = {16: fake_gpio.HIGH}
        switch = ManualSwitchIO(
            enabled=True,
            poll_dt=0.005,
            debounce_s=0.05,
            gpio_config={
                "inputs": {"emergency": 16},
                "outputs": {"red_light": 25},
                "input_pull": "up",
                "input_active_levels": {"emergency": "high"},
                "output_active_level": "low",
            },
            log=logging.getLogger("test"),
        )

        self.assertTrue(switch.setup())
        state = switch.update()

        self.assertTrue(state["emergency"])
        self.assertEqual(fake_gpio.outputs[25], fake_gpio.LOW)

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
        self.assertEqual(fake_gpio.outputs[23], fake_gpio.LOW)

    def test_runtime_session_lock_rejects_active_owner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lock_path = Path(tmp_dir) / "runtime.pid"
            lock_path.write_text(str(os.getpid()), encoding="ascii")

            lock = _RuntimeSessionLock(lock_path, log=logging.getLogger("test"))

            # Test lock policy with a known PID state, not a real POSIX signal.
            # On Windows, os.kill(pid, 0) sends CTRL_C_EVENT instead of probing.
            with patch("rpi.runtime_control._pid_is_running", return_value=True) as running:
                with self.assertRaises(SystemExit):
                    lock.acquire()
                running.assert_called_once_with(os.getpid())

            self.assertEqual(lock_path.read_text(encoding="ascii").strip(), str(os.getpid()))

    def test_runtime_session_lock_removes_stale_owner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lock_path = Path(tmp_dir) / "runtime.pid"
            lock_path.write_text("99999999", encoding="ascii")

            lock = _RuntimeSessionLock(lock_path, log=logging.getLogger("test"))
            with patch("rpi.runtime_control._pid_is_running", return_value=False) as running:
                lock.acquire()
                running.assert_called_once_with(99999999)
            try:
                self.assertEqual(lock_path.read_text(encoding="ascii").strip(), str(os.getpid()))
            finally:
                lock.release()

            self.assertFalse(lock_path.exists())

    def test_pid_liveness_probe_contract_without_real_signals(self):
        for pid in (0, -1):
            with self.subTest(pid=pid), patch("rpi.runtime_control.os.kill") as probe:
                self.assertFalse(_pid_is_running(pid))
                probe.assert_not_called()
        for error, expected in ((None, True), (ProcessLookupError(), False),
                                (PermissionError(), True)):
            with self.subTest(error=error), patch("rpi.runtime_control.os.kill", side_effect=error) as probe:
                self.assertEqual(_pid_is_running(12345), expected)
                probe.assert_called_once_with(12345, 0)


if __name__ == "__main__":
    unittest.main()
