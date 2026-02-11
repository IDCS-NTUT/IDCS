from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import RPi.GPIO as GPIO
import smbus
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.gimbal.mks_servo42_rs485 import MksServo42Axis
from common.serial_io import SerialReplySubscriber, SerialUpdatePublisher

LOGGER = logging.getLogger("rpi.control_daemon")
ADC_ADDR = 0x48


@dataclass
class SupervisorState:
    active: bool
    emergency: bool
    manual_mode: bool
    active_changed: bool
    emergency_entered: bool
    emergency_exited: bool


class GpioSupervisor:
    S = 24
    S1 = 22
    S2 = 23

    L1 = 17
    L2 = 27
    J = 26
    OUT25 = 25

    INPUTS = (S, S1, S2)
    OUTPUTS = (L1, L2, J, OUT25)

    def __init__(self, *, poll_dt: float = 0.005, debounce_s: float = 0.05) -> None:
        self.poll_dt = poll_dt
        self.debounce_s = debounce_s

        self.active = False
        self.emergency = False
        self.saved_active = False
        self.prev_s_press = GPIO.HIGH
        self.last_s_ts = 0.0
        self.prev_in: dict[int, int] = {}
        self._last_out = {
            self.L1: GPIO.HIGH,
            self.L2: GPIO.HIGH,
            self.J: GPIO.LOW,
            self.OUT25: GPIO.HIGH,
        }

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.S, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.S1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.S2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.L1, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.L2, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.J, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.OUT25, GPIO.OUT, initial=GPIO.HIGH)
        self.prev_in = {p: GPIO.input(p) for p in self.INPUTS}

    def set_out(self, pin: int, level: int) -> None:
        if self._last_out.get(pin) != level:
            GPIO.output(pin, level)
            self._last_out[pin] = level
            LOGGER.info("[OUTPUT] GPIO%s set -> %s", pin, level)

    def apply_normal_outputs(self, *, manual_mode: bool) -> None:
        self.set_out(self.L1, GPIO.HIGH)
        self.set_out(self.L2, GPIO.LOW if self.active else GPIO.HIGH)
        self.set_out(self.J, GPIO.HIGH if self.active else GPIO.LOW)

        if not self.active:
            self.set_out(self.OUT25, GPIO.HIGH)
        else:
            self.set_out(self.OUT25, GPIO.LOW if manual_mode else GPIO.HIGH)

    def _enter_emergency(self) -> None:
        if self.emergency:
            return
        self.emergency = True
        self.saved_active = self.active
        LOGGER.warning("[EMERGENCY] triggered")
        self.set_out(self.L1, GPIO.LOW)
        self.set_out(self.J, GPIO.LOW)
        self.set_out(self.OUT25, GPIO.HIGH)

    def _exit_emergency(self, *, manual_mode: bool) -> None:
        self.emergency = False
        self.active = self.saved_active
        LOGGER.warning("[EMERGENCY] released")
        self.apply_normal_outputs(manual_mode=manual_mode)

    def _maintain_emergency(self) -> None:
        self.set_out(self.L1, GPIO.LOW)
        self.set_out(self.J, GPIO.LOW)
        self.set_out(self.OUT25, GPIO.HIGH)

    def poll(self) -> SupervisorState:
        s = GPIO.input(self.S)
        s1 = GPIO.input(self.S1)
        s2 = GPIO.input(self.S2)

        for p, v in ((self.S, s), (self.S1, s1), (self.S2, s2)):
            if v != self.prev_in[p]:
                LOGGER.info("[INPUT ] GPIO%s changed -> %s", p, v)
                self.prev_in[p] = v

        emergency_entered = False
        emergency_exited = False
        active_changed = False

        if s2 == GPIO.HIGH:
            was_emergency = self.emergency
            self._enter_emergency()
            emergency_entered = not was_emergency and self.emergency
        elif self.emergency:
            self._exit_emergency(manual_mode=(s1 == GPIO.HIGH))
            emergency_exited = True

        if self.emergency:
            self._maintain_emergency()
        else:
            now = time.monotonic()
            if self.prev_s_press == GPIO.HIGH and s == GPIO.LOW and (now - self.last_s_ts >= self.debounce_s):
                self.last_s_ts = now
                self.active = not self.active
                active_changed = True
                LOGGER.info("[LOGIC ] ACTIVE -> %s", self.active)
            self.apply_normal_outputs(manual_mode=(s1 == GPIO.HIGH))

        self.prev_s_press = s

        return SupervisorState(
            active=self.active,
            emergency=self.emergency,
            manual_mode=(s1 == GPIO.HIGH),
            active_changed=active_changed,
            emergency_entered=emergency_entered,
            emergency_exited=emergency_exited,
        )

    def cleanup(self) -> None:
        for pin, level in ((self.L1, GPIO.HIGH), (self.L2, GPIO.HIGH), (self.J, GPIO.LOW), (self.OUT25, GPIO.HIGH)):
            GPIO.output(pin, level)
        GPIO.cleanup()


class ManualGimbalController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.adc_bus = smbus.SMBus(1)
        self.reply_sub = SerialReplySubscriber(
            args.serial_reply_endpoint,
            topics=[f"serial.reply.{args.serial_target}"],
        )
        self.update_pub = SerialUpdatePublisher(args.serial_update_endpoint)
        self.enabled = False

    def _build_pitch_pair_speed_commands(
        self,
        *,
        cmd_prefix: str,
        pitch_rate: float,
        priority: str,
    ) -> list[dict[str, object]]:
        now_ns = time.time_ns()
        commands: list[dict[str, object]] = [
            {
                "cmd_id": f"{cmd_prefix}:pitch_a:{now_ns}",
                "func": "F6",
                "addr": self.args.pitch_a_addr,
                "payload": MksServo42Axis._encode_speed_payload(
                    pitch_rate,
                    self.args.pitch_accel_byte,
                    self.args.pitch_gear_ratio,
                ),
                "expect_reply": False,
                "expected_len": None,
                "priority": priority,
                "target": self.args.serial_target,
            },
            {
                "cmd_id": f"{cmd_prefix}:pitch_b:{now_ns}",
                "func": "F6",
                "addr": self.args.pitch_b_addr,
                "payload": MksServo42Axis._encode_speed_payload(
                    -pitch_rate,
                    self.args.pitch_accel_byte,
                    self.args.pitch_gear_ratio,
                ),
                "expect_reply": False,
                "expected_len": None,
                "priority": priority,
                "target": self.args.serial_target,
            },
        ]
        if self.args.pitch_sync_enabled:
            commands.append(
                {
                    "cmd_id": f"{cmd_prefix}:pitch_sync:{now_ns}",
                    "func": self.args.pitch_sync_func,
                    "addr": self.args.pitch_sync_addr,
                    "payload": [],
                    "expect_reply": False,
                    "expected_len": None,
                    "priority": priority,
                    "target": self.args.serial_target,
                }
            )
        return commands

    @staticmethod
    def _extract_status(reply: dict[str, object]) -> int | None:
        payload = reply.get("reply")
        if isinstance(payload, dict):
            parsed = payload.get("parsed")
            if isinstance(parsed, dict):
                status = parsed.get("status")
                if isinstance(status, int):
                    return status
            raw = payload.get("bytes")
            if isinstance(raw, list) and raw:
                head = raw[0]
                if isinstance(head, int):
                    return head
        return None

    @staticmethod
    def read_adc(bus: smbus.SMBus, ch: int) -> int:
        ctrl = 0x40 | ch
        bus.write_byte(ADC_ADDR, ctrl)
        bus.read_byte(ADC_ADDR)
        return bus.read_byte(ADC_ADDR)

    @staticmethod
    def map_value_to_rate(value: int, *, deadzone: int, max_rad_s: float) -> float:
        center = 128
        diff = value - center
        if abs(diff) < deadzone:
            return 0.0
        scale = min(max(abs(diff) / 128.0, 0.0), 1.0)
        return math.copysign(scale * max_rad_s, diff)

    def _send_update(self, commands: list[dict[str, object]]) -> None:
        self.update_pub.send_update(
            {
                "type": "SerialUpdate",
                "source": "rpi.control_daemon",
                "target": self.args.serial_target,
                "fields": {},
                "commands": commands,
                "update_ts_ms": int(time.time() * 1000),
            }
        )

    def enable_motors(self) -> None:
        if self.enabled:
            return
        self._send_update([
            {"cmd_id": "enable:yaw", "func": "F3", "addr": self.args.yaw_addr, "payload": [0x01], "expect_reply": False, "expected_len": None, "priority": "critical", "target": self.args.serial_target},
            {"cmd_id": "enable:pitch_a", "func": "F3", "addr": self.args.pitch_a_addr, "payload": [0x01], "expect_reply": False, "expected_len": None, "priority": "critical", "target": self.args.serial_target},
            {"cmd_id": "enable:pitch_b", "func": "F3", "addr": self.args.pitch_b_addr, "payload": [0x01], "expect_reply": False, "expected_len": None, "priority": "critical", "target": self.args.serial_target},
        ])
        self.enabled = True

    def startup_motor_check(self, *, timeout_s: float) -> bool:
        req_id = time.time_ns()
        expected = {
            f"startup-status:yaw:{req_id}": self.args.yaw_addr,
            f"startup-status:pitch_a:{req_id}": self.args.pitch_a_addr,
            f"startup-status:pitch_b:{req_id}": self.args.pitch_b_addr,
        }
        received: dict[str, int] = {}
        self._send_update(
            [
                {
                    "cmd_id": f"startup-status:yaw:{req_id}",
                    "func": "F1",
                    "addr": self.args.yaw_addr,
                    "payload": [],
                    "expect_reply": True,
                    "expected_len": 1,
                    "priority": "critical",
                    "target": self.args.serial_target,
                },
                {
                    "cmd_id": f"startup-status:pitch_a:{req_id}",
                    "func": "F1",
                    "addr": self.args.pitch_a_addr,
                    "payload": [],
                    "expect_reply": True,
                    "expected_len": 1,
                    "priority": "critical",
                    "target": self.args.serial_target,
                },
                {
                    "cmd_id": f"startup-status:pitch_b:{req_id}",
                    "func": "F1",
                    "addr": self.args.pitch_b_addr,
                    "payload": [],
                    "expect_reply": True,
                    "expected_len": 1,
                    "priority": "critical",
                    "target": self.args.serial_target,
                },
            ]
        )

        deadline = time.monotonic() + max(timeout_s, 0.1)
        while time.monotonic() < deadline and len(received) < len(expected):
            for reply in self.reply_sub.recv_nowait():
                cmd_id = reply.get("cmd_id")
                if not isinstance(cmd_id, str) or cmd_id not in expected:
                    continue
                if str(reply.get("func", "")).upper() != "F1":
                    continue
                status = self._extract_status(reply)
                if status is None:
                    LOGGER.warning("Startup check reply missing status: cmd_id=%s", cmd_id)
                    continue
                if status == 0:
                    LOGGER.error(
                        "Motor startup check failed: addr=%s status=0 (query failed)",
                        expected[cmd_id],
                    )
                    return False
                LOGGER.info("Motor startup check OK: addr=%s status=%s", expected[cmd_id], status)
                received[cmd_id] = status
            time.sleep(0.01)

        if len(received) != len(expected):
            missing = [cmd_id for cmd_id in expected if cmd_id not in received]
            LOGGER.error("Motor startup check timed out waiting for replies: %s", missing)
            return False
        return True

    def disable_motors(self) -> None:
        self._send_update([
            {"cmd_id": "disable:yaw", "func": "F3", "addr": self.args.yaw_addr, "payload": [0x00], "expect_reply": False, "expected_len": None, "priority": "critical", "target": self.args.serial_target},
            {"cmd_id": "disable:pitch_a", "func": "F3", "addr": self.args.pitch_a_addr, "payload": [0x00], "expect_reply": False, "expected_len": None, "priority": "critical", "target": self.args.serial_target},
            {"cmd_id": "disable:pitch_b", "func": "F3", "addr": self.args.pitch_b_addr, "payload": [0x00], "expect_reply": False, "expected_len": None, "priority": "critical", "target": self.args.serial_target},
        ])
        self.enabled = False

    def send_rates(self, yaw: float, pitch: float) -> None:
        commands = [
            {"cmd_id": f"speed:yaw:{time.time_ns()}", "func": "F6", "addr": self.args.yaw_addr, "payload": MksServo42Axis._encode_speed_payload(yaw, self.args.yaw_accel_byte, self.args.yaw_gear_ratio), "expect_reply": False, "expected_len": None, "priority": "high", "target": self.args.serial_target},
            *self._build_pitch_pair_speed_commands(cmd_prefix="speed", pitch_rate=pitch, priority="high"),
        ]
        self._send_update(commands)

    def stop(self) -> None:
        commands = [
            {"cmd_id": "stop:yaw", "func": "F6", "addr": self.args.yaw_addr, "payload": MksServo42Axis._encode_speed_payload(0.0, self.args.yaw_accel_byte, self.args.yaw_gear_ratio), "expect_reply": False, "expected_len": None, "priority": "critical", "target": self.args.serial_target},
            *self._build_pitch_pair_speed_commands(cmd_prefix="stop", pitch_rate=0.0, priority="critical"),
        ]
        self._send_update(commands)

    def read_joystick_rates(self) -> tuple[int, int, float, float]:
        joy_x = self.read_adc(self.adc_bus, 0)
        joy_y = self.read_adc(self.adc_bus, 1)
        yaw = self.map_value_to_rate(joy_x, deadzone=self.args.deadzone, max_rad_s=self.args.max_rate_rad_s)
        pitch = self.map_value_to_rate(joy_y, deadzone=self.args.deadzone, max_rad_s=self.args.max_rate_rad_s)
        if self.args.invert_yaw:
            yaw *= -1.0
        if self.args.invert_pitch:
            pitch *= -1.0
        return joy_x, joy_y, yaw, pitch

    def close(self) -> None:
        self.reply_sub.close()
        self.update_pub.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Raspberry Pi GPIO + joystick gimbal control daemon")
    parser.add_argument("--config", default=None)
    parser.add_argument("--poll-dt", type=float, default=0.05)
    parser.add_argument("--gpio-poll-dt", type=float, default=0.005)
    parser.add_argument("--debounce-s", type=float, default=0.05)
    parser.add_argument("--manual-only", action="store_true", help="Ignore ACTIVE/MANUAL switches and always drive joystick")
    parser.add_argument("--gpio-only", action="store_true", help="Run only GPIO supervisor (no serial commands)")

    parser.add_argument("--serial-update-endpoint", default="tcp://127.0.0.1:5571")
    parser.add_argument("--serial-reply-endpoint", default="tcp://127.0.0.1:5572")
    parser.add_argument("--serial-target", default="gimbal")
    parser.add_argument("--yaw-addr", default=1, type=int)
    parser.add_argument("--pitch-a-addr", default=2, type=int)
    parser.add_argument("--pitch-b-addr", default=3, type=int)
    parser.add_argument(
        "--no-pitch-sync",
        dest="pitch_sync_enabled",
        action="store_false",
        help="Disable synchronous-motion trigger after pitch A/B speed writes",
    )
    parser.set_defaults(pitch_sync_enabled=True)
    parser.add_argument("--pitch-sync-addr", default=0x00, type=int)
    parser.add_argument("--pitch-sync-func", default="0xFF")
    parser.add_argument("--yaw-accel-byte", default=10, type=int)
    parser.add_argument("--pitch-accel-byte", default=10, type=int)
    parser.add_argument("--yaw-gear-ratio", default=1.0, type=float)
    parser.add_argument("--pitch-gear-ratio", default=1.0, type=float)
    parser.add_argument("--max-rate-rad-s", default=10.0, type=float)
    parser.add_argument("--deadzone", default=8, type=int)
    parser.add_argument("--invert-yaw", action="store_true")
    parser.add_argument("--invert-pitch", action="store_true")
    parser.add_argument("--startup-check-timeout-s", default=2.0, type=float)
    return parser


def install_stop_event() -> Event:
    stop_event = Event()

    def _handler(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return stop_event


def _auto_control_enabled(config_path: str | None) -> bool:
    if not config_path:
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return bool((cfg.get("gimbal") or {}).get("auto_control_enabled", False))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to parse config %s (%s); defaulting to manual daemon mode", config_path, exc)
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    if _auto_control_enabled(args.config):
        LOGGER.warning("auto_control_enabled=true; skipping control daemon manual path")
        return 0

    stop_event = install_stop_event()
    gpio = GpioSupervisor(poll_dt=args.gpio_poll_dt, debounce_s=args.debounce_s)
    gimbal = None if args.gpio_only else ManualGimbalController(args)

    if gimbal is not None and not gimbal.startup_motor_check(timeout_s=args.startup_check_timeout_s):
        LOGGER.error("Aborting control daemon due to failed motor startup check")
        gimbal.close()
        gpio.cleanup()
        return 1

    emergency_stop_sent = False
    motors_enabled = False
    last_log = 0.0

    try:
        while not stop_event.is_set():
            state = gpio.poll()

            if gimbal is None:
                time.sleep(args.poll_dt)
                continue

            if state.emergency_entered:
                gimbal.stop()
                gimbal.disable_motors()
                emergency_stop_sent = True
                motors_enabled = False
            if state.emergency:
                if not emergency_stop_sent:
                    gimbal.stop()
                    emergency_stop_sent = True
                time.sleep(args.poll_dt)
                continue
            if state.emergency_exited:
                emergency_stop_sent = False

            if args.manual_only:
                active = True
                manual_mode = True
            else:
                active = state.active
                manual_mode = state.manual_mode

            if state.active_changed and active and not motors_enabled:
                gimbal.enable_motors()
                motors_enabled = True
            if not active:
                gimbal.stop()
                time.sleep(args.poll_dt)
                continue

            if not motors_enabled:
                gimbal.enable_motors()
                motors_enabled = True

            if manual_mode:
                try:
                    joy_x, joy_y, yaw_rate, pitch_rate = gimbal.read_joystick_rates()
                except OSError as exc:
                    LOGGER.warning("ADC read failed (%s); forcing stop", exc)
                    gimbal.stop()
                    time.sleep(args.poll_dt)
                    continue
                gimbal.send_rates(yaw_rate, pitch_rate)
                now = time.time()
                if now - last_log >= 0.5:
                    last_log = now
                    LOGGER.info(
                        "joy raw yaw=%3d pitch=%3d | cmd yaw=%.3f rad/s pitch=%.3f rad/s",
                        joy_x,
                        joy_y,
                        yaw_rate,
                        pitch_rate,
                    )
            else:
                gimbal.stop()

            time.sleep(args.poll_dt)
    except Exception:  # noqa: BLE001
        LOGGER.exception("control daemon failed")
        return 1
    finally:
        if gimbal is not None:
            try:
                gimbal.stop()
                gimbal.disable_motors()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed stop/disable during shutdown")
            gimbal.close()
        gpio.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
