"""RPi runtime manual-control uplink for Jetson full-pipeline operation.

Reads joystick + switch IO on the Pi and publishes structured manual-control
state updates to Jetson over ZMQ so Jetson can arbitrate behavior.

Unlike ``rpi.manual_control``, this module does not publish serial commands.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Mapping, Optional

import smbus  # type: ignore[import-not-found]
import yaml
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config_sync import (  # noqa: E402
    ConfigSyncError,
    expand_config_paths,
    merge_config_maps,
    parse_config_text,
    request_startup_state,
    read_snapshot,
    resolve_config_sync_endpoint,
    sync_as_client,
)
from common.schemas import ManualControlState  # noqa: E402
from rpi.manual_control import ManualSwitchIO, map_value_to_rate, read_adc, resolve_gpio_config  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml", help="Base YAML config path")
    parser.add_argument(
        "--config-extra",
        default="configs/perception.yaml,configs/control.yaml,configs/system.yaml",
        help="Comma-separated YAML configs merged over --config",
    )
    parser.add_argument(
        "--manual-state-endpoint",
        default=None,
        help="Override net.zmq_manual_state endpoint",
    )
    parser.add_argument(
        "--config-sync-timeout",
        type=float,
        default=None,
        help="Seconds to wait for Jetson config sync (default: wait indefinitely)",
    )
    parser.add_argument(
        "--config-sync-peer-id",
        default="rpi",
        help="Peer id used for config sync handshake",
    )
    parser.add_argument(
        "--session-lock-path",
        default="/tmp/idcs-rpi-runtime-control.pid",
        help="PID file used to prevent multiple runtime_control sessions",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=20.0,
        help="Manual state publish rate in Hz",
    )
    parser.add_argument(
        "--max-rate-rad-s",
        default=1.0,
        type=float,
        help="Clamp joystick output to this magnitude (rad/s)",
    )
    parser.add_argument(
        "--deadzone",
        default=8,
        type=int,
        help="Ignore joystick deltas smaller than this ADC count",
    )
    parser.add_argument(
        "--invert-yaw",
        dest="invert_yaw",
        action="store_true",
        help="Invert yaw joystick sign",
    )
    parser.add_argument(
        "--no-invert-yaw",
        dest="invert_yaw",
        action="store_false",
        help="Disable yaw joystick inversion",
    )
    parser.add_argument(
        "--invert-pitch",
        dest="invert_pitch",
        action="store_true",
        help="Invert pitch joystick sign",
    )
    parser.add_argument(
        "--no-invert-pitch",
        dest="invert_pitch",
        action="store_false",
        help="Disable pitch joystick inversion",
    )
    parser.add_argument(
        "--switch-io",
        dest="switch_io",
        action="store_true",
        help="Enable RPi GPIO switch/emergency control",
    )
    parser.add_argument(
        "--no-switch-io",
        dest="switch_io",
        action="store_false",
        help="Disable RPi GPIO switch/emergency control",
    )
    parser.set_defaults(switch_io=True, invert_yaw=None, invert_pitch=None)
    parser.add_argument(
        "--switch-poll-dt-s",
        default=0.005,
        type=float,
        help="GPIO switch polling interval in seconds",
    )
    parser.add_argument(
        "--switch-debounce-s",
        default=0.05,
        type=float,
        help="Debounce interval reserved for GPIO switch handling in seconds",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def install_stop_event() -> Event:
    stop_event = Event()

    def _handler(signum, _frame):
        stop_event.set()
        return None

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return stop_event


def _load_and_optionally_sync(
    *,
    config_paths: list[Path],
    timeout_s: Optional[float],
    peer_id: str,
    log: logging.Logger,
) -> Mapping[str, Any]:
    initial_snapshots = {path: read_snapshot(path) for path in config_paths}

    preview_cfg = merge_config_maps(
        *(parse_config_text(snapshot.text, str(path)) for path, snapshot in initial_snapshots.items())
    )

    final_texts = {path: snapshot.text for path, snapshot in initial_snapshots.items()}

    source_spec = str(preview_cfg.get("source", "") or "").strip().lower()
    sync_endpoint = resolve_config_sync_endpoint(preview_cfg)
    startup_probe_wait: Optional[float]
    if timeout_s is not None:
        startup_probe_wait = timeout_s
    else:
        startup_probe_wait = 1.0
    try:
        startup_state = request_startup_state(
            sync_endpoint,
            peer_id=peer_id,
            max_wait=startup_probe_wait,
            retry_interval=0.2,
        )
        startup_source = str(startup_state.get("effective_source", "") or "").strip().lower()
        if startup_source:
            if startup_source != source_spec:
                log.info(
                    "Config sync: startup source override from Jetson is %s (local=%s)",
                    startup_source,
                    source_spec or "<unset>",
                )
            source_spec = startup_source
    except ConfigSyncError as exc:
        log.info("Config sync: startup probe unavailable; using local source (%s)", exc)

    log.info(
        "Config sync: source=%s requires peer=%s, endpoint=%s",
        source_spec or "<unset>",
        peer_id,
        sync_endpoint,
    )
    try:
        for path in config_paths:
            log.info(
                "Config sync: requesting %s as peer=%s",
                path.name,
                peer_id,
            )
            final_text, _ = sync_as_client(
                path,
                sync_endpoint,
                config_id=path.name,
                peer_id=peer_id,
                retry_interval=0.2,
                max_wait=timeout_s,
            )
            final_texts[path] = final_text
            log.info(
                "Config sync: completed %s as peer=%s",
                path.name,
                peer_id,
            )
    except ConfigSyncError as exc:
        raise SystemExit(f"config synchronization failed: {exc}") from exc

    return merge_config_maps(*(parse_config_text(final_texts[path], str(path)) for path in config_paths))


def _coerce_publish_period_s(rate_hz: float) -> float:
    if rate_hz <= 0:
        return 0.05
    return max(1.0 / rate_hz, 0.01)


def _coerce_bool(name: str, raw: Any, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"rpi.runtime_control.{name} must be a boolean, got {raw!r}")


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        if pid == os.getpid():
            return True
        try:
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL

            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return True
                return int(exit_code.value) == still_active
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class _RuntimeSessionLock:
    def __init__(self, path: Path, *, log: logging.Logger) -> None:
        self._path = path
        self._log = log
        self._fd: int | None = None
        self._acquired = False

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                owner_pid = self._read_owner_pid()
                if owner_pid is not None and not _pid_is_running(owner_pid):
                    try:
                        self._path.unlink()
                    except FileNotFoundError:
                        continue
                    self._log.warning(
                        "removed stale runtime_control session lock %s owned by dead pid %s",
                        self._path,
                        owner_pid,
                    )
                    continue
                owner = str(owner_pid) if owner_pid is not None else "unknown"
                raise SystemExit(
                    "rpi.runtime_control already appears to be running "
                    f"(pid={owner}, lock={self._path}); stop the old session or remove a stale lock"
                )
            break

        os.write(self._fd, str(os.getpid()).encode("ascii"))
        os.close(self._fd)
        self._fd = None
        self._acquired = True
        self._log.info("runtime_control session lock acquired: %s", self._path)

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if not self._acquired:
            return
        try:
            current_pid = self._read_owner_pid()
            if current_pid == os.getpid():
                self._path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._acquired = False

    def _read_owner_pid(self) -> int | None:
        try:
            raw = self._path.read_text(encoding="ascii").strip()
            return int(raw)
        except (OSError, ValueError):
            return None


class _AdcReader:
    def __init__(self, bus: smbus.SMBus, *, poll_period_s: float, log: logging.Logger) -> None:
        self._bus = bus
        self._poll_period_s = max(float(poll_period_s), 0.01)
        self._log = log
        self._stop = Event()
        self._lock = Lock()
        self._latest: tuple[int, int] | None = None
        self._latest_mono: float | None = None
        self._last_error: str | None = None
        self._error_active = False
        self._last_error_log_mono = 0.0
        self._error_log_interval_s = 5.0
        self._thread = Thread(target=self._run, name="rpi-adc-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def snapshot(self) -> tuple[int, int, Optional[str]]:
        with self._lock:
            latest = self._latest
            latest_mono = self._latest_mono
            last_error = self._last_error

        if latest is None:
            return 128, 128, (f"adc_error:{last_error}" if last_error else "adc_unavailable")

        age_s = 0.0 if latest_mono is None else max(0.0, time.monotonic() - latest_mono)
        if age_s > max(1.0, self._poll_period_s * 10.0):
            return latest[0], latest[1], f"adc_stale:{age_s:.2f}s"
        if last_error:
            return latest[0], latest[1], f"adc_error:{last_error}"
        return latest[0], latest[1], None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                joy_x = read_adc(self._bus, 0)
                joy_y = read_adc(self._bus, 1)
                now = time.monotonic()
                if self._error_active:
                    self._log.info("ADC read recovered")
                    self._error_active = False
                with self._lock:
                    self._latest = (int(joy_x), int(joy_y))
                    self._latest_mono = now
                    self._last_error = None
            except OSError as exc:
                err = str(exc)
                now = time.monotonic()
                if (not self._error_active) or ((now - self._last_error_log_mono) >= self._error_log_interval_s):
                    self._log.warning("ADC read failed: %s", err)
                    self._last_error_log_mono = now
                self._error_active = True
                with self._lock:
                    self._last_error = err
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                now = time.monotonic()
                if (not self._error_active) or ((now - self._last_error_log_mono) >= self._error_log_interval_s):
                    self._log.warning("ADC reader error: %s", err)
                    self._last_error_log_mono = now
                self._error_active = True
                with self._lock:
                    self._last_error = err
            finally:
                time.sleep(self._poll_period_s)


def main() -> int:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("rpi.runtime_control")

    if args.config_sync_timeout is not None and args.config_sync_timeout < 0:
        raise SystemExit("--config-sync-timeout must be >= 0")

    stop_event = install_stop_event()
    session_lock = _RuntimeSessionLock(Path(args.session_lock_path), log=log)
    session_lock.acquire()

    adc_bus = None
    try:
        config_paths = expand_config_paths(args.config, args.config_extra)

        cfg = _load_and_optionally_sync(
            config_paths=config_paths,
            timeout_s=args.config_sync_timeout,
            peer_id=str(args.config_sync_peer_id),
            log=log,
        )

        net_cfg = cfg.get("net") if isinstance(cfg, Mapping) else None
        if not isinstance(net_cfg, Mapping):
            raise SystemExit("config missing net section")

        endpoint = args.manual_state_endpoint or net_cfg.get("zmq_manual_state")
        if not endpoint:
            raise SystemExit("manual state endpoint not configured (net.zmq_manual_state)")

        rpi_cfg = cfg.get("rpi") if isinstance(cfg, Mapping) else None
        runtime_cfg_raw = rpi_cfg.get("runtime_control") if isinstance(rpi_cfg, Mapping) else None
        runtime_cfg = runtime_cfg_raw if isinstance(runtime_cfg_raw, Mapping) else {}
        gpio_cfg_raw = rpi_cfg.get("gpio") if isinstance(rpi_cfg, Mapping) else None
        gpio_cfg = gpio_cfg_raw if isinstance(gpio_cfg_raw, Mapping) else None

        invert_yaw_cfg = _coerce_bool("invert_yaw", runtime_cfg.get("invert_yaw"), default=False)
        invert_pitch_cfg = _coerce_bool("invert_pitch", runtime_cfg.get("invert_pitch"), default=True)
        invert_yaw = bool(args.invert_yaw) if args.invert_yaw is not None else invert_yaw_cfg
        invert_pitch = bool(args.invert_pitch) if args.invert_pitch is not None else invert_pitch_cfg
        try:
            gpio_layout = resolve_gpio_config(gpio_cfg)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        log.info("GPIO layout resolved: inputs=%s outputs=%s", gpio_layout["inputs"], gpio_layout["outputs"])

        log.info("opening ADC I2C bus 1")
        adc_bus = smbus.SMBus(1)

        switch_io = ManualSwitchIO(
            enabled=args.switch_io,
            poll_dt=args.switch_poll_dt_s,
            debounce_s=args.switch_debounce_s,
            gpio_config=gpio_layout,
            log=log,
        )

        ctx = zmq.Context.instance()
        push = ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.SNDHWM, 1)
        push.setsockopt(zmq.LINGER, 0)
        push.connect(str(endpoint))

        publish_period_s = _coerce_publish_period_s(float(args.publish_hz))
        adc_reader = _AdcReader(adc_bus, poll_period_s=min(0.05, publish_period_s), log=log)

        adc_reader.start()
        log.info("initializing switch GPIO")
        switch_io.setup()
        log.info("publishing ManualControlState to %s @ %.1f Hz", endpoint, 1.0 / publish_period_s)
        log.info(
            "joystick inversion resolved: yaw=%s pitch=%s",
            invert_yaw,
            invert_pitch,
        )

        switch_state: dict[str, bool] = {
            "active": True,
            "active_changed": False,
            "emergency": False,
            "emergency_entered": False,
            "emergency_exited": False,
            "control_cmd_enabled": False,
            "control_cmd_changed": False,
        }
        next_publish_tick = time.monotonic()
        next_switch_tick = next_publish_tick
        last_log = 0.0

        while not stop_event.is_set():
            now_loop = time.monotonic()
            if now_loop >= next_switch_tick:
                switch_state = switch_io.update()
                next_switch_tick = now_loop + switch_io.poll_dt

            if now_loop < next_publish_tick:
                sleep_s = min(next_publish_tick - now_loop, max(0.0, next_switch_tick - now_loop))
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue

            joy_x, joy_y, note = adc_reader.snapshot()

            yaw_rate = map_value_to_rate(
                joy_x, deadzone=args.deadzone, max_rad_s=args.max_rate_rad_s
            )
            pitch_rate = map_value_to_rate(
                joy_y, deadzone=args.deadzone, max_rad_s=args.max_rate_rad_s
            )
            if invert_yaw:
                yaw_rate *= -1.0
            if invert_pitch:
                pitch_rate *= -1.0

            payload = ManualControlState(
                src_ts_ms=int(time.time() * 1000),
                source="rpi.runtime_control",
                active=bool(switch_state.get("active", True)),
                emergency=bool(switch_state.get("emergency", False)),
                active_changed=bool(switch_state.get("active_changed", False)),
                emergency_entered=bool(switch_state.get("emergency_entered", False)),
                emergency_exited=bool(switch_state.get("emergency_exited", False)),
                control_cmd_enabled=bool(switch_state.get("control_cmd_enabled", False)),
                control_cmd_changed=bool(switch_state.get("control_cmd_changed", False)),
                joystick_raw=(int(joy_x), int(joy_y)),
                joystick_rate_cmd=(float(yaw_rate), float(pitch_rate)),
                serial_local_mode=False,
                note=note,
            )

            try:
                push.send_string(payload.model_dump_json(exclude_none=True), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            now = time.monotonic()
            if (
                payload.emergency
                or payload.active_changed
                or payload.emergency_entered
                or payload.emergency_exited
                or payload.control_cmd_changed
                or (now - last_log) >= 1.0
            ):
                last_log = now
                log.info(
                    "manual state active=%s emergency=%s cmd_enabled=%s joy=(%d,%d) rate=(%.3f,%.3f)%s",
                    payload.active,
                    payload.emergency,
                    payload.control_cmd_enabled,
                    payload.joystick_raw[0],
                    payload.joystick_raw[1],
                    payload.joystick_rate_cmd[0],
                    payload.joystick_rate_cmd[1],
                    f" note={payload.note}" if payload.note else "",
                )

            next_publish_tick += publish_period_s
            if next_publish_tick < time.monotonic():
                next_publish_tick = time.monotonic()

    except Exception as exc:  # noqa: BLE001
        log.error("runtime control failed: %s", exc)
        return 1
    finally:
        if "adc_reader" in locals():
            adc_reader.stop()
        if "switch_io" in locals():
            switch_io.cleanup()
        if adc_bus is not None and hasattr(adc_bus, "close"):
            try:
                adc_bus.close()
            except Exception:
                pass
        try:
            if "push" in locals():
                push.close(0)
        except Exception:
            pass
        session_lock.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
