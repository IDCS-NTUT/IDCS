"""RPi runtime manual-control uplink for Jetson full-pipeline operation.

Reads joystick + switch IO on the Pi and publishes structured manual-control
state updates to Jetson over ZMQ so Jetson can arbitrate behavior.

Unlike ``rpi.manual_control``, this module does not publish serial commands.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any, Mapping

import smbus  # type: ignore[import-not-found]
import yaml
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config_sync import (  # noqa: E402
    ConfigSyncError,
    acquire_config_sync_lock,
    merge_config_maps,
    parse_config_text,
    read_snapshot,
    resolve_config_sync_endpoint,
    sync_as_client,
)
from common.schemas import ManualControlState  # noqa: E402
from rpi.manual_control import ManualSwitchIO, map_value_to_rate, read_adc  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dev.yaml", help="Base YAML config path")
    parser.add_argument(
        "--config-extra",
        default="configs/dev_extra.yaml",
        help="Optional second YAML config merged over --config",
    )
    parser.add_argument(
        "--manual-state-endpoint",
        default=None,
        help="Override net.zmq_manual_state endpoint",
    )
    parser.add_argument(
        "--config-sync-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for Jetson config sync when source != sim",
    )
    parser.add_argument(
        "--config-sync-peer-id",
        default="rpi",
        help="Peer id used for config sync handshake",
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
        help="Debounce interval for S press toggle in seconds",
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
    config_path: Path,
    extra_path: Path | None,
    timeout_s: float,
    peer_id: str,
    log: logging.Logger,
) -> Mapping[str, Any]:
    config_paths = [config_path] + ([extra_path] if extra_path else [])
    initial_snapshots = {path: read_snapshot(path) for path in config_paths}

    preview_cfg = merge_config_maps(
        *(parse_config_text(snapshot.text, str(path)) for path, snapshot in initial_snapshots.items())
    )

    source_spec = str(preview_cfg.get("source", "") or "").strip().lower()
    is_sim = source_spec.startswith("sim")

    final_texts = {path: snapshot.text for path, snapshot in initial_snapshots.items()}

    if not is_sim:
        sync_endpoint = resolve_config_sync_endpoint(preview_cfg)
        log.info(
            "Config sync: source=%s requires peer=%s, endpoint=%s",
            source_spec or "<unset>",
            peer_id,
            sync_endpoint,
        )
        try:
            with acquire_config_sync_lock(config_path, timeout_s):
                for path in config_paths:
                    final_text, _ = sync_as_client(
                        path,
                        sync_endpoint,
                        config_id=path.name,
                        peer_id=peer_id,
                        max_wait=timeout_s,
                    )
                    final_texts[path] = final_text
        except ConfigSyncError as exc:
            raise SystemExit(f"config synchronization failed: {exc}") from exc
    else:
        log.info("Config sync: source=sim, skipping rpi sync handshake")

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

    config_path = Path(args.config)
    extra_path = Path(args.config_extra) if args.config_extra else None

    cfg = _load_and_optionally_sync(
        config_path=config_path,
        extra_path=extra_path,
        timeout_s=float(args.config_sync_timeout),
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

    invert_yaw_cfg = _coerce_bool("invert_yaw", runtime_cfg.get("invert_yaw"), default=False)
    invert_pitch_cfg = _coerce_bool("invert_pitch", runtime_cfg.get("invert_pitch"), default=True)
    invert_yaw = bool(args.invert_yaw) if args.invert_yaw is not None else invert_yaw_cfg
    invert_pitch = bool(args.invert_pitch) if args.invert_pitch is not None else invert_pitch_cfg

    adc_bus = smbus.SMBus(1)

    switch_io = ManualSwitchIO(
        enabled=args.switch_io,
        poll_dt=args.switch_poll_dt_s,
        debounce_s=args.switch_debounce_s,
        log=log,
    )

    ctx = zmq.Context.instance()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.SNDHWM, 1)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(str(endpoint))

    publish_period_s = _coerce_publish_period_s(float(args.publish_hz))

    try:
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

            joy_x = 128
            joy_y = 128
            note = None
            try:
                joy_x = read_adc(adc_bus, 0)
                joy_y = read_adc(adc_bus, 1)
            except OSError as exc:
                note = f"adc_error:{exc}"

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
                or (now - last_log) >= 1.0
            ):
                last_log = now
                log.info(
                    "manual state active=%s emergency=%s joy=(%d,%d) rate=(%.3f,%.3f)%s",
                    payload.active,
                    payload.emergency,
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
        switch_io.cleanup()
        try:
            push.close(0)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
