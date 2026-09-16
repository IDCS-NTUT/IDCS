"""Config-driven, control-free DeepStream video runtime."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from common.config import (
    ConfigError,
    load_config_bundle,
    resolve_active_return_video_profile,
    resolve_config_paths,
)


@dataclass(frozen=True)
class RuntimeSettings:
    input_mode: str
    rtp_input_port: int | None
    nvinfer_config: Path
    header_bind: str
    result_bind: str | None
    snapshot_bind: str
    return_host: str
    return_port: int
    return_width: int
    return_height: int
    return_fps: int
    return_bitrate_kbps: int
    target_selection: bool
    argus_sensor_id: int = 0
    argus_sensor_mode: int = 4
    argus_width: int = 1280
    argus_height: int = 720
    argus_fps: int = 60


def _port(endpoint: str, name: str) -> int:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "tcp" or not parsed.hostname:
        raise ValueError(f"{name} must be a tcp://host:port endpoint")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError(f"{name} must include a valid port")
    return port


def resolve_runtime_base_dir(primary_config: Path, *, cwd: Path) -> Path:
    """Resolve repository-relative runtime assets from the active config tree.

    Deployed overlays keep their primary file under ``<root>/configs``. Using
    the process working directory in that case can silently select a stale
    nvinfer profile from another checkout.
    """

    primary_config = primary_config.expanduser().resolve()
    if primary_config.parent.name == "configs":
        return primary_config.parent.parent
    return cwd.resolve()


def load_settings(config: Mapping[str, Any], *, base_dir: Path) -> RuntimeSettings:
    net, ds = config.get("net"), config.get("deepstream")
    if not isinstance(net, Mapping) or not isinstance(ds, Mapping):
        raise ValueError("configuration requires net and deepstream mappings")
    mode = str(ds.get("input_mode", "rtp")).lower()
    if mode not in {"rtp", "argus"}:
        raise ValueError("deepstream.input_mode must be 'rtp' or 'argus'")
    path = Path(str(ds.get("nvinfer_config", "")))
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"DeepStream nvinfer config does not exist: {path}")
    host = str(net.get("return_ip") or net.get("pc_ip") or "").strip()
    if not host:
        raise ValueError("net.return_ip or net.pc_ip is required")
    try:
        rtp_port, return_port = int(net["rtp_port"]), int(net["rtp_return_port"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("net RTP ports must be configured integers") from exc
    if not 1 <= rtp_port <= 65535 or not 1 <= return_port <= 65535:
        raise ValueError("net RTP ports must be valid")
    return_video: Mapping[str, Any] = {
        "width": 1280,
        "height": 720,
        "fps": 60,
        "bitrate_kbps": 8000,
    }
    if isinstance(config.get("video"), Mapping):
        try:
            return_video, _ = resolve_active_return_video_profile(config)
        except ConfigError as exc:
            raise ValueError(str(exc)) from exc

    def positive_return(name: str) -> int:
        try:
            value = int(return_video[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"return video {name} must be a configured integer") from exc
        if value <= 0:
            raise ValueError(f"return video {name} must be positive")
        return value

    def positive(name: str, default: int) -> int:
        value = int(ds.get(name, default))
        if value <= 0:
            raise ValueError(f"deepstream.{name} must be positive")
        return value
    sensor_id = int(ds.get("argus_sensor_id", 0))
    if sensor_id < 0:
        raise ValueError("deepstream.argus_sensor_id must be non-negative")
    return RuntimeSettings(
        mode,
        rtp_port if mode == "rtp" else None,
        path,
        f"tcp://0.0.0.0:{_port(str(net.get('header_push', '')), 'net.header_push')}",
        (
            f"tcp://0.0.0.0:{_port(str(net.get('zmq_results', '')), 'net.zmq_results')}"
            if bool(ds.get("legacy_display_output", False))
            else None
        ),
        f"tcp://0.0.0.0:{_port(str(net.get('zmq_perception_v2', '')), 'net.zmq_perception_v2')}",
        host,
        return_port,
        positive_return("width"),
        positive_return("height"),
        positive_return("fps"),
        positive_return("bitrate_kbps"),
        bool(ds.get("target_selection", False)),
        sensor_id,
        positive("argus_sensor_mode", 4),
        positive("argus_width", 1280),
        positive("argus_height", 720),
        positive("argus_fps", 60),
    )


def build_pipeline_argv(settings: RuntimeSettings, paths: Sequence[Path], duration_s: float | None = None,
                        report: Path | None = None, ready_file: Path | None = None,
                        health_file: Path | None = None) -> list[str]:
    argv = [
        "--nvsort", "--gpu-osd", "--return-h264", "--return-udp-host",
        settings.return_host, "--return-udp-port", str(settings.return_port),
        "--nvinfer-config", str(settings.nvinfer_config),
        "--snapshot-result-bind", settings.snapshot_bind,
        "--return-width", str(settings.return_width),
        "--return-height", str(settings.return_height),
        "--return-fps", str(settings.return_fps),
        "--return-bitrate-kbps", str(settings.return_bitrate_kbps),
    ]
    if settings.result_bind is not None:
        argv.extend(["--shadow-result-bind", settings.result_bind])
    if settings.input_mode == "rtp":
        argv.extend(["--rtp-input-port", str(settings.rtp_input_port), "--shadow-header-bind", settings.header_bind])
    else:
        argv.extend([
            "--live-argus", "--argus-sensor-id", str(settings.argus_sensor_id),
            "--argus-sensor-mode", str(settings.argus_sensor_mode), "--argus-width",
            str(settings.argus_width), "--argus-height", str(settings.argus_height),
            "--argus-fps", str(settings.argus_fps),
        ])
    if settings.target_selection:
        argv.append("--shadow-target-selection")
        for path in paths:
            argv.extend(["--idcs-config", str(path)])
    if duration_s is not None:
        argv.extend(["--duration-s", str(duration_s)])
    if report is not None:
        argv.extend(["--report", str(report)])
    if ready_file is not None:
        argv.extend(["--ready-file", str(ready_file)])
    if health_file is not None:
        argv.extend(["--health-file", str(health_file)])
    return argv


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/network.yaml")
    parser.add_argument("--config-extra", default="configs/perception.yaml,configs/control.yaml,configs/system.yaml,configs/deepstream_runtime.yaml")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--ready-file", type=Path, help="create only after first DeepStream metadata frame")
    parser.add_argument("--health-file", type=Path, help="refresh while DeepStream metadata frames arrive")
    parser.add_argument("--check", action="store_true", help="validate settings without opening sockets or video")
    args = parser.parse_args(argv)
    paths = resolve_config_paths(args.config, args.config_extra)
    try:
        bundle = load_config_bundle(paths, required_sections=("net", "deepstream"))
        settings = load_settings(
            bundle.data,
            base_dir=resolve_runtime_base_dir(bundle.paths[0], cwd=Path.cwd()),
        )
    except (ConfigError, ValueError) as exc:
        parser.error(str(exc))
    pipeline_argv = build_pipeline_argv(
        settings, bundle.paths, args.duration_s, args.report, args.ready_file,
        args.health_file,
    )
    if args.check:
        print(json.dumps({
            "settings": asdict(settings),
            "pipeline_argv": pipeline_argv,
            **bundle.provenance(),
        }, default=str, indent=2))
        return 0
    from jetson.deepstream.pipeline import run as run_pipeline
    print("[deepstream.runtime] starting control-free video runtime", flush=True)
    return run_pipeline(pipeline_argv)


if __name__ == "__main__":
    raise SystemExit(run())
