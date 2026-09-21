"""Validate the isolated DeepStream 60-FPS proof-of-concept prerequisites.

This performs no camera access and starts no IDCS services. Run it on the
Jetson after building the custom parser library:

    python -m jetson.deepstream.preflight
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
# Keep the no-argument preflight aligned with the feature-gated runtime.
# Alternative smoke profiles remain available through --engine/--config.
DEFAULT_ENGINE = REPO_ROOT / "assets/models/yolo/small_736.engine"
DEFAULT_CONFIG = REPO_ROOT / "configs/deepstream/nvinfer_yolo26s_736_drone_person_smoke.txt"
DEFAULT_PARSER = Path(__file__).with_name("libnvdsinfer_yolo26_parser.so")
DEFAULT_TRACKER_LIBRARY = Path("/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
REQUIRED_GST_ELEMENTS = ("nvinfer", "nvtracker", "nvstreammux", "nvv4l2decoder", "nvdsosd")


def _check_path(label: str, path: Path, errors: list[str]) -> None:
    if path.is_file():
        print(f"OK   {label}: {path}")
    else:
        errors.append(f"missing {label}: {path}")


def _check_gst_element(name: str, errors: list[str]) -> None:
    result = subprocess.run(
        ["gst-inspect-1.0", name], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        print(f"OK   GStreamer element: {name}")
    else:
        errors.append(f"missing or unloadable GStreamer element: {name}")


def _check_engine(engine: Path, errors: list[str]) -> None:
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec).exists():
        errors.append("trtexec is unavailable; cannot validate engine compatibility")
        return
    result = subprocess.run(
        [trtexec, f"--loadEngine={engine}", "--getPlanVersionOnly"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        print("OK   TensorRT engine deserializes on this Jetson")
    else:
        output = f"{result.stdout}\n{result.stderr}".lower()
        if (
            "no cuda-capable device" in output
            or "nvrmgpulibopen failed" in output
            or "nvrm_gpu" in output
        ):
            errors.append(
                "Jetson GPU runtime is unavailable; TensorRT engine compatibility "
                "was not evaluated"
            )
        else:
            errors.append("TensorRT cannot deserialize the configured engine")


def _check_shared_library_dependencies(label: str, library: Path, errors: list[str]) -> None:
    """Detect missing tracker dependencies before a replay run stalls."""

    if not library.is_file():
        errors.append(f"missing {label}: {library}")
        return
    ldd = shutil.which("ldd")
    if ldd is None:
        errors.append("ldd is unavailable; cannot validate tracker dependencies")
        return
    result = subprocess.run([ldd, str(library)], capture_output=True, text=True, check=False)
    unresolved = [line.strip() for line in result.stdout.splitlines() if "=> not found" in line]
    if result.returncode == 0 and not unresolved:
        print(f"OK   {label} dependencies resolve")
    else:
        errors.append(
            f"unresolved {label} dependencies: " + "; ".join(unresolved or [result.stderr.strip()])
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parser", type=Path, default=DEFAULT_PARSER)
    parser.add_argument("--tracker-library", type=Path, default=DEFAULT_TRACKER_LIBRARY)
    args = parser.parse_args(argv)

    errors: list[str] = []
    _check_path("nvinfer config", args.config, errors)
    _check_path("TensorRT engine", args.engine, errors)
    _check_path("YOLO26 parser library", args.parser, errors)
    for element in REQUIRED_GST_ELEMENTS:
        _check_gst_element(element, errors)
    _check_shared_library_dependencies("NvMultiObjectTracker library", args.tracker_library, errors)
    if args.engine.is_file():
        _check_engine(args.engine, errors)

    if errors:
        print("FAILED preflight:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASS preflight: safe to run the isolated DeepStream file/replay smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
