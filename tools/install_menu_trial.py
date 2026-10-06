"""Install an uncommitted menu trial without modifying tagged checkouts.

Run on each Linux host after extracting a source archive under IDCS-trials.
Backs up only the affected unit files; --rollback restores those files. It
does not start, stop, enable, or disable any service.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


UNITS = {
    "pc": ("idcs-pc-mode-agent.service", "idcs-camera-ui.service", "idcs-sim.target",
           "idcs-sim-session.target", "idcs-sim-swarm.target",
           "idcs-sim-streamer.service", "idcs-sim-controller.service", "idcs-sim-panel.service",
           "idcs-sim-ui.service", "idcs-sim-recorder.service"),
    "jetson": ("idcs-operator-agent.service", "idcs-deepstream-camera.service",
               "idcs-deepstream-video.service", "idcs-controller-camera.service", "idcs-camera.target"),
    "pi": ("idcs-operator-display.service",),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=UNITS, required=True)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    snapshot = Path(__file__).resolve().parents[1]
    project = Path.home() / "Desktop/project"
    if snapshot.parent != project / "IDCS-trials":
        parser.error("snapshot must be a direct child of ~/Desktop/project/IDCS-trials")
    runtime = project / "IDCS-runtime"
    system = args.host == "jetson"
    destination = Path("/etc/systemd/system") if system else Path.home() / ".config/systemd/user"
    state = snapshot / ".deployment"
    backup = state / "backup"
    generated = state / "units"
    manifest = state / f"{args.host}.json"
    sudo = ["sudo", "-n"] if system else []
    if args.rollback:
        record = json.loads(manifest.read_text())
        for name, existed in record["units"].items():
            if name not in UNITS[args.host]:
                raise RuntimeError("unexpected unit in rollback record")
            target = destination / name
            if existed:
                subprocess.run([*sudo, "cp", str(backup / name), str(target)], check=True)
            else:
                subprocess.run([*sudo, "rm", "-f", "--", str(target)], check=True)
    else:
        if manifest.exists():
            parser.error("trial already installed; use a new snapshot or --rollback")
        backup.mkdir(parents=True)
        generated.mkdir()
        destination.mkdir(parents=True, exist_ok=True)
        source_host = "rpi" if args.host == "pi" else args.host
        record = {"snapshot": str(snapshot), "runtime": str(runtime), "units": {}}
        for name in UNITS[args.host]:
            target = destination / name
            if target.is_symlink():
                raise RuntimeError(f"refusing to replace symlink unit {target}")
            record["units"][name] = target.exists()
            if target.exists():
                shutil.copy2(target, backup / name)
            source = snapshot / "deploy/systemd" / source_host / name
            text = source.read_text().replace(str(runtime), str(snapshot))
            (generated / name).write_text(text)
        # Write the rollback record before the first installation write.
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        for name in UNITS[args.host]:
            subprocess.run([*sudo, "cp", str(generated / name), str(destination / name)], check=True)
        assets = snapshot / "assets"
        if not assets.exists():
            assets.symlink_to(runtime / "assets", target_is_directory=True)
        native = snapshot / "jetson/deepstream/libnvdsinfer_yolo26_parser.so"
        original = runtime / "jetson/deepstream/libnvdsinfer_yolo26_parser.so"
        if original.exists() and not native.exists():
            native.symlink_to(original)
    subprocess.run([*sudo, "systemctl", *([] if system else ["--user"]), "daemon-reload"], check=True)
    print(json.dumps({"host": args.host, "snapshot": str(snapshot), "rollback": args.rollback,
                      "services_restarted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
