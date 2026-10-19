"""Scoped software-service guard for an isolated live visual-lock trial.

Run via a transient systemd unit; it restores only previously active software
units after the bounded child exits. Physical/HIL units are read-only checks.
No production unit/config edits, GPIO changes or motor commands.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

JETSON_SOFTWARE_UNITS = (
    "idcs-operator-agent.service", "idcs-controller-camera.service",
    "idcs-deepstream-camera.service", "idcs-deepstream-video.service",
)


def video_owners(host, proc=Path("/proc")):
    """Only actual python -m video owners, not guards listing child arguments."""
    modules = {"rpi.operator_display"} if host == "pi" else {"pc.streamer"} if host == "pc" else {
        "jetson.deepstream.runtime","jetson.visual_lock.live_trial","jetson.server",
        "tools.verify_visual_lock","tools.replay_visual_lock_capture","tools.verify_nvdcf_handoff",
        "tools.review_roi_seeds"}
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
        try: argv = (entry/"cmdline").read_bytes().decode().split("\0")
        except (OSError,UnicodeError): continue
        # The outer reviewer is waiting for this guard, not running video.
        # Keep unrelated review runners visible to duplicate-owner checks.
        if int(entry.name) == os.getppid() and argv[1:3] == ["-m", "tools.review_roi_seeds"]:
            continue
        if len(argv)>2 and argv[1] == "-m" and argv[2] in modules:
            found.append(dict(pid=int(entry.name),module=argv[2]))
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host",choices=("pc","jetson","pi"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("command",nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command: p.error("child command required")
    args.output.mkdir(parents=True,exist_ok=False)
    control = ["systemctl","--user"] if args.host != "jetson" else ["sudo","systemctl"]
    units = (["idcs-pc-mode-agent.service"] if args.host == "pc" else
             ["idcs-operator-display.service"] if args.host == "pi" else list(JETSON_SOFTWARE_UNITS))
    if args.host == "jetson":
        for unit in ("idcs-bridge.service","idcs-serial.service","idcs-controller.service","idcs-hil.target"):
            if subprocess.run(["systemctl","is-active","--quiet",unit]).returncode == 0:
                raise RuntimeError(f"physical stack active; refusing trial without touching {unit}")
    active = [u for u in units if subprocess.run(control+["is-active","--quiet",u]).returncode == 0]
    child = None
    # Single trial guard per host, including its startup/restoration window.
    import fcntl
    lock = open(f"/tmp/idcs-visual-lock-{args.host}.lock","a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status = dict(command=command,host=args.host,previously_active=active,started_wall_time=time.time())
    def stopping(*_):
        if child and child.poll() is None: child.terminate()
    signal.signal(signal.SIGTERM,stopping)
    signal.signal(signal.SIGINT,stopping)
    try:
        if active: subprocess.run(control+["stop",*active],check=True,timeout=40)
        owners = video_owners(args.host)
        if owners: raise RuntimeError(f"existing video owner; refusing duplicate: {owners}")
        with (args.output/"runtime.log").open("w") as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ)
            status["pid"] = child.pid
            (args.output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
            status["returncode"] = child.wait()
    finally:
        if child and child.poll() is None:
            child.terminate()
            child.wait(timeout=20)
        # Reverse stop order: video ready first, then shadow controller/operator.
        status["restore"] = {}
        for unit in reversed(active):
            done = subprocess.run(control+["start",unit],capture_output=True,text=True,timeout=40)
            status["restore"][unit] = dict(returncode=done.returncode,stderr=done.stderr)
        status["finished_wall_time"] = time.time()
        (args.output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
    return status.get("returncode",1)


if __name__ == "__main__":
    raise SystemExit(main())
