"""Repeat real VPI CUDA/NVMM gates, accepting only normal child-process exit.

Run alone on Jetson, with other IDCS GPU owners paused. No sockets, services,
device reset, finalizer bypass or motor commands. Build the optional test surface
first. This tests standalone ownership, not an existing live pipeline's lifetime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",required=True)
    parser.add_argument("--surface-library",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--repeat",type=int,default=5)
    parser.add_argument("--provenance",required=True)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 10: parser.error("repeat must be 1..10")
    report_path = Path(args.output)
    if report_path.exists(): parser.error("output must not exist")
    library,surface = Path(args.library).resolve(),Path(args.surface_library).resolve()
    hashes = {"library":hashlib.sha256(library.read_bytes()).hexdigest(),
              "test_surface":hashlib.sha256(surface.read_bytes()).hexdigest()}
    env = dict(os.environ,IDCS_VPI_LIBRARY=str(library),IDCS_VPI_TEST_SURFACE=str(surface),
               IDCS_VPI_CUDA_INTEROP="1",IDCS_VPI_STANDALONE_EGL="1")
    commands = [[sys.executable,"-m","tools.probe_vpi_cuda_interop"]]
    commands += [[sys.executable,"-m","pytest","-q","tests/test_visual_lock.py",
                  "tests/test_vpi_visual_lock_native.py"] for _ in range(args.repeat)]
    rows = []
    for command in commands:
        try:
            p = subprocess.run(command,env=env,capture_output=True,text=True,timeout=45)
            row = dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        except subprocess.TimeoutExpired as exc:
            row = dict(command=command,returncode="timeout",stdout=str(exc.stdout),stderr=str(exc.stderr))
        rows.append(row)
        print(json.dumps(row),flush=True)
        if row["returncode"] != 0: break
    passed = len(rows) == len(commands) and all(r["returncode"] == 0 for r in rows)
    report = dict(host=socket.gethostname(),platform=platform.platform(),source=args.provenance,
                  hashes=hashes,scope="standalone process with early explicit EGL owner",
                  repetitions=args.repeat,results=rows,acceptance="pass" if passed else "fail")
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x",encoding="utf-8") as f:
        f.write(json.dumps(report,indent=2)+"\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
