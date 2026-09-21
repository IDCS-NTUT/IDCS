"""Reject a runtime launch when the same Python module already owns a process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence


def command_runs_module(argv: Sequence[str], module: str) -> bool:
    return any(
        arg == "-m" and index + 1 < len(argv) and argv[index + 1] == module
        for index, arg in enumerate(argv)
    )


def iter_process_commands(proc_root: Path = Path("/proc")) -> Iterable[tuple[int, list[str]]]:
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if argv:
            yield pid, argv


def find_module_owners(
    modules: Sequence[str], *, proc_root: Path = Path("/proc"), self_pid: int | None = None
) -> list[dict[str, object]]:
    own_pid = os.getpid() if self_pid is None else self_pid
    owners: list[dict[str, object]] = []
    for pid, argv in iter_process_commands(proc_root):
        if pid == own_pid:
            continue
        matched = [module for module in modules if command_runs_module(argv, module)]
        if matched:
            owners.append({"pid": pid, "modules": matched, "argv": argv})
    return owners


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    owners = find_module_owners(args.module, proc_root=args.proc_root)
    result = {"modules": args.module, "owners": owners, "safe_to_start": not owners}
    print(json.dumps(result, separators=(",", ":")))
    return 1 if owners else 0


if __name__ == "__main__":
    raise SystemExit(main())
