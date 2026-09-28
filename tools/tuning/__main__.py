"""Command line for the tuning procedure (docs/tuning_procedure.md).

    python -m tools.tuning init RUN_DIR [--plan configs/tuning/plan.yaml]
    python -m tools.tuning latency RUN_DIR --trace TRACE.jsonl [...]
    python -m tools.tuning sysid|limits|hardware RUN_DIR      (Jetson, stack stopped)
    python -m tools.tuning fit|sim|agreement|emit RUN_DIR     (any host)
    python -m tools.tuning live-ab RUN_DIR --streamer-check CHECK.json   (Jetson, stack running)
    python -m tools.tuning status RUN_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.tuning import stages


def _print_stage(report: dict) -> int:
    gate = report["gate"]
    print(json.dumps({"passed": gate["passed"], "failures": gate["failures"]}, indent=2))
    return 0 if gate["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.tuning", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("run_dir", type=Path)
    init.add_argument("--plan", type=Path, default=Path("configs/tuning/plan.yaml"))
    latency = sub.add_parser("latency")
    latency.add_argument("run_dir", type=Path)
    latency.add_argument("--trace", type=Path, action="append", required=True)
    for name in ("sysid", "fit", "limits", "sim", "hardware", "agreement", "emit", "status"):
        sub.add_parser(name).add_argument("run_dir", type=Path)
    live = sub.add_parser("live-ab")
    live.add_argument("run_dir", type=Path)
    live.add_argument("--streamer-check", type=Path, required=True,
                      help="output of the running HIL streamer's command with --check")
    live.add_argument("--duration-s", type=int, default=30)
    live.add_argument("--controller-unit", default="idcs-controller")
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            run = stages.Run.init(args.run_dir, args.plan)
            print(f"initialized {run.root} (plan sha256 {run.manifest['plan_sha256'][:12]})")
            return 0
        run = stages.Run(args.run_dir)
        if args.command == "status":
            for name in stages.ORDER:
                entry = run.manifest["stages"].get(name)
                state = "-" if entry is None else ("passed" if entry["passed"] else "FAILED")
                print(f"{name:10s} {state}" + ("" if not entry or entry["passed"] else f"  {entry['failures']}"))
            print("qualified_config.yaml present" if (run.root / "qualified_config.yaml").exists()
                  else "not qualified yet")
            return 0
        if args.command == "latency":
            return _print_stage(stages.stage_latency(run, args.trace))
        if args.command == "live-ab":
            return _print_stage(stages.stage_live_ab(run, args.streamer_check, duration_s=args.duration_s,
                                                     controller_unit=args.controller_unit))
        return _print_stage(getattr(stages, f"stage_{args.command}")(run))
    except stages.StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
