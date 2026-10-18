"""Inspect and change the current host's overlay through its running agent."""
from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import sys

import yaml

from common.config import load_config_bundle, merge_config_layers
from common.config_schema import validate_fields
from common.live_config import (ConfigAgent, command_endpoint, digest, flatten, read_overlay, put,
                                remove, unit_stacks, uses_key)


def fold_patch(root: Path, overlay: dict, host=None) -> str:
    """Patch the last declaring source of every affected host unit stack."""
    owners, contents = {}, {}
    stacks = unit_stacks(root, host)
    paths = sorted({p for spec in stacks.values() for p in spec["paths"]}) if stacks else sorted((root / "configs").rglob("*.yaml"))
    # Prefer base authoring ownership, not generated bench/tuned output.
    paths.sort(key=lambda p: (p.parent.name == "base", str(p)))
    for path in paths:
        value = load_config_bundle([path]).mutable_copy()
        contents[path] = value
        for key in flatten(value):
            owners[key] = path
    edits = {}
    for key, value in flatten(overlay).items():
        targets = set()
        for spec in stacks.values():
            if uses_key(spec["module"], key):
                candidates = [p for p in spec["paths"] if key in flatten(contents[p])]
                if candidates:
                    targets.add(candidates[-1])
                else:
                    parents = [p for p in spec["paths"] if key.split(".")[0] in contents[p]]
                    if parents:
                        targets.add(parents[0])
        if not targets:
            if key not in owners:
                raise ValueError(f"{key}: no authoring owner; add a reviewed overlay in the next release")
            targets.add(owners[key])
        for owner in targets:
            edits.setdefault(owner, merge_config_layers(contents[owner]))
            put(edits[owner], key, value)
    patches = []
    for path, value in sorted(edits.items()):
        # Preserve source comments and formatting with YAML node byte ranges.
        source = path.read_text(encoding="utf-8")
        replacements = []
        def visit(node, prefix=""):
            if not isinstance(node, yaml.MappingNode):
                return
            for key_node, value_node in node.value:
                key = f"{prefix}.{key_node.value}" if prefix else key_node.value
                if key in flatten(overlay) and key in flatten(value):
                    replacement = yaml.safe_dump(flatten(value)[key], default_flow_style=True).strip()
                    if replacement.endswith("\n..."):
                        replacement = replacement[:-4]
                    if value_node.end_mark.line > value_node.start_mark.line:
                        replacement += "\n" + " " * value_node.end_mark.column
                    replacements.append((value_node.start_mark.index, value_node.end_mark.index, replacement))
                else:
                    visit(value_node, key)
        visit(yaml.compose(source))
        after_source = source
        for start, end, replacement in sorted(replacements, reverse=True):
            after_source = after_source[:start] + replacement + after_source[end:]
        if yaml.safe_load(after_source) != value:
            # A new optional key has no source span. Produce an applicable
            # structural YAML patch; comment changes remain visible to review.
            after_source = yaml.safe_dump(value, sort_keys=False)
        before = source.splitlines(True)
        after = after_source.splitlines(True)
        name = path.relative_to(root).as_posix()
        patches.extend(difflib.unified_diff(before, after, fromfile="a/" + name, tofile="b/" + name))
    return "".join(patches)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("pc", "jetson"), default="pc")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--actor", default=os.environ.get("USER") or os.environ.get("USERNAME") or "operator")
    parser.add_argument("--endpoint", help="agent REQ/REP endpoint; default is host-local IPC")
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("get", "status", "diff", "fold", "clear"):
        sub = commands.add_parser(name)
        if name == "get":
            sub.add_argument("key", nargs="?")
    sub = commands.add_parser("set")
    sub.add_argument("key")
    sub.add_argument("value", help="YAML scalar or list")
    sub = commands.add_parser("set_many")
    sub.add_argument("values", help="YAML mapping of dotted keys to values")
    sub.add_argument("--unset", nargs="*", default=[], dest="unset_keys")
    sub = commands.add_parser("unset")
    sub.add_argument("key")
    sub = commands.add_parser("revert")
    sub.add_argument("n", nargs="?", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        overlay = read_overlay()
        request = dict(type="ConfigCommand", action=args.action, actor=args.actor,
                       expected_digest=digest(overlay))
        if hasattr(args, "key") and args.key:
            request["key"] = args.key
        if args.action == "set":
            # Duplicate YAML keys are rejected even in a structured CLI value.
            from common.config import _unique_key_loader
            request["value"] = yaml.load(args.value, Loader=_unique_key_loader())
        if args.action == "set_many":
            from common.config import _unique_key_loader
            request["values"] = yaml.load(args.values, Loader=_unique_key_loader())
            request["unset_keys"] = args.unset_keys
            if not isinstance(request["values"], dict):
                raise ValueError("set_many values must be a mapping")
        if args.action == "revert":
            request["n"] = args.n
        # Local preflight uses the exact host stack table before transport.
        # The agent repeats it against its own release and overlay.
        candidate = merge_config_layers(overlay)
        if args.action == "set":
            put(candidate, args.key, request["value"])
        elif args.action == "set_many":
            for key, value in request["values"].items():
                put(candidate, key, value)
            for key in args.unset_keys:
                remove(candidate, key)
        elif args.action == "unset":
            remove(candidate, args.key)
        elif args.action == "clear":
            candidate = {}
        elif args.action == "revert":
            from common.live_config import live_path
            records = [json.loads(line) for line in live_path().with_name("live-journal.jsonl").read_text(
                encoding="utf-8").splitlines()]
            transactions = [r for r in records if r.get("type") == "result"]
            if args.n < 1 or args.n > len(transactions):
                raise ValueError("revert count outside journal history")
            candidate = transactions[-args.n]["overlay_before"]
        validate_fields(candidate, overlay=True)
        validator = ConfigAgent(args.root, args.host, run=lambda _: (1, "preflight only"))
        validator.bundles(candidate)
        if args.action == "fold":
            print(fold_patch(args.root, overlay, args.host), end="")
            return 0
        import zmq
        with zmq.Context() as context:
            with context.socket(zmq.REQ) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDTIMEO, 5000)
                socket.setsockopt(zmq.RCVTIMEO, 10000)
                socket.connect(args.endpoint or command_endpoint())
                socket.send_json(request)
                reply = socket.recv_json()
        if args.action == "get" and args.key and reply.get("ok"):
            reply = {"ok": True, "key": args.key,
                     "overrides": {u: flatten(b.mutable_copy()).get(args.key)
                                   for u, b in validator.bundles(reply["overlay"]).items()}}
        print(json.dumps(reply, indent=2, sort_keys=True))
        return 0 if reply.get("ok") else 1
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
