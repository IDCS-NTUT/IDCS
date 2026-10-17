"""Local-only merged-stack proof against a named commit; writes task evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import yaml

from common.config import load_config_bundle, merge_config_layers
from common.live_config import changed, digest, flatten, runtime_bundle, unit_stacks


def audit(root, base="245d103", output=Path(".task-tmp/live-config-audit")):
    before_maps, after_maps, rows = {}, {}, []
    for unit, spec in unit_stacks(root).items():
        before = {}
        for path in spec["paths"]:
            name = path.relative_to(root).as_posix()
            text = subprocess.check_output(["git", "show", f"{base}:{name}"], cwd=root).decode("utf-8")
            before = merge_config_layers(before, yaml.safe_load(text) or {})
        after = runtime_bundle(spec["paths"], overlay={}).mutable_copy()
        authored = load_config_bundle(spec["paths"]).mutable_copy()
        differences = changed(before, after)
        if differences:
            raise AssertionError((unit, differences))
        removed = sorted(set(flatten(before)) - set(flatten(authored)))
        before_maps[unit], after_maps[unit] = before, after
        rows.append(dict(unit=unit, stack=[p.relative_to(root).as_posix() for p in spec["paths"]],
                         effective_changes=0, fixed_leaves_removed=len(removed),
                         removed_paths=removed, digest=digest(after)))
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("before", before_maps), ("after", after_maps), ("summary", rows)):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} unit stacks: zero effective value changes from {base}")
    for row in rows:
        print(row["unit"], row["fixed_leaves_removed"], row["digest"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="245d103")
    parser.add_argument("--output", type=Path, default=Path(".task-tmp/live-config-audit"))
    args = parser.parse_args()
    audit(Path(__file__).resolve().parents[1], args.base, args.output)


if __name__ == "__main__":
    main()
