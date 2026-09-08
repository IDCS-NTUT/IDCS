"""Extract a raw TensorRT plan from an Ultralytics ``.engine`` export.

Ultralytics prefixes its engine with a little-endian metadata-length field and
JSON document. DeepStream ``nvinfer`` requires the following raw TensorRT plan
instead. This tool preserves the original export and writes a separate plan.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def extract_plan(source: Path, target: Path) -> dict[str, object]:
    blob = source.read_bytes()
    if len(blob) < 8:
        raise ValueError(f"engine export is too short: {source}")
    metadata_bytes = struct.unpack("<I", blob[:4])[0]
    metadata_end = 4 + metadata_bytes
    if metadata_end >= len(blob):
        raise ValueError(f"invalid Ultralytics metadata length in {source}")
    metadata = json.loads(blob[4:metadata_end])
    raw_plan = blob[metadata_end:]
    if raw_plan[:4] != b"ftrt":
        raise ValueError("payload is not a TensorRT plan; refusing to write it")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw_plan)
    return {
        "source": str(source),
        "target": str(target),
        "metadata_bytes": metadata_bytes,
        "plan_bytes": len(raw_plan),
        "description": metadata.get("description"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(json.dumps(extract_plan(args.source, args.target), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
