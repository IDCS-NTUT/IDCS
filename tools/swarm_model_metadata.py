"""Check model/normalization binding, or bind a just-built derived artifact."""
import argparse
import json
from pathlib import Path

from common.swarm_metadata import bind_derived_model, load_normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--built-from", type=Path,
                        help="exact source just used for export/build; reviewed build provenance required")
    args = parser.parse_args()
    result = (bind_derived_model(args.model, args.built_from) if args.built_from else
              load_normalization(args.model))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
