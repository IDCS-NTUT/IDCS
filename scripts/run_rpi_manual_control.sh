#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

CONFIG_PATH=${1:-configs/dev_extra.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

cd "${REPO_ROOT}"

python -m rpi.manual_control --config "${CONFIG_PATH}" "$@"
