"""Compatibility wrapper for legacy GPIO switch test entrypoint.

Use rpi.control_daemon as the single implementation and run GPIO supervisor only.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from control_daemon import main as control_daemon_main


if __name__ == "__main__":
    raise SystemExit(control_daemon_main(["--gpio-only", *sys.argv[1:]]))
