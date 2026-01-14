"""Deprecated wrapper for the RPi manual input client.

Use ``python -m rpi.manual_input`` instead. This module forwards to the new
manual input implementation to keep legacy entry points working.
"""

from __future__ import annotations

import logging
import sys

from rpi.manual_input import main


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("rpi.manual_control").warning(
        "rpi.manual_control is deprecated; use rpi.manual_input instead"
    )
    sys.exit(main())
