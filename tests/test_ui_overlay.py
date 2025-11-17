import time
import unittest

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from common.control import ControlDebugOverlayConfig
from common.schemas import ControlCmd, MpcAxisDiagnostic

from pc.ui import MpcDebugOverlay  # noqa: E402  (cv2 import guarded above)


class MpcDebugOverlayTests(unittest.TestCase):
    def test_overlay_records_history_and_renders(self) -> None:
        cfg = ControlDebugOverlayConfig(
            enabled=True,
            history_window_s=0.5,
            opacity=0.9,
            bar_height_px=24,
            show_terms=("theta", "omega"),
        )
        overlay = MpcDebugOverlay(cfg)

        cmd = ControlCmd(
            frame_id=1,
            src_ts_ms=0,
            cmd_ts_ms=1,
            target_ok=True,
            target_uv=(0.0, 0.0),
            err_uv=(0.0, 0.0),
            err_rad=(0.0, 0.0),
            pan_rate_cmd=0.0,
            tilt_rate_cmd=0.0,
            controller_mode="mpc",
            mpc={
                "yaw": MpcAxisDiagnostic(
                    status="optimal",
                    cost=0.5,
                    u0=0.2,
                    terms={"theta": 0.3, "omega": 0.1},
                )
            },
        )

        now = time.time()
        overlay.ingest(cmd, now)
        self.assertGreaterEqual(len(overlay._history["yaw"]), 1)  # type: ignore[attr-defined]

        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        overlay.render(frame, now + 0.1)
        self.assertGreaterEqual(len(overlay._history["yaw"]), 1)  # type: ignore[attr-defined]

        overlay.render(frame, now + 1.0)
        self.assertEqual(len(overlay._history["yaw"]), 0)  # type: ignore[attr-defined]
