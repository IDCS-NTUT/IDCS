"""Isolated moving-texture source and one-shot test-operator geometry.

Not imported by production tracking. Geometry is used only for an explicit
operator seed in the matching source frame, never as tracker updates. Run via
the software trial guard in place of pc.streamer with the same arguments.
"""
import math


def box_at(source_time_ns):
    if type(source_time_ns) is not int or source_time_ns < 0:
        raise ValueError("exact nonnegative source timestamp required")
    phase = (source_time_ns % 24_000_000_000) / 24_000_000_000 * math.tau
    return (round(560 + 110 * math.sin(phase)),
            round(290 + 45 * math.cos(phase * 2)), 128, 96)


def seed_for(request, *, blank=False):
    if (request.get("type") != "visual_roi_seed_v1"
            or request.get("source_clock_domain") != "pc_monotonic"
            or not request.get("tracker_session")
            or type(request.get("frame_id")) is not int
            or request["frame_id"] < 0
            or (request.get("width"), request.get("height")) != (1280, 720)):
        raise ValueError("exact fixture source identity required")
    box = box_at(request["source_time_ns"])
    return {**request, **dict(zip(("x", "y", "w", "h"), (80, 80, 128, 96) if blank else box))}


def main():
    import time
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from pc import streamer

    class Capture:
        def __init__(self, spec, w, h, fps, cfg=None, **kwargs):
            if spec != "sim" or (w, h) != (1280, 720) or kwargs.get("sim_control_enabled"):
                raise ValueError("fixture requires passive 1280x720 SIM, no control")
            self.period = 1 / fps
            self.deadline = time.monotonic()
            self.gen = SimpleNamespace(drain_planner_events=lambda: [])
            rng = np.random.default_rng(736)
            self.texture = cv2.resize(rng.integers(30, 240, (24, 32, 3), dtype=np.uint8),
                                      (128, 96), interpolation=cv2.INTER_NEAREST)

        def isOpened(self): return True
        def release(self): pass

        def read(self):
            time.sleep(max(0, self.deadline - time.monotonic()))
            self.deadline = max(self.deadline + self.period, time.monotonic())
            self.last_frame_source_ns = time.monotonic_ns()
            x, y, w, h = box_at(self.last_frame_source_ns)
            image = np.full((720, 1280, 3), 32, np.uint8)
            image[y:y+h, x:x+w] = self.texture
            return True, image

    streamer.open_source = Capture
    streamer.main()


if __name__ == "__main__":
    main()
