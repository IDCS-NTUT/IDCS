import time
import unittest

from common.schemas import Box
from jetson.multi_target_tracker import MultiTargetTracker


class MultiTargetTrackerTests(unittest.TestCase):
    @staticmethod
    def _box(x: float, y: float, w: float = 0.08, h: float = 0.08, cls: str = "drone") -> Box:
        return Box(x=x, y=y, w=w, h=h, cls=cls, conf=0.9)

    def test_single_object_keeps_same_id(self) -> None:
        tracker = MultiTargetTracker(min_hits=2, max_missed=5, iou_gate=0.05, center_dist_gate_px=120.0)
        seen_ids = []
        t0 = 100.0
        for frame in range(12):
            detections = [self._box(0.10 + frame * 0.01, 0.25)]
            result = tracker.update(detections, img_w=1280, img_h=720, timestamp_s=t0 + frame * 0.033)
            track = result.detection_to_track.get(0)
            self.assertIsNotNone(track)
            assert track is not None
            seen_ids.append(track.track_id)

        self.assertEqual(len(set(seen_ids)), 1)
        self.assertGreaterEqual(len(result.active_tracks), 1)
        self.assertTrue(result.active_tracks[0].confirmed)

    def test_two_objects_tracked_simultaneously(self) -> None:
        tracker = MultiTargetTracker(min_hits=1, max_missed=5, iou_gate=0.05, center_dist_gate_px=140.0)
        id_pairs = []
        t0 = 200.0
        for frame in range(10):
            detections = [
                self._box(0.10 + frame * 0.005, 0.20),
                self._box(0.70 - frame * 0.006, 0.60),
            ]
            result = tracker.update(detections, img_w=1280, img_h=720, timestamp_s=t0 + frame * 0.033)
            id_a = result.detection_to_track[0].track_id
            id_b = result.detection_to_track[1].track_id
            id_pairs.append((id_a, id_b))

        self.assertTrue(all(a != b for a, b in id_pairs))
        self.assertEqual(len({pair[0] for pair in id_pairs}), 1)
        self.assertEqual(len({pair[1] for pair in id_pairs}), 1)

    def test_short_miss_does_not_kill_track(self) -> None:
        tracker = MultiTargetTracker(min_hits=1, max_missed=3, iou_gate=0.05, center_dist_gate_px=150.0)
        t0 = 300.0

        first = tracker.update([self._box(0.30, 0.30)], img_w=1280, img_h=720, timestamp_s=t0)
        first_id = first.detection_to_track[0].track_id

        tracker.update([], img_w=1280, img_h=720, timestamp_s=t0 + 0.033)

        third = tracker.update([self._box(0.31, 0.305)], img_w=1280, img_h=720, timestamp_s=t0 + 0.066)
        third_id = third.detection_to_track[0].track_id
        self.assertEqual(first_id, third_id)

    def test_simple_crossing_has_limited_switching(self) -> None:
        tracker = MultiTargetTracker(min_hits=1, max_missed=4, iou_gate=0.05, center_dist_gate_px=130.0)
        t0 = 400.0

        id_a_history = []
        id_b_history = []
        for frame in range(14):
            xa = 0.15 + frame * 0.035
            xb = 0.85 - frame * 0.035
            detections = [
                self._box(xa, 0.40),
                self._box(xb, 0.45),
            ]
            result = tracker.update(detections, img_w=1280, img_h=720, timestamp_s=t0 + frame * 0.033)
            id_a_history.append(result.detection_to_track[0].track_id)
            id_b_history.append(result.detection_to_track[1].track_id)

        switches_a = sum(1 for i in range(1, len(id_a_history)) if id_a_history[i] != id_a_history[i - 1])
        switches_b = sum(1 for i in range(1, len(id_b_history)) if id_b_history[i] != id_b_history[i - 1])
        self.assertLessEqual(switches_a, 1)
        self.assertLessEqual(switches_b, 1)

    def test_tracker_runtime_is_lightweight(self) -> None:
        tracker = MultiTargetTracker(min_hits=1, max_missed=5, iou_gate=0.05, center_dist_gate_px=140.0)
        frames = 200
        dets_per_frame = 10

        start = time.perf_counter()
        for frame in range(frames):
            detections = [
                self._box(
                    x=0.05 + 0.08 * i + 0.002 * frame,
                    y=0.10 + 0.05 * (i % 4),
                    w=0.05,
                    h=0.06,
                )
                for i in range(dets_per_frame)
            ]
            tracker.update(detections, img_w=1280, img_h=720, timestamp_s=500.0 + frame * 0.033)

        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / frames) * 1000.0
        self.assertLess(avg_ms, 10.0)


if __name__ == "__main__":
    unittest.main()
