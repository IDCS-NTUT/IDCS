"""Single-SIM controller input arbitration; no tracker or simulator truth join.

Normal Single keeps synthetic observations. During a Jetson ROI lock its
normal 5574 publisher relays only that visual selection. A stale lock withholds
selection until a new Jetson detector frame explicitly gives authority back.
"""
from common.perception import perception_snapshot_from_json, perception_snapshot_to_json


class SimRoiSelection:
    def __init__(self):
        self.snapshot = None
        self.received_s = None

    def observe(self, payload, now_s):
        try:
            snapshot = perception_snapshot_from_json(payload)
        except (ValueError, TypeError):
            return
        frame = snapshot.frame
        if (snapshot.roi_state is None or frame.source_clock_domain != "pc_monotonic"
                or frame.source_identity_verified is not True):
            return
        if self.snapshot is not None and frame.tracker_session == self.snapshot.frame.tracker_session:
            if frame.frame_id <= self.snapshot.frame.frame_id:
                return
        self.snapshot, self.received_s = snapshot, now_s

    def payload(self, ordinary_payload, now_s):
        snapshot = self.snapshot
        if snapshot is None or not snapshot.roi_state.owns_selection:
            return ordinary_payload
        if not 0 <= now_s-self.received_s <= .25:
            snapshot = snapshot.model_copy(update={"tracks": (), "assessments": (), "selection": None})
        return perception_snapshot_to_json(snapshot)
