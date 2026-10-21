"""Pure panel authority for explicit manual ROI selection.

The fire-control contact is lamp-powered. An unreadable sample is never an
OFF edge, and an ON level seen at entry can never lock a target. No actuator
or tracker socket belongs to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


RoiAction = Literal["lock", "release"]


@dataclass
class RoiPanelWorkflow:
    active: bool = False
    state: Literal["camera", "selecting", "pending", "tracking"] = "camera"
    request_id: str | None = None
    saw_readable_off: bool = False
    previous_on: bool | None = None

    def open(self) -> None:
        self.active = True
        self.state = "camera"
        self.request_id = None
        self.saw_readable_off = False
        self.previous_on = None

    def close(self) -> RoiAction | None:
        action: RoiAction | None = "release" if self.request_id is not None else None
        self.active = False
        self.state = "camera"
        self.request_id = None
        self.saw_readable_off = False
        self.previous_on = None
        return action

    def update(self, panel: Mapping, *, available: bool, allow_auto: bool = False) -> RoiAction | None:
        if not self.active:
            return None
        unsafe = (not available or panel.get("emergency") is not False
                  or panel.get("manual_active") is not True and not allow_auto)
        valid = (panel.get("inputs_valid") or {}).get("fire_control") is True
        physical_on = (panel.get("inputs") or {}).get("fire_control")
        if unsafe or not valid or type(physical_on) is not bool:
            release = self.request_id is not None
            self.state = "camera"
            self.request_id = None
            self.saw_readable_off = False
            self.previous_on = None
            return "release" if release else None

        if allow_auto and panel.get('manual_active') is not True:
            # Leaving Manual may follow an already explicit lock through the
            # normal auto gate. It must never create a seed or move an editor.
            if self.state in ('pending','tracking') and physical_on:
                return None
            release=self.request_id is not None
            self.state='camera';self.request_id=None
            self.saw_readable_off=False;self.previous_on=None
            return 'release' if release else None

        # PanelState.control_cmd_enabled is deliberately suppressed while ROI
        # is open. Use the separately sampled physical contact for the edge.
        on = physical_on
        if not on:
            release = self.request_id is not None
            self.state = "selecting"
            self.request_id = None
            self.saw_readable_off = True
            self.previous_on = False
            return "release" if release else None
        if self.state == "selecting" and self.saw_readable_off and self.previous_on is False:
            self.saw_readable_off = False
            self.previous_on = True
            return "lock"
        self.previous_on = True
        return None

    def submitted(self, request_id: str) -> None:
        if not self.active or self.state != "selecting" or self.previous_on is not True:
            raise ValueError("ROI lock is not awaiting a submitted request")
        self.request_id = request_id
        self.state = "pending"

    def result(self, request_id: str, accepted: bool) -> bool:
        if (not self.active or self.state not in ("pending", "tracking")
                or accepted and self.state != "pending" or request_id != self.request_id):
            return False
        if accepted:
            self.state = "tracking"
        else:
            self.request_id = None
            self.state = "camera"
        return True

    def disarm(self) -> str | None:
        """Abandon pending/tracking authority, requiring a new readable OFF."""
        request_id = self.request_id
        self.state = "camera"
        self.request_id = None
        self.saw_readable_off = False
        self.previous_on = None
        return request_id

    def sim_lease(self) -> dict:
        # Isolated trials never grant a camera-follow lease. The managed
        # display builds its separate SIM-only lease from all fresh gates.
        return {"type": "roi_sim_lease_v1", "enabled": False,
                "request_id": None}
