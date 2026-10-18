"""Panel-only editing policy and asynchronous host-overlay workflow.

This module has no I/O. Transport, time and fresh local panel authority are
injected by OperatorDisplay. Only confirmed replies update the requested values;
service application is always described separately by the agent's receipts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time

from common.config_schema import FIELDS
from rpi.display.menu import MenuRow, MenuView, NavEvent


# Fixed UI policy: (step, minimum, maximum). Runtime/schema caps can only narrow
# these bands. CLI retains the full schema domain and null/advanced settings.
EDIT_BANDS = {
    "controller.rate_limit_rad_s": (.05, .25, .8),
    "controller.accel_limit_rad_s2": (.5, .5, 10),
    "controller.manual_rate_limit_rad_s": (.05, .25, .8),
    "controller.manual_accel_limit_rad_s2": (.5, .5, 10),
    "controller.coast_s": (.1, .1, 2),
    "operator.lock_lost_s": (.1, .1, 5),
    "controller.idle_return_s": (.1, .1, 10),
    "controller.idle_return_rate_rad_s": (.05, .05, .8),
    "deepstream.shadow_tracks.max_age_frames": (5, 0, 300),
    "deepstream.shadow_tracks.min_confidence": (.05, 0, 1),
    "deepstream.id_stitching.max_gap_frames": (5, 1, 300),
    "threat_eval.zones.*.radius_m": (5, 0, 100),
    "swarm_eval.max_engage_distance_m": (5, 0, 200),
    "swarm_eval.damage_by_class.*": (.1, 0, 10),
    "laser.dir_cam.x": (.0005, -.05, .05),
    "laser.dir_cam.y": (.0005, -.05, .05),
    "laser.dir_cam.z": (.0005, .95, 1.05),
    "laser.offset_m.x": (.001, -.5, .5),
    "laser.offset_m.y": (.001, -.9, .1),
    "laser.offset_m.z": (.001, -.5, .5),
    "camera.known_size_ranging.class_sizes_m.*": (.05, .05, 5),
}
GROUPS = ("Speed limits", "Lost target", "Detection persistence", "Engagement rules", "Calibration")
BOOLEAN_KEYS = ("deepstream.shadow_tracks.enabled", "deepstream.id_stitching.enabled", "swarm_eval.enabled")
CALIBRATION = ("laser.dir_cam.", "laser.offset_m.", "camera.known_size_ranging.class_sizes_m.")
RULES = ("threat_eval.zones.", "swarm_eval.")
EXCLUDED = "swarm_eval.excluded_target_classes"


def band_key(key):
    if key in EDIT_BANDS:
        return key
    for pattern in EDIT_BANDS:
        parts, candidate = pattern.split("."), key.split(".")
        if len(parts) == len(candidate) and all(a == "*" or a == b for a, b in zip(parts, candidate)):
            return pattern
    return None


def flat(value, prefix=""):
    result = {}
    for k, v in value.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flat(v, path))
        else:
            result[path] = v
    return result


@dataclass(frozen=True)
class Entry:
    key: str
    member: str | None = None

    @property
    def label(self):
        return f"Exclude {self.member}" if self.member else self.key


def fire_control_enabled(status, now):
    """Only fresh, explicitly readable switch-off grants editing authority."""
    try:
        panel = status.panel_state
        return not (status.panel.live(now)
                    and panel.get("inputs_valid", {}).get("fire_control") is True
                    and panel.get("inputs", {}).get("fire_control") is False)
    except (AttributeError, TypeError):
        return True


class ConfigMenu:
    def __init__(self, submit, fire_control_enabled, clock=time.monotonic):
        self.submit, self.fire_control_enabled, self.clock = submit, fire_control_enabled, clock
        self.active = False
        self.screen, self.cursor, self.group = "root", 0, 0
        self.effective, self.release, self.overlay, self.units = {}, {}, {}, {}
        self.effective_stacks, self.release_stacks = {}, {}
        self.digest = None
        self.pending = None
        self.next_refresh = 0.
        self.message = "Loading agent status"
        self.result_lines = []
        self.last_units, self.last_digest, self.pc_gap = {}, None, False
        self.entry = None
        self.proposed = None
        self.typed = None
        self.confirm_request = None
        self.confirm_return = "group"
        self.session = None
        self.touched = set()
        self.leave_after_reply = False
        self.last_step = None
        self.repeats = 0

    def open(self):
        self.active, self.screen, self.cursor = True, "root", 0
        self.refresh()

    def refresh(self):
        if self.pending is None:
            self._send(dict(action="get"), status=True)

    def tick(self):
        if (self.active and self.pending is None and self.screen in ("root", "group", "result")
                and self.clock() >= self.next_refresh):
            self._send(dict(action="status"), status=True)

    def _guard(self, keys):
        try:
            enabled = self.fire_control_enabled()
        except Exception:
            enabled = True
        if enabled is not False and any(k.startswith(RULES + CALIBRATION) for k in keys):
            self.message = "Refused: fire control enabled or unreadable"
            return False
        return True

    def request_allowed(self, request):
        if request.get("action") in ("get", "status"):
            return True
        keys = list(request.get("values", {})) + request.get("unset_keys", [])
        keys += [request["key"]] if "key" in request else []
        keys += list(request.get("entry_values", {}))
        if request.get("action") == "calibration_save":
            keys.append("laser.dir_cam.x")
        return self._guard(keys)

    def _send(self, request, *, status=False):
        if self.pending is not None:
            self.message = "Waiting for agent reply"
            return False
        if not status:
            if not self.request_allowed(request):
                return False
            if self.digest is None:
                self.message = "Agent offline; refresh required"
                self.refresh()
                return False
            request = dict(request, expected_digest=self.digest)
        actor = "pi-panel:calibration-preview" if self.session is not None else "pi-panel"
        request = dict(type="ConfigCommand", actor=actor, **request)
        if self.session is not None and request["action"] == "set_many":
            request["actor"] = "pi-panel:calibration-discard"
        if request["action"] == "calibration_save":
            request["actor"] = "pi-panel:calibration-save"
        if not self.submit(request):
            self.message = "Agent offline or command channel busy"
            return False
        self.pending = request
        if not status:
            self.message = "Waiting for agent reply"
        return True

    def on_reply(self, reply):
        request, self.pending = self.pending, None
        if request is None:
            return
        self.next_refresh = self.clock() + .8
        if not reply.get("ok"):
            self.message = "Rejected: " + str(reply.get("error", "agent offline"))
            self.result_lines = [self.message]
            self.leave_after_reply = False
            # Do not retry a mutation, especially a timed-out preview. A get is
            # the only automatic recovery; the operator must review again.
            self.digest = None
            self.last_units = {}
            if request["action"] != "get":
                self.refresh()
            return
        previous_digest = self.digest
        self.digest = reply.get("digest", self.digest)
        if request["action"] in ("get", "status"):
            if request["action"] == "get":
                effective = reply.get("effective", {})
                release = reply.get("release", {})
                self.effective_stacks, self.release_stacks = effective, release
                self.effective = self._canonical(effective)
                self.release = self._canonical(release)
                self.overlay = flat(reply.get("overlay", {}))
            self.units = reply.get("units", {})
            if self.digest == self.last_digest and not self.message.startswith(("Rejected", "Refused")):
                for unit, result in self.last_units.items():
                    status = self.units.get(unit, {})
                    if not status.get("stale", True) and status.get("requested_digest") == self.last_digest:
                        result["state"] = ("rejected" if status.get("config_error") or status.get("selection_config_error") else
                                           "pending_worker" if status.get("pending_selection_config_digest") else
                                           "pending" if status.get("pending_config_keys") else
                                           "restart_required" if status.get("restart_config_keys") else "applied")
                        if status.get("config_error") or status.get("selection_config_error"):
                            result["reason"] = status.get("config_error") or status.get("selection_config_error")
                if self.last_units:
                    self.result_lines = self._results(self.last_units)
                    if self.pc_gap:
                        self.result_lines.append("PC not updated (host-local overlay)")
                    self.message = "; ".join(self.result_lines)
            if self.message == "Waiting for agent reply":
                self.message = "Agent status refreshed"
            if request["action"] == "status" and self.digest != previous_digest:
                self.refresh()
            return
        self.last_units, self.last_digest = deepcopy(reply.get("units", {})), self.digest
        self.result_lines = self._results(self.last_units) or [
            "Calibration saved; previews retained" if request["action"] == "calibration_save" else
            "Overlay recorded; no affected active unit"]
        keys = reply.get("changed_keys", []) + list(request.get("entry_values", {}))
        self.pc_gap = any(k.startswith(("laser.", "threat_eval.zones.")) for k in keys)
        if self.pc_gap:
            self.result_lines.append("PC not updated (host-local overlay)")
        self.message = "; ".join(self.result_lines)
        if self.session is None:
            self.screen, self.cursor = "result", 0
        if self.leave_after_reply:
            self.session, self.touched = None, set()
            self.group, self.screen, self.cursor = 0, "result", 0
            self.leave_after_reply = False
        self.refresh()

    @staticmethod
    def _canonical(stacks):
        values = {}
        for key in {k for v in stacks.values() for k in v}:
            if key.startswith("controller."):
                preferred = ("idcs-controller-camera.service", "idcs-controller.service")
            elif key.startswith("operator."):
                preferred = ("idcs-operator-agent.service",)
            else:
                preferred = ("idcs-deepstream-camera.service", "idcs-deepstream-video.service",
                             "idcs-controller-camera.service")
            def priority(unit):
                return next((n for n, name in enumerate(preferred) if unit.endswith("/" + name)), len(preferred)), unit
            unit = min((u for u in stacks if key in stacks[u]), key=priority)
            values[key] = deepcopy(stacks[unit][key])
        return values

    @staticmethod
    def _results(units):
        lines = []
        names = {"pending": "pending until idle", "awaiting_service": "awaiting service reply",
                 "cannot_restart": "restart required", "pending_worker": "pending worker"}
        for unit, value in units.items():
            state = value.get("state", "unknown")
            detail = (value.get("reason") or value.get("config_error")
                      or value.get("selection_config_error") or value.get("message"))
            lines.append(f"{unit}: {names.get(state, state)}" + (f" ({detail})" if detail else ""))
        return lines

    def keys(self):
        return sorted(k for k in self.effective if band_key(k) or k in BOOLEAN_KEYS or k == EXCLUDED)

    def entries(self, group=None):
        group = self.group if group is None else group
        def belongs(k):
            return (k in list(EDIT_BANDS)[:4] if group == 0 else
                    k in list(EDIT_BANDS)[4:8] if group == 1 else
                    k.startswith("deepstream.") if group == 2 else
                    k.startswith(RULES) if group == 3 else k.startswith(CALIBRATION))
        entries = [Entry(k) for k in self.keys() if belongs(k) and k != EXCLUDED]
        if group == 3 and EXCLUDED in self.effective:
            classes = {v for k, v in self.effective.items() if k.startswith("perception.class_labels.")}
            classes.update(k.rsplit(".", 1)[1] for k in self.effective if k.startswith("swarm_eval.damage_by_class."))
            classes.update(self.effective[EXCLUDED])
            entries.extend(Entry(EXCLUDED, name) for name in sorted(classes))
        return entries

    def value(self, entry, source=None):
        value = (self.effective if source is None else source).get(entry.key)
        return entry.member in (value or []) if entry.member is not None else value

    def bounds(self, key):
        step, low, high = EDIT_BANDS[band_key(key)]
        spec = FIELDS[key]
        low, high = max(low, spec.get("min", low)), min(high, spec.get("max", high))
        if key in ("controller.rate_limit_rad_s", "controller.manual_rate_limit_rad_s"):
            caps = [v for k, v in self.release.items() if k in
                    ("gimbal.yaw_rate_limit_rad_s", "gimbal.pitch_rate_limit_rad_s")]
            caps += [v for values in self.release_stacks.values() for k, v in values.items()
                     if k in ("gimbal.yaw_rate_limit_rad_s", "gimbal.pitch_rate_limit_rad_s")]
            high = min([high, *caps])
        if key == "controller.idle_return_rate_rad_s":
            high = min(high, self.effective["controller.rate_limit_rad_s"])
        return step, low, high

    def _edit(self, entry):
        if not self._guard([entry.key]) or self.digest is None or self.pending:
            return
        self.entry, self.typed = entry, None
        self.result_lines = []
        self.proposed = deepcopy(self.effective[entry.key])
        self.last_step, self.repeats = None, 0
        if entry.member:
            if entry.member in self.proposed:
                self.proposed.remove(entry.member)
            else:
                self.proposed.append(entry.member)
            self._confirm(dict(action="set", key=entry.key, value=self.proposed))
        elif isinstance(self.proposed, bool):
            self.proposed = not self.proposed
            self._confirm(dict(action="set", key=entry.key, value=self.proposed))
        else:
            _, low, high = self.bounds(entry.key)
            if low > high:
                self.message = "No panel editing band within release caps; use CLI"
                return
            self.screen, self.cursor = "edit", 0

    def _confirm(self, request, return_to="group"):
        self.confirm_request, self.confirm_return = request, return_to
        self.screen, self.cursor = "confirm", 0

    def _preview(self, value):
        key = self.entry.key
        if self._send(dict(action="set", key=key, value=value)):
            # Track requests even if their reply is lost: they might have landed.
            self.touched.add(key)
            return True
        return False

    def text(self, char):
        if self.screen != "edit" or self.pending:
            return False
        if char == "\b":
            self.typed = (self.typed or "")[:-1]
        elif char in "0123456789.-":
            self.typed = (self.typed or "") + char
        else:
            return False
        return True

    def _typed_value(self):
        try:
            value = float(self.typed) if self.typed is not None else self.proposed
            _, low, high = self.bounds(self.entry.key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError()
            if FIELDS[self.entry.key].get("integer"):
                if value != int(value):
                    raise ValueError()
                value = int(value)
            return value
        except (ValueError, TypeError):
            self.message = "Invalid number or outside editing band; CLI for other values"
            return None

    def _leave_calibration(self):
        self.screen, self.cursor = "exit", 0

    def _finish(self, save):
        keys = sorted(self.touched)
        if not self._guard(keys):
            return
        entry_values, overlay = self.session
        if save:
            request = dict(action="calibration_save", entry_values={k: entry_values[k] for k in keys})
        else:
            request = dict(action="set_many", values={k: overlay[k] for k in keys if k in overlay},
                           unset_keys=[k for k in keys if k not in overlay])
        if not keys and not save:
            self.session = None
            self.screen, self.cursor = "root", 0
            return
        if self._send(request):
            self.leave_after_reply = True

    def _rows(self):
        if self.screen == "root":
            return [(g, ">", lambda n=n: self._enter_group(n)) for n, g in enumerate(GROUPS)] + [
                ("Last result", ">", lambda: self._show_result()),
                ("Remaining overlay keys", ">", self._remaining),
                ("Revert all", None, self._revert_all),
                ("Refresh", None, self.refresh)]
        if self.screen == "group":
            rows = []
            if self.group == 4:
                rows.append(("Finish calibration", ">", self._leave_calibration))
            for entry in self.entries():
                marker = "* " if self.value(entry) != self.value(entry, self.release) else ""
                label = entry.label if entry.member else entry.key.split(".", 1)[1]
                rows.append((marker + label, str(self.value(entry)), lambda e=entry: self._edit(e)))
            return rows
        if self.screen == "confirm":
            rows = [("Cancel", None, self._cancel_confirm), ("Confirm", None, self._commit)]
            if self.entry and self.confirm_request.get("action") == "set":
                rows.append(("Revert to release", None, lambda: self._confirm(
                    dict(action="unset", key=self.entry.key), self.confirm_return)))
            return rows
        if self.screen == "exit":
            return [("Keep editing", None, lambda: self._enter_group(4)),
                    ("Save", None, lambda: self._finish(True)),
                    ("Discard", None, lambda: self._finish(False))]
        return []

    def _enter_group(self, group):
        if group in (3, 4) and not self._guard(["swarm_eval.enabled" if group == 3 else "laser.dir_cam.x"]):
            return
        if self.digest is None or self.pending:
            return
        self.group, self.screen, self.cursor = group, "group", 0
        if group == 4 and self.session is None:
            self.session = deepcopy(self.effective), deepcopy(self.overlay)
            self.touched = set()

    def _show_result(self):
        self.screen, self.cursor = "result", 0

    def _remaining(self):
        self.result_lines = ["Remaining overlay keys:", *(sorted(set(self.overlay) - set(self.keys())) or ["none"])]
        self._show_result()

    def _revert_all(self):
        keys = [k for k in self.keys() if k in self.overlay]
        if not keys:
            self.message = "No managed overrides"
            return
        if self._guard(keys):
            self.entry = None
            self._confirm(dict(action="set_many", values={}, unset_keys=keys), "root")

    def _cancel_confirm(self):
        self.screen, self.cursor = self.confirm_return, 0

    def _commit(self):
        request = self.confirm_request
        if self._send(request):
            if self.session is not None:
                self.touched.update([request["key"]] if "key" in request else [])
            self.screen, self.cursor = self.confirm_return, 0

    def handle(self, event):
        if isinstance(event, str):
            return self.text(event)
        if self.pending is not None and self.pending["action"] not in ("get", "status"):
            return True
        if event in (NavEvent.BACK, NavEvent.LEFT, NavEvent.MENU):
            if self.screen == "confirm":
                self._cancel_confirm()
            elif self.screen == "edit" and event != NavEvent.MENU:
                self.screen, self.cursor = "group", 0
            elif self.session is not None:
                self._leave_calibration()
            elif self.screen == "root" or event == NavEvent.MENU:
                self.active = False
            else:
                self.screen, self.cursor = "root", 0
            return True
        if self.screen == "edit":
            if event in (NavEvent.UP, NavEvent.DOWN):
                if not self._guard([self.entry.key]) or self.pending:
                    return True
                now = self.clock()
                direction = 1 if event is NavEvent.UP else -1
                same = self.last_step is not None and self.last_step[0] == event and now - self.last_step[1] < .4
                self.repeats = self.repeats + 1 if same else 0
                self.last_step = event, now
                step, low, high = self.bounds(self.entry.key)
                base = self._typed_value()
                if base is None:
                    base = self.effective[self.entry.key]
                if base is None:
                    base = low
                value = round(max(low, min(high, base + direction * step * (5 if self.repeats >= 8 else 1))), 8)
                if FIELDS[self.entry.key].get("integer"):
                    value = int(value)
                self.proposed, self.typed = value, None
                if self.session is not None:
                    self._preview(value)
            elif event in (NavEvent.SELECT, NavEvent.RIGHT):
                value = self._typed_value()
                if value is not None:
                    self._confirm(dict(action="set", key=self.entry.key, value=value), "edit")
            return True
        rows = self._rows()
        if not rows:
            if event in (NavEvent.UP, NavEvent.DOWN):
                self.cursor = max(0, self.cursor + (1 if event is NavEvent.DOWN else -1))
            return True
        self.cursor = min(self.cursor, len(rows)-1)
        if event in (NavEvent.UP, NavEvent.DOWN):
            self.cursor = (self.cursor + (1 if event is NavEvent.DOWN else -1)) % len(rows)
        elif event in (NavEvent.SELECT, NavEvent.RIGHT):
            rows[self.cursor][2]()
        return True

    def view(self):
        rows = self._rows()
        title = "Settings" + (" / " + GROUPS[self.group] if self.screen in ("group", "edit") else "")
        lines = []
        if self.screen == "edit":
            key = self.entry.key
            step, low, high = self.bounds(key)
            lines = [("* " if self.effective.get(key) != self.release.get(key) else "") + key, f"Effective (requested): {self.effective.get(key)}",
                     f"Release: {self.release.get(key)}", f"Proposed: {self.typed if self.typed is not None else self.proposed}",
                     f"Step {step}; band [{low}, {high}]", "up/down nudge; select review; left cancel"]
            if self.session:
                lines[-1] = "Nudges preview live; left returns; Menu finishes"
            if any(key in s.get("pending_config_keys", []) and not s.get("stale", True) for s in self.units.values()):
                lines.append("held until idle")
            variants = {(values.get(key), self.release_stacks.get(unit, {}).get(key))
                        for unit, values in self.effective_stacks.items() if key in values}
            if len(variants) > 1:
                lines.append("Per-stack requested / release:")
                lines.extend(f"{unit.split('/')[-1]}: {values.get(key)} / {self.release_stacks.get(unit, {}).get(key)}"
                             for unit, values in sorted(self.effective_stacks.items()) if key in values)
            lines.extend(self.result_lines or [self.message])
        elif self.screen == "confirm":
            title = "Confirm configuration"
            request = self.confirm_request
            lines = [str(request.get("key", "Managed menu overrides")),
                     "Proposed: " + str(request.get("value", request.get("unset_keys", "release"))), self.message]
        elif self.screen == "result":
            title = "Settings / Last result"
            lines = self.result_lines or [self.message]
        else:
            lines = [self.message]
        if self.screen == "root":
            remaining = sorted(set(self.overlay) - set(self.keys()))
            lines += ["Remaining overlay: " + (", ".join(remaining) if remaining else "none")]
        return MenuView(title, tuple(MenuRow(label, value) for label, value, _ in rows),
                        min(self.cursor, len(rows)-1) if rows else None, tuple(lines), 2,
                        kind="config" if rows else "info", scroll=self.cursor if not rows and self.screen == "result" else 0)
