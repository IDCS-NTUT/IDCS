"""Source-scoped target catalogs and selection policy; never motor authority."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from common.perception import PerceptionSnapshotV2, PerceptionTrackV2, TargetSelectionV2
from common.operator_commands import OperatorSelection, parse_selection


class TargetCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: Literal["camera", "sim"]
    session: str = Field(min_length=1, max_length=80)
    sequence: int = Field(ge=0)
    created_monotonic_ns: int = Field(ge=0)
    tracks: tuple[PerceptionTrackV2, ...] = Field(default=(), max_length=512)
    selected_id: int | None = Field(default=None, ge=0)
    policy: str | None = None


class CatalogEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    catalog: TargetCatalog
    age_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)


def catalog_from_snapshot(snapshot: PerceptionSnapshotV2, source: str, session: str,
                          now_ns: int) -> TargetCatalog:
    selection = snapshot.selection
    return TargetCatalog(source=source, session=session, sequence=snapshot.sequence,
        created_monotonic_ns=now_ns, tracks=snapshot.tracks[:512],
        selected_id=None if selection is None else selection.track_id,
        policy=None if selection is None else selection.policy)


class CatalogCache:
    def __init__(self, stale_s: float = .5):
        self.stale_s = stale_s
        self.value: CatalogEnvelope | None = None
        self.received_s: float | None = None
        self.invalid = False
        self.retired: set[str] = set()

    def invalidate(self):
        self.invalid = True

    def accept(self, value: CatalogEnvelope, now_s: float) -> bool:
        if value.catalog.session in self.retired:
            return False
        if self.value is not None:
            old = self.value.catalog
            if old.session == value.catalog.session and value.catalog.sequence <= old.sequence:
                return False
            if old.session != value.catalog.session:
                self.retired.add(old.session)
        self.value, self.received_s, self.invalid = value, now_s, False
        return True

    def forward(self, now_s: float) -> CatalogEnvelope | None:
        if self.invalid or self.value is None or self.received_s is None or now_s < self.received_s:
            return None
        age = self.value.age_ms + (now_s - self.received_s) * 1000
        return CatalogEnvelope(catalog=self.value.catalog, age_ms=age) if age < self.stale_s * 1000 else None


def apply_operator_selection(snapshot: PerceptionSnapshotV2, *, mode: str,
                             track_id: int | None, source: str, session: str,
                             expected_source: str, expected_session: str,
                             fresh: bool = True) -> PerceptionSnapshotV2:
    """Operator mode never falls back to the snapshot's planner choice."""
    if mode != "operator":
        return snapshot
    selection = None
    if (fresh and source == expected_source and session == expected_session
            and any(t.track_id == track_id for t in snapshot.tracks)):
        f = snapshot.frame
        selection = TargetSelectionV2(track_id=track_id, source_frame_id=f.frame_id,
            applied_frame_id=f.frame_id, selected_time_ns=f.observed_time_ns,
            selection_clock_domain=f.observation_clock_domain, policy="operator_lock")
    return snapshot.model_copy(update={"selection": selection})


class SelectionReceiver:
    """Keep Operator mode fail-closed during silence; reject superseded authorities."""
    def __init__(self):
        self.state: OperatorSelection | None = None
        self.received_s: float | None = None
        self.retired: set[str] = set()

    def receive(self, raw: bytes | str, now_s: float) -> bool:
        state = parse_selection(raw)
        if state is None or state.authority_session in self.retired:
            return False
        if self.state is not None:
            if state.authority_session == self.state.authority_session and state.sequence < self.state.sequence:
                return False
            if state.authority_session != self.state.authority_session:
                self.retired.add(self.state.authority_session)
        self.state, self.received_s = state, now_s
        return True

    def apply(self, snapshot: PerceptionSnapshotV2, source: str, session: str, now_s: float):
        state = self.state
        if state is None:
            # The configured authority may already be in persisted Operator
            # mode. Never assume Planner while waiting for its first message.
            return snapshot.model_copy(update={"selection": None})
        fresh = self.received_s is not None and 0 <= now_s - self.received_s <= 2.0
        return apply_operator_selection(snapshot, mode=state.selection_mode,
            track_id=state.lock_track_id, source=state.source, session=state.source_session,
            expected_source=source, expected_session=session, fresh=fresh)
