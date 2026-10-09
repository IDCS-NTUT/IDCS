"""Simulator-owned, display-only score and session-scoped counter reset."""
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_SCORE_ENDPOINT = 'tcp://127.0.0.1:5580'


class SwarmScore(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    epoch: int = Field(default=0, strict=True, ge=0)
    active: int = Field(strict=True, ge=0)
    spawned: int = Field(strict=True, ge=0)
    eliminated: int = Field(strict=True, ge=0)
    breached: int = Field(strict=True, ge=0)


class ScoreResetRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    type: Literal['SimScoreReset'] = 'SimScoreReset'
    source_session: str = Field(min_length=1, max_length=80)
    client_session: str = Field(min_length=1, max_length=80)
    request_id: int = Field(strict=True, ge=1)


class ScoreResetHandler:
    """Called on the renderer thread; retries cannot reset a score twice."""
    def __init__(self, source_session, camera):
        self.source_session, self.camera = source_session, camera
        self.receipts = {}

    def handle(self, raw):
        try:
            if len(raw) > 4096:
                raise ValueError('oversized score request')
            request = ScoreResetRequest.model_validate_json(raw)
            if request.source_session != self.source_session:
                raise ValueError('simulator session changed')
            previous = self.receipts.get(request.client_session)
            if previous is not None:
                if request.request_id < previous[0]:
                    raise ValueError('retired score request')
                if request.request_id == previous[0]:
                    return previous[1]
            elif len(self.receipts) >= 64:
                raise ValueError('score client-session limit; restart simulator')
            epoch = self.camera.clear_planner_eval_score()
            reply = dict(type='SimScoreResetReply', ok=True, request_id=request.request_id,
                         source_session=self.source_session, epoch=epoch)
            self.receipts[request.client_session] = request.request_id, reply
            return reply
        except (ValueError, TypeError) as exc:
            return dict(type='SimScoreResetReply', ok=False, message=str(exc)[:160])


def simulation_metric(pose, diagnostics, diagnostics_age_s):
    """One shared HUD fact: Swarm outcome counts OR Single raw aim error.

    Error is the latest controller observation relative to the parallax aim
    point, not Kalman prediction or simulator truth; never frame-join it here.
    """
    if not pose:
        return None
    if pose.get('sim_scene') == 'swarm':
        raw = pose.get('swarm_score')
        if raw is None:
            return 'swarm_score', 'SWARM --'
        score = SwarmScore.model_validate(raw)
        return 'swarm_score', f'SWARM HIT {score.eliminated}  BREACH {score.breached}  LIVE {score.active}'
    if pose.get('sim_scene') != 'single':
        return None
    valid = (diagnostics and diagnostics.get('reason') == 'tracking'
             and diagnostics.get('track_id') is not None
             and diagnostics_age_s is not None and 0 <= diagnostics_age_s <= .25)
    capture_age_ms = None if not diagnostics else (diagnostics.get('timing') or {}).get('source_frame_age_ms')
    valid = (valid and isinstance(capture_age_ms, (int, float)) and not isinstance(capture_age_ms, bool)
             and math.isfinite(capture_age_ms) and capture_age_ms >= 0
             and capture_age_ms / 1000 + diagnostics_age_s <= .25)
    errors = [] if not valid else [(diagnostics.get(axis) or {}).get('raw_error_rad') for axis in ('yaw', 'pitch')]
    if len(errors) != 2 or not all(isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(e) for e in errors):
        return 'tracking_error', 'ERR --'
    return 'tracking_error', f'ERR {math.degrees(errors[0]):+.2f}\u00b0 / {math.degrees(errors[1]):+.2f}\u00b0'
