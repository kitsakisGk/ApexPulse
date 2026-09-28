"""Response models for the HTTP API.

Deliberately separate from the internal domain models. The wire format is a
contract with the dashboard and any other consumer, so it should change when the
product needs it to — not every time an internal field is renamed. These models
are also what FastAPI turns into the OpenAPI schema, so the field descriptions
here are the API documentation.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - pydantic resolves field types at runtime

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    """Base response model."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ScoreLine(ApiModel):
    """Match score."""

    ct: int = Field(ge=0, le=64, description="Rounds won by the Counter-Terrorists.")
    t: int = Field(ge=0, le=64, description="Rounds won by the Terrorists.")

    @property
    def rounds_played(self) -> int:
        """Total rounds decided so far."""
        return self.ct + self.t


class RoundSummary(ApiModel):
    """State of the round currently in progress."""

    number: int = Field(ge=1, le=64, description="Round number within the match.")
    phase: str = Field(description="One of freezetime, live, bomb_planted, or over.")
    seconds_remaining: float = Field(ge=0.0, description="Round clock.")
    bomb_planted: bool = Field(description="Whether the bomb is down.")
    bomb_seconds_remaining: float | None = Field(
        default=None, description="Detonation countdown; null before a plant."
    )


class TeamSummary(ApiModel):
    """Per-side state the dashboard renders."""

    alive: int = Field(ge=0, le=5, description="Players still alive.")
    health: int = Field(ge=0, description="Combined health of living players.")
    money: int = Field(ge=0, description="Combined money held by the side.")
    equipment_value: int = Field(ge=0, description="Combined value of carried equipment.")
    consecutive_losses: int = Field(ge=0, description="Rounds lost in a row.")


class PlayerSummary(ApiModel):
    """One player, as shown in the status grid."""

    player_id: str
    name: str
    team: str
    health: int = Field(ge=0, le=100)
    armour: int = Field(ge=0, le=100)
    money: int = Field(ge=0)
    weapon: str | None = Field(default=None, description="Primary weapon, if carrying one.")
    alive: bool
    kills: int = Field(ge=0)
    deaths: int = Field(ge=0)


class MomentumSummary(ApiModel):
    """Recent form over the sliding window."""

    kills_ct: int = Field(ge=0)
    kills_t: int = Field(ge=0)
    kill_delta: int = Field(description="CT kills minus T kills; positive favours CT.")
    window_seconds: float = Field(ge=0.0, description="Match time the window spans.")


class PredictionSummary(ApiModel):
    """The live win probability."""

    ct_win_probability: float = Field(ge=0.0, le=1.0)
    t_win_probability: float = Field(ge=0.0, le=1.0)
    favoured_side: str = Field(description="CT, T, or even.")
    confidence: float = Field(ge=0.0, le=1.0, description="How far from an even call.")
    latency_ms: float = Field(ge=0.0, description="Time taken to produce this prediction.")


class MatchSnapshot(ApiModel):
    """Everything the dashboard needs for one match at one instant."""

    match_id: str
    map_name: str
    sequence: int = Field(ge=0, description="Monotonic tick counter within the match.")
    timestamp: datetime
    score: ScoreLine
    round: RoundSummary
    ct: TeamSummary
    t: TeamSummary
    players: tuple[PlayerSummary, ...]
    momentum: MomentumSummary
    prediction: PredictionSummary | None = Field(
        default=None, description="Null when no model is loaded or the tick is unscoreable."
    )


class RoundOutcome(ApiModel):
    """One completed round."""

    round_number: int = Field(ge=1, le=64)
    winner: str
    reason: str
    score: ScoreLine
    timestamp: datetime


class MatchHistoryResponse(ApiModel):
    """A match's completed rounds."""

    match_id: str
    rounds_played: int = Field(ge=0)
    rounds: tuple[RoundOutcome, ...]
    ct_wins: int = Field(ge=0)
    t_wins: int = Field(ge=0)
    streak_side: str | None = Field(default=None, description="Side on a winning run.")
    streak_length: int = Field(default=0, ge=0)


class MatchListItem(ApiModel):
    """One entry in the live match list."""

    match_id: str
    map_name: str
    score: ScoreLine
    round_number: int = Field(ge=1, le=64)
    phase: str
    ct_win_probability: float | None = Field(default=None)


class MatchListResponse(ApiModel):
    """Every match with live state."""

    count: int = Field(ge=0)
    matches: tuple[MatchListItem, ...]


class ComponentHealth(ApiModel):
    """Health of one dependency."""

    name: str
    healthy: bool
    detail: str | None = Field(default=None, description="Populated when unhealthy.")


class HealthResponse(ApiModel):
    """Liveness and dependency status."""

    status: str = Field(description="ok or degraded.")
    version: str
    environment: str
    components: tuple[ComponentHealth, ...]

    @property
    def is_healthy(self) -> bool:
        """Whether every component reported healthy."""
        return all(component.healthy for component in self.components)


class ReadinessResponse(ApiModel):
    """Whether the service can serve traffic.

    Distinct from health on purpose: a process can be alive while unable to
    answer, and an orchestrator needs to tell those apart before routing to it.
    """

    ready: bool
    reason: str | None = Field(default=None, description="Populated when not ready.")


class MetricsResponse(ApiModel):
    """Operational counters."""

    live_matches: int = Field(ge=0)
    predictions_served: int = Field(ge=0)
    model_loaded: bool
    model_trained_at: str | None = Field(default=None)
    inference_p50_ms: float = Field(ge=0.0)
    inference_p95_ms: float = Field(ge=0.0)
    inference_p99_ms: float = Field(ge=0.0)
    uptime_seconds: float = Field(ge=0.0)


class ErrorResponse(ApiModel):
    """A failed request."""

    detail: str
    match_id: str | None = Field(default=None)
