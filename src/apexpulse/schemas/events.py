"""Telemetry events carried over the broker.

Events form a discriminated union keyed on ``event_type``, so a consumer can parse
an arbitrary payload off a topic and receive a correctly-typed model without
inspecting the JSON itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, TypeAdapter, model_validator

from apexpulse.schemas import constants
from apexpulse.schemas.enums import EventType, MapName, RoundEndReason, Team, Weapon
from apexpulse.schemas.models import ApexModel, MatchState, Position, utcnow

SCHEMA_VERSION = "1.0"
"""Bumped when a change breaks wire compatibility; travels in the broker headers."""


class BaseEvent(ApexModel):
    """Fields shared by every telemetry event."""

    match_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime = Field(default_factory=utcnow)
    sequence: int = Field(ge=0, description="Monotonic counter, unique within a match.")


class MatchStartEvent(BaseEvent):
    """Emitted once, before the first round."""

    event_type: Literal[EventType.MATCH_START] = EventType.MATCH_START
    map_name: MapName
    team_ct_name: str = Field(min_length=1, max_length=64)
    team_t_name: str = Field(min_length=1, max_length=64)


class RoundStartEvent(BaseEvent):
    """Emitted at the start of freezetime."""

    event_type: Literal[EventType.ROUND_START] = EventType.ROUND_START
    round_number: int = Field(ge=1, le=64)
    score_ct: int = Field(ge=0, le=64)
    score_t: int = Field(ge=0, le=64)


class TickEvent(BaseEvent):
    """A periodic full-state snapshot.

    Ticks are the pipeline's heartbeat: the feature extractor scores one vector per
    tick, so this event carries the complete state rather than a delta.
    """

    event_type: Literal[EventType.TICK] = EventType.TICK
    state: MatchState

    @model_validator(mode="after")
    def _state_belongs_to_this_match(self) -> Self:
        if self.state.match_id != self.match_id:
            raise ValueError("state.match_id must match the event's match_id")
        return self


class KillEvent(BaseEvent):
    """A player death."""

    event_type: Literal[EventType.KILL] = EventType.KILL
    round_number: int = Field(ge=1, le=64)
    killer_id: str | None = Field(
        default=None, description="Absent for suicides and world damage."
    )
    victim_id: str = Field(min_length=1, max_length=64)
    assister_id: str | None = None
    weapon: Weapon
    headshot: bool = False
    victim_team: Team
    position: Position | None = None

    @model_validator(mode="after")
    def _a_player_cannot_assist_their_own_kill(self) -> Self:
        if self.assister_id is not None and self.assister_id in (self.killer_id, self.victim_id):
            raise ValueError("assister_id must differ from the killer and the victim")
        return self


class BombPlantedEvent(BaseEvent):
    """The bomb was planted, changing the round's win condition."""

    event_type: Literal[EventType.BOMB_PLANTED] = EventType.BOMB_PLANTED
    round_number: int = Field(ge=1, le=64)
    planter_id: str = Field(min_length=1, max_length=64)
    site: Literal["A", "B"]
    position: Position | None = None


class BombDefusedEvent(BaseEvent):
    """The bomb was defused."""

    event_type: Literal[EventType.BOMB_DEFUSED] = EventType.BOMB_DEFUSED
    round_number: int = Field(ge=1, le=64)
    defuser_id: str = Field(min_length=1, max_length=64)
    seconds_remaining: float = Field(ge=0.0, le=constants.BOMB_TIMER_SECONDS)


class RoundEndEvent(BaseEvent):
    """A round was decided. Supplies the label for offline training."""

    event_type: Literal[EventType.ROUND_END] = EventType.ROUND_END
    round_number: int = Field(ge=1, le=64)
    winner: Team
    reason: RoundEndReason
    score_ct: int = Field(ge=0, le=64)
    score_t: int = Field(ge=0, le=64)

    @model_validator(mode="after")
    def _reason_is_consistent_with_the_winner(self) -> Self:
        """Reject impossible pairings, e.g. a CT win by bomb detonation."""
        ct_only = {RoundEndReason.T_ELIMINATED, RoundEndReason.BOMB_DEFUSED}
        t_only = {RoundEndReason.CT_ELIMINATED, RoundEndReason.BOMB_EXPLODED}

        if self.winner is Team.CT and self.reason in t_only:
            raise ValueError(f"CT cannot win by {self.reason.value}")
        if self.winner is Team.T and self.reason in ct_only:
            raise ValueError(f"T cannot win by {self.reason.value}")
        # Time expiry is a CT win by definition: the Ts failed to detonate.
        if self.reason is RoundEndReason.TIME_EXPIRED and self.winner is not Team.CT:
            raise ValueError("time_expired always resolves to a CT win")
        return self


class MatchEndEvent(BaseEvent):
    """Emitted once the match is decided."""

    event_type: Literal[EventType.MATCH_END] = EventType.MATCH_END
    winner: Team
    score_ct: int = Field(ge=0, le=64)
    score_t: int = Field(ge=0, le=64)

    @model_validator(mode="after")
    def _winner_holds_the_higher_score(self) -> Self:
        if self.winner is Team.CT and self.score_ct <= self.score_t:
            raise ValueError("declared CT win does not match the final score")
        if self.winner is Team.T and self.score_t <= self.score_ct:
            raise ValueError("declared T win does not match the final score")
        return self


TelemetryEvent: TypeAlias = Annotated[
    MatchStartEvent
    | RoundStartEvent
    | TickEvent
    | KillEvent
    | BombPlantedEvent
    | BombDefusedEvent
    | RoundEndEvent
    | MatchEndEvent,
    Field(discriminator="event_type"),
]
"""Any telemetry event, resolved by its ``event_type`` discriminator."""

TelemetryEventAdapter: TypeAdapter[TelemetryEvent] = TypeAdapter(TelemetryEvent)
"""Reusable parser; constructing a TypeAdapter per message is measurably slower."""


def parse_event(payload: bytes | str) -> TelemetryEvent:
    """Deserialise ``payload`` into the concrete event type it declares.

    Args:
        payload: JSON produced by :func:`serialise_event`.

    Raises:
        pydantic.ValidationError: If the payload is malformed or violates a rule.
    """
    return TelemetryEventAdapter.validate_json(payload)


def serialise_event(event: TelemetryEvent) -> bytes:
    """Serialise ``event`` to compact JSON bytes for the broker."""
    return TelemetryEventAdapter.dump_json(event)
