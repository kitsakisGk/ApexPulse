"""Message models for the WebSocket channel.

Every frame carries a ``type`` discriminator so a client can switch on it without
inspecting the payload, and the same shape is used for every message kind — a
dashboard should never have to guess what arrived.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - pydantic resolves field types at runtime
from enum import StrEnum

from pydantic import Field

from apexpulse.api.schemas import (
    ApiModel,
    MatchSnapshot,
    PredictionSummary,
    RoundOutcome,
)


class MessageType(StrEnum):
    """Discriminator for a WebSocket frame."""

    WELCOME = "welcome"
    """Sent once on connect, carrying the current state so the client can render
    immediately instead of waiting for the next tick."""

    TICK = "tick"
    PREDICTION = "prediction"
    ROUND_END = "round_end"
    MATCH_END = "match_end"
    ERROR = "error"


class WelcomeMessage(ApiModel):
    """First frame after a successful connection."""

    type: MessageType = MessageType.WELCOME
    match_id: str = Field(description="Match being watched, or '*' for every match.")
    server_version: str
    model_loaded: bool
    snapshot: MatchSnapshot | None = Field(
        default=None, description="Current state, when the match already has some."
    )


class TickMessage(ApiModel):
    """A full state update."""

    type: MessageType = MessageType.TICK
    match_id: str
    sequence: int = Field(ge=0)
    timestamp: datetime
    snapshot: MatchSnapshot


class PredictionMessage(ApiModel):
    """A win-probability update without the full state.

    Lighter than a tick: the gauge moves far more often than the player grid
    needs redrawing, so a client that only wants the number can subscribe to
    these alone.
    """

    type: MessageType = MessageType.PREDICTION
    match_id: str
    sequence: int = Field(ge=0)
    timestamp: datetime
    prediction: PredictionSummary


class RoundEndMessage(ApiModel):
    """A round was decided."""

    type: MessageType = MessageType.ROUND_END
    match_id: str
    timestamp: datetime
    outcome: RoundOutcome


class MatchEndMessage(ApiModel):
    """The match finished."""

    type: MessageType = MessageType.MATCH_END
    match_id: str
    timestamp: datetime
    winner: str
    score_ct: int = Field(ge=0, le=64)
    score_t: int = Field(ge=0, le=64)


class ErrorMessage(ApiModel):
    """Something went wrong, reported over the channel rather than by closing it."""

    type: MessageType = MessageType.ERROR
    detail: str
    match_id: str | None = None
