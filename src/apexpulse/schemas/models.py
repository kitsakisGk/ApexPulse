"""Core domain models for match, round, and player state.

These types are the contract between every stage of the pipeline: the simulator
produces them, the broker carries them as JSON, and the feature extractor consumes
them. Validation is deliberately strict — a malformed tick should fail at the
producer rather than silently corrupt a feature vector downstream.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from apexpulse.schemas import constants
from apexpulse.schemas.enums import MapName, RoundPhase, Team, Weapon

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
"""A value constrained to the closed unit interval."""


def utcnow() -> datetime:
    """Return the current UTC time; patchable in tests."""
    return datetime.now(UTC)


class ApexModel(BaseModel):
    """Base model applying the project-wide serialisation policy.

    Models are frozen so a snapshot cannot be mutated after validation, and unknown
    fields are rejected so a producer typo fails at the boundary rather than
    silently dropping data.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        ser_json_timedelta="float",
    )


class DerivedFieldsModel(ApexModel):
    """A model that serialises computed fields alongside its own.

    Computed fields are emitted by ``model_dump`` but are not constructor
    arguments, so a strict round trip would reject its own output. Stripping them
    on input keeps the payload self-describing for the dashboard while leaving
    them derived, never authoritative.
    """

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            computed = {field.alias or name for name, field in cls.model_computed_fields.items()}
            if computed & data.keys():
                return {key: value for key, value in data.items() if key not in computed}
        return data


class Position(ApexModel):
    """A player's location on the map in world units.

    CS2 maps span roughly ±4096 units per axis; the bounds are generous enough to
    admit every playable location while still rejecting obviously corrupt data.
    """

    x: float = Field(ge=-8192.0, le=8192.0)
    y: float = Field(ge=-8192.0, le=8192.0)
    z: float = Field(ge=-2048.0, le=2048.0)

    def distance_to(self, other: Position) -> float:
        """Return the Euclidean distance to ``other``."""
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


class PlayerState(DerivedFieldsModel):
    """A single player's state at one instant."""

    player_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    team: Team

    health: int = Field(ge=0, le=constants.MAX_HEALTH)
    armour: int = Field(default=0, ge=0, le=constants.MAX_ARMOUR)
    has_helmet: bool = False
    has_defuse_kit: bool = False

    money: int = Field(ge=0, le=constants.MAX_MONEY)
    primary_weapon: Weapon | None = None
    position: Position | None = None

    kills: int = Field(default=0, ge=0)
    deaths: int = Field(default=0, ge=0)
    assists: int = Field(default=0, ge=0)
    damage_dealt: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_alive(self) -> bool:
        """Whether the player is alive this round."""
        return self.health > 0

    @model_validator(mode="after")
    def _dead_players_hold_nothing(self) -> Self:
        """A dead player cannot carry armour, a kit, or a weapon."""
        if self.health == 0 and (self.armour > 0 or self.has_helmet or self.has_defuse_kit):
            raise ValueError("a dead player cannot retain armour, a helmet, or a defuse kit")
        return self

    @model_validator(mode="after")
    def _helmet_requires_armour(self) -> Self:
        if self.has_helmet and self.armour == 0:
            raise ValueError("has_helmet requires non-zero armour")
        return self

    @model_validator(mode="after")
    def _only_cts_carry_defuse_kits(self) -> Self:
        if self.has_defuse_kit and self.team is not Team.CT:
            raise ValueError("only CT players can carry a defuse kit")
        return self


class TeamEconomy(DerivedFieldsModel):
    """Aggregate economic state for one side."""

    team: Team
    money: int = Field(ge=0, description="Combined money held by living and dead players.")
    equipment_value: int = Field(ge=0, description="Combined value of carried equipment.")
    consecutive_losses: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def loss_bonus(self) -> int:
        """Per-player payout should this team lose the current round."""
        return constants.loss_bonus(self.consecutive_losses)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_full_buy(self) -> bool:
        """Whether the side can afford a full loadout."""
        return self.money >= constants.FULL_BUY_THRESHOLD * constants.PLAYERS_PER_TEAM


class RoundState(ApexModel):
    """Everything about the round currently in progress."""

    round_number: int = Field(ge=1, le=64)
    phase: RoundPhase
    seconds_remaining: float = Field(ge=0.0, le=constants.ROUND_SECONDS)
    bomb_planted: bool = False
    bomb_seconds_remaining: float | None = Field(
        default=None, ge=0.0, le=constants.BOMB_TIMER_SECONDS
    )
    bomb_position: Position | None = None

    @model_validator(mode="after")
    def _bomb_timer_tracks_the_plant(self) -> Self:
        """The bomb countdown exists exactly while the bomb is planted."""
        if self.bomb_planted and self.bomb_seconds_remaining is None:
            raise ValueError("bomb_seconds_remaining is required once the bomb is planted")
        if not self.bomb_planted and self.bomb_seconds_remaining is not None:
            raise ValueError("bomb_seconds_remaining is only valid while the bomb is planted")
        return self

    @model_validator(mode="after")
    def _planted_phase_matches_the_flag(self) -> Self:
        if self.phase is RoundPhase.BOMB_PLANTED and not self.bomb_planted:
            raise ValueError("phase 'bomb_planted' requires bomb_planted=True")
        return self


class MatchState(DerivedFieldsModel):
    """A complete, self-contained snapshot of a match at one tick.

    This is the payload the dashboard renders and the feature extractor scores, so
    it carries everything needed for both without a second lookup.
    """

    match_id: str = Field(min_length=1, max_length=64)
    map_name: MapName
    timestamp: datetime = Field(default_factory=utcnow)

    score_ct: int = Field(default=0, ge=0, le=64)
    score_t: int = Field(default=0, ge=0, le=64)

    round_state: RoundState
    players: tuple[PlayerState, ...] = Field(min_length=1, max_length=32)
    economy_ct: TeamEconomy
    economy_t: TeamEconomy

    @model_validator(mode="after")
    def _economies_describe_their_own_side(self) -> Self:
        if self.economy_ct.team is not Team.CT or self.economy_t.team is not Team.T:
            raise ValueError("economy_ct and economy_t must describe CT and T respectively")
        return self

    @model_validator(mode="after")
    def _player_ids_are_unique(self) -> Self:
        ids = [player.player_id for player in self.players]
        if len(ids) != len(set(ids)):
            raise ValueError("player_id must be unique within a match snapshot")
        return self

    def players_on(self, team: Team) -> tuple[PlayerState, ...]:
        """Return every player on ``team``."""
        return tuple(player for player in self.players if player.team is team)

    def alive_count(self, team: Team) -> int:
        """Return how many players on ``team`` are still alive."""
        return sum(1 for player in self.players if player.team is team and player.is_alive)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def alive_ct(self) -> int:
        """Living CT players."""
        return self.alive_count(Team.CT)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def alive_t(self) -> int:
        """Living T players."""
        return self.alive_count(Team.T)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def man_advantage(self) -> int:
        """Living CT players minus living T players; positive favours CT."""
        return self.alive_ct - self.alive_t
