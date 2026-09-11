"""Live match-state manager.

Wraps the generic key/value :class:`~apexpulse.storage.state.StateStore` in the
domain operations the pipeline actually performs: publish the current snapshot,
append a completed round, list what is live, and read any of it back as validated
models rather than raw dictionaries.

Keeping this above the store — rather than inside it — means the same manager runs
unchanged on Redis, SQLite, or memory.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - pydantic resolves field types at runtime
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field

from apexpulse.config import get_settings
from apexpulse.logging import get_logger
from apexpulse.schemas.enums import (  # noqa: TC001 - pydantic field types
    RoundEndReason,
    Team,
)
from apexpulse.schemas.models import ApexModel, MatchState
from apexpulse.storage import keys

if TYPE_CHECKING:
    from apexpulse.config import Settings
    from apexpulse.storage.state import StateStore

logger = get_logger(__name__)

MAX_TRACKED_ROUNDS = 64
"""Cap on retained rounds, bounding a pathological match's memory use."""


class MomentumSnapshot(ApexModel):
    """Windowed momentum accompanying a live snapshot."""

    kills_ct: int = Field(default=0, ge=0)
    kills_t: int = Field(default=0, ge=0)
    kill_delta: int = 0
    headshot_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    kills_per_second: float = Field(default=0.0, ge=0.0)
    window_seconds: float = Field(default=0.0, ge=0.0)


class LiveSnapshot(ApexModel):
    """The current state of one match, as published for the API and dashboard."""

    match_id: str = Field(min_length=1, max_length=64)
    sequence: int = Field(ge=0)
    timestamp: datetime
    state: MatchState
    momentum: MomentumSnapshot = Field(default_factory=MomentumSnapshot)

    @property
    def is_live(self) -> bool:
        """Whether the snapshot describes a round still in progress."""
        return self.state.round_state.phase.value != "over"


class RoundResult(ApexModel):
    """One completed round, retained as match history."""

    round_number: int = Field(ge=1, le=64)
    winner: Team
    reason: RoundEndReason
    score_ct: int = Field(ge=0, le=64)
    score_t: int = Field(ge=0, le=64)
    timestamp: datetime


class MatchHistory(ApexModel):
    """Every completed round of a match, oldest first."""

    match_id: str = Field(min_length=1, max_length=64)
    rounds: tuple[RoundResult, ...] = ()

    @property
    def rounds_played(self) -> int:
        """Number of completed rounds."""
        return len(self.rounds)

    def wins_for(self, team: Team) -> int:
        """Count rounds won by ``team``."""
        return sum(1 for result in self.rounds if result.winner is team)

    def current_streak(self) -> tuple[Team, int] | None:
        """Return the side on a winning streak and its length, or ``None`` if empty."""
        if not self.rounds:
            return None
        winner = self.rounds[-1].winner
        streak = 0
        for result in reversed(self.rounds):
            if result.winner is not winner:
                break
            streak += 1
        return winner, streak


class MatchStateManager:
    """Domain operations over the live state store.

    Args:
        store: A started :class:`StateStore`.
        settings: Runtime configuration; defaults to the process settings.
    """

    def __init__(self, store: StateStore, settings: Settings | None = None) -> None:
        self._store = store
        self._settings = settings or get_settings()

    @property
    def store(self) -> StateStore:
        """The underlying key/value store."""
        return self._store

    @property
    def _ttl(self) -> int:
        return self._settings.redis_state_ttl_seconds

    # -- Live snapshot --------------------------------------------------------

    async def publish_snapshot(self, snapshot: LiveSnapshot) -> None:
        """Overwrite ``snapshot``'s match with its current state."""
        await self._store.set(
            keys.match_key(snapshot.match_id),
            snapshot.model_dump(mode="json"),
            ttl=self._ttl,
        )

    async def get_snapshot(self, match_id: str) -> LiveSnapshot | None:
        """Read ``match_id``'s current snapshot, or ``None`` if absent or expired."""
        raw = await self._store.get(keys.match_key(match_id))
        if raw is None:
            return None
        return self._parse(LiveSnapshot, raw, match_id=match_id, kind="snapshot")

    # -- Round history --------------------------------------------------------

    async def append_round(self, match_id: str, result: RoundResult) -> MatchHistory:
        """Append ``result`` to ``match_id``'s history and return the updated history.

        Appending is idempotent per round number: replaying a round-end event —
        which happens whenever a consumer restarts and re-reads the topic — updates
        the existing entry instead of duplicating it.
        """
        history = await self.get_history(match_id)
        retained = [entry for entry in history.rounds if entry.round_number != result.round_number]
        retained.append(result)
        retained.sort(key=lambda entry: entry.round_number)

        updated = MatchHistory(
            match_id=match_id,
            rounds=tuple(retained[-MAX_TRACKED_ROUNDS:]),
        )
        await self._store.set(
            keys.rounds_key(match_id),
            updated.model_dump(mode="json"),
            ttl=self._ttl,
        )
        return updated

    async def get_history(self, match_id: str) -> MatchHistory:
        """Read ``match_id``'s round history, empty when none has been recorded."""
        raw = await self._store.get(keys.rounds_key(match_id))
        if raw is None:
            return MatchHistory(match_id=match_id)
        parsed = self._parse(MatchHistory, raw, match_id=match_id, kind="history")
        return parsed if parsed is not None else MatchHistory(match_id=match_id)

    # -- Discovery ------------------------------------------------------------

    async def live_match_ids(self) -> list[str]:
        """Return the ids of every match holding a live snapshot."""
        found = await self._store.keys(keys.match_pattern())
        ids = [keys.match_id_from_key(key) for key in found]
        return sorted(match_id for match_id in ids if match_id is not None)

    async def live_snapshots(self) -> list[LiveSnapshot]:
        """Read every live snapshot, skipping any that expired mid-scan."""
        snapshots = [await self.get_snapshot(match_id) for match_id in await self.live_match_ids()]
        return [snapshot for snapshot in snapshots if snapshot is not None]

    async def drop(self, match_id: str) -> bool:
        """Remove a match's snapshot and history; return whether anything existed."""
        removed_snapshot = await self._store.delete(keys.match_key(match_id))
        removed_history = await self._store.delete(keys.rounds_key(match_id))
        return removed_snapshot or removed_history

    # -- Health ---------------------------------------------------------------

    async def healthy(self) -> bool:
        """Return whether the backing store is reachable."""
        return await self._store.ping()

    # -- Internals ------------------------------------------------------------

    def _parse[T: ApexModel](
        self,
        model: type[T],
        raw: dict[str, Any],
        *,
        match_id: str,
        kind: str,
    ) -> T | None:
        """Validate stored JSON, logging and discarding anything malformed.

        Stored state can predate a schema change, so a validation failure means
        stale data rather than a bug: drop it and let the next tick republish.
        """
        try:
            return model.model_validate(raw)
        except Exception as exc:  # stale or corrupt state must not crash a reader
            logger.warning(
                "discarded_malformed_state",
                match_id=match_id,
                kind=kind,
                error=str(exc),
            )
            return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None
