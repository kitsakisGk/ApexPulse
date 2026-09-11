"""Live match-state tracking.

Bridges the consumer to the state manager: every tick overwrites the match's
current snapshot and every completed round is appended to its history, so the API
can answer "what is happening right now" with a single read rather than replaying
the stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from apexpulse.config import get_settings
from apexpulse.logging import get_logger
from apexpulse.schemas.events import MatchEndEvent, RoundEndEvent, TickEvent
from apexpulse.storage.keys import match_key
from apexpulse.storage.match_state import (
    LiveSnapshot,
    MatchStateManager,
    MomentumSnapshot,
    RoundResult,
)
from apexpulse.stream.window import TelemetryWindow

if TYPE_CHECKING:
    from apexpulse.config import Settings
    from apexpulse.schemas.events import TelemetryEvent
    from apexpulse.storage.match_state import MatchHistory
    from apexpulse.storage.state import StateStore

logger = get_logger(__name__)

__all__ = ["MatchTracker", "match_key"]


@dataclass
class MatchTracker:
    """Maintain live state and windowed momentum for each in-flight match.

    Registers itself against a :class:`~apexpulse.stream.consumer.TelemetryConsumer`
    and writes a snapshot per tick. One tracker handles many concurrent matches,
    keyed by ``match_id``.

    Args:
        store: Started state store to write through.
        settings: Runtime configuration; defaults to the process settings.
        window_seconds: Match time retained for momentum metrics.
    """

    store: StateStore
    settings: Settings = field(default_factory=get_settings)
    window_seconds: float = 15.0

    _manager: MatchStateManager = field(init=False, repr=False)
    _windows: dict[str, TelemetryWindow] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._manager = MatchStateManager(self.store, self.settings)

    @property
    def manager(self) -> MatchStateManager:
        """The state manager this tracker writes through."""
        return self._manager

    def window_for(self, match_id: str) -> TelemetryWindow:
        """Return the window tracking ``match_id``, creating it on first sight."""
        if match_id not in self._windows:
            self._windows[match_id] = TelemetryWindow(span_seconds=self.window_seconds)
        return self._windows[match_id]

    async def handle(self, event: TelemetryEvent) -> None:
        """Fold ``event`` into the tracked state for its match."""
        window = self.window_for(event.match_id)
        window.observe(event)

        if isinstance(event, TickEvent):
            await self._persist(event, window)
        elif isinstance(event, RoundEndEvent):
            await self._record_round(event)
        elif isinstance(event, MatchEndEvent):
            await self._finalise(event)

    async def _persist(self, tick: TickEvent, window: TelemetryWindow) -> None:
        """Publish the current snapshot, enriched with windowed momentum."""
        metrics = window.metrics()
        await self._manager.publish_snapshot(
            LiveSnapshot(
                match_id=tick.match_id,
                sequence=tick.sequence,
                timestamp=tick.timestamp,
                state=tick.state,
                momentum=MomentumSnapshot(
                    kills_ct=metrics.kills_ct,
                    kills_t=metrics.kills_t,
                    kill_delta=metrics.kill_delta,
                    headshot_rate=round(metrics.headshot_rate, 4),
                    kills_per_second=round(metrics.kills_per_second, 4),
                    window_seconds=round(metrics.span_seconds, 2),
                ),
            )
        )

    async def _record_round(self, event: RoundEndEvent) -> None:
        """Append a completed round to the match's history."""
        await self._manager.append_round(
            event.match_id,
            RoundResult(
                round_number=event.round_number,
                winner=event.winner,
                reason=event.reason,
                score_ct=event.score_ct,
                score_t=event.score_t,
                timestamp=event.timestamp,
            ),
        )

    async def _finalise(self, event: MatchEndEvent) -> None:
        """Record the result and release the match's window."""
        self._windows.pop(event.match_id, None)
        history = await self._manager.get_history(event.match_id)
        logger.info(
            "match_finalised",
            match_id=event.match_id,
            winner=event.winner.value,
            score_ct=event.score_ct,
            score_t=event.score_t,
            rounds=history.rounds_played,
        )

    async def snapshot(self, match_id: str) -> LiveSnapshot | None:
        """Read back the stored snapshot for ``match_id``."""
        return await self._manager.get_snapshot(match_id)

    async def history(self, match_id: str) -> MatchHistory:
        """Read back the completed-round history for ``match_id``."""
        return await self._manager.get_history(match_id)

    async def live_match_ids(self) -> list[str]:
        """Return the ids of every match with a stored snapshot."""
        return await self._manager.live_match_ids()
