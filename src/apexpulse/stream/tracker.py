"""Live match-state tracking.

Bridges the consumer to the state store: every tick overwrites the match's current
snapshot, so the API can answer "what is happening right now" with a single read
rather than replaying the stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apexpulse.config import get_settings
from apexpulse.logging import get_logger
from apexpulse.schemas.events import MatchEndEvent, RoundEndEvent, TickEvent
from apexpulse.stream.window import TelemetryWindow

if TYPE_CHECKING:
    from apexpulse.config import Settings
    from apexpulse.schemas.events import TelemetryEvent
    from apexpulse.storage.state import StateStore

logger = get_logger(__name__)

MATCH_KEY_PREFIX = "apexpulse:match:"
"""Key namespace for live match snapshots."""


def match_key(match_id: str) -> str:
    """Return the state-store key holding ``match_id``'s live snapshot."""
    return f"{MATCH_KEY_PREFIX}{match_id}"


@dataclass
class MatchTracker:
    """Maintain live state and windowed momentum for each in-flight match.

    Registers itself against a :class:`~apexpulse.stream.consumer.TelemetryConsumer`
    and writes a snapshot per tick. One tracker handles many concurrent matches,
    keyed by ``match_id``.

    Args:
        store: Started state store to write snapshots into.
        settings: Runtime configuration; defaults to the process settings.
        window_seconds: Match time retained for momentum metrics.
    """

    store: StateStore
    settings: Settings = field(default_factory=get_settings)
    window_seconds: float = 15.0

    _windows: dict[str, TelemetryWindow] = field(default_factory=dict, init=False, repr=False)
    _rounds_seen: dict[str, int] = field(default_factory=dict, init=False, repr=False)

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
            self._rounds_seen[event.match_id] = event.round_number
        elif isinstance(event, MatchEndEvent):
            await self._finalise(event)

    async def _persist(self, tick: TickEvent, window: TelemetryWindow) -> None:
        """Write the current snapshot, enriched with windowed momentum."""
        metrics = window.metrics()
        snapshot: dict[str, Any] = {
            "match_id": tick.match_id,
            "sequence": tick.sequence,
            "timestamp": tick.timestamp.isoformat(),
            "state": tick.state.model_dump(mode="json"),
            "momentum": {
                "kills_ct": metrics.kills_ct,
                "kills_t": metrics.kills_t,
                "kill_delta": metrics.kill_delta,
                "headshot_rate": round(metrics.headshot_rate, 4),
                "kills_per_second": round(metrics.kills_per_second, 4),
                "window_seconds": round(metrics.span_seconds, 2),
            },
        }

        await self.store.set(
            match_key(tick.match_id),
            snapshot,
            ttl=self.settings.redis_state_ttl_seconds,
        )

    async def _finalise(self, event: MatchEndEvent) -> None:
        """Record the result and release the match's window."""
        self._windows.pop(event.match_id, None)
        logger.info(
            "match_finalised",
            match_id=event.match_id,
            winner=event.winner.value,
            score_ct=event.score_ct,
            score_t=event.score_t,
            rounds=self._rounds_seen.get(event.match_id, 0),
        )

    async def snapshot(self, match_id: str) -> dict[str, Any] | None:
        """Read back the stored snapshot for ``match_id``."""
        return await self.store.get(match_key(match_id))

    async def live_match_ids(self) -> list[str]:
        """Return the ids of every match with a stored snapshot."""
        keys = await self.store.keys(f"{MATCH_KEY_PREFIX}*")
        return [key.removeprefix(MATCH_KEY_PREFIX) for key in keys]
