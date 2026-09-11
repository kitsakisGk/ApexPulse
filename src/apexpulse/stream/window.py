"""Sliding-window aggregation over the telemetry stream.

A single tick says what is true *now*. Momentum — three kills in the last ten
seconds, a collapsing economy — only exists across time, so the feature extractor
needs a bounded view of recent history rather than one snapshot.

The window is time-based rather than count-based: at a configurable tick rate, a
fixed number of events would cover a different span of match time at 4 Hz than at
16 Hz, which would silently change what the model sees.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from apexpulse.schemas.enums import EventType, Team
from apexpulse.schemas.events import KillEvent, TickEvent

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    from apexpulse.schemas.events import TelemetryEvent

DEFAULT_WINDOW_SECONDS = 15.0
"""Span of match time retained; long enough for momentum, short enough to stay hot."""


@dataclass
class SlidingWindow[T]:
    """A time-bounded buffer of timestamped items.

    Items are appended in timestamp order and evicted once they fall outside
    ``span_seconds`` of the newest item. Eviction is driven by event time, not
    wall-clock time, so a replay running faster than real time behaves identically
    to a live feed.

    Args:
        span_seconds: How much match time to retain.
    """

    span_seconds: float = DEFAULT_WINDOW_SECONDS
    _items: deque[tuple[datetime, T]] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.span_seconds <= 0:
            raise ValueError("span_seconds must be positive")

    def append(self, timestamp: datetime, item: T) -> None:
        """Add ``item`` and evict anything now outside the window."""
        self._items.append((timestamp, item))
        self._evict_before(timestamp)

    def _evict_before(self, newest: datetime) -> None:
        cutoff = newest.timestamp() - self.span_seconds
        while self._items and self._items[0][0].timestamp() < cutoff:
            self._items.popleft()

    def clear(self) -> None:
        """Drop every retained item."""
        self._items.clear()

    @property
    def items(self) -> tuple[T, ...]:
        """Retained items, oldest first."""
        return tuple(item for _, item in self._items)

    @property
    def newest(self) -> T | None:
        """Most recently appended item, or ``None`` when empty."""
        return self._items[-1][1] if self._items else None

    @property
    def oldest(self) -> T | None:
        """Oldest retained item, or ``None`` when empty."""
        return self._items[0][1] if self._items else None

    @property
    def span(self) -> float:
        """Match-time seconds between the oldest and newest retained items."""
        if len(self._items) < 2:
            return 0.0
        return self._items[-1][0].timestamp() - self._items[0][0].timestamp()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __bool__(self) -> bool:
        return bool(self._items)


@dataclass
class WindowedMetrics:
    """Aggregates computed over the current window.

    These are the momentum signals the model consumes alongside instantaneous
    state: who has been winning fights lately, and how fast the round is moving.
    """

    kills_ct: int = 0
    kills_t: int = 0
    headshots: int = 0
    ticks: int = 0
    span_seconds: float = 0.0

    @property
    def kill_delta(self) -> int:
        """CT kills minus T kills in the window; positive favours CT."""
        return self.kills_ct - self.kills_t

    @property
    def total_kills(self) -> int:
        """Kills by either side in the window."""
        return self.kills_ct + self.kills_t

    @property
    def kills_per_second(self) -> float:
        """Engagement pace; 0.0 when the window covers no time."""
        if self.span_seconds <= 0:
            return 0.0
        return self.total_kills / self.span_seconds

    @property
    def headshot_rate(self) -> float:
        """Share of window kills that were headshots; 0.0 when there were none."""
        return 0.0 if self.total_kills == 0 else self.headshots / self.total_kills


@dataclass
class TelemetryWindow:
    """Maintains sliding windows over one match's event stream.

    Ticks and kills are kept separately: ticks are dense and drive state, kills
    are sparse and drive momentum. Mixing them into one buffer would let a burst
    of ticks evict the kills the metrics depend on.

    Args:
        span_seconds: Match time retained by both windows.
    """

    span_seconds: float = DEFAULT_WINDOW_SECONDS
    _ticks: SlidingWindow[TickEvent] = field(init=False, repr=False)
    _kills: SlidingWindow[KillEvent] = field(init=False, repr=False)
    _current_round: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ticks = SlidingWindow(span_seconds=self.span_seconds)
        self._kills = SlidingWindow(span_seconds=self.span_seconds)

    def observe(self, event: TelemetryEvent) -> None:
        """Fold ``event`` into the window, resetting at each round boundary."""
        if event.event_type is EventType.ROUND_START:
            # Momentum does not carry across rounds: everyone respawns, so last
            # round's kills say nothing about this round's balance.
            self.reset()
            self._current_round = getattr(event, "round_number", None)
        elif isinstance(event, TickEvent):
            self._ticks.append(event.timestamp, event)
            self._current_round = event.state.round_state.round_number
        elif isinstance(event, KillEvent):
            self._kills.append(event.timestamp, event)

    def reset(self) -> None:
        """Clear both windows."""
        self._ticks.clear()
        self._kills.clear()

    @property
    def latest_tick(self) -> TickEvent | None:
        """Most recent tick, or ``None`` before the first one arrives."""
        return self._ticks.newest

    @property
    def current_round(self) -> int | None:
        """Round the window is tracking."""
        return self._current_round

    @property
    def tick_count(self) -> int:
        """Ticks currently retained."""
        return len(self._ticks)

    @property
    def kill_count(self) -> int:
        """Kills currently retained."""
        return len(self._kills)

    def metrics(self) -> WindowedMetrics:
        """Compute aggregates over the retained events."""
        kills = self._kills.items
        # A CT kill is one whose victim is a T, which is what the event records.
        kills_ct = sum(1 for kill in kills if kill.victim_team is Team.T)

        return WindowedMetrics(
            kills_ct=kills_ct,
            kills_t=len(kills) - kills_ct,
            headshots=sum(1 for kill in kills if kill.headshot),
            ticks=len(self._ticks),
            span_seconds=max(self._ticks.span, self._kills.span),
        )
