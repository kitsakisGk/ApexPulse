"""Bridges the telemetry stream to connected WebSocket clients.

Registers against the consumer, so every event that reaches the pipeline also
reaches whoever is watching. This is the join that makes the dashboard live:
without it the API only answers questions, and a client has to ask repeatedly.

Broadcasting is throttled. Ticks arrive at 8 Hz and a browser cannot usefully
repaint a player grid that often, so state frames are rate-limited while the
lighter prediction frames pass through — the gauge is what viewers watch move.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from apexpulse.api.routes.matches import to_snapshot
from apexpulse.api.schemas import PredictionSummary, RoundOutcome, ScoreLine
from apexpulse.api.ws_schemas import (
    MatchEndMessage,
    PredictionMessage,
    RoundEndMessage,
    TickMessage,
)
from apexpulse.logging import get_logger
from apexpulse.schemas.events import MatchEndEvent, RoundEndEvent, TickEvent
from apexpulse.storage.match_state import LiveSnapshot, MomentumSnapshot
from apexpulse.stream.window import TelemetryWindow

if TYPE_CHECKING:
    from apexpulse.api.dependencies import AppState
    from apexpulse.schemas.events import TelemetryEvent

logger = get_logger(__name__)

STATE_FRAME_INTERVAL = 0.25
"""Minimum wall-clock seconds between full state frames, per match.

Four a second. A browser cannot usefully repaint ten player rows at 8 Hz, and
sending frames it will discard wastes bandwidth on every connected client.
"""


@dataclass
class BroadcastStats:
    """Counters describing what was pushed."""

    ticks_seen: int = 0
    state_frames: int = 0
    prediction_frames: int = 0
    round_frames: int = 0
    throttled: int = 0

    @property
    def throttle_rate(self) -> float:
        """Share of ticks suppressed by the state-frame interval."""
        return 0.0 if self.ticks_seen == 0 else self.throttled / self.ticks_seen


@dataclass
class StreamBroadcaster:
    """Pushes consumed telemetry to WebSocket subscribers.

    Args:
        state: Shared application state, holding the connection registry and the
            inference engine.
        window_seconds: Match time retained for momentum features.
    """

    state: AppState
    window_seconds: float = 15.0

    _windows: dict[str, TelemetryWindow] = field(default_factory=dict, init=False, repr=False)
    _last_state_frame: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _stats: BroadcastStats = field(default_factory=BroadcastStats, init=False, repr=False)

    @property
    def stats(self) -> BroadcastStats:
        """Counters for this broadcaster."""
        return self._stats

    def window_for(self, match_id: str) -> TelemetryWindow:
        """Return the window tracking ``match_id``, creating it on first sight."""
        if match_id not in self._windows:
            self._windows[match_id] = TelemetryWindow(span_seconds=self.window_seconds)
        return self._windows[match_id]

    async def handle(self, event: TelemetryEvent) -> None:
        """Fold ``event`` into the window and push whatever it warrants."""
        window = self.window_for(event.match_id)
        window.observe(event)

        if isinstance(event, TickEvent):
            await self._on_tick(event, window)
        elif isinstance(event, RoundEndEvent):
            await self._on_round_end(event)
        elif isinstance(event, MatchEndEvent):
            await self._on_match_end(event)

    async def _on_tick(self, tick: TickEvent, window: TelemetryWindow) -> None:
        """Push a prediction every tick, and a full state frame on the interval."""
        self._stats.ticks_seen += 1

        # Nobody is watching this match, so skip the projection work entirely.
        if self.state.connections.subscriber_count(tick.match_id) == 0:
            return

        metrics = window.metrics()
        prediction: PredictionSummary | None = None

        if self.state.engine is not None:
            scored = self.state.engine.predict(tick.state, metrics)
            if scored.scored:
                self.state.predictions_served += 1
                prediction = PredictionSummary(
                    ct_win_probability=scored.ct_win_probability,
                    t_win_probability=scored.t_win_probability,
                    favoured_side=scored.favoured_side,
                    confidence=scored.confidence,
                    latency_ms=scored.latency_ms,
                )

        if prediction is not None:
            prediction_frame = PredictionMessage(
                match_id=tick.match_id,
                sequence=tick.sequence,
                timestamp=tick.timestamp,
                prediction=prediction,
            )
            await self.state.connections.broadcast(
                tick.match_id, prediction_frame.model_dump(mode="json")
            )
            self._stats.prediction_frames += 1

        now = time.monotonic()
        last = self._last_state_frame.get(tick.match_id, 0.0)
        if now - last < STATE_FRAME_INTERVAL:
            self._stats.throttled += 1
            return
        self._last_state_frame[tick.match_id] = now

        snapshot = LiveSnapshot(
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
        state_frame = TickMessage(
            match_id=tick.match_id,
            sequence=tick.sequence,
            timestamp=tick.timestamp,
            snapshot=to_snapshot(snapshot, self.state),
        )
        await self.state.connections.broadcast(tick.match_id, state_frame.model_dump(mode="json"))
        self._stats.state_frames += 1

    async def _on_round_end(self, event: RoundEndEvent) -> None:
        """Push the round result. Never throttled: a round ends once."""
        message = RoundEndMessage(
            match_id=event.match_id,
            timestamp=event.timestamp,
            outcome=RoundOutcome(
                round_number=event.round_number,
                winner=event.winner.value,
                reason=event.reason.value,
                score=ScoreLine(ct=event.score_ct, t=event.score_t),
                timestamp=event.timestamp,
            ),
        )
        await self.state.connections.broadcast(event.match_id, message.model_dump(mode="json"))
        self._stats.round_frames += 1

    async def _on_match_end(self, event: MatchEndEvent) -> None:
        """Push the final result and release the match's state."""
        message = MatchEndMessage(
            match_id=event.match_id,
            timestamp=event.timestamp,
            winner=event.winner.value,
            score_ct=event.score_ct,
            score_t=event.score_t,
        )
        await self.state.connections.broadcast(event.match_id, message.model_dump(mode="json"))

        self._windows.pop(event.match_id, None)
        self._last_state_frame.pop(event.match_id, None)
        logger.info(
            "match_broadcast_finished",
            match_id=event.match_id,
            state_frames=self._stats.state_frames,
            prediction_frames=self._stats.prediction_frames,
        )
