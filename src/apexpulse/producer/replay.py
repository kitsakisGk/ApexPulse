"""Replay simulated matches onto the broker.

Bridges the pure, synchronous :class:`MatchSimulator` to the async transport,
pacing events so a consumer sees traffic shaped like a real broadcast rather than
a burst dump.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from apexpulse.config import get_settings
from apexpulse.logging import get_logger
from apexpulse.producer.simulator import MatchSimulator
from apexpulse.schemas.events import SCHEMA_VERSION, TickEvent, serialise_event

if TYPE_CHECKING:
    from collections.abc import Iterator

    from apexpulse.broker.base import EventBroker
    from apexpulse.config import Settings
    from apexpulse.schemas.events import TelemetryEvent

logger = get_logger(__name__)


@dataclass
class ReplayStats:
    """Counters describing one replay run."""

    events_published: int = 0
    ticks_published: int = 0
    rounds_completed: int = 0
    duration_seconds: float = 0.0

    @property
    def events_per_second(self) -> float:
        """Sustained publish rate; 0.0 when nothing was published."""
        if self.duration_seconds <= 0:
            return 0.0
        return self.events_published / self.duration_seconds


@dataclass
class TelemetryReplayer:
    """Publish a simulated match to the configured telemetry topic.

    Args:
        broker: Started transport to publish through.
        settings: Runtime configuration; defaults to the process settings.
        speed: Wall-clock multiplier. ``1.0`` replays in real time, ``0.0``
            publishes as fast as possible, which is what tests and dataset
            generation want.
    """

    broker: EventBroker
    settings: Settings = field(default_factory=get_settings)
    speed: float = 1.0

    def __post_init__(self) -> None:
        if self.speed < 0:
            raise ValueError("speed must be non-negative")

    async def replay(
        self,
        simulator: MatchSimulator | None = None,
        *,
        max_events: int | None = None,
    ) -> ReplayStats:
        """Publish every event of a match.

        Args:
            simulator: Match to replay; a default simulator is built when omitted.
            max_events: Stop after this many events. Useful for smoke tests and
                bounded demos.

        Returns:
            Counters for the run.
        """
        simulator = simulator or MatchSimulator(tick_rate_hz=self.settings.tick_rate_hz)
        topic = self.settings.kafka_telemetry_topic
        stats = ReplayStats()
        started = time.perf_counter()

        # Pace only on ticks: they carry the match clock, so sleeping on them
        # reproduces real timing without stalling on bursts of kills.
        tick_interval = (1.0 / simulator.tick_rate_hz) * self.speed if self.speed else 0.0

        events: Iterator[TelemetryEvent] = simulator.run()
        for event in events:
            await self.broker.publish(
                topic,
                serialise_event(event),
                key=event.match_id,
                headers={"event_type": event.event_type.value, "schema": SCHEMA_VERSION},
            )
            stats.events_published += 1

            if isinstance(event, TickEvent):
                stats.ticks_published += 1
                if tick_interval:
                    await asyncio.sleep(tick_interval)
                else:
                    # Yield control so consumers drain instead of starving.
                    await asyncio.sleep(0)
            elif event.event_type.value == "round_end":
                stats.rounds_completed += 1

            if max_events is not None and stats.events_published >= max_events:
                logger.info("replay_truncated", published=stats.events_published)
                break

        await self.broker.flush()
        stats.duration_seconds = time.perf_counter() - started

        logger.info(
            "replay_complete",
            match_id=simulator.match_id,
            events=stats.events_published,
            ticks=stats.ticks_published,
            rounds=stats.rounds_completed,
            duration_s=round(stats.duration_seconds, 3),
            events_per_s=round(stats.events_per_second, 1),
        )
        return stats
