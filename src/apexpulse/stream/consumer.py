"""Streaming telemetry consumer.

Reads telemetry off the broker, dispatches each event to registered handlers, and
shuts down cleanly when asked. Handler failures are contained: one bad handler
must not take down the pipeline mid-match.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import ValidationError

from apexpulse.config import get_settings
from apexpulse.logging import get_logger
from apexpulse.schemas.enums import EventType  # noqa: TC001 - used as a runtime dict key
from apexpulse.schemas.events import parse_event

if TYPE_CHECKING:
    from apexpulse.broker.base import EventBroker
    from apexpulse.config import Settings
    from apexpulse.schemas.events import TelemetryEvent

logger = get_logger(__name__)

EventHandler = Callable[["TelemetryEvent"], Awaitable[None]]
"""Coroutine invoked for each consumed event."""


@dataclass
class ConsumerStats:
    """Counters describing a consumer run."""

    events_consumed: int = 0
    events_failed: int = 0
    handler_errors: int = 0
    last_sequence: int = -1

    @property
    def success_rate(self) -> float:
        """Share of messages that parsed successfully; 1.0 when none were seen."""
        total = self.events_consumed + self.events_failed
        return 1.0 if total == 0 else self.events_consumed / total


@dataclass
class TelemetryConsumer:
    """Consume telemetry events and fan them out to handlers.

    Handlers are registered per event type, or with ``None`` to receive every
    event. A handler that raises is logged and skipped — the stream continues,
    because dropping one dashboard update is far better than halting ingestion.

    Args:
        broker: Started transport to read from.
        settings: Runtime configuration; defaults to the process settings.
        group: Consumer group id.
    """

    broker: EventBroker
    settings: Settings = field(default_factory=get_settings)
    group: str | None = None

    _handlers: dict[EventType | None, list[EventHandler]] = field(
        default_factory=dict, init=False, repr=False
    )
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _stats: ConsumerStats = field(default_factory=ConsumerStats, init=False, repr=False)

    @property
    def stats(self) -> ConsumerStats:
        """Counters for the current or most recent run."""
        return self._stats

    @property
    def is_stopping(self) -> bool:
        """Whether a shutdown has been requested."""
        return self._stop_event.is_set()

    def on(self, event_type: EventType | None, handler: EventHandler) -> None:
        """Register ``handler`` for ``event_type``, or for every event if ``None``."""
        self._handlers.setdefault(event_type, []).append(handler)

    def stop(self) -> None:
        """Request a graceful shutdown; the run loop exits after the current event."""
        self._stop_event.set()

    async def _dispatch(self, event: TelemetryEvent) -> None:
        """Invoke every handler registered for ``event``, isolating failures."""
        handlers = [*self._handlers.get(event.event_type, []), *self._handlers.get(None, [])]

        for handler in handlers:
            try:
                await handler(event)
            except Exception as exc:  # one bad handler must not stop ingestion
                self._stats.handler_errors += 1
                logger.error(
                    "handler_failed",
                    handler=getattr(handler, "__name__", repr(handler)),
                    event_type=event.event_type.value,
                    sequence=event.sequence,
                    error=str(exc),
                    exc_info=True,
                )

    async def run(self, *, max_events: int | None = None) -> ConsumerStats:
        """Consume until stopped, or until ``max_events`` have been handled.

        Malformed payloads are counted and skipped rather than raised: a single
        corrupt message on a topic should not end the run.
        """
        self._stats = ConsumerStats()
        self._stop_event.clear()
        topic = self.settings.kafka_telemetry_topic
        group = self.group or self.settings.kafka_consumer_group

        logger.info("consumer_starting", topic=topic, group=group)

        try:
            async for message in self.broker.consume(topic, group=group):
                if self.is_stopping:
                    break

                try:
                    event = parse_event(message.value)
                except ValidationError as exc:
                    self._stats.events_failed += 1
                    logger.warning(
                        "malformed_event",
                        offset=message.offset,
                        errors=exc.error_count(),
                    )
                    continue

                self._stats.events_consumed += 1
                self._stats.last_sequence = event.sequence
                await self._dispatch(event)

                if max_events is not None and self._stats.events_consumed >= max_events:
                    break
        except asyncio.CancelledError:
            # Cancellation is a legitimate shutdown path; report what was done.
            logger.info("consumer_cancelled", consumed=self._stats.events_consumed)
            raise
        finally:
            logger.info(
                "consumer_stopped",
                consumed=self._stats.events_consumed,
                failed=self._stats.events_failed,
                handler_errors=self._stats.handler_errors,
                last_sequence=self._stats.last_sequence,
            )

        return self._stats

    async def run_until_signalled(self, *, max_events: int | None = None) -> ConsumerStats:
        """Run, stopping cleanly on SIGINT or SIGTERM.

        Signal handlers are only installed where the event loop supports them;
        on Windows ``add_signal_handler`` raises, so the plain run loop is used
        and Ctrl+C surfaces as :class:`KeyboardInterrupt` instead.
        """
        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, AttributeError, ValueError):
                loop.add_signal_handler(sig, self.stop)
                installed.append(sig)

        if installed:
            logger.info("signal_handlers_installed", signals=[s.name for s in installed])

        try:
            return await self.run(max_events=max_events)
        finally:
            for sig in installed:
                with contextlib.suppress(NotImplementedError, ValueError):
                    loop.remove_signal_handler(sig)
