"""Real-time inference worker.

Consumes telemetry, scores each tick, and publishes the prediction back onto the
broker for the API to broadcast. The worker owns the window per match, so the
momentum features the model was trained on are present at serving time too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import orjson

from apexpulse.config import get_settings
from apexpulse.logging import get_logger
from apexpulse.schemas.events import TickEvent
from apexpulse.stream.window import TelemetryWindow

if TYPE_CHECKING:
    from apexpulse.broker.base import EventBroker
    from apexpulse.config import Settings
    from apexpulse.inference.engine import InferenceEngine, Prediction
    from apexpulse.schemas.events import TelemetryEvent
    from apexpulse.storage.state import StateStore

logger = get_logger(__name__)

PREDICTION_KEY_PREFIX = "apexpulse:prediction:"
"""State-store namespace for the latest prediction per match."""


def prediction_key(match_id: str) -> str:
    """Return the key holding ``match_id``'s latest prediction."""
    return f"{PREDICTION_KEY_PREFIX}{match_id}"


@dataclass
class WorkerStats:
    """Counters describing a worker run."""

    ticks_seen: int = 0
    predictions_published: int = 0
    skipped: int = 0
    errors: int = 0

    @property
    def publish_rate(self) -> float:
        """Share of observed ticks that produced a published prediction."""
        return 0.0 if self.ticks_seen == 0 else self.predictions_published / self.ticks_seen


@dataclass
class InferenceWorker:
    """Score live telemetry and publish the results.

    Registers against a :class:`~apexpulse.stream.consumer.TelemetryConsumer`;
    every tick is scored and the prediction is both published to the prediction
    topic and written to the state store for the API to read.

    Args:
        engine: A loaded inference engine.
        broker: Transport to publish predictions on; omit to skip publishing.
        store: State store for the latest prediction; omit to skip writing.
        settings: Runtime configuration; defaults to the process settings.
        window_seconds: Match time retained for momentum features.
    """

    engine: InferenceEngine
    broker: EventBroker | None = None
    store: StateStore | None = None
    settings: Settings = field(default_factory=get_settings)
    window_seconds: float = 15.0

    _windows: dict[str, TelemetryWindow] = field(default_factory=dict, init=False, repr=False)
    _stats: WorkerStats = field(default_factory=WorkerStats, init=False, repr=False)
    _latest: dict[str, Prediction] = field(default_factory=dict, init=False, repr=False)

    @property
    def stats(self) -> WorkerStats:
        """Counters for this worker."""
        return self._stats

    def latest(self, match_id: str) -> Prediction | None:
        """Return the most recent prediction for ``match_id``, if any."""
        return self._latest.get(match_id)

    def window_for(self, match_id: str) -> TelemetryWindow:
        """Return the window tracking ``match_id``, creating it on first sight."""
        if match_id not in self._windows:
            self._windows[match_id] = TelemetryWindow(span_seconds=self.window_seconds)
        return self._windows[match_id]

    async def handle(self, event: TelemetryEvent) -> None:
        """Fold ``event`` into the window and score it when it is a tick."""
        window = self.window_for(event.match_id)
        window.observe(event)

        if not isinstance(event, TickEvent):
            if event.event_type.value == "match_end":
                self._windows.pop(event.match_id, None)
                self._latest.pop(event.match_id, None)
            return

        self._stats.ticks_seen += 1
        prediction = self.engine.predict(event.state, window.metrics())

        if not prediction.scored:
            self._stats.skipped += 1
            return

        self._latest[event.match_id] = prediction
        await self._publish(event, prediction)

    async def _publish(self, tick: TickEvent, prediction: Prediction) -> None:
        """Send a prediction to the broker and the state store.

        Failures are counted and logged rather than raised: a dropped prediction
        costs one dashboard frame, while a raise would stop ingestion entirely.
        """
        payload = {
            "match_id": prediction.match_id,
            "sequence": tick.sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "round_number": prediction.round_number,
            "ct_win_probability": round(prediction.ct_win_probability, 6),
            "t_win_probability": round(prediction.t_win_probability, 6),
            "favoured_side": prediction.favoured_side,
            "confidence": round(prediction.confidence, 6),
            "latency_ms": round(prediction.latency_ms, 4),
        }

        try:
            if self.broker is not None:
                await self.broker.publish(
                    self.settings.kafka_prediction_topic,
                    orjson.dumps(payload),
                    key=prediction.match_id,
                )
            if self.store is not None:
                await self.store.set(
                    prediction_key(prediction.match_id),
                    payload,
                    ttl=self.settings.redis_state_ttl_seconds,
                )
        except Exception as exc:  # a dropped frame beats halting the pipeline
            self._stats.errors += 1
            logger.error(
                "prediction_publish_failed",
                match_id=prediction.match_id,
                error=str(exc),
            )
            return

        self._stats.predictions_published += 1

    async def read_prediction(self, match_id: str) -> dict[str, Any] | None:
        """Read the stored prediction for ``match_id``."""
        if self.store is None:
            return None
        return await self.store.get(prediction_key(match_id))
