"""In-process broker backed by ``asyncio`` queues.

Mirrors the semantics the pipeline actually depends on — durable per-group offsets,
independent fan-out to each consumer group, and monotonic offsets within a topic —
without requiring a running Redpanda cluster. This is the default transport on
developer machines and in unit tests.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from typing import TYPE_CHECKING

from apexpulse.broker.base import BrokerMessage, EventBroker
from apexpulse.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)

_SENTINEL = object()


class InMemoryBroker(EventBroker):
    """Async publish/subscribe over per-group in-memory queues.

    Each consumer group receives every message published to a topic, matching
    Kafka's fan-out. Messages published while no group is subscribed are dropped,
    exactly as they would be for a Kafka consumer starting at ``latest``.
    """

    def __init__(self, *, maxsize: int = 10_000) -> None:
        self._maxsize = maxsize
        self._queues: dict[str, dict[str, asyncio.Queue[BrokerMessage | object]]] = defaultdict(
            dict
        )
        self._offsets: dict[str, int] = defaultdict(int)
        self._running = False
        self._dropped = 0

    async def start(self) -> None:
        self._running = True
        logger.info("broker_started", backend="memory", maxsize=self._maxsize)

    async def stop(self) -> None:
        """Signal every subscriber to finish, then drop all queues."""
        self._running = False
        for groups in self._queues.values():
            for queue in groups.values():
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(_SENTINEL)
        self._queues.clear()
        logger.info("broker_stopped", backend="memory", dropped=self._dropped)

    async def publish(
        self,
        topic: str,
        value: bytes,
        *,
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not self._running:
            raise RuntimeError("broker is not started")

        offset = self._offsets[topic]
        self._offsets[topic] = offset + 1
        message = BrokerMessage(
            topic=topic,
            value=value,
            key=key,
            offset=offset,
            headers=dict(headers or {}),
        )

        for group, queue in self._queues[topic].items():
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Shed load rather than stalling the producer, and make it visible:
                # a silently wedged simulator is far harder to diagnose.
                self._dropped += 1
                logger.warning("broker_queue_full", topic=topic, group=group, offset=offset)

    async def consume(
        self,
        topic: str,
        *,
        group: str,
    ) -> AsyncIterator[BrokerMessage]:
        queue: asyncio.Queue[BrokerMessage | object] = asyncio.Queue(maxsize=self._maxsize)
        self._queues[topic][group] = queue
        logger.info("consumer_subscribed", topic=topic, group=group)

        try:
            while self._running:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                assert isinstance(item, BrokerMessage)
                yield item
        finally:
            self._queues.get(topic, {}).pop(group, None)
            logger.info("consumer_unsubscribed", topic=topic, group=group)

    async def flush(self, timeout: float = 5.0) -> int:
        """Wait for subscribers to drain; return messages still queued."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            pending = sum(q.qsize() for groups in self._queues.values() for q in groups.values())
            if pending == 0:
                return 0
            await asyncio.sleep(0.01)
        return sum(q.qsize() for groups in self._queues.values() for q in groups.values())

    @property
    def dropped_messages(self) -> int:
        """Messages discarded because a subscriber queue was full."""
        return self._dropped
