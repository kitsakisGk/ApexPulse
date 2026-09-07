"""Redpanda/Kafka transport.

Imported lazily by :func:`~apexpulse.broker.factory.create_broker` so that
``confluent-kafka`` — which ships no wheel for some platforms — is only required
when the Kafka backend is actually selected.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from apexpulse.broker.base import BrokerMessage, EventBroker
from apexpulse.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from apexpulse.config import Settings

logger = get_logger(__name__)


class KafkaBroker(EventBroker):
    """Publish/subscribe over the Kafka protocol via ``confluent-kafka``.

    The underlying client is synchronous and releases the GIL during I/O, so calls
    are dispatched to a worker thread to keep the event loop responsive.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._producer: Any = None
        self._consumers: list[Any] = []
        self._running = False

    async def start(self) -> None:
        from confluent_kafka import Producer

        self._producer = Producer(
            {
                "bootstrap.servers": self._settings.kafka_bootstrap_servers,
                "linger.ms": 5,
                "batch.size": 32 * 1024,
                "compression.type": "lz4",
                "enable.idempotence": True,
                "acks": "all",
            }
        )
        self._running = True
        logger.info(
            "broker_started",
            backend="kafka",
            bootstrap_servers=self._settings.kafka_bootstrap_servers,
        )

    async def stop(self) -> None:
        self._running = False
        if self._producer is not None:
            await asyncio.to_thread(self._producer.flush, 10.0)
            self._producer = None
        for consumer in self._consumers:
            consumer.close()
        self._consumers.clear()
        logger.info("broker_stopped", backend="kafka")

    def _on_delivery(self, err: Any, msg: Any) -> None:
        """Delivery report callback; fired from the librdkafka poll thread."""
        if err is not None:
            logger.error("publish_failed", error=str(err), topic=msg.topic() if msg else None)

    async def publish(
        self,
        topic: str,
        value: bytes,
        *,
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if self._producer is None:
            raise RuntimeError("broker is not started")

        self._producer.produce(
            topic=topic,
            value=value,
            key=key.encode() if key else None,
            headers=list((headers or {}).items()) or None,
            on_delivery=self._on_delivery,
        )
        # Serve delivery callbacks without blocking the loop.
        self._producer.poll(0)

    async def consume(
        self,
        topic: str,
        *,
        group: str,
    ) -> AsyncIterator[BrokerMessage]:
        from confluent_kafka import Consumer, KafkaError

        consumer = Consumer(
            {
                "bootstrap.servers": self._settings.kafka_bootstrap_servers,
                "group.id": group,
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
            }
        )
        consumer.subscribe([topic])
        self._consumers.append(consumer)
        logger.info("consumer_subscribed", topic=topic, group=group)

        try:
            while self._running:
                msg = await asyncio.to_thread(consumer.poll, 0.5)
                if msg is None:
                    continue

                error = msg.error()
                if error is not None:
                    # End-of-partition is informational, not a failure.
                    if error.code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("consume_error", error=str(error))
                    continue

                raw_value = msg.value()
                msg_topic = msg.topic()
                if not isinstance(raw_value, bytes) or msg_topic is None:
                    continue

                raw_key = msg.key()
                decoded_key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key

                # librdkafka returns headers as a list of pairs; normalise to str/str.
                raw_headers = msg.headers() or []
                header_pairs = raw_headers.items() if isinstance(raw_headers, dict) else raw_headers
                decoded_headers = {
                    name: raw.decode() if isinstance(raw, bytes) else str(raw)
                    for name, raw in header_pairs
                    if raw is not None
                }

                offset = msg.offset()
                partition = msg.partition()

                yield BrokerMessage(
                    topic=msg_topic,
                    value=raw_value,
                    key=decoded_key,
                    partition=partition if partition is not None else 0,
                    offset=offset if offset is not None else -1,
                    headers=decoded_headers,
                )
        finally:
            consumer.close()
            if consumer in self._consumers:
                self._consumers.remove(consumer)
            logger.info("consumer_unsubscribed", topic=topic, group=group)

    async def flush(self, timeout: float = 5.0) -> int:
        if self._producer is None:
            return 0
        remaining: int = await asyncio.to_thread(self._producer.flush, timeout)
        return remaining
