"""Tests for the pluggable event transport."""

from __future__ import annotations

import asyncio

import pytest

from apexpulse.broker import BrokerMessage, InMemoryBroker, create_broker
from apexpulse.config import Settings


async def _drain(broker: InMemoryBroker, topic: str, group: str, count: int) -> list[BrokerMessage]:
    """Collect exactly ``count`` messages, failing fast if they never arrive."""
    received: list[BrokerMessage] = []

    async def _consume() -> None:
        async for message in broker.consume(topic, group=group):
            received.append(message)
            if len(received) >= count:
                return

    await asyncio.wait_for(_consume(), timeout=2.0)
    return received


async def test_publish_then_consume_round_trips_payload() -> None:
    async with InMemoryBroker() as broker:
        task = asyncio.create_task(_drain(broker, "ticks", "g1", 1))
        await asyncio.sleep(0.01)  # let the subscriber register
        await broker.publish("ticks", b'{"tick":1}', key="match-1")

        messages = await task

    assert messages[0].value == b'{"tick":1}'
    assert messages[0].key == "match-1"
    assert messages[0].topic == "ticks"


async def test_offsets_increase_monotonically_per_topic() -> None:
    async with InMemoryBroker() as broker:
        task = asyncio.create_task(_drain(broker, "ticks", "g1", 3))
        await asyncio.sleep(0.01)
        for index in range(3):
            await broker.publish("ticks", str(index).encode())

        messages = await task

    assert [m.offset for m in messages] == [0, 1, 2]


async def test_each_consumer_group_receives_every_message() -> None:
    """Fan-out must match Kafka: groups are independent, not competing consumers."""
    async with InMemoryBroker() as broker:
        first = asyncio.create_task(_drain(broker, "ticks", "group-a", 2))
        second = asyncio.create_task(_drain(broker, "ticks", "group-b", 2))
        await asyncio.sleep(0.01)

        await broker.publish("ticks", b"one")
        await broker.publish("ticks", b"two")

        got_a, got_b = await asyncio.gather(first, second)

    assert [m.value for m in got_a] == [b"one", b"two"]
    assert [m.value for m in got_b] == [b"one", b"two"]


async def test_headers_survive_the_round_trip() -> None:
    async with InMemoryBroker() as broker:
        task = asyncio.create_task(_drain(broker, "ticks", "g1", 1))
        await asyncio.sleep(0.01)
        await broker.publish("ticks", b"x", headers={"schema": "v1"})

        messages = await task

    assert messages[0].headers == {"schema": "v1"}


async def test_publishing_before_start_is_rejected() -> None:
    broker = InMemoryBroker()

    with pytest.raises(RuntimeError, match="not started"):
        await broker.publish("ticks", b"x")


async def test_full_queue_drops_rather_than_blocking_the_producer() -> None:
    """A wedged consumer must not stall the simulator."""
    broker = InMemoryBroker(maxsize=2)
    await broker.start()

    # Subscribe, then take a single message so the queue registers and then stalls.
    consumer = broker.consume("ticks", group="slow")
    task = asyncio.create_task(consumer.__anext__())
    await asyncio.sleep(0.01)

    for _ in range(10):
        await broker.publish("ticks", b"x")

    assert broker.dropped_messages > 0

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await broker.stop()


async def test_flush_returns_zero_once_drained() -> None:
    async with InMemoryBroker() as broker:
        assert await broker.flush(timeout=0.2) == 0


def test_factory_returns_memory_broker_by_default() -> None:
    broker = create_broker(Settings(broker_backend="memory"))

    assert isinstance(broker, InMemoryBroker)


def test_factory_rejects_an_unknown_backend() -> None:
    settings = Settings()
    object.__setattr__(settings, "broker_backend", "rabbitmq")

    with pytest.raises(ValueError, match="unknown broker backend"):
        create_broker(settings)
