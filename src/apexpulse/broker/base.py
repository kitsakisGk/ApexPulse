"""Transport-agnostic broker interface.

Every implementation is an async context manager that publishes bytes to a named
topic and streams them back to subscribed consumer groups.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType


@dataclass(frozen=True, slots=True)
class BrokerMessage:
    """A single event as delivered to a consumer."""

    topic: str
    value: bytes
    key: str | None = None
    partition: int = 0
    offset: int = -1
    headers: dict[str, str] = field(default_factory=dict)


class EventBroker(ABC):
    """Abstract publish/subscribe transport."""

    @abstractmethod
    async def start(self) -> None:
        """Establish connections and prepare topics."""

    @abstractmethod
    async def stop(self) -> None:
        """Flush pending writes and release resources."""

    @abstractmethod
    async def publish(
        self,
        topic: str,
        value: bytes,
        *,
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Publish ``value`` to ``topic``."""

    @abstractmethod
    def consume(
        self,
        topic: str,
        *,
        group: str,
    ) -> AsyncIterator[BrokerMessage]:
        """Stream messages from ``topic`` for the consumer group ``group``.

        The iterator runs until the caller stops consuming or :meth:`stop` is
        called, whichever happens first.
        """

    @abstractmethod
    async def flush(self, timeout: float = 5.0) -> int:
        """Block until queued writes are delivered; return the count still pending."""

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()
