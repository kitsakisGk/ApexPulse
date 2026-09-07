"""Broker construction from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apexpulse.broker.memory import InMemoryBroker
from apexpulse.config import get_settings

if TYPE_CHECKING:
    from apexpulse.broker.base import EventBroker
    from apexpulse.config import Settings


def create_broker(settings: Settings | None = None) -> EventBroker:
    """Return the broker implementation named by ``settings.broker_backend``."""
    settings = settings or get_settings()

    if settings.broker_backend == "memory":
        return InMemoryBroker()

    if settings.broker_backend == "kafka":
        try:
            from apexpulse.broker.kafka import KafkaBroker
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "broker_backend='kafka' requires the 'stream' extra: uv sync --extra stream"
            ) from exc
        return KafkaBroker(settings)

    raise ValueError(f"unknown broker backend: {settings.broker_backend!r}")
