"""Streaming consumer and sliding-window tick aggregation."""

from apexpulse.stream.consumer import ConsumerStats, EventHandler, TelemetryConsumer
from apexpulse.stream.tracker import MatchTracker, match_key
from apexpulse.stream.window import (
    DEFAULT_WINDOW_SECONDS,
    SlidingWindow,
    TelemetryWindow,
    WindowedMetrics,
)

__all__ = [
    "DEFAULT_WINDOW_SECONDS",
    "ConsumerStats",
    "EventHandler",
    "MatchTracker",
    "SlidingWindow",
    "TelemetryConsumer",
    "TelemetryWindow",
    "WindowedMetrics",
    "match_key",
]
