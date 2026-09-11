"""Live state manager, pluggable state stores, and the DuckDB historical sink."""

from apexpulse.storage import keys
from apexpulse.storage.match_state import (
    LiveSnapshot,
    MatchHistory,
    MatchStateManager,
    MomentumSnapshot,
    RoundResult,
)
from apexpulse.storage.state import (
    InMemoryStateStore,
    RedisStateStore,
    SqliteStateStore,
    StateStore,
    create_state_store,
)

__all__ = [
    "InMemoryStateStore",
    "LiveSnapshot",
    "MatchHistory",
    "MatchStateManager",
    "MomentumSnapshot",
    "RedisStateStore",
    "RoundResult",
    "SqliteStateStore",
    "StateStore",
    "create_state_store",
    "keys",
]
