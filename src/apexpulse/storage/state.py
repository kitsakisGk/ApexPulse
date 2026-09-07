"""Pluggable live match-state store.

Holds the current snapshot per match: the hot, small, frequently-overwritten state
the dashboard reads. Historical telemetry goes to DuckDB instead (see the batch
sink), which is append-only and column-oriented.

Three backends share one interface — ``memory`` for tests, ``sqlite`` for local
development where state should survive a restart, and ``redis`` in production.
"""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Self

from apexpulse.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from apexpulse.config import Settings

logger = get_logger(__name__)

_COMPACT_JSON = (",", ":")


class StateStore(ABC):
    """Key/value store for live match snapshots, with per-key expiry."""

    @abstractmethod
    async def start(self) -> None:
        """Open connections and create schema if required."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources held by the store."""

    @abstractmethod
    async def set(self, key: str, value: dict[str, Any], *, ttl: int | None = None) -> None:
        """Store ``value`` under ``key``, expiring after ``ttl`` seconds."""

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """Return the value for ``key``, or ``None`` if absent or expired."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove ``key``; return whether it existed."""

    @abstractmethod
    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching a glob ``pattern``."""

    @abstractmethod
    async def ping(self) -> bool:
        """Return whether the backing store is reachable."""

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


class InMemoryStateStore(StateStore):
    """Process-local store. Fast, isolated per test, lost on restart."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[dict[str, Any], float | None]] = {}

    async def start(self) -> None:
        logger.info("state_store_started", backend="memory")

    async def stop(self) -> None:
        self._data.clear()

    async def set(self, key: str, value: dict[str, Any], *, ttl: int | None = None) -> None:
        expires_at = time.monotonic() + ttl if ttl else None
        self._data[key] = (dict(value), expires_at)

    async def get(self, key: str) -> dict[str, Any] | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() >= expires_at:
            del self._data[key]
            return None
        return dict(value)

    async def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    async def keys(self, pattern: str = "*") -> list[str]:
        now = time.monotonic()
        live = [
            key
            for key, (_, expires_at) in self._data.items()
            if expires_at is None or now < expires_at
        ]
        return sorted(key for key in live if fnmatch(key, pattern))

    async def ping(self) -> bool:
        return True


class SqliteStateStore(StateStore):
    """File-backed store using SQLite in WAL mode.

    The default on machines without Redis: state survives process restarts, and a
    single file keeps operational overhead at zero. Expiry is enforced on read and
    swept on write, mirroring Redis' lazy-expiry behaviour.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None

    async def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        # WAL keeps a reader (the API) from blocking the writer (the consumer).
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS match_state (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                expires_at REAL
            )
            """
        )
        self._conn.commit()
        logger.info("state_store_started", backend="sqlite", path=str(self._path))

    async def stop(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("state store is not started")
        return self._conn

    async def set(self, key: str, value: dict[str, Any], *, ttl: int | None = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        self._db.execute(
            "INSERT INTO match_state (key, value, expires_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, expires_at=excluded.expires_at",
            (key, json.dumps(value, separators=_COMPACT_JSON), expires_at),
        )
        self._db.execute(
            "DELETE FROM match_state WHERE expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),),
        )
        self._db.commit()

    async def get(self, key: str) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT value, expires_at FROM match_state WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value, expires_at = row
        if expires_at is not None and time.time() >= expires_at:
            await self.delete(key)
            return None
        parsed: dict[str, Any] = json.loads(value)
        return parsed

    async def delete(self, key: str) -> bool:
        cursor = self._db.execute("DELETE FROM match_state WHERE key = ?", (key,))
        self._db.commit()
        return cursor.rowcount > 0

    async def keys(self, pattern: str = "*") -> list[str]:
        rows = self._db.execute(
            "SELECT key FROM match_state WHERE expires_at IS NULL OR expires_at >= ?",
            (time.time(),),
        ).fetchall()
        return sorted(row[0] for row in rows if fnmatch(row[0], pattern))

    async def ping(self) -> bool:
        try:
            self._db.execute("SELECT 1").fetchone()
        except sqlite3.Error:
            return False
        return True


class RedisStateStore(StateStore):
    """Redis-backed store used when the container stack is available."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any = None

    async def start(self) -> None:
        from redis.asyncio import from_url

        self._client = from_url(str(self._settings.redis_url), decode_responses=True)
        logger.info("state_store_started", backend="redis", url=str(self._settings.redis_url))

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def _redis(self) -> Any:
        if self._client is None:
            raise RuntimeError("state store is not started")
        return self._client

    async def set(self, key: str, value: dict[str, Any], *, ttl: int | None = None) -> None:
        payload = json.dumps(value, separators=_COMPACT_JSON)
        if ttl:
            await self._redis.set(key, payload, ex=ttl)
        else:
            await self._redis.set(key, payload)

    async def get(self, key: str) -> dict[str, Any] | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        parsed: dict[str, Any] = json.loads(raw)
        return parsed

    async def delete(self, key: str) -> bool:
        deleted: int = await self._redis.delete(key)
        return deleted > 0

    async def keys(self, pattern: str = "*") -> list[str]:
        found: list[str] = [key async for key in self._redis.scan_iter(match=pattern)]
        return sorted(found)

    async def ping(self) -> bool:
        try:
            result: bool = await self._redis.ping()
        except Exception:  # any transport failure means unreachable
            return False
        return result


def create_state_store(settings: Settings | None = None) -> StateStore:
    """Return the store implementation named by ``settings.state_backend``."""
    from apexpulse.config import get_settings

    settings = settings or get_settings()

    if settings.state_backend == "memory":
        return InMemoryStateStore()
    if settings.state_backend == "sqlite":
        return SqliteStateStore(settings.sqlite_state_path)
    if settings.state_backend == "redis":
        try:
            import redis.asyncio  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "state_backend='redis' requires the 'stream' extra: uv sync --extra stream"
            ) from exc
        return RedisStateStore(settings)

    raise ValueError(f"unknown state backend: {settings.state_backend!r}")
