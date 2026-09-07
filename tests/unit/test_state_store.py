"""Contract tests for the pluggable live-state store.

Every backend is exercised through the same assertions: if the implementations are
genuinely interchangeable, one suite must pass against all of them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from apexpulse.config import Settings
from apexpulse.storage import (
    InMemoryStateStore,
    SqliteStateStore,
    StateStore,
    create_state_store,
)

SNAPSHOT = {"match_id": "m-1", "round": 7, "score": {"ct": 4, "t": 2}}


@pytest.fixture(params=["memory", "sqlite"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[StateStore]:
    """Yield each backend in turn, started and torn down around the test."""
    backend: StateStore
    if request.param == "memory":
        backend = InMemoryStateStore()
    else:
        backend = SqliteStateStore(tmp_path / "state.sqlite3")

    async with backend as started:
        yield started


async def test_set_then_get_round_trips_the_snapshot(store: StateStore) -> None:
    await store.set("match:m-1", SNAPSHOT)

    assert await store.get("match:m-1") == SNAPSHOT


async def test_missing_key_returns_none(store: StateStore) -> None:
    assert await store.get("match:absent") is None


async def test_set_overwrites_the_previous_snapshot(store: StateStore) -> None:
    await store.set("match:m-1", {"round": 1})
    await store.set("match:m-1", {"round": 2})

    assert await store.get("match:m-1") == {"round": 2}


async def test_delete_reports_whether_the_key_existed(store: StateStore) -> None:
    await store.set("match:m-1", SNAPSHOT)

    assert await store.delete("match:m-1") is True
    assert await store.delete("match:m-1") is False
    assert await store.get("match:m-1") is None


async def test_keys_filters_by_glob_pattern(store: StateStore) -> None:
    await store.set("match:m-1", {"a": 1})
    await store.set("match:m-2", {"a": 2})
    await store.set("player:p-1", {"a": 3})

    assert await store.keys("match:*") == ["match:m-1", "match:m-2"]
    assert await store.keys() == ["match:m-1", "match:m-2", "player:p-1"]


async def test_expired_entries_are_not_returned(store: StateStore) -> None:
    await store.set("match:ttl", SNAPSHOT, ttl=1)
    assert await store.get("match:ttl") is not None

    await asyncio.sleep(1.05)

    assert await store.get("match:ttl") is None
    assert "match:ttl" not in await store.keys()


async def test_ping_reports_reachable(store: StateStore) -> None:
    assert await store.ping() is True


async def test_stored_values_are_copied_not_aliased(store: StateStore) -> None:
    """Mutating the caller's dict must not corrupt what the store holds."""
    payload = {"round": 1}
    await store.set("match:m-1", payload)
    payload["round"] = 99

    assert await store.get("match:m-1") == {"round": 1}


async def test_sqlite_state_survives_a_reopen(tmp_path: Path) -> None:
    """The sqlite backend is the one that must outlive the process."""
    path = tmp_path / "state.sqlite3"

    async with SqliteStateStore(path) as store:
        await store.set("match:m-1", SNAPSHOT)

    async with SqliteStateStore(path) as reopened:
        assert await reopened.get("match:m-1") == SNAPSHOT


async def test_using_sqlite_before_start_is_rejected(tmp_path: Path) -> None:
    store = SqliteStateStore(tmp_path / "state.sqlite3")

    with pytest.raises(RuntimeError, match="not started"):
        await store.get("match:m-1")


def test_factory_honours_the_configured_backend(tmp_path: Path) -> None:
    memory = create_state_store(Settings(state_backend="memory"))
    sqlite = create_state_store(
        Settings(state_backend="sqlite", sqlite_state_path=tmp_path / "s.sqlite3")
    )

    assert isinstance(memory, InMemoryStateStore)
    assert isinstance(sqlite, SqliteStateStore)


def test_factory_rejects_an_unknown_backend() -> None:
    settings = Settings()
    object.__setattr__(settings, "state_backend", "cassandra")

    with pytest.raises(ValueError, match="unknown state backend"):
        create_state_store(settings)
