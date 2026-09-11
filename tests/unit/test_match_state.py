"""Tests for the live match-state manager and its key layout."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from apexpulse.config import Settings
from apexpulse.schemas.enums import MapName, RoundEndReason, RoundPhase, Team
from apexpulse.schemas.models import MatchState, PlayerState, RoundState, TeamEconomy
from apexpulse.storage import (
    InMemoryStateStore,
    LiveSnapshot,
    MatchHistory,
    MatchStateManager,
    MomentumSnapshot,
    RoundResult,
    SqliteStateStore,
    keys,
)

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def at(offset: float) -> datetime:
    return BASE_TIME + timedelta(seconds=offset)


def make_state(match_id: str = "m-1", **overrides: object) -> MatchState:
    players = tuple(
        PlayerState(player_id=f"ct{i}", name=f"CT{i}", team=Team.CT, health=100, money=1000)
        for i in range(5)
    ) + tuple(
        PlayerState(player_id=f"t{i}", name=f"T{i}", team=Team.T, health=100, money=1000)
        for i in range(5)
    )
    fields: dict[str, object] = {
        "match_id": match_id,
        "map_name": MapName.MIRAGE,
        "round_state": RoundState(round_number=3, phase=RoundPhase.LIVE, seconds_remaining=60.0),
        "players": players,
        "economy_ct": TeamEconomy(team=Team.CT, money=5000, equipment_value=12000),
        "economy_t": TeamEconomy(team=Team.T, money=4000, equipment_value=9000),
    }
    fields.update(overrides)
    return MatchState(**fields)  # type: ignore[arg-type]


def make_snapshot(match_id: str = "m-1", sequence: int = 1, **momentum: object) -> LiveSnapshot:
    return LiveSnapshot(
        match_id=match_id,
        sequence=sequence,
        timestamp=at(sequence),
        state=make_state(match_id),
        momentum=MomentumSnapshot(**momentum),  # type: ignore[arg-type]
    )


def make_round(number: int, winner: Team = Team.CT, *, ct: int = 1, t: int = 0) -> RoundResult:
    reason = RoundEndReason.T_ELIMINATED if winner is Team.CT else RoundEndReason.CT_ELIMINATED
    return RoundResult(
        round_number=number,
        winner=winner,
        reason=reason,
        score_ct=ct,
        score_t=t,
        timestamp=at(number * 100),
    )


@pytest.fixture
async def manager():
    """A manager backed by an in-memory store."""
    async with InMemoryStateStore() as store:
        yield MatchStateManager(store, Settings(state_backend="memory"))


# -- Key layout ---------------------------------------------------------------


def test_keys_are_namespaced_per_entity() -> None:
    assert keys.match_key("m-1") == "apexpulse:match:m-1"
    assert keys.rounds_key("m-1") == "apexpulse:match:m-1:rounds"


def test_a_snapshot_key_round_trips_to_its_match_id() -> None:
    assert keys.match_id_from_key(keys.match_key("apex-99")) == "apex-99"


def test_facet_keys_are_not_mistaken_for_snapshots() -> None:
    """`live_match_ids` scans one prefix, so facets must be distinguishable."""
    assert keys.match_id_from_key(keys.rounds_key("m-1")) is None
    assert keys.match_id_from_key("some:other:key") is None


# -- Snapshots ----------------------------------------------------------------


async def test_a_published_snapshot_reads_back_as_a_typed_model(manager) -> None:
    await manager.publish_snapshot(make_snapshot(sequence=7))

    restored = await manager.get_snapshot("m-1")

    assert isinstance(restored, LiveSnapshot)
    assert restored.sequence == 7
    assert restored.state.map_name is MapName.MIRAGE
    assert restored.state.alive_ct == 5


async def test_reading_an_unknown_match_returns_none(manager) -> None:
    assert await manager.get_snapshot("never-seen") is None


async def test_publishing_overwrites_the_previous_snapshot(manager) -> None:
    await manager.publish_snapshot(make_snapshot(sequence=1))
    await manager.publish_snapshot(make_snapshot(sequence=2))

    restored = await manager.get_snapshot("m-1")

    assert restored is not None
    assert restored.sequence == 2


async def test_momentum_survives_the_round_trip(manager) -> None:
    await manager.publish_snapshot(
        make_snapshot(kills_ct=3, kills_t=1, kill_delta=2, headshot_rate=0.5)
    )

    restored = await manager.get_snapshot("m-1")

    assert restored is not None
    assert restored.momentum.kills_ct == 3
    assert restored.momentum.kill_delta == 2
    assert restored.momentum.headshot_rate == 0.5


async def test_malformed_stored_state_is_discarded_rather_than_raised(manager) -> None:
    """Stored state can predate a schema change; a reader must not crash on it."""
    await manager.store.set(keys.match_key("m-1"), {"match_id": "m-1", "garbage": True})

    assert await manager.get_snapshot("m-1") is None


# -- Round history ------------------------------------------------------------


async def test_history_starts_empty(manager) -> None:
    history = await manager.get_history("m-1")

    assert isinstance(history, MatchHistory)
    assert history.rounds_played == 0
    assert history.current_streak() is None


async def test_rounds_accumulate_in_order(manager) -> None:
    await manager.append_round("m-1", make_round(1, Team.CT, ct=1, t=0))
    await manager.append_round("m-1", make_round(2, Team.T, ct=1, t=1))
    history = await manager.append_round("m-1", make_round(3, Team.CT, ct=2, t=1))

    assert history.rounds_played == 3
    assert [entry.round_number for entry in history.rounds] == [1, 2, 3]
    assert history.wins_for(Team.CT) == 2
    assert history.wins_for(Team.T) == 1


async def test_replaying_a_round_updates_it_rather_than_duplicating(manager) -> None:
    """A consumer restart re-reads the topic, so appends must be idempotent."""
    await manager.append_round("m-1", make_round(1, Team.CT, ct=1, t=0))
    history = await manager.append_round("m-1", make_round(1, Team.T, ct=0, t=1))

    assert history.rounds_played == 1
    assert history.rounds[0].winner is Team.T


async def test_out_of_order_rounds_are_sorted(manager) -> None:
    await manager.append_round("m-1", make_round(3))
    await manager.append_round("m-1", make_round(1))
    history = await manager.append_round("m-1", make_round(2))

    assert [entry.round_number for entry in history.rounds] == [1, 2, 3]


async def test_the_current_streak_counts_consecutive_wins(manager) -> None:
    await manager.append_round("m-1", make_round(1, Team.T))
    await manager.append_round("m-1", make_round(2, Team.CT))
    await manager.append_round("m-1", make_round(3, Team.CT))
    history = await manager.append_round("m-1", make_round(4, Team.CT))

    assert history.current_streak() == (Team.CT, 3)


async def test_a_streak_resets_when_the_other_side_wins(manager) -> None:
    await manager.append_round("m-1", make_round(1, Team.CT))
    await manager.append_round("m-1", make_round(2, Team.CT))
    history = await manager.append_round("m-1", make_round(3, Team.T))

    assert history.current_streak() == (Team.T, 1)


async def test_history_is_capped_to_bound_memory(manager) -> None:
    from apexpulse.storage.match_state import MAX_TRACKED_ROUNDS

    for number in range(1, MAX_TRACKED_ROUNDS + 5):
        history = await manager.append_round("m-1", make_round(min(number, 64)))

    assert history.rounds_played <= MAX_TRACKED_ROUNDS


# -- Discovery ----------------------------------------------------------------


async def test_live_match_ids_lists_only_snapshot_keys(manager) -> None:
    await manager.publish_snapshot(make_snapshot("alpha"))
    await manager.publish_snapshot(make_snapshot("beta"))
    await manager.append_round("alpha", make_round(1))
    await manager.store.set("unrelated:key", {"noise": True})

    assert await manager.live_match_ids() == ["alpha", "beta"]


async def test_live_snapshots_returns_every_tracked_match(manager) -> None:
    await manager.publish_snapshot(make_snapshot("alpha", sequence=1))
    await manager.publish_snapshot(make_snapshot("beta", sequence=2))

    snapshots = await manager.live_snapshots()

    assert [snapshot.match_id for snapshot in snapshots] == ["alpha", "beta"]


async def test_dropping_a_match_removes_snapshot_and_history(manager) -> None:
    await manager.publish_snapshot(make_snapshot("m-1"))
    await manager.append_round("m-1", make_round(1))

    assert await manager.drop("m-1") is True
    assert await manager.get_snapshot("m-1") is None
    assert (await manager.get_history("m-1")).rounds_played == 0
    assert await manager.drop("m-1") is False


async def test_health_reports_the_backing_store(manager) -> None:
    assert await manager.healthy() is True


# -- Backend parity -----------------------------------------------------------


async def test_the_manager_behaves_identically_on_sqlite(tmp_path) -> None:
    """The manager sits above the store, so swapping backends changes nothing."""
    async with SqliteStateStore(tmp_path / "state.sqlite3") as store:
        manager = MatchStateManager(store, Settings(state_backend="sqlite"))

        await manager.publish_snapshot(make_snapshot(sequence=5))
        await manager.append_round("m-1", make_round(1, Team.CT))

        snapshot = await manager.get_snapshot("m-1")
        history = await manager.get_history("m-1")

    assert snapshot is not None
    assert snapshot.sequence == 5
    assert history.rounds_played == 1


async def test_state_survives_a_store_restart(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    settings = Settings(state_backend="sqlite")

    async with SqliteStateStore(path) as store:
        await MatchStateManager(store, settings).publish_snapshot(make_snapshot(sequence=9))

    async with SqliteStateStore(path) as reopened:
        restored = await MatchStateManager(reopened, settings).get_snapshot("m-1")

    assert restored is not None
    assert restored.sequence == 9


async def test_snapshots_expire_with_the_configured_ttl() -> None:
    settings = Settings(state_backend="memory", redis_state_ttl_seconds=1)

    async with InMemoryStateStore() as store:
        manager = MatchStateManager(store, settings)
        await manager.publish_snapshot(make_snapshot())
        assert await manager.get_snapshot("m-1") is not None

        await asyncio.sleep(1.05)

        assert await manager.get_snapshot("m-1") is None
