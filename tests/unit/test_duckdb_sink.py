"""Tests for the DuckDB historical sink.

The sink produces the supervised dataset the model trains on, so these assertions
focus on label correctness and on the join that pairs each tick with its outcome.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from apexpulse.config import Settings
from apexpulse.producer import MatchSimulator
from apexpulse.schemas.enums import MapName, RoundEndReason, RoundPhase, Team
from apexpulse.schemas.events import MatchEndEvent, RoundEndEvent, TickEvent
from apexpulse.schemas.models import MatchState, PlayerState, RoundState, TeamEconomy
from apexpulse.storage import DuckDBSink

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def at(offset: float) -> datetime:
    return BASE_TIME + timedelta(seconds=offset)


def make_tick(
    sequence: int,
    *,
    match_id: str = "m-1",
    round_number: int = 1,
    phase: RoundPhase = RoundPhase.LIVE,
    alive_ct: int = 5,
    alive_t: int = 5,
) -> TickEvent:
    """Build a tick with the given liveness on each side."""
    players = tuple(
        PlayerState(
            player_id=f"ct{i}",
            name=f"CT{i}",
            team=Team.CT,
            health=100 if i < alive_ct else 0,
            money=1000,
        )
        for i in range(5)
    ) + tuple(
        PlayerState(
            player_id=f"t{i}",
            name=f"T{i}",
            team=Team.T,
            health=100 if i < alive_t else 0,
            money=800,
        )
        for i in range(5)
    )

    state = MatchState(
        match_id=match_id,
        map_name=MapName.MIRAGE,
        timestamp=at(sequence),
        round_state=RoundState(round_number=round_number, phase=phase, seconds_remaining=90.0),
        players=players,
        economy_ct=TeamEconomy(team=Team.CT, money=5000, equipment_value=12000),
        economy_t=TeamEconomy(team=Team.T, money=4000, equipment_value=9000),
    )
    return TickEvent(match_id=match_id, timestamp=at(sequence), sequence=sequence, state=state)


def make_round(
    round_number: int = 1, winner: Team = Team.CT, *, match_id: str = "m-1"
) -> RoundEndEvent:
    reason = RoundEndReason.T_ELIMINATED if winner is Team.CT else RoundEndReason.CT_ELIMINATED
    return RoundEndEvent(
        match_id=match_id,
        timestamp=at(round_number * 100),
        sequence=round_number * 1000,
        round_number=round_number,
        winner=winner,
        reason=reason,
        score_ct=1 if winner is Team.CT else 0,
        score_t=0 if winner is Team.CT else 1,
    )


@pytest.fixture
async def sink():
    """An in-memory sink with a small batch so flushes are observable."""
    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=10) as active:
        yield active


# -- Schema and lifecycle -----------------------------------------------------


async def test_the_schema_is_created_on_start(sink) -> None:
    assert sink.count("ticks") == 0
    assert sink.count("rounds") == 0
    assert sink.count("matches") == 0


async def test_using_the_sink_before_start_is_rejected() -> None:
    unopened = DuckDBSink(path=":memory:")

    with pytest.raises(RuntimeError, match="not started"):
        unopened.count("ticks")


async def test_a_non_positive_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        DuckDBSink(path=":memory:", batch_size=0)


async def test_an_unknown_table_is_rejected(sink) -> None:
    with pytest.raises(ValueError, match="unknown table"):
        sink.count("robert'); DROP TABLE ticks;--")


# -- Buffering ----------------------------------------------------------------


async def test_ticks_buffer_until_the_batch_is_full(sink) -> None:
    for sequence in range(9):
        await sink.write_tick(make_tick(sequence))

    assert sink.pending == 9
    assert sink.count("ticks") == 0, "nothing is written before the batch fills"

    await sink.write_tick(make_tick(9))

    assert sink.pending == 0
    assert sink.count("ticks") == 10


async def test_an_explicit_flush_writes_a_partial_batch(sink) -> None:
    for sequence in range(3):
        await sink.write_tick(make_tick(sequence))

    assert await sink.flush() == 3
    assert sink.count("ticks") == 3


async def test_flushing_an_empty_buffer_is_a_no_op(sink) -> None:
    assert await sink.flush() == 0


async def test_stopping_flushes_buffered_ticks(tmp_path) -> None:
    """A consumer shutting down must not silently drop its last partial batch."""
    path = tmp_path / "telemetry.duckdb"

    async with DuckDBSink(path=path, settings=Settings(), batch_size=1000) as sink:
        for sequence in range(5):
            await sink.write_tick(make_tick(sequence))
        assert sink.pending == 5

    async with DuckDBSink(path=path, settings=Settings()) as reopened:
        assert reopened.count("ticks") == 5


# -- Rounds and matches -------------------------------------------------------


async def test_rounds_are_written_immediately(sink) -> None:
    """The label must never lag the features it explains."""
    await sink.write_round(make_round(1))

    assert sink.count("rounds") == 1


async def test_replaying_a_round_updates_it_rather_than_duplicating(sink) -> None:
    await sink.write_round(make_round(1, Team.CT))
    await sink.write_round(make_round(1, Team.T))

    assert sink.count("rounds") == 1


async def test_writing_a_match_flushes_pending_ticks(sink) -> None:
    await sink.write_tick(make_tick(1))
    assert sink.pending == 1

    await sink.write_match(
        MatchEndEvent(
            match_id="m-1", timestamp=at(999), sequence=999, winner=Team.CT, score_ct=13, score_t=7
        )
    )

    assert sink.pending == 0
    assert sink.count("ticks") == 1
    assert sink.count("matches") == 1


# -- Training data ------------------------------------------------------------


async def test_a_tick_is_labelled_by_its_round_outcome(sink) -> None:
    await sink.write_tick(make_tick(1, round_number=1))
    await sink.write_round(make_round(1, Team.CT))
    await sink.flush()

    frame = sink.training_frame()

    assert len(frame) == 1
    assert frame.iloc[0]["ct_won"] == 1
    assert frame.iloc[0]["round_winner"] == "CT"


async def test_a_t_round_is_labelled_zero(sink) -> None:
    await sink.write_tick(make_tick(1, round_number=1))
    await sink.write_round(make_round(1, Team.T))
    await sink.flush()

    assert sink.training_frame().iloc[0]["ct_won"] == 0


async def test_ticks_without_an_outcome_are_excluded(sink) -> None:
    """An in-flight round has no label yet, so it cannot be trained on."""
    await sink.write_tick(make_tick(1, round_number=1))
    await sink.write_tick(make_tick(2, round_number=2))
    await sink.write_round(make_round(1, Team.CT))
    await sink.flush()

    assert sink.count("ticks") == 2
    assert sink.count("training_data") == 1, "round 2 has not ended"


async def test_freezetime_ticks_are_excluded_from_training(sink) -> None:
    """Nothing has happened yet, so freezetime state cannot explain the outcome."""
    await sink.write_tick(make_tick(1, phase=RoundPhase.FREEZETIME))
    await sink.write_tick(make_tick(2, phase=RoundPhase.LIVE))
    await sink.write_round(make_round(1, Team.CT))
    await sink.flush()

    assert sink.count("training_data") == 1


async def test_replayed_ticks_are_deduplicated_by_the_view(sink) -> None:
    """A consumer restart re-reads the topic; the dataset must not double-count."""
    await sink.write_tick(make_tick(1))
    await sink.flush()
    await sink.write_tick(make_tick(1))
    await sink.flush()
    await sink.write_round(make_round(1, Team.CT))

    assert sink.count("ticks") == 2, "the raw table is append-only"
    assert sink.count("training_data") == 1, "the view collapses the replay"


async def test_label_balance_reports_both_classes(sink) -> None:
    await sink.write_tick(make_tick(1, round_number=1))
    await sink.write_tick(make_tick(2, round_number=2))
    await sink.write_round(make_round(1, Team.CT))
    await sink.write_round(make_round(2, Team.T))
    await sink.flush()

    ct_wins, t_wins = sink.label_balance()

    assert (ct_wins, t_wins) == (1, 1)


async def test_label_balance_is_zero_on_an_empty_dataset(sink) -> None:
    assert sink.label_balance() == (0, 0)


async def test_alive_counts_are_persisted_accurately(sink) -> None:
    await sink.write_tick(make_tick(1, alive_ct=3, alive_t=1))
    await sink.write_round(make_round(1, Team.CT))
    await sink.flush()

    row = sink.training_frame().iloc[0]

    assert row["alive_ct"] == 3
    assert row["alive_t"] == 1
    assert row["health_ct"] == 300
    assert row["health_t"] == 100


# -- Export -------------------------------------------------------------------


async def test_training_data_exports_to_parquet(sink, tmp_path) -> None:
    await sink.write_tick(make_tick(1))
    await sink.write_round(make_round(1, Team.CT))
    await sink.flush()

    destination = sink.export_parquet(tmp_path / "out" / "training.parquet")

    assert destination.is_file()
    assert destination.stat().st_size > 0


async def test_exporting_an_unknown_table_is_rejected(sink, tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown table"):
        sink.export_parquet(tmp_path / "x.parquet", table="; DROP TABLE ticks")


# -- End to end ---------------------------------------------------------------


async def test_a_simulated_match_produces_a_usable_dataset() -> None:
    """One match must land in every table with a coherent shape."""
    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=500) as sink:
        for event in MatchSimulator(seed=4, tick_rate_hz=2.0, start_time=BASE_TIME).run():
            await sink.handle(event)
        await sink.flush()

        assert sink.count("matches") == 1
        assert sink.count("rounds") >= 13
        assert sink.count("training_data") > 100

        ct_wins, t_wins = sink.label_balance()

        assert ct_wins > 0 and t_wins > 0, "both outcomes must appear"
        assert ct_wins + t_wins == sink.count("training_data")


async def test_the_dataset_carries_a_learnable_man_advantage_signal() -> None:
    """CT should win more often while ahead on players.

    Measured across several matches on purpose. Within a single match the effect
    is swamped by bomb state — a post-plant 5v3 still favours the Ts — so a
    one-match assertion tests noise rather than signal.
    """
    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=2_000) as sink:
        for seed in range(8):
            simulator = MatchSimulator(
                match_id=f"m-{seed}", seed=seed, tick_rate_hz=2.0, start_time=BASE_TIME
            )
            for event in simulator.run():
                await sink.handle(event)
        await sink.flush()

        frame = sink.training_frame()

    ahead = frame[frame.alive_ct > frame.alive_t].ct_won.mean()
    level = frame[frame.alive_ct == frame.alive_t].ct_won.mean()
    behind = frame[frame.alive_ct < frame.alive_t].ct_won.mean()

    assert ahead > level > behind, (
        f"expected monotonic signal: ahead={ahead:.3f} level={level:.3f} behind={behind:.3f}"
    )
    assert ahead - behind > 0.1, "the effect must be large enough for a model to learn"


async def test_bulk_writes_sustain_a_useful_throughput() -> None:
    """Guards the Arrow bulk-append path; a row-wise upsert managed ~130 rows/s."""
    import time

    ticks = [make_tick(sequence) for sequence in range(5_000)]

    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=1_000) as sink:
        started = time.perf_counter()
        for tick in ticks:
            await sink.write_tick(tick)
        await sink.flush()
        elapsed = time.perf_counter() - started

        assert sink.count("ticks") == 5_000

    assert elapsed < 5.0, f"5k ticks took {elapsed:.2f}s"
