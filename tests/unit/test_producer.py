"""Tests for the match simulator and the replay publisher.

The simulator is the pipeline's data source, so these assertions target the
properties everything downstream assumes: determinism, ordered sequences, valid
events, and match structure that obeys the CS2 ruleset.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from itertools import islice

import pytest

from apexpulse.broker import InMemoryBroker
from apexpulse.config import Settings
from apexpulse.producer import MatchSimulator, TelemetryReplayer
from apexpulse.schemas import constants
from apexpulse.schemas.enums import MapName, RoundEndReason, RoundPhase, Team
from apexpulse.schemas.events import (
    BombPlantedEvent,
    KillEvent,
    MatchEndEvent,
    MatchStartEvent,
    RoundEndEvent,
    RoundStartEvent,
    TickEvent,
    parse_event,
)

# A low tick rate keeps the suite fast; structure is unaffected.
FAST_TICK_RATE = 2.0

FIXED_START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
"""Pinned so event timestamps are reproducible across runs."""


@pytest.fixture(scope="module")
def match() -> list:
    """One complete match, reused across assertions to keep the suite quick."""
    return list(MatchSimulator(seed=7, tick_rate_hz=FAST_TICK_RATE).run())


# -- Determinism --------------------------------------------------------------


def test_the_same_seed_reproduces_the_same_match() -> None:
    """Identical seed and start time must reproduce the stream byte for byte."""

    def prefix() -> list[str]:
        simulator = MatchSimulator(seed=99, tick_rate_hz=FAST_TICK_RATE, start_time=FIXED_START)
        return [event.model_dump_json() for event in islice(simulator.run(), 400)]

    assert prefix() == prefix()


def test_timestamps_advance_with_the_simulated_clock_not_wall_time() -> None:
    """A replay is driven by the match clock, so start_time fully determines stamps."""
    first = next(MatchSimulator(seed=99, start_time=FIXED_START).run())
    second = next(MatchSimulator(seed=99, start_time=FIXED_START).run())

    assert first.timestamp == second.timestamp == FIXED_START


def test_different_seeds_produce_different_matches() -> None:
    def prefix(seed: int) -> list[str]:
        simulator = MatchSimulator(seed=seed, tick_rate_hz=FAST_TICK_RATE, start_time=FIXED_START)
        return [event.model_dump_json() for event in islice(simulator.run(), 400)]

    assert prefix(1) != prefix(2)


def test_a_non_positive_tick_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="tick_rate_hz must be positive"):
        MatchSimulator(tick_rate_hz=0.0)


# -- Event stream structure ---------------------------------------------------


def test_sequences_are_unique_and_monotonic(match: list) -> None:
    """Downstream ordering and de-duplication depend on this."""
    sequences = [event.sequence for event in match]

    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


def test_timestamps_never_move_backwards(match: list) -> None:
    stamps = [event.timestamp for event in match]

    assert stamps == sorted(stamps)


def test_the_match_is_bracketed_by_start_and_end_events(match: list) -> None:
    assert isinstance(match[0], MatchStartEvent)
    assert isinstance(match[-1], MatchEndEvent)
    assert sum(isinstance(e, MatchStartEvent) for e in match) == 1
    assert sum(isinstance(e, MatchEndEvent) for e in match) == 1


def test_every_round_start_is_matched_by_a_round_end(match: list) -> None:
    starts = [e for e in match if isinstance(e, RoundStartEvent)]
    ends = [e for e in match if isinstance(e, RoundEndEvent)]

    assert len(starts) == len(ends)
    assert [e.round_number for e in starts] == list(range(1, len(starts) + 1))
    assert [e.round_number for e in ends] == list(range(1, len(ends) + 1))


def test_every_event_carries_the_match_id(match: list) -> None:
    assert {event.match_id for event in match} == {"apex-001"}


# -- Match rules --------------------------------------------------------------


def test_the_match_ends_once_a_side_reaches_the_win_threshold(match: list) -> None:
    final = match[-1]

    assert isinstance(final, MatchEndEvent)
    assert max(final.score_ct, final.score_t) >= constants.ROUNDS_TO_WIN
    assert final.score_ct + final.score_t <= constants.MAX_REGULATION_ROUNDS + 1


def test_the_declared_winner_holds_the_higher_score(match: list) -> None:
    final = match[-1]
    assert isinstance(final, MatchEndEvent)

    if final.winner is Team.CT:
        assert final.score_ct > final.score_t
    else:
        assert final.score_t > final.score_ct


def test_the_running_score_increments_by_exactly_one_per_round(match: list) -> None:
    ends = [e for e in match if isinstance(e, RoundEndEvent)]

    previous_ct = previous_t = 0
    for event in ends:
        delta = (event.score_ct - previous_ct) + (event.score_t - previous_t)
        assert delta == 1, f"round {event.round_number} changed the score by {delta}"
        previous_ct, previous_t = event.score_ct, event.score_t


def test_the_final_scoreline_matches_the_rounds_played(match: list) -> None:
    ends = [e for e in match if isinstance(e, RoundEndEvent)]
    final = match[-1]
    assert isinstance(final, MatchEndEvent)

    ct_wins = sum(1 for e in ends if e.winner is Team.CT)
    t_wins = sum(1 for e in ends if e.winner is Team.T)

    assert (final.score_ct, final.score_t) == (ct_wins, t_wins)


def test_time_expiry_always_resolves_to_a_ct_win(match: list) -> None:
    """The Ts must detonate; running the clock down is a CT win by definition."""
    expiries = [
        e for e in match if isinstance(e, RoundEndEvent) and e.reason is RoundEndReason.TIME_EXPIRED
    ]

    assert all(event.winner is Team.CT for event in expiries)


def test_a_defused_bomb_was_planted_first(match: list) -> None:
    planted_rounds = {e.round_number for e in match if isinstance(e, BombPlantedEvent)}
    defused_rounds = {
        e.round_number
        for e in match
        if isinstance(e, RoundEndEvent) and e.reason is RoundEndReason.BOMB_DEFUSED
    }

    assert defused_rounds <= planted_rounds


def test_bomb_detonation_only_follows_a_plant(match: list) -> None:
    planted_rounds = {e.round_number for e in match if isinstance(e, BombPlantedEvent)}
    exploded_rounds = {
        e.round_number
        for e in match
        if isinstance(e, RoundEndEvent) and e.reason is RoundEndReason.BOMB_EXPLODED
    }

    assert exploded_rounds <= planted_rounds


def test_at_most_one_plant_per_round(match: list) -> None:
    rounds = Counter(e.round_number for e in match if isinstance(e, BombPlantedEvent))

    assert all(count == 1 for count in rounds.values())


# -- Tick snapshots -----------------------------------------------------------


def test_ticks_carry_a_consistent_state(match: list) -> None:
    ticks = [e for e in match if isinstance(e, TickEvent)]

    assert ticks, "a match must emit tick snapshots"
    for tick in ticks[:200]:
        state = tick.state
        assert state.match_id == tick.match_id
        assert state.alive_ct + state.alive_t <= 2 * constants.PLAYERS_PER_TEAM
        assert state.man_advantage == state.alive_ct - state.alive_t


def test_every_round_opens_with_a_freezetime_snapshot(match: list) -> None:
    phases_by_round: dict[int, list[RoundPhase]] = {}
    for event in match:
        if isinstance(event, TickEvent):
            phases_by_round.setdefault(event.state.round_state.round_number, []).append(
                event.state.round_state.phase
            )

    for round_number, phases in phases_by_round.items():
        assert phases[0] is RoundPhase.FREEZETIME, f"round {round_number} skipped freezetime"


def test_the_bomb_countdown_only_exists_after_a_plant(match: list) -> None:
    for event in match:
        if not isinstance(event, TickEvent):
            continue
        round_state = event.state.round_state
        assert (round_state.bomb_seconds_remaining is not None) == round_state.bomb_planted


def test_players_are_restored_to_full_health_each_round(match: list) -> None:
    """A round must start with ten live players."""
    seen: set[int] = set()
    for event in match:
        if not isinstance(event, TickEvent):
            continue
        round_number = event.state.round_state.round_number
        if round_number in seen:
            continue
        seen.add(round_number)
        assert event.state.alive_ct == constants.PLAYERS_PER_TEAM
        assert event.state.alive_t == constants.PLAYERS_PER_TEAM


def test_the_tick_rate_controls_snapshot_density() -> None:
    """Freezetime is a fixed number of seconds, so its tick count scales with the rate."""

    def first_round_freezetime_ticks(rate: float) -> int:
        count = 0
        for event in MatchSimulator(seed=3, tick_rate_hz=rate).run():
            if not isinstance(event, TickEvent):
                continue
            if event.state.round_state.phase is not RoundPhase.FREEZETIME:
                break
            count += 1
        return count

    assert first_round_freezetime_ticks(1.0) == int(constants.FREEZETIME_SECONDS)
    assert first_round_freezetime_ticks(4.0) == int(constants.FREEZETIME_SECONDS * 4)


# -- Kills and economy --------------------------------------------------------


def test_kills_reference_players_on_opposing_sides(match: list) -> None:
    kills = [e for e in match if isinstance(e, KillEvent)]

    assert kills, "a match must produce kills"
    for kill in kills:
        assert kill.killer_id != kill.victim_id
        killer_is_ct = kill.killer_id is not None and kill.killer_id.startswith("ct")
        assert killer_is_ct == (kill.victim_team is Team.T)


def test_player_money_stays_within_the_game_bounds(match: list) -> None:
    for event in match:
        if isinstance(event, TickEvent):
            for player in event.state.players:
                assert 0 <= player.money <= constants.MAX_MONEY


def test_a_losing_streak_raises_the_loss_bonus(match: list) -> None:
    """The economy model must reward consecutive losses, or buys are meaningless."""
    observed = {
        event.state.economy_t.consecutive_losses: event.state.economy_t.loss_bonus
        for event in match
        if isinstance(event, TickEvent)
    }

    for losses, bonus in observed.items():
        assert bonus == constants.loss_bonus(losses)


def test_map_selection_is_honoured() -> None:
    start = next(MatchSimulator(seed=5, tick_rate_hz=1.0, map_name=MapName.NUKE).run())

    assert isinstance(start, MatchStartEvent)
    assert start.map_name is MapName.NUKE


# -- Replay publisher ---------------------------------------------------------


async def test_replay_publishes_every_event_to_the_topic() -> None:
    settings = Settings(broker_backend="memory")

    async with InMemoryBroker() as broker:
        stats = await TelemetryReplayer(broker=broker, settings=settings, speed=0.0).replay(
            MatchSimulator(seed=11, tick_rate_hz=1.0)
        )

    assert stats.events_published > 0
    assert stats.ticks_published > 0
    assert stats.rounds_completed >= constants.ROUNDS_TO_WIN


async def test_replayed_payloads_survive_the_round_trip() -> None:
    """What lands on the topic must parse back into typed events."""
    settings = Settings(broker_backend="memory")
    received: list[bytes] = []

    async with InMemoryBroker() as broker:
        import asyncio

        async def _collect() -> None:
            async for message in broker.consume(settings.kafka_telemetry_topic, group="test"):
                received.append(message.value)

        consumer = asyncio.create_task(_collect())
        await asyncio.sleep(0.01)

        await TelemetryReplayer(broker=broker, settings=settings, speed=0.0).replay(
            MatchSimulator(seed=13, tick_rate_hz=1.0), max_events=50
        )
        await asyncio.sleep(0.05)
        consumer.cancel()

    assert received
    for payload in received:
        parse_event(payload)


async def test_max_events_bounds_the_run() -> None:
    settings = Settings(broker_backend="memory")

    async with InMemoryBroker() as broker:
        stats = await TelemetryReplayer(broker=broker, settings=settings, speed=0.0).replay(
            MatchSimulator(seed=17, tick_rate_hz=1.0), max_events=25
        )

    assert stats.events_published == 25


async def test_a_negative_speed_is_rejected() -> None:
    async with InMemoryBroker() as broker:
        with pytest.raises(ValueError, match="speed must be non-negative"):
            TelemetryReplayer(broker=broker, speed=-1.0)


def test_replay_stats_report_zero_rate_before_any_run() -> None:
    from apexpulse.producer import ReplayStats

    assert ReplayStats().events_per_second == 0.0
