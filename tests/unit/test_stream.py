"""Tests for the streaming consumer, sliding window, and match tracker."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from apexpulse.broker import InMemoryBroker
from apexpulse.config import Settings
from apexpulse.producer import MatchSimulator, TelemetryReplayer
from apexpulse.schemas.enums import EventType, RoundEndReason, Team, Weapon
from apexpulse.schemas.events import (
    KillEvent,
    MatchEndEvent,
    RoundEndEvent,
    RoundStartEvent,
    TickEvent,
    serialise_event,
)
from apexpulse.storage import InMemoryStateStore
from apexpulse.stream import (
    MatchTracker,
    SlidingWindow,
    TelemetryConsumer,
    TelemetryWindow,
    match_key,
)

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def at(offset: float) -> datetime:
    """Return ``offset`` seconds after the fixed base time."""
    return BASE_TIME + timedelta(seconds=offset)


def make_kill(offset: float, *, victim_team: Team = Team.T, headshot: bool = False) -> KillEvent:
    return KillEvent(
        match_id="m-1",
        timestamp=at(offset),
        sequence=int(offset * 10),
        round_number=1,
        killer_id="ct0" if victim_team is Team.T else "t0",
        victim_id="t0" if victim_team is Team.T else "ct0",
        weapon=Weapon.AK47,
        headshot=headshot,
        victim_team=victim_team,
    )


# -- SlidingWindow ------------------------------------------------------------


def test_a_window_retains_items_inside_its_span() -> None:
    window: SlidingWindow[str] = SlidingWindow(span_seconds=10.0)

    window.append(at(0), "a")
    window.append(at(5), "b")
    window.append(at(9), "c")

    assert window.items == ("a", "b", "c")


def test_a_window_evicts_items_that_fall_outside_its_span() -> None:
    window: SlidingWindow[str] = SlidingWindow(span_seconds=10.0)

    window.append(at(0), "old")
    window.append(at(5), "mid")
    window.append(at(12), "new")

    assert window.items == ("mid", "new"), "the 12s item should evict the 0s item"


def test_eviction_follows_event_time_not_wall_time() -> None:
    """A replay faster than real time must window identically to a live feed.

    The window is inclusive at both ends: with the newest item at t=19 and a 5s
    span, the cutoff is t=14 and that item is retained, so the retained set spans
    exactly ``span_seconds``.
    """
    window: SlidingWindow[int] = SlidingWindow(span_seconds=5.0)

    for second in range(20):
        window.append(at(second), second)

    assert window.items == (14, 15, 16, 17, 18, 19)
    assert window.span == pytest.approx(5.0)


def test_window_reports_its_extremes_and_span() -> None:
    window: SlidingWindow[str] = SlidingWindow(span_seconds=30.0)
    window.append(at(0), "first")
    window.append(at(8), "last")

    assert window.oldest == "first"
    assert window.newest == "last"
    assert window.span == pytest.approx(8.0)


def test_an_empty_window_is_falsy_and_reports_no_span() -> None:
    window: SlidingWindow[str] = SlidingWindow()

    assert not window
    assert len(window) == 0
    assert window.newest is None
    assert window.oldest is None
    assert window.span == 0.0


def test_a_non_positive_span_is_rejected() -> None:
    with pytest.raises(ValueError, match="span_seconds must be positive"):
        SlidingWindow(span_seconds=0.0)


# -- TelemetryWindow ----------------------------------------------------------


def test_kill_delta_counts_each_side_correctly() -> None:
    window = TelemetryWindow(span_seconds=60.0)

    window.observe(make_kill(1, victim_team=Team.T))
    window.observe(make_kill(2, victim_team=Team.T))
    window.observe(make_kill(3, victim_team=Team.CT))

    metrics = window.metrics()

    assert metrics.kills_ct == 2, "a T victim means a CT got the kill"
    assert metrics.kills_t == 1
    assert metrics.kill_delta == 1


def test_headshot_rate_is_computed_over_window_kills() -> None:
    window = TelemetryWindow(span_seconds=60.0)

    window.observe(make_kill(1, headshot=True))
    window.observe(make_kill(2, headshot=True))
    window.observe(make_kill(3, headshot=False))
    window.observe(make_kill(4, headshot=False))

    assert window.metrics().headshot_rate == pytest.approx(0.5)


def test_metrics_are_zero_before_any_kills() -> None:
    metrics = TelemetryWindow().metrics()

    assert metrics.total_kills == 0
    assert metrics.kill_delta == 0
    assert metrics.headshot_rate == 0.0
    assert metrics.kills_per_second == 0.0


def test_a_round_start_clears_accumulated_momentum() -> None:
    """Last round's kills say nothing about this round: everyone respawns."""
    window = TelemetryWindow(span_seconds=600.0)
    window.observe(make_kill(1))
    window.observe(make_kill(2))
    assert window.metrics().total_kills == 2

    window.observe(
        RoundStartEvent(
            match_id="m-1", timestamp=at(3), sequence=99, round_number=2, score_ct=1, score_t=0
        )
    )

    assert window.metrics().total_kills == 0
    assert window.current_round == 2


def test_old_kills_leave_the_window_as_the_match_clock_advances() -> None:
    window = TelemetryWindow(span_seconds=10.0)

    window.observe(make_kill(0))
    window.observe(make_kill(5))
    window.observe(make_kill(20))

    assert window.metrics().total_kills == 1


def test_dense_ticks_do_not_evict_sparse_kills() -> None:
    """Ticks and kills are windowed separately, so a tick burst cannot hide kills."""
    window = TelemetryWindow(span_seconds=30.0)
    simulator = MatchSimulator(seed=5, tick_rate_hz=8.0, start_time=BASE_TIME)

    window.observe(make_kill(0))
    ticks = 0
    for event in simulator.run():
        if isinstance(event, TickEvent):
            window.observe(event)
            ticks += 1
            if ticks >= 50:
                break

    assert window.metrics().kills_ct == 1, "50 ticks must not evict the kill"


# -- TelemetryConsumer --------------------------------------------------------


async def test_the_consumer_dispatches_events_to_a_typed_handler() -> None:
    settings = Settings(broker_backend="memory")
    seen: list[int] = []

    async def on_kill(event: KillEvent) -> None:  # type: ignore[override]
        seen.append(event.sequence)

    async with InMemoryBroker() as broker:
        consumer = TelemetryConsumer(broker=broker, settings=settings)
        consumer.on(EventType.KILL, on_kill)  # type: ignore[arg-type]

        task = asyncio.create_task(consumer.run(max_events=2))
        await asyncio.sleep(0.01)

        await broker.publish(settings.kafka_telemetry_topic, serialise_event(make_kill(1)))
        await broker.publish(settings.kafka_telemetry_topic, serialise_event(make_kill(2)))

        stats = await asyncio.wait_for(task, timeout=2.0)

    assert stats.events_consumed == 2
    assert seen == [10, 20]


async def test_a_wildcard_handler_receives_every_event_type() -> None:
    settings = Settings(broker_backend="memory")
    kinds: list[str] = []

    async def on_any(event: object) -> None:
        kinds.append(event.event_type.value)  # type: ignore[attr-defined]

    async with InMemoryBroker() as broker:
        consumer = TelemetryConsumer(broker=broker, settings=settings)
        consumer.on(None, on_any)  # type: ignore[arg-type]

        task = asyncio.create_task(consumer.run(max_events=2))
        await asyncio.sleep(0.01)

        await broker.publish(settings.kafka_telemetry_topic, serialise_event(make_kill(1)))
        await broker.publish(
            settings.kafka_telemetry_topic,
            serialise_event(
                RoundEndEvent(
                    match_id="m-1",
                    timestamp=at(2),
                    sequence=2,
                    round_number=1,
                    winner=Team.CT,
                    reason=RoundEndReason.T_ELIMINATED,
                    score_ct=1,
                    score_t=0,
                )
            ),
        )

        await asyncio.wait_for(task, timeout=2.0)

    assert kinds == ["kill", "round_end"]


async def test_malformed_payloads_are_counted_and_skipped() -> None:
    """One corrupt message must not end the run."""
    settings = Settings(broker_backend="memory")

    async with InMemoryBroker() as broker:
        consumer = TelemetryConsumer(broker=broker, settings=settings)
        task = asyncio.create_task(consumer.run(max_events=1))
        await asyncio.sleep(0.01)

        await broker.publish(settings.kafka_telemetry_topic, b'{"event_type":"nonsense"}')
        await broker.publish(settings.kafka_telemetry_topic, serialise_event(make_kill(1)))

        stats = await asyncio.wait_for(task, timeout=2.0)

    assert stats.events_failed == 1
    assert stats.events_consumed == 1
    assert stats.success_rate == pytest.approx(0.5)


async def test_a_failing_handler_does_not_stop_ingestion() -> None:
    settings = Settings(broker_backend="memory")
    survived: list[int] = []

    async def explodes(event: object) -> None:
        raise RuntimeError("handler bug")

    async def records(event: object) -> None:
        survived.append(event.sequence)  # type: ignore[attr-defined]

    async with InMemoryBroker() as broker:
        consumer = TelemetryConsumer(broker=broker, settings=settings)
        consumer.on(None, explodes)  # type: ignore[arg-type]
        consumer.on(None, records)  # type: ignore[arg-type]

        task = asyncio.create_task(consumer.run(max_events=2))
        await asyncio.sleep(0.01)

        await broker.publish(settings.kafka_telemetry_topic, serialise_event(make_kill(1)))
        await broker.publish(settings.kafka_telemetry_topic, serialise_event(make_kill(2)))

        stats = await asyncio.wait_for(task, timeout=2.0)

    assert stats.events_consumed == 2
    assert survived == [10, 20], "the healthy handler still ran"
    assert stats.handler_errors == 2


async def test_stop_requests_a_graceful_shutdown() -> None:
    settings = Settings(broker_backend="memory")

    async with InMemoryBroker() as broker:
        consumer = TelemetryConsumer(broker=broker, settings=settings)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.01)

        await broker.publish(settings.kafka_telemetry_topic, serialise_event(make_kill(1)))
        await asyncio.sleep(0.02)

        assert consumer.is_stopping is False
        consumer.stop()
        assert consumer.is_stopping is True

        await broker.publish(settings.kafka_telemetry_topic, serialise_event(make_kill(2)))
        stats = await asyncio.wait_for(task, timeout=2.0)

    assert stats.events_consumed == 1, "the event after stop() must not be handled"


async def test_run_until_signalled_completes_without_signal_support() -> None:
    """Windows event loops reject add_signal_handler; the run must proceed anyway."""
    settings = Settings(broker_backend="memory")

    async with InMemoryBroker() as broker:
        consumer = TelemetryConsumer(broker=broker, settings=settings)
        task = asyncio.create_task(consumer.run_until_signalled(max_events=1))
        await asyncio.sleep(0.01)

        await broker.publish(settings.kafka_telemetry_topic, serialise_event(make_kill(1)))
        stats = await asyncio.wait_for(task, timeout=2.0)

    assert stats.events_consumed == 1


def test_success_rate_is_one_before_any_messages() -> None:
    from apexpulse.stream import ConsumerStats

    assert ConsumerStats().success_rate == 1.0


# -- MatchTracker -------------------------------------------------------------


async def test_the_tracker_persists_a_snapshot_per_tick() -> None:
    settings = Settings(broker_backend="memory", state_backend="memory")

    async with InMemoryStateStore() as store:
        tracker = MatchTracker(store=store, settings=settings)

        for event in MatchSimulator(seed=3, tick_rate_hz=2.0, start_time=BASE_TIME).run():
            await tracker.handle(event)
            if isinstance(event, TickEvent) and event.sequence > 20:
                break

        snapshot = await tracker.snapshot("apex-001")

    assert snapshot is not None
    assert snapshot["match_id"] == "apex-001"
    assert "state" in snapshot
    assert "momentum" in snapshot


async def test_the_stored_snapshot_carries_windowed_momentum() -> None:
    settings = Settings(state_backend="memory")

    async with InMemoryStateStore() as store:
        tracker = MatchTracker(store=store, settings=settings, window_seconds=60.0)
        window = tracker.window_for("m-1")
        window.observe(make_kill(1, victim_team=Team.T, headshot=True))

        tick = next(
            event
            for event in MatchSimulator(seed=3, tick_rate_hz=2.0, start_time=BASE_TIME).run()
            if isinstance(event, TickEvent)
        )
        await tracker._persist(tick, window)

        snapshot = await tracker.snapshot(tick.match_id)

    assert snapshot is not None
    assert snapshot["momentum"]["kills_ct"] == 1
    assert snapshot["momentum"]["headshot_rate"] == 1.0


async def test_live_match_ids_lists_tracked_matches() -> None:
    settings = Settings(state_backend="memory")

    async with InMemoryStateStore() as store:
        tracker = MatchTracker(store=store, settings=settings)
        await store.set(match_key("alpha"), {"match_id": "alpha"})
        await store.set(match_key("beta"), {"match_id": "beta"})
        await store.set("unrelated:key", {"noise": True})

        assert await tracker.live_match_ids() == ["alpha", "beta"]


async def test_a_finished_match_releases_its_window() -> None:
    settings = Settings(state_backend="memory")

    async with InMemoryStateStore() as store:
        tracker = MatchTracker(store=store, settings=settings)
        tracker.window_for("m-1").observe(make_kill(1))
        assert "m-1" in tracker._windows

        await tracker.handle(
            MatchEndEvent(
                match_id="m-1",
                timestamp=at(10),
                sequence=999,
                winner=Team.CT,
                score_ct=13,
                score_t=7,
            )
        )

        assert "m-1" not in tracker._windows


# -- End to end ---------------------------------------------------------------


async def test_a_replayed_match_flows_through_to_stored_state() -> None:
    """Producer to broker to consumer to state store, with nothing lost."""
    settings = Settings(broker_backend="memory", state_backend="memory")

    async with InMemoryBroker() as broker, InMemoryStateStore() as store:
        tracker = MatchTracker(store=store, settings=settings)
        consumer = TelemetryConsumer(broker=broker, settings=settings)
        consumer.on(None, tracker.handle)

        consumer_task = asyncio.create_task(consumer.run(max_events=400))
        await asyncio.sleep(0.01)

        replayer = TelemetryReplayer(broker=broker, settings=settings, speed=0.0)
        await replayer.replay(
            MatchSimulator(seed=9, tick_rate_hz=4.0, start_time=BASE_TIME), max_events=400
        )

        stats = await asyncio.wait_for(consumer_task, timeout=5.0)
        snapshot = await tracker.snapshot("apex-001")

    assert stats.events_consumed == 400
    assert stats.events_failed == 0
    assert snapshot is not None
    assert snapshot["state"]["match_id"] == "apex-001"
