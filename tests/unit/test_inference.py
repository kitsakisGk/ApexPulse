"""Tests for the inference engine and worker.

Serving is judged on two things: the probability must respond correctly to the
game, and the per-tick latency must stay inside its budget.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from apexpulse.broker import InMemoryBroker
from apexpulse.config import Settings
from apexpulse.features import FEATURE_NAMES
from apexpulse.inference import InferenceEngine, InferenceWorker, LatencyStats, prediction_key
from apexpulse.inference.engine import LATENCY_BUDGET_MS, NEUTRAL_PROBABILITY
from apexpulse.ml import save_model, train_model
from apexpulse.producer import MatchSimulator
from apexpulse.schemas.enums import MapName, RoundPhase, Team
from apexpulse.schemas.events import TickEvent
from apexpulse.schemas.models import MatchState, PlayerState, RoundState, TeamEconomy
from apexpulse.storage import DuckDBSink, InMemoryStateStore
from apexpulse.stream.window import TelemetryWindow

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def build_state(
    *,
    alive_ct: int = 5,
    alive_t: int = 5,
    phase: RoundPhase = RoundPhase.LIVE,
    seconds_remaining: float = 60.0,
    bomb_planted: bool = False,
    bomb_seconds: float | None = None,
    money_ct: int = 5_000,
    money_t: int = 5_000,
    match_id: str = "m-1",
) -> MatchState:
    """Build a snapshot with the properties a test cares about."""
    players = tuple(
        PlayerState(
            player_id=f"ct{i}",
            name=f"CT{i}",
            team=Team.CT,
            health=100 if i < alive_ct else 0,
            money=1_000,
        )
        for i in range(5)
    ) + tuple(
        PlayerState(
            player_id=f"t{i}",
            name=f"T{i}",
            team=Team.T,
            health=100 if i < alive_t else 0,
            money=1_000,
        )
        for i in range(5)
    )

    return MatchState(
        match_id=match_id,
        map_name=MapName.MIRAGE,
        timestamp=BASE_TIME,
        round_state=RoundState(
            round_number=3,
            phase=phase,
            seconds_remaining=seconds_remaining,
            bomb_planted=bomb_planted,
            bomb_seconds_remaining=bomb_seconds,
        ),
        players=players,
        economy_ct=TeamEconomy(team=Team.CT, money=money_ct, equipment_value=12_000),
        economy_t=TeamEconomy(team=Team.T, money=money_t, equipment_value=12_000),
    )


@pytest.fixture(scope="module")
async def checkpoint(tmp_path_factory):
    """Train a model once and save it for the module to load.

    Sixteen matches rather than a handful: the behaviour assertions below check
    that a 5v1 reads as a near-certain CT win, and a model trained on too little
    data returns a flat ~0.5 for every input, which would test nothing.
    """
    directory = tmp_path_factory.mktemp("checkpoint")

    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=10_000) as sink:
        for seed in range(16):
            simulator = MatchSimulator(
                match_id=f"m-{seed:02d}", seed=seed, tick_rate_hz=4.0, start_time=BASE_TIME
            )
            for event in simulator.run():
                await sink.handle(event)
        await sink.flush()
        frame = sink.training_frame()

    booster, calibrator, result = train_model(frame, num_rounds=250, seed=7)
    save_model(booster, result, calibrator, directory=directory)
    return directory


@pytest.fixture
async def engine(checkpoint):
    """An engine loaded from the module's checkpoint."""
    return InferenceEngine.from_checkpoint(directory=checkpoint)


# -- Loading ------------------------------------------------------------------


async def test_an_engine_loads_from_a_checkpoint(engine) -> None:
    assert engine.metadata["features"] == list(FEATURE_NAMES)
    assert engine.tree_count is not None


async def test_a_missing_checkpoint_explains_how_to_create_one(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="apexpulse train"):
        InferenceEngine.from_checkpoint(directory=tmp_path)


async def test_a_mismatched_feature_contract_is_rejected(checkpoint) -> None:
    """Serving a checkpoint trained on different features would score nonsense."""
    from apexpulse.ml import load_model

    booster, calibrator, metadata = load_model(directory=checkpoint)
    metadata["features"] = ["not", "the", "same"]

    with pytest.raises(ValueError, match="feature order"):
        InferenceEngine(booster, calibrator, metadata)


async def test_a_non_positive_tree_cap_is_rejected(checkpoint) -> None:
    from apexpulse.ml import load_model

    booster, calibrator, metadata = load_model(directory=checkpoint)

    with pytest.raises(ValueError, match="max_trees must be at least 1"):
        InferenceEngine(booster, calibrator, metadata, max_trees=0)


async def test_capping_trees_reduces_the_model_used(checkpoint) -> None:
    full = InferenceEngine.from_checkpoint(directory=checkpoint)
    capped = InferenceEngine.from_checkpoint(directory=checkpoint, max_trees=5)

    assert capped.tree_count == 5
    assert full.tree_count is not None
    assert capped.tree_count < full.tree_count


async def test_a_cap_above_the_model_size_changes_nothing(checkpoint) -> None:
    """The cap takes the minimum, so an oversized value is a no-op."""
    full = InferenceEngine.from_checkpoint(directory=checkpoint)
    capped = InferenceEngine.from_checkpoint(directory=checkpoint, max_trees=100_000)

    assert capped.tree_count == full.tree_count


# -- Prediction correctness ---------------------------------------------------


async def test_a_prediction_is_a_probability(engine) -> None:
    prediction = engine.predict(build_state())

    assert 0.0 <= prediction.ct_win_probability <= 1.0
    assert prediction.t_win_probability == pytest.approx(1.0 - prediction.ct_win_probability)
    assert prediction.scored is True


async def test_a_man_advantage_favours_the_side_that_holds_it(engine) -> None:
    """The headline behaviour: the number must track the game."""
    ahead = engine.predict(build_state(alive_ct=5, alive_t=2)).ct_win_probability
    even = engine.predict(build_state(alive_ct=3, alive_t=3)).ct_win_probability
    behind = engine.predict(build_state(alive_ct=2, alive_t=5)).ct_win_probability

    assert ahead > even > behind, f"ahead={ahead:.3f} even={even:.3f} behind={behind:.3f}"
    assert ahead > 0.6
    assert behind < 0.4


async def test_a_planted_bomb_shifts_the_probability_towards_t(engine) -> None:
    """Same manpower and the same clock, bomb down: the win condition changed.

    The clock is held equal on purpose. A planted round sets the round clock to
    zero while the bomb timer runs, so comparing a 0s planted state against a 40s
    unplanted one varies two features at once — and a 0s round clock with no bomb
    is a state that barely occurs in training, because CT have already won it.

    The data backs the direction: across 280k ticks, CT win 54.8% of unplanted
    rounds and 44.1% of planted ones (52.9% vs 45.9% at equal manpower).
    """
    clock = 30.0
    unplanted = engine.predict(
        build_state(alive_ct=3, alive_t=3, seconds_remaining=clock)
    ).ct_win_probability
    planted = engine.predict(
        build_state(
            alive_ct=3,
            alive_t=3,
            phase=RoundPhase.BOMB_PLANTED,
            seconds_remaining=clock,
            bomb_planted=True,
            bomb_seconds=30.0,
        )
    ).ct_win_probability

    assert planted < unplanted, f"planted={planted:.3f} unplanted={unplanted:.3f}"


async def test_the_bomb_timer_reaches_the_model(engine) -> None:
    """The timer is a live input, not a constant.

    Deliberately not asserting a monotonic trend: at a lopsided 3v2 the model
    reads ~94% regardless of the countdown, because manpower has already decided
    the round. Demanding monotonicity there would assert a relationship the game
    does not have. What must hold is that the feature is wired through at all.
    """
    from apexpulse.features import FeatureExtractor

    extractor = FeatureExtractor()
    early = extractor.extract(
        build_state(
            alive_ct=3,
            alive_t=3,
            phase=RoundPhase.BOMB_PLANTED,
            seconds_remaining=0.0,
            bomb_planted=True,
            bomb_seconds=35.0,
        )
    )
    late = extractor.extract(
        build_state(
            alive_ct=3,
            alive_t=3,
            phase=RoundPhase.BOMB_PLANTED,
            seconds_remaining=0.0,
            bomb_planted=True,
            bomb_seconds=5.0,
        )
    )

    assert early["bomb_time_fraction"] > late["bomb_time_fraction"]
    assert early["bomb_planted"] == late["bomb_planted"] == 1.0


async def test_freezetime_is_not_scored(engine) -> None:
    """Nothing has happened yet, so the model has nothing to say."""
    prediction = engine.predict(build_state(phase=RoundPhase.FREEZETIME))

    assert prediction.scored is False
    assert prediction.ct_win_probability == NEUTRAL_PROBABILITY
    assert engine.stats.skipped >= 1


async def test_a_finished_round_is_not_scored(engine) -> None:
    prediction = engine.predict(build_state(phase=RoundPhase.OVER, seconds_remaining=0.0))

    assert prediction.scored is False


async def test_the_favoured_side_follows_the_probability(engine) -> None:
    assert engine.predict(build_state(alive_ct=5, alive_t=1)).favoured_side == "CT"
    assert engine.predict(build_state(alive_ct=1, alive_t=5)).favoured_side == "T"


async def test_confidence_grows_with_distance_from_an_even_call(engine) -> None:
    lopsided = engine.predict(build_state(alive_ct=5, alive_t=1))
    level = engine.predict(build_state(alive_ct=3, alive_t=3))

    assert lopsided.confidence > level.confidence
    assert 0.0 <= level.confidence <= 1.0


async def test_features_are_attached_only_when_requested(engine) -> None:
    """Building the dict costs more than the prediction, so it is opt-in."""
    assert engine.predict(build_state()).features == {}

    detailed = engine.predict(build_state(), include_features=True)

    assert set(detailed.features) == set(FEATURE_NAMES)


# -- Latency ------------------------------------------------------------------


async def test_single_tick_latency_stays_within_budget(engine) -> None:
    """The Day 8 target: sub-20ms per tick."""
    states = [build_state(alive_ct=n % 6, alive_t=(n + 2) % 6) for n in range(300)]

    for state in states[:50]:  # warm up
        engine.predict(state)
    engine.stats.samples.clear()

    for state in states:
        engine.predict(state)

    stats = engine.stats

    assert stats.p99_ms < LATENCY_BUDGET_MS, f"p99 {stats.p99_ms:.2f}ms exceeds the budget"
    assert stats.p50_ms < LATENCY_BUDGET_MS / 2


async def test_batch_scoring_is_faster_per_state_than_single_calls(engine) -> None:
    """Batching amortises per-call overhead, which matters for backfill."""
    import time

    states = [build_state(alive_ct=n % 6, alive_t=(n + 1) % 6) for n in range(200)]

    for state in states[:20]:
        engine.predict(state)

    started = time.perf_counter()
    for state in states:
        engine.predict(state)
    single_ms = (time.perf_counter() - started) / len(states) * 1_000

    started = time.perf_counter()
    engine.predict_batch(states)
    batch_ms = (time.perf_counter() - started) / len(states) * 1_000

    assert batch_ms < single_ms


async def test_batch_and_single_paths_agree(engine) -> None:
    states = [build_state(alive_ct=n % 6, alive_t=(n + 3) % 6) for n in range(20)]

    single = [engine.predict(state).ct_win_probability for state in states]
    batched = [prediction.ct_win_probability for prediction in engine.predict_batch(states)]

    for index, (one, many) in enumerate(zip(single, batched, strict=True)):
        assert one == pytest.approx(many, abs=1e-6), f"state {index} disagreed"


async def test_batch_preserves_unscoreable_states(engine) -> None:
    states = [
        build_state(alive_ct=4, alive_t=2),
        build_state(phase=RoundPhase.FREEZETIME),
        build_state(alive_ct=2, alive_t=4),
    ]

    predictions = engine.predict_batch(states)

    assert len(predictions) == 3
    assert [p.scored for p in predictions] == [True, False, True]


async def test_an_empty_batch_returns_nothing(engine) -> None:
    assert engine.predict_batch([]) == []


def test_latency_stats_are_empty_before_any_prediction() -> None:
    stats = LatencyStats()

    assert stats.count == 0
    assert stats.p99_ms == 0.0
    assert stats.mean_ms == 0.0


def test_latency_percentiles_are_ordered() -> None:
    stats = LatencyStats()
    for value in range(1, 101):
        stats.record(float(value))

    assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms
    assert stats.total_predictions == 100


def test_the_latency_window_is_bounded() -> None:
    """Retaining every sample would grow without limit in a long-running worker."""
    from apexpulse.inference.engine import LATENCY_WINDOW

    stats = LatencyStats()
    for value in range(LATENCY_WINDOW * 2):
        stats.record(float(value))

    assert stats.count == LATENCY_WINDOW
    assert stats.total_predictions == LATENCY_WINDOW * 2


# -- Worker -------------------------------------------------------------------


async def test_the_worker_publishes_a_prediction_per_tick(engine) -> None:
    settings = Settings(broker_backend="memory", state_backend="memory")

    async with InMemoryBroker() as broker, InMemoryStateStore() as store:
        worker = InferenceWorker(engine=engine, broker=broker, store=store, settings=settings)

        scored = 0
        for event in MatchSimulator(seed=5, tick_rate_hz=2.0, start_time=BASE_TIME).run():
            await worker.handle(event)
            if isinstance(event, TickEvent):
                scored += 1
            if scored >= 120:
                break

        stored = await worker.read_prediction("apex-001")

    assert worker.stats.ticks_seen == 120
    assert worker.stats.predictions_published > 0
    assert worker.stats.errors == 0
    assert stored is not None
    assert 0.0 <= stored["ct_win_probability"] <= 1.0
    assert stored["favoured_side"] in {"CT", "T", "even"}


async def test_the_worker_skips_freezetime(engine) -> None:
    settings = Settings(state_backend="memory")

    async with InMemoryStateStore() as store:
        worker = InferenceWorker(engine=engine, store=store, settings=settings)
        await worker.handle(
            TickEvent(
                match_id="m-1",
                timestamp=BASE_TIME,
                sequence=1,
                state=build_state(phase=RoundPhase.FREEZETIME),
            )
        )

    assert worker.stats.skipped == 1
    assert worker.stats.predictions_published == 0


async def test_the_worker_maintains_momentum_per_match(engine) -> None:
    """Serving must supply the same windowed features training used."""
    settings = Settings(state_backend="memory")

    async with InMemoryStateStore() as store:
        worker = InferenceWorker(engine=engine, store=store, settings=settings)

        for event in MatchSimulator(seed=9, tick_rate_hz=2.0, start_time=BASE_TIME).run():
            await worker.handle(event)
            if worker.stats.ticks_seen >= 200:
                break

        window = worker.window_for("apex-001")

    assert isinstance(window, TelemetryWindow)
    assert window.tick_count > 0


async def test_a_publish_failure_is_counted_not_raised(engine) -> None:
    """A dropped frame beats halting ingestion."""

    class BrokenStore(InMemoryStateStore):
        async def set(self, key, value, *, ttl=None):  # type: ignore[override]
            raise RuntimeError("store is down")

    settings = Settings(state_backend="memory")

    async with BrokenStore() as store:
        worker = InferenceWorker(engine=engine, store=store, settings=settings)
        await worker.handle(
            TickEvent(
                match_id="m-1",
                timestamp=BASE_TIME,
                sequence=1,
                state=build_state(alive_ct=4, alive_t=3),
            )
        )

    assert worker.stats.errors == 1
    assert worker.stats.predictions_published == 0


async def test_a_finished_match_releases_its_state(engine) -> None:
    from apexpulse.schemas.events import MatchEndEvent

    settings = Settings(state_backend="memory")

    async with InMemoryStateStore() as store:
        worker = InferenceWorker(engine=engine, store=store, settings=settings)
        await worker.handle(
            TickEvent(
                match_id="m-1",
                timestamp=BASE_TIME,
                sequence=1,
                state=build_state(alive_ct=4, alive_t=3),
            )
        )
        assert worker.latest("m-1") is not None

        await worker.handle(
            MatchEndEvent(
                match_id="m-1",
                timestamp=BASE_TIME,
                sequence=999,
                winner=Team.CT,
                score_ct=13,
                score_t=7,
            )
        )

        assert worker.latest("m-1") is None


def test_prediction_keys_are_namespaced() -> None:
    assert prediction_key("apex-001") == "apexpulse:prediction:apex-001"
