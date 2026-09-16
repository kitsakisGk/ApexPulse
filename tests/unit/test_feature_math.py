"""Mathematical validation of the feature vector.

The tests in ``test_features.py`` cover behaviour at chosen points. These assert
the *properties* that must hold everywhere: symmetry, monotonicity, boundedness,
and exact agreement between the offline and online paths — including momentum,
which is the feature most easily left inert.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import pytest

from apexpulse.config import Settings
from apexpulse.features import (
    FEATURE_NAMES,
    FeatureExtractor,
    build_feature_frame,
)
from apexpulse.producer import MatchSimulator
from apexpulse.schemas import constants
from apexpulse.schemas.enums import MapName, RoundPhase, Team
from apexpulse.schemas.models import MatchState, PlayerState, RoundState, TeamEconomy
from apexpulse.storage import DuckDBSink
from apexpulse.stream.window import WindowedMetrics

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

# Features that describe a side-relative balance and must therefore flip sign
# when the two teams are swapped.
SYMMETRIC_FEATURES = (
    "alive_delta",
    "alive_ratio",
    "health_ratio",
    "money_ratio",
    "equipment_ratio",
    "loss_streak_delta",
    "score_delta",
    "kill_delta_window",
)


def build_state(
    *,
    alive_ct: int,
    alive_t: int,
    health_ct: int = 100,
    health_t: int = 100,
    money_ct: int = 5_000,
    money_t: int = 5_000,
    equipment_ct: int = 10_000,
    equipment_t: int = 10_000,
    losses_ct: int = 0,
    losses_t: int = 0,
    score_ct: int = 0,
    score_t: int = 0,
    seconds_remaining: float = 90.0,
    bomb_planted: bool = False,
    bomb_seconds: float | None = None,
) -> MatchState:
    """Build a snapshot with independently controlled per-side values."""
    players = tuple(
        PlayerState(
            player_id=f"ct{i}",
            name=f"CT{i}",
            team=Team.CT,
            health=health_ct if i < alive_ct else 0,
            money=1_000,
        )
        for i in range(constants.PLAYERS_PER_TEAM)
    ) + tuple(
        PlayerState(
            player_id=f"t{i}",
            name=f"T{i}",
            team=Team.T,
            health=health_t if i < alive_t else 0,
            money=1_000,
        )
        for i in range(constants.PLAYERS_PER_TEAM)
    )

    phase = RoundPhase.BOMB_PLANTED if bomb_planted else RoundPhase.LIVE
    return MatchState(
        match_id="m-1",
        map_name=MapName.MIRAGE,
        timestamp=BASE_TIME,
        score_ct=score_ct,
        score_t=score_t,
        round_state=RoundState(
            round_number=max(1, score_ct + score_t + 1),
            phase=phase,
            seconds_remaining=seconds_remaining,
            bomb_planted=bomb_planted,
            bomb_seconds_remaining=bomb_seconds,
        ),
        players=players,
        economy_ct=TeamEconomy(
            team=Team.CT,
            money=money_ct,
            equipment_value=equipment_ct,
            consecutive_losses=losses_ct,
        ),
        economy_t=TeamEconomy(
            team=Team.T,
            money=money_t,
            equipment_value=equipment_t,
            consecutive_losses=losses_t,
        ),
    )


def mirror(**kwargs: object) -> dict[str, object]:
    """Swap every CT value with its T counterpart."""
    swapped: dict[str, object] = {}
    for key, value in kwargs.items():
        if key.endswith("_ct"):
            swapped[key.removesuffix("_ct") + "_t"] = value
        elif key.endswith("_t"):
            swapped[key.removesuffix("_t") + "_ct"] = value
        else:
            swapped[key] = value
    return swapped


# -- Symmetry -----------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"alive_ct": 5, "alive_t": 2},
        {"alive_ct": 1, "alive_t": 4},
        {"alive_ct": 3, "alive_t": 3, "money_ct": 16_000, "money_t": 800},
        {"alive_ct": 3, "alive_t": 3, "equipment_ct": 25_000, "equipment_t": 3_000},
        {"alive_ct": 3, "alive_t": 3, "losses_ct": 5, "losses_t": 1},
        {"alive_ct": 3, "alive_t": 3, "score_ct": 11, "score_t": 4},
        {"alive_ct": 4, "alive_t": 4, "health_ct": 30, "health_t": 100},
    ],
)
def test_swapping_sides_negates_every_symmetric_feature(params: dict) -> None:
    """A feature describing balance must be antisymmetric under a side swap.

    If it is not, the model can learn which side is which rather than who is
    ahead, and would fail the moment teams switch at half time.
    """
    extractor = FeatureExtractor()

    original = extractor.extract(build_state(**params)).as_dict()
    swapped = extractor.extract(build_state(**mirror(**params))).as_dict()

    for name in SYMMETRIC_FEATURES:
        assert original[name] == pytest.approx(-swapped[name], abs=1e-12), (
            f"{name} is not antisymmetric: {original[name]} vs {swapped[name]}"
        )


def test_an_even_position_yields_zero_for_every_symmetric_feature() -> None:
    """With both sides identical, every balance feature must read exactly 0."""
    vector = FeatureExtractor().extract(build_state(alive_ct=3, alive_t=3)).as_dict()

    for name in SYMMETRIC_FEATURES:
        assert vector[name] == 0.0, f"{name}={vector[name]} on an even position"


# -- Monotonicity -------------------------------------------------------------


def test_alive_features_increase_with_each_surviving_ct() -> None:
    extractor = FeatureExtractor()

    values = [
        extractor.extract(build_state(alive_ct=n, alive_t=3))["alive_ratio"]
        for n in range(1, constants.PLAYERS_PER_TEAM + 1)
    ]

    assert values == sorted(values)
    assert values[0] < values[-1]


def test_money_ratio_increases_as_ct_gets_richer() -> None:
    extractor = FeatureExtractor()

    values = [
        extractor.extract(build_state(alive_ct=3, alive_t=3, money_ct=money, money_t=8_000))[
            "money_ratio"
        ]
        for money in (0, 2_000, 8_000, 16_000, 40_000)
    ]

    assert values == sorted(values)


def test_time_fraction_decreases_as_the_clock_runs_down() -> None:
    extractor = FeatureExtractor()

    values = [
        extractor.extract(build_state(alive_ct=3, alive_t=3, seconds_remaining=seconds))[
            "time_fraction"
        ]
        for seconds in (115.0, 90.0, 45.0, 10.0, 0.0)
    ]

    assert values == sorted(values, reverse=True)


# -- Boundedness over random input --------------------------------------------


def test_no_feature_escapes_its_bounds_under_random_states() -> None:
    """Fuzz the extractor: a NaN or an unbounded value must never appear."""
    rng = random.Random(1234)
    extractor = FeatureExtractor()

    for _ in range(400):
        planted = rng.random() < 0.3
        state = build_state(
            alive_ct=rng.randint(0, 5),
            alive_t=rng.randint(0, 5),
            health_ct=rng.randint(1, 100),
            health_t=rng.randint(1, 100),
            money_ct=rng.randint(0, constants.MAX_MONEY),
            money_t=rng.randint(0, constants.MAX_MONEY),
            equipment_ct=rng.randint(0, 30_000),
            equipment_t=rng.randint(0, 30_000),
            losses_ct=rng.randint(0, 9),
            losses_t=rng.randint(0, 9),
            score_ct=rng.randint(0, 12),
            score_t=rng.randint(0, 12),
            seconds_remaining=0.0 if planted else rng.uniform(0.0, constants.ROUND_SECONDS),
            bomb_planted=planted,
            bomb_seconds=rng.uniform(0.0, constants.BOMB_TIMER_SECONDS) if planted else None,
        )
        metrics = WindowedMetrics(
            kills_ct=rng.randint(0, 5),
            kills_t=rng.randint(0, 5),
            headshots=rng.randint(0, 5),
            ticks=rng.randint(0, 200),
            span_seconds=rng.uniform(0.0, 30.0),
        )

        for name, value in extractor.extract(state, metrics).as_dict().items():
            assert math.isfinite(value), f"{name} is not finite: {value}"
            assert -1.0 <= value <= 1.0, f"{name}={value} escaped [-1, 1]"


def test_extreme_inputs_saturate_rather_than_overflow() -> None:
    """Values beyond the expected range clamp instead of leaking out of bounds."""
    extractor = FeatureExtractor()

    vector = extractor.extract(
        build_state(
            alive_ct=5,
            alive_t=0,
            money_ct=constants.MAX_MONEY * 5,
            money_t=0,
            losses_ct=99,
            losses_t=0,
            score_ct=32,
            score_t=0,
        ),
        WindowedMetrics(kills_ct=999, kills_t=0, span_seconds=0.001),
    ).as_dict()

    for name, value in vector.items():
        assert -1.0 <= value <= 1.0, f"{name}={value}"


# -- Offline / online parity, including momentum ------------------------------


async def test_persisted_momentum_reproduces_the_online_features() -> None:
    """The offline path must recompute momentum, not emit zeros.

    Momentum was previously dropped on the floor when a tick was persisted, so
    two of fifteen features were constant zero in training while the live model
    received real values — a silent train/serve skew.
    """
    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=5_000) as sink:
        for event in MatchSimulator(seed=21, tick_rate_hz=4.0, start_time=BASE_TIME).run():
            await sink.handle(event)
        await sink.flush()

        frame = sink.training_frame()

    assert {"kills_ct_window", "kills_t_window", "window_seconds"} <= set(frame.columns)

    features = build_feature_frame(frame)

    assert features["kill_delta_window"].abs().max() > 0.0, "momentum is still inert"
    assert features["engagement_pace"].max() > 0.0
    assert features["kill_delta_window"].between(-1.0, 1.0).all()
    assert features["engagement_pace"].between(0.0, 1.0).all()


async def test_offline_momentum_matches_the_extractor_row_for_row() -> None:
    """Recompute a sample of stored rows through the online path and compare."""
    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=5_000) as sink:
        for event in MatchSimulator(seed=33, tick_rate_hz=4.0, start_time=BASE_TIME).run():
            await sink.handle(event)
        await sink.flush()

        frame = sink.training_frame()

    offline = build_feature_frame(frame)
    extractor = FeatureExtractor()

    sampled = frame.sample(n=min(150, len(frame)), random_state=7)
    for position, (_, row) in enumerate(sampled.iterrows()):
        metrics = WindowedMetrics(
            kills_ct=int(row["kills_ct_window"]),
            kills_t=int(row["kills_t_window"]),
            span_seconds=float(row["window_seconds"]),
        )
        state = build_state(
            alive_ct=int(row["alive_ct"]),
            alive_t=int(row["alive_t"]),
            money_ct=int(row["money_ct"]),
            money_t=int(row["money_t"]),
            equipment_ct=int(row["equipment_ct"]),
            equipment_t=int(row["equipment_t"]),
            losses_ct=int(row["losses_ct"]),
            losses_t=int(row["losses_t"]),
            score_ct=int(row["score_ct"]),
            score_t=int(row["score_t"]),
            seconds_remaining=float(row["seconds_remaining"]),
            bomb_planted=bool(row["bomb_planted"]),
            bomb_seconds=(float(row["bomb_seconds"]) if row["bomb_planted"] else None),
        )
        online = extractor.extract(state, metrics).as_dict()
        stored = offline.loc[sampled.index[position]]

        for name in ("kill_delta_window", "engagement_pace", "alive_ratio", "money_ratio"):
            assert stored[name] == pytest.approx(online[name], abs=1e-9), (
                f"row {position}: {name} offline={stored[name]} online={online[name]}"
            )


def test_missing_momentum_columns_fall_back_to_zero() -> None:
    """An older database without the window columns must still be readable."""
    import pandas as pd

    row = {
        "alive_ct": 5,
        "alive_t": 3,
        "health_ct": 500,
        "health_t": 300,
        "seconds_remaining": 60.0,
        "bomb_planted": False,
        "bomb_seconds": None,
        "money_ct": 5_000,
        "money_t": 4_000,
        "equipment_ct": 10_000,
        "equipment_t": 8_000,
        "losses_ct": 0,
        "losses_t": 1,
        "score_ct": 3,
        "score_t": 2,
    }

    features = build_feature_frame(pd.DataFrame([row]))

    assert features["kill_delta_window"].iloc[0] == 0.0
    assert features["engagement_pace"].iloc[0] == 0.0
    assert list(features.columns) == list(FEATURE_NAMES)


# -- Dataset quality ----------------------------------------------------------


async def test_no_feature_is_constant_across_a_real_dataset() -> None:
    """A zero-variance column teaches the model nothing and skews importances."""
    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=5_000) as sink:
        for seed in range(4):
            simulator = MatchSimulator(
                match_id=f"m-{seed}", seed=seed, tick_rate_hz=2.0, start_time=BASE_TIME
            )
            for event in simulator.run():
                await sink.handle(event)
        await sink.flush()

        frame = sink.training_frame()

    features = build_feature_frame(frame)
    constant = [name for name in FEATURE_NAMES if features[name].std() == 0.0]

    assert not constant, f"constant features carry no signal: {constant}"


async def test_no_feature_contains_a_missing_value() -> None:
    """A NaN reaching the model is a silent corruption, not a training error."""
    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=5_000) as sink:
        for event in MatchSimulator(seed=8, tick_rate_hz=2.0, start_time=BASE_TIME).run():
            await sink.handle(event)
        await sink.flush()

        features = build_feature_frame(sink.training_frame())

    assert not features.isna().to_numpy().any()


async def test_the_dataset_splits_cleanly_along_match_boundaries() -> None:
    from apexpulse.features import split_by_match

    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=5_000) as sink:
        for seed in range(5):
            simulator = MatchSimulator(
                match_id=f"m-{seed}",
                seed=seed,
                tick_rate_hz=2.0,
                start_time=BASE_TIME + timedelta(hours=seed),
            )
            for event in simulator.run():
                await sink.handle(event)
        await sink.flush()

        frame = sink.training_frame()

    train, test = split_by_match(frame, test_fraction=0.4)

    assert set(train["match_id"]).isdisjoint(set(test["match_id"]))
    assert len(train) > 0
    assert len(test) > 0
    assert train["ct_won"].nunique() == 2, "both outcomes must survive the split"
