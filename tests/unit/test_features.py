"""Tests for feature extraction.

Features are where a model quietly goes wrong, so these assertions cover three
things: the arithmetic is correct, the vector is bounded, and the offline and
online paths agree exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from apexpulse.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FeatureExtractor,
    FeatureVector,
    build_feature_frame,
    build_training_set,
    describe_features,
    is_scoreable,
    split_by_match,
)
from apexpulse.schemas import constants
from apexpulse.schemas.enums import MapName, RoundPhase, Team
from apexpulse.schemas.models import MatchState, PlayerState, RoundState, TeamEconomy
from apexpulse.stream.window import WindowedMetrics

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def make_state(
    *,
    alive_ct: int = 5,
    alive_t: int = 5,
    health: int = 100,
    money_ct: int = 5_000,
    money_t: int = 5_000,
    equipment_ct: int = 12_000,
    equipment_t: int = 12_000,
    losses_ct: int = 0,
    losses_t: int = 0,
    score_ct: int = 0,
    score_t: int = 0,
    phase: RoundPhase = RoundPhase.LIVE,
    seconds_remaining: float = 115.0,
    bomb_planted: bool = False,
    bomb_seconds: float | None = None,
) -> MatchState:
    """Build a snapshot with precisely the properties a test cares about."""
    players = tuple(
        PlayerState(
            player_id=f"ct{i}",
            name=f"CT{i}",
            team=Team.CT,
            health=health if i < alive_ct else 0,
            money=1_000,
        )
        for i in range(constants.PLAYERS_PER_TEAM)
    ) + tuple(
        PlayerState(
            player_id=f"t{i}",
            name=f"T{i}",
            team=Team.T,
            health=health if i < alive_t else 0,
            money=1_000,
        )
        for i in range(constants.PLAYERS_PER_TEAM)
    )

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


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


# -- Vector shape -------------------------------------------------------------


def test_the_vector_has_the_declared_width(extractor: FeatureExtractor) -> None:
    vector = extractor.extract(make_state())

    assert len(vector) == FEATURE_COUNT
    assert len(FEATURE_NAMES) == FEATURE_COUNT


def test_feature_names_are_unique() -> None:
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_a_vector_of_the_wrong_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 15 features"):
        FeatureVector((0.0, 1.0))


def test_features_are_addressable_by_name(extractor: FeatureExtractor) -> None:
    vector = extractor.extract(make_state(alive_ct=5, alive_t=2))

    assert vector["alive_delta"] == pytest.approx(0.6)
    assert vector.as_dict()["alive_delta"] == pytest.approx(0.6)


def test_an_unknown_feature_name_raises(extractor: FeatureExtractor) -> None:
    with pytest.raises(KeyError):
        extractor.extract(make_state())["not_a_feature"]


# -- Bounds -------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        make_state(),
        make_state(alive_ct=0, alive_t=5),
        make_state(alive_ct=5, alive_t=0),
        make_state(money_ct=0, money_t=constants.MAX_MONEY),
        make_state(money_ct=0, money_t=0, equipment_ct=0, equipment_t=0),
        make_state(losses_ct=5, losses_t=0),
        make_state(score_ct=13, score_t=0),
        make_state(bomb_planted=True, bomb_seconds=40.0, phase=RoundPhase.BOMB_PLANTED),
        make_state(seconds_remaining=0.0),
    ],
)
def test_every_feature_stays_within_unit_bounds(state: MatchState) -> None:
    """No feature may dominate by scale, so all live in [-1, 1]."""
    vector = FeatureExtractor().extract(state)

    for name, value in vector.as_dict().items():
        assert -1.0 <= value <= 1.0, f"{name}={value} escaped [-1, 1]"


def test_a_zeroed_economy_is_a_tie_not_a_crash(extractor: FeatureExtractor) -> None:
    """Both sides broke: the ratio is undefined, and must resolve to 0.0."""
    vector = extractor.extract(make_state(money_ct=0, money_t=0))

    assert vector["money_ratio"] == 0.0


# -- Manpower -----------------------------------------------------------------


def test_alive_delta_is_symmetric(extractor: FeatureExtractor) -> None:
    ahead = extractor.extract(make_state(alive_ct=4, alive_t=2))["alive_delta"]
    behind = extractor.extract(make_state(alive_ct=2, alive_t=4))["alive_delta"]

    assert ahead == pytest.approx(-behind)


def test_health_distinguishes_wounded_from_healthy(extractor: FeatureExtractor) -> None:
    """Five players at 20 HP is a weaker position than five at full."""
    healthy = extractor.extract(make_state(alive_ct=5, alive_t=5, health=100))
    wounded = extractor.extract(make_state(alive_ct=5, alive_t=5, health=100))

    assert healthy["alive_delta"] == wounded["alive_delta"]

    # Same alive counts, different health: only health_ratio should move.
    lopsided = make_state(alive_ct=5, alive_t=5)
    players = tuple(
        PlayerState(player_id=f"ct{i}", name=f"CT{i}", team=Team.CT, health=20, money=0)
        for i in range(5)
    ) + tuple(
        PlayerState(player_id=f"t{i}", name=f"T{i}", team=Team.T, health=100, money=0)
        for i in range(5)
    )
    hurt = lopsided.model_copy(update={"players": players})

    assert extractor.extract(hurt)["health_ratio"] < 0


# -- Clock --------------------------------------------------------------------


def test_time_fraction_falls_as_the_round_runs_down(extractor: FeatureExtractor) -> None:
    early = extractor.extract(make_state(seconds_remaining=115.0))["time_fraction"]
    late = extractor.extract(make_state(seconds_remaining=10.0))["time_fraction"]

    assert early == pytest.approx(1.0)
    assert late < early


def test_the_bomb_flag_and_timer_move_together(extractor: FeatureExtractor) -> None:
    unplanted = extractor.extract(make_state())
    planted = extractor.extract(
        make_state(
            phase=RoundPhase.BOMB_PLANTED,
            bomb_planted=True,
            bomb_seconds=20.0,
            seconds_remaining=0.0,
        )
    )

    assert unplanted["bomb_planted"] == 0.0
    assert unplanted["bomb_time_fraction"] == 0.0
    assert planted["bomb_planted"] == 1.0
    assert planted["bomb_time_fraction"] == pytest.approx(0.5)


# -- Economy ------------------------------------------------------------------


def test_money_ratio_favours_the_richer_side(extractor: FeatureExtractor) -> None:
    rich_ct = extractor.extract(make_state(money_ct=15_000, money_t=3_000))

    assert rich_ct["money_ratio"] > 0


def test_loss_streak_delta_tracks_the_difference(extractor: FeatureExtractor) -> None:
    vector = extractor.extract(make_state(losses_ct=4, losses_t=0))

    assert vector["loss_streak_delta"] == pytest.approx(0.8)


# -- Momentum -----------------------------------------------------------------


def test_momentum_is_zero_without_a_window(extractor: FeatureExtractor) -> None:
    vector = extractor.extract(make_state())

    assert vector["kill_delta_window"] == 0.0
    assert vector["engagement_pace"] == 0.0


def test_momentum_reflects_recent_kills(extractor: FeatureExtractor) -> None:
    metrics = WindowedMetrics(kills_ct=3, kills_t=1, headshots=2, ticks=40, span_seconds=10.0)

    vector = extractor.extract(make_state(), metrics)

    assert vector["kill_delta_window"] == pytest.approx(0.4)
    assert vector["engagement_pace"] == pytest.approx(0.4)


def test_momentum_can_be_disabled() -> None:
    metrics = WindowedMetrics(kills_ct=5, kills_t=0, span_seconds=5.0)

    vector = FeatureExtractor(include_momentum=False).extract(make_state(), metrics)

    assert vector["kill_delta_window"] == 0.0


def test_an_extreme_kill_burst_saturates_rather_than_escaping(
    extractor: FeatureExtractor,
) -> None:
    metrics = WindowedMetrics(kills_ct=50, kills_t=0, span_seconds=1.0)

    vector = extractor.extract(make_state(), metrics)

    assert vector["kill_delta_window"] == 1.0
    assert vector["engagement_pace"] == 1.0


# -- Batch --------------------------------------------------------------------


def test_batch_extraction_preserves_order(extractor: FeatureExtractor) -> None:
    states = [make_state(alive_ct=n, alive_t=5) for n in range(1, 5)]

    vectors = extractor.extract_batch(states)

    assert [v["alive_ct"] for v in vectors] == [0.2, 0.4, 0.6, 0.8]


def test_mismatched_batch_lengths_are_rejected(extractor: FeatureExtractor) -> None:
    with pytest.raises(ValueError, match="same length"):
        extractor.extract_batch([make_state()], [None, None])


# -- Scoreability -------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        (RoundPhase.FREEZETIME, False),
        (RoundPhase.LIVE, True),
        (RoundPhase.OVER, False),
    ],
)
def test_only_live_phases_are_scoreable(phase: RoundPhase, expected: bool) -> None:
    assert is_scoreable(make_state(phase=phase)) is expected


def test_a_planted_round_is_scoreable() -> None:
    state = make_state(
        phase=RoundPhase.BOMB_PLANTED, bomb_planted=True, bomb_seconds=30.0, seconds_remaining=0.0
    )

    assert is_scoreable(state) is True


# -- Offline parity -----------------------------------------------------------


def _row_from_state(state: MatchState) -> dict[str, object]:
    """Flatten a state into the column shape of the training view."""
    return {
        "match_id": state.match_id,
        "alive_ct": state.alive_ct,
        "alive_t": state.alive_t,
        "health_ct": sum(p.health for p in state.players_on(Team.CT)),
        "health_t": sum(p.health for p in state.players_on(Team.T)),
        "seconds_remaining": state.round_state.seconds_remaining,
        "bomb_planted": state.round_state.bomb_planted,
        "bomb_seconds": state.round_state.bomb_seconds_remaining,
        "money_ct": state.economy_ct.money,
        "money_t": state.economy_t.money,
        "equipment_ct": state.economy_ct.equipment_value,
        "equipment_t": state.economy_t.equipment_value,
        "losses_ct": state.economy_ct.consecutive_losses,
        "losses_t": state.economy_t.consecutive_losses,
        "score_ct": state.score_ct,
        "score_t": state.score_t,
        "ct_won": 1,
    }


def test_offline_and_online_extraction_agree_exactly() -> None:
    """Train/serve skew is the failure this guards against.

    The offline path is vectorised for speed and the online path is per-tick, so
    the arithmetic exists twice. If the two ever disagree, a model trained on one
    would be scored with the other.
    """
    import pandas as pd

    states = [
        make_state(),
        make_state(alive_ct=1, alive_t=4, health=40),
        make_state(money_ct=16_000, money_t=800, losses_ct=0, losses_t=5),
        make_state(score_ct=11, score_t=9, seconds_remaining=12.5),
        make_state(
            phase=RoundPhase.BOMB_PLANTED,
            bomb_planted=True,
            bomb_seconds=17.5,
            seconds_remaining=0.0,
            alive_ct=2,
            alive_t=3,
        ),
        make_state(alive_ct=0, alive_t=5),
        make_state(money_ct=0, money_t=0, equipment_ct=0, equipment_t=0),
    ]

    offline = build_feature_frame(pd.DataFrame([_row_from_state(s) for s in states]))
    extractor = FeatureExtractor()

    for index, state in enumerate(states):
        online = extractor.extract(state).as_dict()
        for name in FEATURE_NAMES:
            assert offline.iloc[index][name] == pytest.approx(online[name], abs=1e-9), (
                f"state {index}: {name} offline={offline.iloc[index][name]} online={online[name]}"
            )


def test_the_offline_frame_has_exactly_the_declared_columns() -> None:
    import pandas as pd

    frame = build_feature_frame(pd.DataFrame([_row_from_state(make_state())]))

    assert list(frame.columns) == list(FEATURE_NAMES)


def test_building_a_training_set_splits_features_from_the_label() -> None:
    import pandas as pd

    rows = pd.DataFrame([_row_from_state(make_state()) for _ in range(3)])

    features, labels = build_training_set(rows)

    assert list(features.columns) == list(FEATURE_NAMES)
    assert labels.tolist() == [1, 1, 1]


def test_unlabelled_rows_are_rejected() -> None:
    import pandas as pd

    rows = pd.DataFrame([_row_from_state(make_state())]).drop(columns=["ct_won"])

    with pytest.raises(KeyError, match="training_data"):
        build_training_set(rows)


# -- Splitting ----------------------------------------------------------------


def test_splitting_keeps_whole_matches_on_one_side() -> None:
    """Row-wise splitting would leak: consecutive ticks are nearly identical."""
    import pandas as pd

    rows = pd.DataFrame(
        [{**_row_from_state(make_state()), "match_id": f"m-{n // 10}"} for n in range(100)]
    )

    train, test = split_by_match(rows, test_fraction=0.3)

    assert set(train["match_id"]) & set(test["match_id"]) == set()
    assert len(train) + len(test) == len(rows)


def test_splitting_requires_more_than_one_match() -> None:
    import pandas as pd

    rows = pd.DataFrame([_row_from_state(make_state())])

    with pytest.raises(ValueError, match="at least two matches"):
        split_by_match(rows)


def test_an_out_of_range_test_fraction_is_rejected() -> None:
    import pandas as pd

    rows = pd.DataFrame([{**_row_from_state(make_state()), "match_id": f"m-{n}"} for n in range(4)])

    with pytest.raises(ValueError, match="between 0 and 1"):
        split_by_match(rows, test_fraction=1.5)


def test_describe_reports_each_feature_range() -> None:
    import pandas as pd

    frame = build_feature_frame(
        pd.DataFrame([_row_from_state(make_state(alive_ct=n)) for n in range(1, 6)])
    )

    described = describe_features(frame)

    assert set(described) == set(FEATURE_NAMES)
    assert described["alive_ct"]["min"] == pytest.approx(0.2)
    assert described["alive_ct"]["max"] == pytest.approx(1.0)
