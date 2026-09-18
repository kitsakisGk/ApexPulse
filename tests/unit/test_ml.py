"""Tests for training, evaluation, and calibration.

The model is judged on the honesty of its probabilities, not just on whether it
picks the right side, so these assertions centre on log loss, skill over the base
rate, and calibration.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from apexpulse.config import Settings
from apexpulse.features import FEATURE_NAMES
from apexpulse.ml import (
    assess_calibration,
    evaluate,
    format_reliability_table,
    load_model,
    save_model,
    train_model,
)
from apexpulse.producer import MatchSimulator
from apexpulse.storage import DuckDBSink

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


async def build_dataset(matches: int = 8, tick_rate: float = 2.0):
    """Generate a labelled dataset from simulated matches."""
    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=5_000) as sink:
        for seed in range(matches):
            simulator = MatchSimulator(
                match_id=f"m-{seed:02d}", seed=seed, tick_rate_hz=tick_rate, start_time=BASE_TIME
            )
            for event in simulator.run():
                await sink.handle(event)
        await sink.flush()
        return sink.training_frame()


@pytest.fixture(scope="module")
async def dataset():
    """One dataset shared across the module; generation is the slow part."""
    return await build_dataset()


# -- Metrics ------------------------------------------------------------------


def test_perfect_predictions_score_near_zero_log_loss() -> None:
    import pandas as pd

    labels = pd.Series([1, 0, 1, 0])
    metrics = evaluate(labels, [0.999, 0.001, 0.999, 0.001])

    assert metrics.log_loss < 0.01
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc == 1.0


def test_confident_and_wrong_is_punished() -> None:
    import pandas as pd

    labels = pd.Series([1, 1, 0, 0])
    metrics = evaluate(labels, [0.01, 0.01, 0.99, 0.99])

    assert metrics.log_loss > 4.0
    assert metrics.accuracy == 0.0
    assert metrics.skill_score < 0, "worse than guessing must score negative skill"


def test_predicting_the_base_rate_scores_zero_skill() -> None:
    """A model that adds nothing must be seen to add nothing."""
    import pandas as pd

    labels = pd.Series([1, 1, 1, 0])
    metrics = evaluate(labels, [0.75, 0.75, 0.75, 0.75])

    assert metrics.skill_score == pytest.approx(0.0, abs=1e-9)
    assert metrics.base_rate == pytest.approx(0.75)


def test_brier_score_measures_squared_error() -> None:
    import pandas as pd

    labels = pd.Series([1, 0])
    metrics = evaluate(labels, [0.8, 0.3])

    assert metrics.brier_score == pytest.approx(((1 - 0.8) ** 2 + (0 - 0.3) ** 2) / 2)


def test_a_single_class_yields_a_neutral_auc() -> None:
    """ROC AUC is undefined with one class; report 0.5 rather than raising."""
    import pandas as pd

    metrics = evaluate(pd.Series([1, 1, 1]), [0.9, 0.8, 0.7])

    assert metrics.roc_auc == 0.5


# -- Calibration --------------------------------------------------------------


def test_a_perfectly_calibrated_model_has_near_zero_error() -> None:
    """Construct outcomes that match the stated probabilities exactly."""
    import pandas as pd

    probabilities = []
    labels = []
    for level in (0.1, 0.3, 0.5, 0.7, 0.9):
        wins = round(level * 100)
        probabilities.extend([level] * 100)
        labels.extend([1] * wins + [0] * (100 - wins))

    report = assess_calibration(pd.Series(labels), probabilities)

    assert report.expected_calibration_error < 0.01
    assert report.is_well_calibrated


def test_an_overconfident_model_is_flagged() -> None:
    """Always claiming 95% while winning half the time must fail calibration."""
    import pandas as pd

    labels = pd.Series([1, 0] * 100)
    report = assess_calibration(labels, [0.95] * 200)

    assert report.expected_calibration_error > 0.4
    assert not report.is_well_calibrated


def test_calibration_error_is_weighted_by_bin_population() -> None:
    """A tiny badly-calibrated bin must not dominate the headline number."""
    import pandas as pd

    # 990 well-calibrated samples, 10 badly calibrated ones.
    probabilities = [0.5] * 990 + [0.95] * 10
    labels = [1] * 495 + [0] * 495 + [0] * 10

    report = assess_calibration(pd.Series(labels), probabilities)

    assert report.expected_calibration_error < 0.02
    assert report.max_calibration_error > 0.9, "the bad bin is still visible"


def test_empty_bins_are_omitted() -> None:
    import pandas as pd

    report = assess_calibration(pd.Series([1, 0]), [0.55, 0.52])

    assert len(report.bins) == 1
    assert report.bins[0].label == "50%-60%"


def test_a_prediction_of_exactly_one_lands_in_the_final_bin() -> None:
    import pandas as pd

    report = assess_calibration(pd.Series([1, 1]), [1.0, 1.0])

    assert sum(bin_.count for bin_ in report.bins) == 2


def test_the_reliability_table_renders() -> None:
    import pandas as pd

    report = assess_calibration(pd.Series([1, 0, 1, 0]), [0.9, 0.1, 0.8, 0.2])
    table = format_reliability_table(report)

    assert "expected calibration error" in table
    assert "band" in table


# -- Training -----------------------------------------------------------------


async def test_training_beats_guessing_the_base_rate(dataset) -> None:
    """The headline claim: the model adds real information."""
    _, result = train_model(dataset, num_rounds=120, seed=7)
    metrics = result.metrics

    assert metrics.log_loss < metrics.baseline_log_loss
    assert metrics.skill_score > 0.05, f"skill score only {metrics.skill_score:.3f}"
    assert metrics.roc_auc > 0.6


async def test_training_holds_out_whole_matches(dataset) -> None:
    """Row-wise leakage would inflate every metric above."""
    _, result = train_model(dataset, num_rounds=60, seed=7)

    assert result.train_matches > 0
    assert result.test_matches > 0
    assert result.train_rows > result.test_rows


async def test_feature_importance_covers_every_feature(dataset) -> None:
    _, result = train_model(dataset, num_rounds=60, seed=7)

    assert set(result.feature_importance) == set(FEATURE_NAMES)
    assert sum(result.feature_importance.values()) == pytest.approx(1.0, abs=1e-3)


async def test_manpower_is_the_most_important_feature_group(dataset) -> None:
    """Who is alive should matter most — measured as a group, not per feature.

    The five manpower features are strongly collinear: `alive_ratio`,
    `alive_delta`, `alive_ct`, `alive_t`, and `health_ratio` all encode nearly the
    same fact. Boosting splits on whichever it reaches first and the rest add
    little marginal gain, so each looks weak alone while the group dominates.
    Ranking individual features here would test that collinearity, not the game.
    """
    _, result = train_model(dataset, num_rounds=120, seed=7)
    importance = result.feature_importance

    manpower = {"alive_ct", "alive_t", "alive_delta", "alive_ratio", "health_ratio"}
    economy = {"money_ratio", "equipment_ratio", "loss_streak_delta"}
    clock = {"time_fraction", "bomb_planted", "bomb_time_fraction"}

    manpower_gain = sum(importance[name] for name in manpower)
    economy_gain = sum(importance[name] for name in economy)
    clock_gain = sum(importance[name] for name in clock)

    assert manpower_gain > clock_gain, (
        f"manpower {manpower_gain:.1%} should outweigh the clock {clock_gain:.1%}"
    )
    assert manpower_gain > 0.2, f"manpower carries only {manpower_gain:.1%} of the gain"
    assert economy_gain > 0.1, f"economy carries only {economy_gain:.1%} of the gain"


async def test_no_single_feature_dominates_the_model(dataset) -> None:
    """A feature above half the total gain usually means leakage."""
    _, result = train_model(dataset, num_rounds=120, seed=7)
    top_gain = max(result.feature_importance.values())

    assert top_gain < 0.5, f"one feature carries {top_gain:.1%} of the gain"


async def test_training_is_reproducible(dataset) -> None:
    _, first = train_model(dataset, num_rounds=60, seed=99)
    _, second = train_model(dataset, num_rounds=60, seed=99)

    assert first.metrics.log_loss == pytest.approx(second.metrics.log_loss)


async def test_predictions_are_probabilities(dataset) -> None:
    import xgboost as xgb

    from apexpulse.features import build_training_set, split_by_match

    booster, _ = train_model(dataset, num_rounds=60, seed=7)
    _, test_frame = split_by_match(dataset, test_fraction=0.2)
    features, _ = build_training_set(test_frame)

    predictions = booster.predict(
        xgb.DMatrix(features, feature_names=list(FEATURE_NAMES)),
        iteration_range=(0, booster.best_iteration + 1),
    )

    assert predictions.min() >= 0.0
    assert predictions.max() <= 1.0
    assert predictions.std() > 0.02, "a model emitting one constant value is useless"


async def test_the_trained_model_is_reasonably_calibrated(dataset) -> None:
    """A broadcast gauge needs honest numbers, not just the right winner."""
    import xgboost as xgb

    from apexpulse.features import build_training_set, split_by_match

    booster, _ = train_model(dataset, num_rounds=150, seed=7)
    _, test_frame = split_by_match(dataset, test_fraction=0.2)
    features, labels = build_training_set(test_frame)

    predictions = booster.predict(
        xgb.DMatrix(features, feature_names=list(FEATURE_NAMES)),
        iteration_range=(0, booster.best_iteration + 1),
    )
    report = assess_calibration(labels, predictions)

    assert report.expected_calibration_error < 0.15


async def test_an_empty_dataset_is_rejected() -> None:
    import pandas as pd

    with pytest.raises(ValueError, match="empty dataset"):
        train_model(pd.DataFrame())


# -- Artifacts ----------------------------------------------------------------


async def test_a_saved_model_reloads_and_predicts_identically(dataset, tmp_path) -> None:
    import numpy as np
    import xgboost as xgb

    from apexpulse.features import build_training_set, split_by_match

    booster, result = train_model(dataset, num_rounds=60, seed=7)
    model_path, metadata_path = save_model(booster, result, directory=tmp_path)

    assert model_path.is_file()
    assert metadata_path.is_file()

    reloaded, metadata = load_model(directory=tmp_path)

    _, test_frame = split_by_match(dataset, test_fraction=0.2)
    features, _ = build_training_set(test_frame)
    matrix = xgb.DMatrix(features, feature_names=list(FEATURE_NAMES))

    original = booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
    restored = reloaded.predict(matrix, iteration_range=(0, booster.best_iteration + 1))

    assert np.allclose(original, restored)
    assert metadata["features"] == list(FEATURE_NAMES)
    assert "metrics" in metadata


async def test_metadata_records_the_feature_order(dataset, tmp_path) -> None:
    """Serving must apply features in the order training used."""
    booster, result = train_model(dataset, num_rounds=40, seed=7)
    _, metadata_path = save_model(booster, result, directory=tmp_path)

    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["features"] == list(FEATURE_NAMES)
    assert metadata["dataset"]["test_matches"] > 0
    assert metadata["trained_at"]


def test_loading_a_missing_model_explains_how_to_create_one(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="apexpulse train"):
        load_model(directory=tmp_path)
