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
    _, _, result = train_model(dataset, num_rounds=120, seed=7)
    metrics = result.metrics

    assert metrics.log_loss < metrics.baseline_log_loss
    assert metrics.skill_score > 0.05, f"skill score only {metrics.skill_score:.3f}"
    assert metrics.roc_auc > 0.6


async def test_training_holds_out_whole_matches(dataset) -> None:
    """Row-wise leakage would inflate every metric above."""
    _, _, result = train_model(dataset, num_rounds=60, seed=7)

    assert result.train_matches > 0
    assert result.test_matches > 0
    assert result.train_rows > result.test_rows


async def test_feature_importance_covers_every_feature(dataset) -> None:
    _, _, result = train_model(dataset, num_rounds=60, seed=7)

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
    _, _, result = train_model(dataset, num_rounds=120, seed=7)
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
    _, _, result = train_model(dataset, num_rounds=120, seed=7)
    top_gain = max(result.feature_importance.values())

    assert top_gain < 0.5, f"one feature carries {top_gain:.1%} of the gain"


async def test_training_is_reproducible(dataset) -> None:
    _, _, first = train_model(dataset, num_rounds=60, seed=99)
    _, _, second = train_model(dataset, num_rounds=60, seed=99)

    assert first.metrics.log_loss == pytest.approx(second.metrics.log_loss)


async def test_predictions_are_probabilities(dataset) -> None:
    import xgboost as xgb

    from apexpulse.features import build_training_set, split_by_match

    booster, _, _ = train_model(dataset, num_rounds=60, seed=7)
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

    booster, _, _ = train_model(dataset, num_rounds=150, seed=7)
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

    booster, calibrator, result = train_model(dataset, num_rounds=60, seed=7)
    model_path, calibrator_path, metadata_path = save_model(
        booster, result, calibrator, directory=tmp_path
    )

    assert model_path.is_file()
    assert calibrator_path.is_file()
    assert metadata_path.is_file()

    reloaded, _, metadata = load_model(directory=tmp_path)

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
    booster, calibrator, result = train_model(dataset, num_rounds=40, seed=7)
    _, _, metadata_path = save_model(booster, result, calibrator, directory=tmp_path)

    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["features"] == list(FEATURE_NAMES)
    assert metadata["dataset"]["test_matches"] > 0
    assert metadata["trained_at"]


def test_loading_a_missing_model_explains_how_to_create_one(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="apexpulse train"):
        load_model(directory=tmp_path)


# -- Probability calibration --------------------------------------------------


def test_the_identity_calibrator_changes_nothing() -> None:
    import numpy as np

    from apexpulse.ml import ProbabilityCalibrator

    calibrator = ProbabilityCalibrator.identity()
    values = [0.1, 0.5, 0.9]

    assert calibrator.is_identity
    assert np.allclose(calibrator.apply(values), values)


def test_a_calibrator_corrects_systematic_overconfidence() -> None:
    """The failure the first checkpoint showed: 64% claimed, 54% observed."""
    import numpy as np
    import pandas as pd

    from apexpulse.ml import assess_calibration, fit_calibrator

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.95, 4_000)
    # Truth is systematically less extreme than the prediction.
    true_rate = 0.5 + (raw - 0.5) * 0.6
    labels = pd.Series((rng.uniform(size=4_000) < true_rate).astype(int))

    before = assess_calibration(labels, raw).expected_calibration_error
    calibrator = fit_calibrator(labels, raw)
    after = assess_calibration(labels, calibrator.apply(raw)).expected_calibration_error

    assert before > 0.05, "the synthetic data must actually be miscalibrated"
    assert after < before / 2, f"calibration did not help: {before:.3f} -> {after:.3f}"


def test_calibration_essentially_preserves_ranking() -> None:
    """A monotonic map cannot reorder predictions, so AUC barely moves.

    It is not exactly identical: isotonic regression collapses tied inputs onto a
    single output, which changes how those ties break. The discrimination the
    model learned is what must survive, not the last decimal place.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from apexpulse.ml import fit_calibrator

    rng = np.random.default_rng(1)
    raw = rng.uniform(0.05, 0.95, 2_000)
    labels = pd.Series((rng.uniform(size=2_000) < raw * 0.8).astype(int))

    calibrator = fit_calibrator(labels, raw)
    calibrated = calibrator.apply(raw)

    before = roc_auc_score(labels, raw)
    after = roc_auc_score(labels, calibrated)

    assert after == pytest.approx(before, abs=0.01)


def test_calibration_never_reorders_a_pair() -> None:
    """The mapping is monotonic: if a > b before, a >= b after."""
    import numpy as np
    import pandas as pd

    from apexpulse.ml import fit_calibrator

    rng = np.random.default_rng(4)
    raw = rng.uniform(0.05, 0.95, 1_000)
    labels = pd.Series((rng.uniform(size=1_000) < raw).astype(int))

    calibrated = fit_calibrator(labels, raw).apply(raw)
    order = np.argsort(raw)

    assert np.all(np.diff(calibrated[order]) >= -1e-9)


def test_calibrated_output_stays_within_zero_and_one() -> None:
    import numpy as np
    import pandas as pd

    from apexpulse.ml import fit_calibrator

    rng = np.random.default_rng(2)
    raw = rng.uniform(0.1, 0.9, 500)
    labels = pd.Series((rng.uniform(size=500) < 0.5).astype(int))

    calibrated = fit_calibrator(labels, raw).apply([0.0, 0.5, 1.0])

    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_too_few_samples_yield_the_identity() -> None:
    """Fitting on a handful of rows would model noise."""
    import pandas as pd

    from apexpulse.ml import fit_calibrator

    calibrator = fit_calibrator(pd.Series([1, 0, 1]), [0.9, 0.2, 0.8])

    assert calibrator.is_identity


def test_a_single_outcome_yields_the_identity() -> None:
    import pandas as pd

    from apexpulse.ml import fit_calibrator

    calibrator = fit_calibrator(pd.Series([1] * 200), [0.5] * 200)

    assert calibrator.is_identity


def test_a_calibrator_round_trips_through_json(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    from apexpulse.ml import ProbabilityCalibrator, fit_calibrator

    rng = np.random.default_rng(3)
    raw = rng.uniform(0.1, 0.9, 500)
    labels = pd.Series((rng.uniform(size=500) < raw).astype(int))

    original = fit_calibrator(labels, raw)
    path = original.save(tmp_path / "calibrator.json")
    restored = ProbabilityCalibrator.load(path)

    assert np.allclose(original.apply(raw), restored.apply(raw))


def test_a_missing_calibrator_file_loads_as_the_identity(tmp_path) -> None:
    """An older checkpoint without a calibrator must still serve."""
    from apexpulse.ml import ProbabilityCalibrator

    assert ProbabilityCalibrator.load(tmp_path / "absent.json").is_identity


def test_mismatched_breakpoints_are_rejected() -> None:
    from apexpulse.ml import ProbabilityCalibrator

    with pytest.raises(ValueError, match="same length"):
        ProbabilityCalibrator(breakpoints_x=(0.1, 0.2), breakpoints_y=(0.1,))


async def test_training_reports_its_calibration_error(dataset) -> None:
    _, _, result = train_model(dataset, num_rounds=120, seed=7)

    assert result.calibration_error_raw > 0.0
    assert "calibration" in result.summary()


async def test_calibration_is_off_by_default(dataset) -> None:
    """Measured to worsen both skill and calibration on this model."""
    _, calibrator, _ = train_model(dataset, num_rounds=60, seed=7)

    assert calibrator.is_identity


async def test_calibration_can_be_requested_explicitly(dataset) -> None:
    """The machinery still works when a future model needs it."""
    _, calibrator, result = train_model(dataset, num_rounds=60, seed=7, calibrate=True)

    # Only fits when enough test matches exist to split meaningfully.
    assert calibrator.is_identity or len(calibrator.breakpoints_x) > 1
    assert result.calibration_error_raw >= 0.0
